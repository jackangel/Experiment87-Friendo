"""
Training and fine-tuning loops with gradient accumulation.

Provides:
  * ``print_gate_stats``           — logging helper for gate/meta stats.
  * ``build_cosine_scheduler``     — linear-warmup + cosine-decay LR schedule.
  * ``run_pretraining``            — single-dataset pretraining.
  * ``run_pretraining_mixed``      — multi-dataset weighted pretraining.
  * ``run_finetuning``             — ChatML masked instruction tuning.
"""

import math
import random

import torch
import torch.nn.functional as F

from .data import (
    _detach_kv_list,
    apply_meta_compute_penalty,
    apply_gaussian_kl_penalty,
    token_generator_from_parquet,
    stream_chatml_from_json,
    mixed_token_generator_from_datasets,
)
from .generation import generate_block_recurrent
from .tokenizer import CHAT_START, CHAT_END


# =============================================================================
# ADVANCED DIAGNOSTICS (comprehensive health monitoring)
# =============================================================================

def print_detailed_diagnostics(model, iteration, optimizer, grad_norm=None, last_loss_tensor=None):
    """Print comprehensive diagnostic metrics every N steps to catch issues early.
    
    Monitors:
    - Loss decomposition (CE vs regularizers)
    - Gradient health (norms, layer-wise flow)
    - Router behavior (tau, logit gaps, diversity)
    - Per-layer gate dynamics (locking patterns, health distribution)
    - Learning dynamics (token variance, effective signal)
    
    Args:
        grad_norm: Total gradient norm (captured before zero_grad), or None
    """
    print(f"\n{'='*80}")
    print(f"[DETAILED DIAGNOSTICS - Step {iteration}]")
    print(f"{'='*80}")
    
    # -------------------------------------------------------------------------
    # 1. LOSS DECOMPOSITION
    # -------------------------------------------------------------------------
    print("\n[Loss Breakdown]")
    ce = getattr(model, "last_ce_loss", None)
    kl = getattr(model, "last_kl_loss", None)
    kl_w = getattr(model, "kl_weight", 0.0)
    
    if ce is not None:
        ce_val = ce.item()
        print(f"  CE (token loss):        {ce_val:.4f}")
        
        if kl is not None:
            kl_contrib = (kl.item() * kl_w)
            kl_pct = (kl_contrib / ce_val * 100) if ce_val > 0 else 0
            print(f"  KL contribution:        {kl_contrib:.4f}  ({kl_pct:.2f}% of CE)")
        
        if getattr(model, "enable_meta_routing", False):
            penalty = getattr(model, "last_meta_compute_penalty", None)
            penalty_w = getattr(model, "compute_penalty_weight", 0.0)
            if penalty is not None:
                penalty_contrib = penalty.item() * penalty_w
                penalty_pct = (penalty_contrib / ce_val * 100) if ce_val > 0 else 0
                print(f"  Penalty contribution:   {penalty_contrib:.4f}  ({penalty_pct:.2f}% of CE)")
            
            ent = getattr(model, "last_meta_entropy", None)
            ent_w = getattr(model.meta_phase, "entropy_weight", 0.0) if hasattr(model, "meta_phase") else 0.0
            if ent is not None:
                ent_contrib = ent.item() * ent_w
                ent_pct = (ent_contrib / ce_val * 100) if ce_val > 0 else 0
                print(f"  Entropy bonus:          {ent_contrib:.4f}  ({ent_pct:.2f}% of CE, SUBTRACTS)")
    
    # -------------------------------------------------------------------------
    # 2. GRADIENT HEALTH
    # -------------------------------------------------------------------------
    print("\n[Gradient Health]")
    if grad_norm is not None:
        total_norm = grad_norm if isinstance(grad_norm, float) else grad_norm.item()
        param_count = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"  Total gradient norm:    {total_norm:.4f}  ({param_count} params with grad)")
        
        if total_norm > 10.0:
            print(f"  ⚠️  WARNING: High gradient norm (may be unstable)")
        elif total_norm < 0.001:
            print(f"  ⚠️  WARNING: Very low gradient norm (may be vanishing)")
    else:
        print(f"  Total gradient norm:    N/A (gradients not captured)")
    
    # -------------------------------------------------------------------------
    # 3. ROUTER DETAILED DIAGNOSTICS
    # -------------------------------------------------------------------------
    if getattr(model, "enable_meta_routing", False) and model.meta_phase is not None:
        print("\n[Meta Router Behavior]")
        
        # Current tau (annealing) - compute from global_training_iteration
        mp = model.meta_phase
        global_step = model.global_training_iteration.item()
        if hasattr(mp, 'gumbel_tau_anneal_steps') and mp.gumbel_tau_anneal_steps > 0:
            progress = min(1.0, float(global_step) / mp.gumbel_tau_anneal_steps)
            current_tau = mp.gumbel_tau_start - progress * (mp.gumbel_tau_start - mp.gumbel_tau_end)
            print(f"  Gumbel tau (annealed):  {current_tau:.3f}  (progress: {progress*100:.1f}%, step={global_step})")
        else:
            print(f"  Gumbel tau (fixed):     {mp.gumbel_tau:.3f}  (step={global_step})")
        
        # Router entropy (per-token diversity)
        # Entropy H ∈ [0, log(2)≈0.693] for binary routing
        # H ≈ 0: collapsed (all tokens choose same option)
        # H ≈ 0.693: maximum diversity (50/50 split)
        entropy_per_step = getattr(mp, "last_entropy_per_step", None)
        if entropy_per_step:
            avg_entropy = sum(entropy_per_step) / len(entropy_per_step)
            max_entropy = 0.693  # log(2) for binary routing
            diversity_pct = (avg_entropy / max_entropy * 100) if max_entropy > 0 else 0
            print(f"  Routing diversity:      {avg_entropy:.4f} / {max_entropy:.3f}  ({diversity_pct:.1f}% of max)")
            if avg_entropy < 0.01:
                print(f"  ⚠️  WARNING: Router has collapsed (entropy near zero)")
        
        # Forced exploration status
        forced = getattr(mp, "last_forced_frac", 0.0)
        if forced > 0:
            print(f"  Forced exploration:     {forced*100:.1f}% of tokens")
    
    # -------------------------------------------------------------------------
    # 4. PER-LAYER GATE BREAKDOWN
    # -------------------------------------------------------------------------
    gates = _collect_all_gates(model)
    if gates and any(g.enable_ablation for _, g in gates):
        print("\n[Per-Layer Forgetting Gate Status]")
        print(f"  {'Layer':<12} {'Locked':<10} {'Health':<10} {'Consistency':<12} {'p* Req':<10}")
        print(f"  {'-'*60}")
        
        for label, gate in gates:
            if not gate.enable_ablation:
                continue
            
            locked_count = gate.is_locked.sum().item()
            locked_pct = (locked_count / gate.num_gated * 100) if gate.num_gated > 0 else 0
            health_avg = (gate.health.sum().item() / gate.num_gated) if gate.num_gated > 0 else 0
            p_obs = gate.consistency_ema.item()
            p_star = gate.p_lock_required
            
            status = "✓" if p_obs >= p_star else "✗"
            print(f"  {label:<12} {locked_pct:>5.1f}%     {health_avg*100:>5.1f}%     {p_obs:>6.3f} {status:<5}  {p_star:>6.3f}")
        
        # Summary stats
        all_health = []
        for _, gate in gates:
            if gate.enable_ablation:
                all_health.extend(gate.health.cpu().tolist())
        
        if all_health:
            import statistics
            health_min = min(all_health)
            health_max = max(all_health)
            health_std = statistics.stdev(all_health) if len(all_health) > 1 else 0
            print(f"\n  Health distribution:    min={health_min:.3f}, max={health_max:.3f}, std={health_std:.3f}")
    
    # -------------------------------------------------------------------------
    # 5. LEARNING DYNAMICS
    # -------------------------------------------------------------------------
    if last_loss_tensor is not None and hasattr(last_loss_tensor, 'shape'):
        print("\n[Learning Dynamics]")
        # Token-level loss variance (if per-token losses available)
        if last_loss_tensor.numel() > 1:
            token_mean = last_loss_tensor.mean().item()
            token_std = last_loss_tensor.std().item()
            token_max = last_loss_tensor.max().item()
            print(f"  Token loss variance:    mean={token_mean:.3f}, std={token_std:.3f}, max={token_max:.3f}")
            if token_std / token_mean > 1.5:
                print(f"  ℹ️  High variance: some tokens much harder than others")
    
    print(f"\n{'='*80}\n")


# =============================================================================
# TRAINING & FINE-TUNING LOOPS
# =============================================================================

def _collect_all_gates(model):
    """Collect EVERY CognitiveForgettingGate in the model graph.

    Phase 1 fix: the original logger only iterated ``model.layers[].mlp_forget_gate``
    and silently missed:
      * ``mlp_forget_gate_layer`` (the graph-reasoning path gate, which is the
        ACTIVE gate in deeper layers when graph reasoning is enabled)
      * the Meta-Dynamic phase's shared ``dynamic_layer`` gate (which sits
        outside ``model.layers`` and ticks up to max_meta_steps× per forward).

    Returns a list of (label, gate) tuples for complete reporting.
    """
    gates = []
    # Fixed-phase layers.
    for i, layer in enumerate(getattr(model, "layers", [])):
        mg = getattr(layer, "mlp_forget_gate", None)
        if mg is not None:
            gates.append((f"L{i:02d}/mlp", mg))
        # mlp_forget_gate_layer always exists; add it as a separate entry so
        # we see both, even when graph_reasoning is the active path.
        lg = getattr(layer, "mlp_forget_gate_layer", None)
        if lg is not None:
            gates.append((f"L{i:02d}/layer", lg))
    # Meta-phase shared dynamic layer (outside model.layers).
    mp = getattr(model, "meta_phase", None)
    if mp is not None:
        dyn = getattr(mp, "dynamic_layer", None)
        if dyn is not None:
            mg = getattr(dyn, "mlp_forget_gate", None)
            if mg is not None:
                gates.append(("META/dyn-mlp", mg))
            lg = getattr(dyn, "mlp_forget_gate_layer", None)
            if lg is not None:
                gates.append(("META/dyn-layer", lg))
    return gates


def print_gate_stats(model, iteration, running_loss, train_steps, scheduler, step_type="CLEAR"):
    current_lr = scheduler.get_last_lr()[0]
    log_str = f"[Step {iteration}] Type: {step_type:5s} | Loss: {running_loss / max(1, train_steps):.4f} | LR: {current_lr:.2e}"

    # --- CE-only loss + clean token PPL (Phase 1.4) ---
    # running_loss above may include KL + meta penalty + entropy bonus, so
    # raw e^loss is NOT a clean perplexity.  Report model.last_ce_loss if the
    # training loop captured it this pass.
    ce = getattr(model, "last_ce_loss", None)
    if ce is not None:
        import math as _math
        ppl = _math.exp(min(ce, 20.0))  # cap to avoid overflow
        log_str += f" | CE={ce:.4f} | PPL={ppl:.1f}"

    # --- Complete gate aggregation (Phase 1.2 + 1.5) ---
    # Iterate ALL gates (fixed mlp, fixed layer-path, meta dynamic layer) so
    # the reported locked/health fractions describe the whole system.
    gates = _collect_all_gates(model)
    total_locked, total_health, total_gated, total_wildcard = 0, 0, 0, 0
    cons_rates = []  # observed per-gate consistency p (EMA-smoothed)
    p_required = []  # theoretically required p* to lock, per gate
    for _label, gate in gates:
        if not gate.enable_ablation:
            continue
        total_locked += gate.is_locked.sum().item()
        total_health += gate.health.sum().item()
        total_gated += gate.num_gated
        total_wildcard += gate.num_wildcard
        cons_rates.append(float(gate.consistency_ema.item()))
        p_required.append(gate.p_lock_required)

    if total_gated > 0:
        locked_pct = (total_locked / total_gated) * 100
        health_avg = (total_health / total_gated) * 100
        log_str += f" | Gated: {total_gated} ({locked_pct:.2f}% locked, {health_avg:.1f}% health)"
        # Consistency summary: mean observed p vs the MINIMUM p* across gates
        # (the easiest layer to lock).  If observed < min(p*) everywhere, the
        # gate is provably dead — the Phase 1 verification gate.
        if cons_rates:
            import math as _math
            mean_p = sum(cons_rates) / len(cons_rates)
            finite_p_req = [v for v in p_required if not _math.isnan(v)]
            min_p_req = min(finite_p_req) if finite_p_req else float('nan')
            log_str += f" | Consistency: p_obs={mean_p:.3f} (need p*>={min_p_req:.3f} to lock)"

    # Meta-Dynamic Routing stats: average exit probability per step + penalty.
    if getattr(model, "enable_meta_routing", False) and model.meta_phase is not None:
        exit_stats = getattr(model.meta_phase, "last_exit_stats", None)
        penalty = getattr(model.meta_phase, "last_compute_penalty", None)
        if exit_stats:
            exit_str = "/".join(f"{p:.2f}" for p in exit_stats)
            avg_exit = sum(exit_stats) / len(exit_stats)
            log_str += f" | Meta[exit/step: {exit_str}, avg={avg_exit:.2f}]"
            if penalty is not None:
                log_str += f" | penalty={penalty.item():.3f} (w={model.compute_penalty_weight:.4f})"
                forced_frac = getattr(model.meta_phase, "last_forced_frac", 0.0)
                if forced_frac > 0.0:
                    # exit_stats already reflect only the router's free choices,
                    # so this is the fraction being forced to explore.
                    log_str += f" | forced={forced_frac:.2f}"
                # Entropy bonus diagnostics (full precision — Phase 1.3).
                # H=0.000 rounds to "noise"; H=0.0006 is a 99.99/0.01 split.
                # 6 decimals lets us tell float-noise (<1e-6) from learned
                # determinism (~1e-3 ⇒ gap ~9 nats).
                ent = getattr(model.meta_phase, "last_entropy", None)
                ew = getattr(model.meta_phase, "entropy_weight", 0.0)
                if ent is not None:
                    kill = getattr(model, "exit_prob_ema", 0.0)
                    log_str += f" | H={ent.item():.6f} (pen_ema={kill:.2f}"
                    if kill > getattr(model, "penalty_collapse_floor", 0.95):
                        log_str += ",KILL"
                    log_str += f", w={ew:.4f})"

    # Gaussian embedding KL divergence diagnostics.
    if getattr(model, "enable_gaussian_embeddings", False):
        kl = getattr(model, "last_kl_loss", None)
        kl_w = getattr(model, "kl_weight", 0.0)
        if kl is not None:
            log_str += f" | KL={kl.item():.4f} (w={kl_w:.4f})"

    print(log_str)


def build_cosine_scheduler(optimizer, start_iteration, peak_lr,
                           warmup_iters=200, max_iters=100000, min_lr_ratio=0.1,
                           forced_lr=None):
    """
    Create a LambdaLR scheduler implementing linear warmup + cosine decay.

    Schedule:
      • step [0, warmup_iters)         → linear ramp  peak_lr * (step+1)/warmup_iters
      • step [warmup_iters, max_iters) → cosine decay peak_lr → peak_lr * min_lr_ratio
      • step >= max_iters              → plateau at   peak_lr * min_lr_ratio

    Resets optimizer param_group LRs to `peak_lr` (scaled per-group by each
    param group's own ``lr_multiplier``, defaulting to 1.0) before binding
    the schedule, which makes checkpoint-resume correct (the
    optimizer_state_dict checkpoint contains an already-decayed LR that must
    NOT become the new base) while preserving any differential LR ratios
    between groups (e.g. a meta-router group intentionally trained at 0.3x
    the base LR).

    Args:
        optimizer:      The torch optimizer.
        start_iteration: When resuming, advance the scheduler to this step.
        peak_lr:        The peak LR reached at the end of warmup (== initial LR).
        warmup_iters:   Number of optimizer steps to linearly warm up over.
        max_iters:      Step at which the cosine reaches the minimum LR.
        min_lr_ratio:   Final LR as a fraction of peak_lr (e.g. 0.1 → 10%).
        forced_lr:      If set, overrides the entire warmup/cosine/plateau
            schedule with a constant LR. Each param group is reset to
            ``forced_lr * pg['lr_multiplier']`` and the schedule's lambda is
            pinned at 1.0, so the forced value survives every subsequent
            ``scheduler.step()`` call unchanged (including replay below and
            every step taken during training).

    Returns:
        torch.optim.lr_scheduler.LambdaLR
    """
    # Reset base LR so the schedule multiplies from the true peak (or the
    # forced override), preserving each group's own relative multiplier
    # instead of flattening every group to the same absolute value.
    #
    # IMPORTANT: also force-reset 'initial_lr' on every param group, not just
    # 'lr'. LambdaLR's __init__ does `group.setdefault('initial_lr', group['lr'])`,
    # which does NOT overwrite an existing 'initial_lr' key. If this optimizer
    # was restored via optimizer.load_state_dict() from an older checkpoint
    # that already had a scheduler bound to it, 'initial_lr' is still present
    # (baked in from whatever peak_lr was used the very first time), so the
    # new scheduler would silently keep computing its schedule from that
    # stale value forever -- completely ignoring the peak_lr/forced_lr passed
    # in here. Explicitly overwriting 'initial_lr' guarantees the fresh value
    # is actually used.
    base_value = forced_lr if forced_lr is not None else peak_lr
    for pg in optimizer.param_groups:
        pg['lr'] = base_value * pg.get('lr_multiplier', 1.0)
        pg['initial_lr'] = pg['lr']

    def lr_lambda(step):
        if forced_lr is not None:
            return 1.0
        if step < warmup_iters:
            return float(step + 1) / float(max(1, warmup_iters))
        if step >= max_iters:
            return min_lr_ratio
        progress = (step - warmup_iters) / float(max(1, max_iters - warmup_iters))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # Advance to the resume point so the LR is correct on continued training.
    for _ in range(start_iteration):
        scheduler.step()
    return scheduler


# =============================================================================
# 9A. Pre-training (Causal LM on Plain Text with Grad Accum)
# =============================================================================

def run_pretraining(model, parquet_files, text_column, tokenizer, optimizer, device,
                    vocab_size, start_iteration=0, chunk_size=512, enable_forgetting=False,
                    batch_size=4, grad_accum_steps=4,
                    peak_lr=4e-4, warmup_iters=200, max_iters=100000, min_lr_ratio=0.1,
                    forced_lr=None):

    print(f"\n--- Starting Pre-training (Parquet) | Device: {device} | Batch Size: {batch_size} | Grad Accum: {grad_accum_steps} ---")
    flash_status = 'Enabled' if model.use_flash_attn else 'Disabled (using Bayesian Signal Attention)'
    print(f"--- Flash Attention: {flash_status} ---")
    if forced_lr is not None:
        print(f"--- FORCED LR OVERRIDE ACTIVE: pinning base LR to {forced_lr:.2e} (schedule below is ignored) ---")
    else:
        print(f"--- LR Schedule: Warmup {warmup_iters} steps → Cosine decay to {min_lr_ratio*100:.0f}% over {max_iters} steps ---")

    iteration = start_iteration
    random.shuffle(parquet_files)

    scheduler = build_cosine_scheduler(optimizer, start_iteration, peak_lr, warmup_iters, max_iters, min_lr_ratio, forced_lr=forced_lr)

    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == torch.float16))

    token_stream = token_generator_from_parquet(parquet_files, text_column, tokenizer)
    buffer = []

    carry_states, past_key_values, abs_pos_offset = None, None, 0
    running_train_loss, train_steps = 0.0, 0

    optimizer.zero_grad(set_to_none=True)

    while True:
        required_len = chunk_size + 1
        advance_len = chunk_size

        batch_chunks = []
        while len(batch_chunks) < batch_size:
            if len(buffer) >= required_len:
                batch_chunks.append(buffer[:required_len])
                buffer = buffer[advance_len:]
            else:
                try:
                    buffer.append(next(token_stream))
                except StopIteration:
                    break

        if len(batch_chunks) < batch_size:
            print("Dataset exhausted.")
            break

        chunk = torch.tensor(batch_chunks, dtype=torch.long, device=device)

        if abs_pos_offset + chunk_size > model.freqs_cis_ext.size(0):
            carry_states, past_key_values, abs_pos_offset = None, None, 0

        detached_carry = [c.detach() for c in carry_states] if carry_states else None
        detached_kv = _detach_kv_list(past_key_values)

        with torch.autocast(device_type=device, dtype=ptdtype):
            x, y = chunk[:, :-1], chunk[:, 1:]

            logits, carry_states, past_key_values = model(
                x=x, carry_states=detached_carry, past_key_values=detached_kv,
                is_training=True, use_cache=True, abs_pos_offset=abs_pos_offset
            )

            loss = F.cross_entropy(logits.view(-1, vocab_size), y.reshape(-1))
            # Phase 1.4: snapshot CE-only loss BEFORE regularizers are added,
            # so the logger can report a clean token PPL (e^CE) unaffected by
            # the KL / meta-penalty terms mixed into the optimised loss.
            model.last_ce_loss = loss.detach()

            # Add Meta-Dynamic Routing compute penalty (no-op if disabled).
            loss = apply_meta_compute_penalty(model, loss)
            # Add Gaussian-embedding KL divergence regularizer (no-op if disabled).
            loss = apply_gaussian_kl_penalty(model, loss)
            abs_pos_offset += chunk_size
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        running_train_loss += loss.item() * grad_accum_steps
        train_steps += 1

        if train_steps % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            iteration += 1
            # Increment global training iteration counter (in-place for buffer persistence)
            model.global_training_iteration.add_(1)

            if iteration % 100 == 0:
                print_gate_stats(model, iteration, running_train_loss, grad_accum_steps * 100, scheduler)
                running_train_loss = 0.0

            # Detailed diagnostics every 500 steps
            if iteration % 500 == 0:
                print_detailed_diagnostics(model, iteration, optimizer, grad_norm=grad_norm)

            if iteration % 2000 == 0:
                model.eval()
                print(f"\n{'='*60}\n[GENERATION SAMPLE (Pre-training Coherence)]\n{'='*60}")
                test_prompt = "The rapid advancement of artificial intelligence has led to"
                gen_ids = generate_block_recurrent(
                    model, tokenizer.encode(test_prompt), tokenizer, device, max_new_tokens=150,
                    chunk_size=chunk_size, temperature=0.7
                )
                print(f"{tokenizer.decode(gen_ids)}\n")
                model.train()

            if iteration % 20000 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iteration, 'chunk_size': chunk_size,
                }, 'checkpoint_ssm_pretrain.pth')


# =============================================================================
# 9A-MIXED. Pre-training with Mixed Datasets (General Language Model)
# =============================================================================

def run_pretraining_mixed(model, dataset_config, tokenizer, optimizer, device,
                         vocab_size, start_iteration=0, chunk_size=512, enable_forgetting=False,
                         batch_size=4, grad_accum_steps=4,
                         text_column_map=None, mix_temperature=1.0, stats_interval=5000,
                         peak_lr=4e-4, warmup_iters=200, max_iters=100000, min_lr_ratio=0.1,
                         forced_lr=None):
    """
    Pre-training with mixed datasets for general language modeling.

    Args:
        dataset_config: Dict of {name: {"path": str, "weight": float}}
        text_column_map: Dict mapping dataset name to text column name (e.g., {"GitHub": "code"})
        mix_temperature: Temperature for dataset mixing (lower = more uniform)
        stats_interval: How often to print mixing statistics
        peak_lr: Peak learning rate (after warmup)
        warmup_iters: Number of warmup steps
        max_iters: Total training steps (where LR reaches minimum)
        min_lr_ratio: Final LR as fraction of peak_lr
        forced_lr: If set, overrides the schedule with a constant pinned LR
            (see build_cosine_scheduler for details).
    """

    print(f"\n--- Starting MIXED Dataset Pre-training | Device: {device} | Batch Size: {batch_size} | Grad Accum: {grad_accum_steps} ---")
    flash_status = 'Enabled' if model.use_flash_attn else 'Disabled (using Bayesian Signal Attention)'
    print(f"--- Flash Attention: {flash_status} ---")
    if forced_lr is not None:
        print(f"--- FORCED LR OVERRIDE ACTIVE: pinning base LR to {forced_lr:.2e} (schedule below is ignored) ---")
    else:
        print(f"--- LR Schedule: Warmup {warmup_iters} steps → Cosine decay to {min_lr_ratio*100:.0f}% over {max_iters} steps ---")

    iteration = start_iteration
    scheduler = build_cosine_scheduler(optimizer, start_iteration, peak_lr, warmup_iters, max_iters, min_lr_ratio, forced_lr=forced_lr)

    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == torch.float16))

    # Initialize mixed token stream
    token_stream, mixer = mixed_token_generator_from_datasets(
        dataset_config, tokenizer, text_column_map, mix_temperature
    )

    buffer = []
    carry_states, past_key_values, abs_pos_offset = None, None, 0
    running_train_loss, train_steps = 0.0, 0

    optimizer.zero_grad(set_to_none=True)

    while True:
        required_len = chunk_size + 1
        advance_len = chunk_size

        batch_chunks = []
        while len(batch_chunks) < batch_size:
            if len(buffer) >= required_len:
                batch_chunks.append(buffer[:required_len])
                buffer = buffer[advance_len:]
            else:
                try:
                    buffer.append(next(token_stream))
                except StopIteration:
                    print("[Mixed Dataset] Stream exhausted.")
                    break

        if len(batch_chunks) < batch_size:
            print("[Mixed Dataset] Insufficient data, ending training.")
            break

        chunk = torch.tensor(batch_chunks, dtype=torch.long, device=device)

        if abs_pos_offset + chunk_size > model.freqs_cis_ext.size(0):
            carry_states, past_key_values, abs_pos_offset = None, None, 0

        detached_carry = [c.detach() for c in carry_states] if carry_states else None
        detached_kv = _detach_kv_list(past_key_values)

        with torch.autocast(device_type=device, dtype=ptdtype):
            x, y = chunk[:, :-1], chunk[:, 1:]

            logits, carry_states, past_key_values = model(
                x=x, carry_states=detached_carry, past_key_values=detached_kv,
                is_training=True, use_cache=True, abs_pos_offset=abs_pos_offset
            )

            loss = F.cross_entropy(logits.view(-1, vocab_size), y.reshape(-1))
            # Phase 1.4: snapshot CE-only loss BEFORE regularizers are added,
            # so the logger can report a clean token PPL (e^CE) unaffected by
            # the KL / meta-penalty terms mixed into the optimised loss.
            model.last_ce_loss = loss.detach()

            # Add Meta-Dynamic Routing compute penalty (no-op if disabled).
            loss = apply_meta_compute_penalty(model, loss)
            # Add Gaussian-embedding KL divergence regularizer (no-op if disabled).
            loss = apply_gaussian_kl_penalty(model, loss)
            abs_pos_offset += chunk_size
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        running_train_loss += loss.item() * grad_accum_steps
        train_steps += 1

        if train_steps % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            iteration += 1
            # Increment global training iteration counter (in-place for buffer persistence)
            model.global_training_iteration.add_(1)

            if iteration % 100 == 0:
                print_gate_stats(model, iteration, running_train_loss, grad_accum_steps * 100, scheduler)
                running_train_loss = 0.0

            # Detailed diagnostics every 500 steps
            if iteration % 500 == 0:
                print_detailed_diagnostics(model, iteration, optimizer, grad_norm=grad_norm)

            if iteration % stats_interval == 0:
                mixer.print_statistics()

            if iteration % 2000 == 0:
                model.eval()
                print(f"\n{'='*60}\n[GENERATION SAMPLE (Mixed Pre-training Coherence)]\n{'='*60}")
                test_prompt = "The rapid advancement of artificial intelligence has led to"
                gen_ids = generate_block_recurrent(
                    model, tokenizer.encode(test_prompt), tokenizer, device, max_new_tokens=150,
                    chunk_size=chunk_size, temperature=0.7
                )
                print(f"{tokenizer.decode(gen_ids)}\n")
                model.train()

            if iteration % 20000 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iteration, 'chunk_size': chunk_size,
                    'mixer_stats': mixer.get_statistics(),
                }, 'checkpoint_ssm_pretrain_mixed.pth')


# =============================================================================
# 9B. Fine-tuning (Masked Instruction Tuning on ChatML JSON with Grad Accum)
# =============================================================================

def run_finetuning(model, json_file, tokenizer, optimizer, device,
                   vocab_size, start_iteration=0, chunk_size=512, enable_forgetting=False,
                   batch_size=2, grad_accum_steps=4, use_streaming=True,
                   peak_lr=1e-4, warmup_iters=100, max_iters=50000, min_lr_ratio=0.1,
                   forced_lr=None):

    print(f"\n--- Starting ChatML Fine-tuning (OpenHermes) | Device: {device} | Batch Size: {batch_size} | Grad Accum: {grad_accum_steps} ---")
    flash_status = 'Enabled' if model.use_flash_attn else 'Disabled (using Bayesian Signal Attention)'
    print(f"--- Flash Attention: {flash_status} ---")
    if forced_lr is not None:
        print(f"--- FORCED LR OVERRIDE ACTIVE: pinning base LR to {forced_lr:.2e} (schedule below is ignored) ---")
    else:
        print(f"--- LR Schedule: Warmup {warmup_iters} steps → Cosine decay to {min_lr_ratio*100:.0f}% over {max_iters} steps ---")

    iteration = start_iteration
    scheduler = build_cosine_scheduler(optimizer, start_iteration, peak_lr, warmup_iters, max_iters, min_lr_ratio, forced_lr=forced_lr)

    token_stream = stream_chatml_from_json(json_file, tokenizer, chunk_size, device, batch_size, use_streaming)

    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == torch.float16))

    carry_states, past_key_values, abs_pos_offset = None, None, 0
    running_train_loss, train_steps = 0.0, 0

    optimizer.zero_grad(set_to_none=True)

    for x_chunk, mask_chunk in token_stream:
        x, y = x_chunk[:, :-1], x_chunk[:, 1:]
        m = mask_chunk[:, 1:]

        if abs_pos_offset + x.size(1) > model.freqs_cis_ext.size(0):
            carry_states, past_key_values, abs_pos_offset = None, None, 0

        detached_carry = [c.detach() for c in carry_states] if carry_states else None
        detached_kv = _detach_kv_list(past_key_values)

        with torch.autocast(device_type=device, dtype=ptdtype):
            logits, carry_states, past_key_values = model(
                x=x, carry_states=detached_carry, past_key_values=detached_kv,
                is_training=True, use_cache=True, abs_pos_offset=abs_pos_offset
            )
            abs_pos_offset += x.size(1)

            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), reduction='none')
            loss = (loss * m.view(-1)).sum() / max(m.sum().item(), 1.0)
            # Phase 1.4: snapshot CE-only loss BEFORE regularizers are added.
            model.last_ce_loss = loss.detach()
            # Add Meta-Dynamic Routing compute penalty (no-op if disabled).
            loss = apply_meta_compute_penalty(model, loss)
            # Add Gaussian-embedding KL divergence regularizer (no-op if disabled).
            loss = apply_gaussian_kl_penalty(model, loss)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        running_train_loss += loss.item() * grad_accum_steps
        train_steps += 1

        if train_steps % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            iteration += 1
            # Increment global training iteration counter (in-place for buffer persistence)
            model.global_training_iteration.add_(1)

            if iteration % 100 == 0:
                print_gate_stats(model, iteration, running_train_loss, grad_accum_steps * 100, scheduler, "CLEAR")
                running_train_loss = 0.0

            # Detailed diagnostics every 500 steps
            if iteration % 500 == 0:
                print_detailed_diagnostics(model, iteration, optimizer, grad_norm=grad_norm)

            if iteration % 2000 == 0:
                model.eval()
                print(f"\n{'='*60}\n[GENERATION SAMPLE (Instruction Following)]\n{'='*60}")
                test_prompt = f"{CHAT_START}user\nWhat is the purpose of AI fine-tuning?{CHAT_END}\n{CHAT_START}assistant\n"
                gen_ids = generate_block_recurrent(
                    model, tokenizer.encode(test_prompt), tokenizer, device, max_new_tokens=150,
                    chunk_size=chunk_size, temperature=0.7, stop_sequence=CHAT_END
                )
                print(f"{tokenizer.decode(gen_ids)}\n")
                model.train()

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iteration, 'chunk_size': chunk_size,
                }, 'checkpoint_ssm_finetune.pth')
