"""
SSMTransformer — the top-level model assembling all layers and the meta phase.
"""

import math

import torch
import torch.nn as nn

from .rope import precompute_freqs_cis
from .block import SSMAttentionBlock
from .meta import MetaDynamicPhase


# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def get_forgetting_config(layer_idx, num_layers, enable_forgetting):
    if not enable_forgetting:
        return None
    depth_ratio = layer_idx / max(1, num_layers - 1)
    return {
        # Decreased from 0.980 to 0.950 (decays faster during grad accum steps)
        'decay_factor': 0.975 + (depth_ratio * 0.018),

        # Phase 2 fix: Lowered floor from 0.90 to 0.85 (allows locking at p~0.65)
        'lock_threshold': 0.75 + (depth_ratio * 0.04),

        'health_floor': 0.1 + (depth_ratio * 0.3),
        'gated_fraction': 0.9 - (depth_ratio * 0.6),
    }


def get_graph_reasoning_config(layer_idx, num_layers, enable_graph_reasoning):
    """
    Configure graph reasoning for specific layers.

    Strategy: Enable only in deeper layers (>= 50% depth) where relational
    reasoning is most valuable. Early layers learn local patterns via SSM +
    attention; deeper layers benefit from explicit relational structure.

    Scales num_rules and top_k_edges down for middle layers, full capacity
    in final layers. This keeps parameter count manageable while still
    granting deep layers relational reasoning power.
    """
    if not enable_graph_reasoning:
        return None

    depth_ratio = layer_idx / max(1, num_layers - 1)

    # Only enable for deeper layers (top 50% of the stack)
    if depth_ratio < 0.5:
        return None

    # Scale capacity with depth
    is_deep = depth_ratio > 0.75
    return {
        'num_rules': 8 if is_deep else 4,
        'graph_steps': 3 if is_deep else 2,
        'top_k_edges': 16 if is_deep else 8,
    }


# =============================================================================
# SSM TRANSFORMER
# =============================================================================

class SSMTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, max_seq_len=512,
                 dropout=0.1, enable_forgetting=False, saliency_decay=0.95,
                 use_flash_attn=False, enable_graph_reasoning=False,
                 enable_meta_routing=False, meta_max_steps=3, meta_gumbel_tau=1.0,
                 meta_force_explore_eps=0.0, meta_entropy_weight=0.05,
                 meta_penalty_collapse_floor=0.95, meta_penalty_collapse_ema=0.99,
                 meta_num_region_centroids=8, meta_num_semantic_anchors=32,
                 meta_temporal_window=4,
                 enable_gaussian_embeddings=True, kl_weight=0.02):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.head_dim = dim // num_heads
        self.saliency_decay = saliency_decay
        self.use_flash_attn = use_flash_attn
        self.enable_meta_routing = enable_meta_routing

        # ------------------------------------------------------------------
        # Gaussian (Probabilistic) Embeddings
        # ------------------------------------------------------------------
        # Each token is represented as a Gaussian N(mu, sigma^2).  During
        # training we sample z = mu + sigma*eps (reparameterization trick);
        # during inference we use the mean deterministically.  KL divergence
        # regularization keeps the distributions close to N(0, 1) and is
        # added to the loss in the training loop.  When disabled, falls back
        # to a single deterministic embedding table (backward-compatible).
        self.enable_gaussian_embeddings = enable_gaussian_embeddings
        self.kl_weight = kl_weight
        # Most-recent KL loss (set during training forward, None otherwise).
        # Read by apply_gaussian_kl_penalty() in the training loop.
        self.last_kl_loss = None
        # Phase 1.4: most-recent CE-only loss (set by the training loop BEFORE
        # regularizers are added), so the logger can report a clean token PPL.
        self.last_ce_loss = None

        self.tok_embeddings_mu = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.tok_embeddings_mu.weight, mean=0.0, std=0.02)
        if enable_gaussian_embeddings:
            self.tok_embeddings_logvar = nn.Embedding(vocab_size, dim)
            # Init log-variance so sigma^2 ≈ 0.02^2 (matches mu init std).
            # log(0.02^2) ≈ -7.82.
            nn.init.constant_(self.tok_embeddings_logvar.weight, math.log(0.02 ** 2))
        else:
            self.tok_embeddings_logvar = None

        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            SSMAttentionBlock(
                dim, num_heads, max_seq_len, num_layers, dropout,
                forgetting_config=get_forgetting_config(i, num_layers, enable_forgetting),
                use_eviction=(i >= 1),
                saliency_decay=saliency_decay,
                use_flash_attn=use_flash_attn,
                graph_reasoning_config=get_graph_reasoning_config(i, num_layers, enable_graph_reasoning),
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        # Output projection is now INDEPENDENT of the embedding tables so the
        # KL regularizer on mu is not confused by tied-gradient from logits.
        self.output = nn.Linear(dim, vocab_size, bias=False)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)

        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len, max_train_len=max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)

        freqs_cis_ext = precompute_freqs_cis(self.head_dim, max_seq_len * 8, max_train_len=max_seq_len)
        self.register_buffer("freqs_cis_ext", freqs_cis_ext)

        # ------------------------------------------------------------------
        # Meta-Dynamic Routing Network (optional)
        # ------------------------------------------------------------------
        # Placed AFTER the fixed layers. Uses a shared dynamic layer +
        # Gumbel-Softmax routing. The compute-penalty counter is incremented
        # by the training loop in SSMTransformer.forward.
        self.meta_phase = None
        # Global training iteration counter (persistent, used for tau annealing and locking)
        self.register_buffer("global_training_iteration", torch.tensor(0, dtype=torch.long), persistent=True)
        self.compute_penalty_weight = 0.0          # current effective weight
        self.compute_penalty_target = 0.01         # Target penalty weight (reduced from 0.05 to prevent collapse)
        self.compute_penalty_warmup = 5000         # steps to ramp 0 -> target
        # --- Penalty kill-switch ---
        # When the router's free exit-prob EMA exceeds `penalty_collapse_floor`,
        # we force penalty_weight to 0 so nothing is pushing EXIT.  This lets
        # the entropy bonus + LM-loss gradient pull the router back without
        # fighting a counter-force.  Once exit-EMA drops below the floor, the
        # warmup schedule resumes normally.
        self.penalty_collapse_floor = float(meta_penalty_collapse_floor)
        self.penalty_collapse_ema_beta = float(meta_penalty_collapse_ema)
        self.exit_prob_ema = 0.0   # running average of router free exit-prob
        if enable_meta_routing:
            self.meta_phase = MetaDynamicPhase(
                dim=dim,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                num_layers=num_layers,
                dropout=dropout,
                max_meta_steps=meta_max_steps,
                gumbel_tau=meta_gumbel_tau,
                force_explore_eps=meta_force_explore_eps,
                entropy_weight=meta_entropy_weight,
                forgetting_config=get_forgetting_config(num_layers, num_layers, enable_forgetting),
                saliency_decay=saliency_decay,
                use_flash_attn=use_flash_attn,
                graph_reasoning_config=get_graph_reasoning_config(num_layers, num_layers, enable_graph_reasoning),
                num_region_centroids=meta_num_region_centroids,
                num_semantic_anchors=meta_num_semantic_anchors,
                temporal_window=meta_temporal_window,
            )

        if enable_meta_routing:
            print(f"\n{'='*60}")
            print(f"Meta-Dynamic Routing Network (Enabled)")
            print(f"{'='*60}")
            print(f"  Max routing steps (M): {meta_max_steps}")
            print(f"  Gumbel tau:            {meta_gumbel_tau}")
            print(f"  Force-explore eps:     {meta_force_explore_eps}  (per-token APPLIED-forced; 0=off)")
            print(f"  Entropy bonus β:       {meta_entropy_weight}  (0=off; pushes router away from collapse)")
            print(f"  Penalty target:        {self.compute_penalty_target} (ramped over {self.compute_penalty_warmup} training iterations)")
            print(f"  Penalty kill-switch:   floor={self.penalty_collapse_floor} (exit-EMA above → penalty=0)")
            print(f"  Shared dynamic layer:  1 (Universal-Transformer style)")
            geo_dim = meta_num_region_centroids + meta_num_semantic_anchors + meta_temporal_window
            print(f"  Geometric features:    {geo_dim} dims "
                  f"(centroids={meta_num_region_centroids}, anchors={meta_num_semantic_anchors}, "
                  f"temporal_k={meta_temporal_window})")
            geo_params = (meta_num_region_centroids + meta_num_semantic_anchors) * dim
            print(f"  Geo reference params:  {geo_params:,} "
                  f"(centroids={meta_num_region_centroids}*{dim}, anchors={meta_num_semantic_anchors}*{dim})")
            router_params = sum(p.numel() for p in self.meta_phase.meta_net.parameters())
            print(f"  Router params:         {router_params:,} (input={dim}+{geo_dim}={dim+geo_dim})")
            print(f"{'='*60}\n")

        # Print forgetting configuration for each layer
        if enable_forgetting:
            print(f"\n{'='*60}")
            print(f"Cognitive Forgetting Gate Configuration (Enabled)")
            print(f"{'='*60}")
            for i in range(num_layers):
                config = get_forgetting_config(i, num_layers, enable_forgetting)
                if config:
                    print(f"Layer {i:2d}: decay={config['decay_factor']:.4f}, lock_thresh={config['lock_threshold']:.3f}, "
                          f"health_floor={config['health_floor']:.2f}, gated_frac={config['gated_fraction']:.2f}")
            print(f"{'='*60}\n")

        # Print graph reasoning configuration for each layer
        if enable_graph_reasoning:
            print(f"\n{'='*60}")
            print(f"Latent Graph Reasoning Configuration (Enabled)")
            print(f"{'='*60}")
            graph_layers = 0
            for i in range(num_layers):
                gr_config = get_graph_reasoning_config(i, num_layers, enable_graph_reasoning)
                if gr_config:
                    graph_layers += 1
                    print(f"Layer {i:2d}: rules={gr_config['num_rules']}, steps={gr_config['graph_steps']}, "
                          f"top_k={gr_config['top_k_edges']}")
            if graph_layers == 0:
                print("  (No layers qualify — increase num_layers or lower depth threshold)")
            else:
                print(f"Active in {graph_layers}/{num_layers} layers (deepest {graph_layers})")
            print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Gaussian Embedding helper
    # ------------------------------------------------------------------
    @property
    def tok_embeddings(self):
        """Backward-compatible accessor for the mean embedding table.

        External code (e.g. validate_vocab_size, whiten_checkpoint) that
        still references ``model.tok_embeddings`` transparently gets the mu
        table now.
        """
        return self.tok_embeddings_mu

    def _gaussian_embed(self, x, is_training):
        """Return the (possibly stochastic) embedding for input token ids ``x``.

        - Deterministic fallback (gaussian disabled)  → mu
        - Inference (is_training=False)                → mu
        - Training + gaussian enabled                  → z = mu + sigma * eps

        When stochastic sampling occurs, ``self.last_kl_loss`` is updated to
        the KL divergence of the sampled distributions vs. N(0, 1) so the
        training loop can add the regularizer.
        """
        mu = self.tok_embeddings_mu(x)
        # Default: deterministic
        self.last_kl_loss = None
        if not self.enable_gaussian_embeddings or not is_training:
            return mu

        logvar = self.tok_embeddings_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        # KL[N(mu, sigma^2) || N(0, 1)] averaged over all elements.
        #   = -0.5 * mean(1 + log(sigma^2) - mu^2 - sigma^2)
        self.last_kl_loss = -0.5 * torch.mean(
            1.0 + logvar - mu.pow(2) - logvar.exp()
        )
        return z

    def _expected_state_length(self):
        """Number of carry/KV entries: fixed layers + 1 for meta phase if enabled."""
        return self.num_layers + (1 if self.enable_meta_routing else 0)

    def _pad_state_list(self, states, expected_len):
        """Pad a carry/KV list up to expected_len so old callers keep working."""
        if states is None:
            return [None] * expected_len
        if len(states) < expected_len:
            states = list(states) + [None] * (expected_len - len(states))
        return states

    def _update_compute_penalty_schedule(self):
        """
        Warmup the compute-penalty weight from 0 → compute_penalty_target over
        `compute_penalty_warmup` optimizer steps.  Starting at 0 lets the
        dynamic layer/router get useful gradients from the language-model loss
        before being forced to be efficient (otherwise the router trivially
        learns to EXIT immediately and the dynamic layer never trains).

        **Kill-switch:** Tracks an EMA of the router's free exit-probability.
        When it exceeds `penalty_collapse_floor` (router has collapsed to
        always-EXIT), we clamp the penalty weight to 0 so the entropy bonus +
        LM-loss can recover the router without resistance.  Once the EMA drops
        below the floor, the warmup schedule resumes.
        """
        if not self.enable_meta_routing:
            self.compute_penalty_weight = 0.0
            return

        # --- Update exit-prob EMA from the most recent forward pass ---
        exit_stats = getattr(self.meta_phase, "last_exit_stats", None)
        if exit_stats:
            cur_avg = sum(exit_stats) / len(exit_stats)
            b = self.penalty_collapse_ema_beta
            self.exit_prob_ema = b * self.exit_prob_ema + (1.0 - b) * cur_avg

        # --- Kill-switch: router collapsed → zero penalty ---
        if self.exit_prob_ema > self.penalty_collapse_floor:
            self.compute_penalty_weight = 0.0
            return

        if self.compute_penalty_warmup <= 0:
            self.compute_penalty_weight = self.compute_penalty_target
            return
        frac = min(1.0, float(self.global_training_iteration.item()) / float(self.compute_penalty_warmup))
        self.compute_penalty_weight = self.compute_penalty_target * frac

    def forward(self, x, carry_states=None, is_training=True, past_key_values=None, use_cache=False, abs_pos_offset=0):
        # Update compute-penalty schedule each forward (cheap, idempotent).
        self._update_compute_penalty_schedule()

        h = self.embed_dropout(self._gaussian_embed(x, is_training))

        expected_len = self._expected_state_length()
        # Pad caller-provided carry/KV lists so the meta phase slot exists.
        carry_states = self._pad_state_list(carry_states, expected_len)
        past_key_values = self._pad_state_list(past_key_values, expected_len)

        new_carry_states = []
        new_key_values = []
        
        # Get current global step for passing to layers
        current_step = self.global_training_iteration.item()

        # --- Fixed Phase: standard sequential layer pipeline ---
        for i, layer in enumerate(self.layers):
            layer_past_kv = past_key_values[i] if past_key_values is not None else None
            h, new_carry, new_kv = layer(
                h, self.freqs_cis_ext, abs_pos_offset=abs_pos_offset,
                carry_state=carry_states[i],
                past_kv=layer_past_kv,
                use_cache=use_cache,
                global_step=current_step
            )
            new_carry_states.append(new_carry)
            new_key_values.append(new_kv)

        # --- Meta Phase (Dynamic Routing) ---
        meta_compute_penalty = None
        if self.enable_meta_routing:
            meta_idx = self.num_layers
            meta_past_kv = [past_key_values[meta_idx]] if (past_key_values is not None and past_key_values[meta_idx] is not None) else None
            meta_carry_in = carry_states[meta_idx]

            h, meta_new_carry, meta_new_kv, meta_compute_penalty, meta_entropy = self.meta_phase(
                h, self.freqs_cis_ext,
                abs_pos_offset=abs_pos_offset,
                carry_state=meta_carry_in,
                past_kv=meta_past_kv,
                use_cache=use_cache,
                is_training=is_training,
                global_step=current_step
            )
            # NOTE: the MetaDynamicPhase packs its KV as a single-element list
            # internally; unpack it back to the element (or None) so that the
            # outer KV cache list matches the (fixed + meta) layout expected by
            # callers like CognitiveMemoryManager / generation loop.
            meta_kv_entry = meta_new_kv[0] if meta_new_kv is not None else None
            new_carry_states.append(meta_new_carry)
            new_key_values.append(meta_kv_entry)

        h = self.norm(h)
        logits = self.output(h)

        # Expose the most recent compute penalty so training loops can add it.
        self.last_meta_compute_penalty = meta_compute_penalty
        # Expose the differentiable entropy sum so the training loop can add
        # the entropy bonus to the loss (loss - entropy_weight * H).
        self.last_meta_entropy = meta_entropy if self.enable_meta_routing else None
        return logits, new_carry_states, new_key_values
