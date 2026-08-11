"""
Meta-Dynamic Routing Network — per-token adaptive compute allocation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import SSMAttentionBlock


# =============================================================================
# META-DYNAMIC ROUTING NETWORK
# =============================================================================

class MetaDynamicPhase(nn.Module):
    """
    Meta-Dynamic Routing Network (a.k.a. 'Meta Phase').

    After the Fixed Phase layers finish processing, the hidden state enters a
    recursive routing loop.  On each step a lightweight Meta Router decides —
    per token — whether to apply one more dynamic layer or to EXIT.  "Easy"
    tokens EXIT early (saving compute) while "hard" tokens loop, allowing
    deeper reasoning.

    Design choices baked into this implementation:

    1.  **Universal-Transformer style dynamic phase:** Rather than having N
        heterogeneous dynamic layers (which would fragment the KV-cache), we use
        a single shared `DynamicLayer` applied up to `max_meta_steps` times.
        This keeps one KV cache per 'virtual' position in the dynamic phase and
        is far more memory-friendly.
    2.  **Gumbel-Softmax Straight-Through Estimator:** During training we sample
        a hard one-hot routing decision forward (so we can mask layer outputs)
        while back-propagating through the soft probabilities.  At inference we
        use a plain argmax and *actually skip the compute* when EXIT is chosen.
    3.  **Step embeddings:** Each routing step injects a learned step embedding
        into the hidden state so the shared dynamic layer knows which iteration
        of the loop it is executing (this prevents the 'confused about position'
        problem described in the design doc).
    4.  **Compute penalty accumulation:** The phase accumulates `(1 - exit_prob)`
        for every step where a token chose NOT to exit.  Callers add
        `penalty_weight * total_penalty` to the CE loss so the router learns to
        be efficient.

    Args:
        dim:               Model hidden dim (must match SSMAttentionBlock.dim).
        num_heads:         Attention heads for the shared dynamic layer.
        max_seq_len:       Maximum KV-cache length (for eviction).
        dropout:           Dropout rate inside the dynamic layer.
        num_layers:        Total fixed-layer count (used for init scaling).
        max_meta_steps:    Maximum number of routing loop iterations (M in the
                           design doc).  Each iteration may apply the dynamic
                           layer OR exit.
        gumbel_tau:        Temperature for the Gumbel-Softmax.  Lower = harder
                           routing decisions.  Anneal during training if desired.
        forgetting_config: Optional config forwarded to CognitiveForgettingGate
                           inside the dynamic layer.
        saliency_decay:    Decay for saliency eviction inside the dynamic layer.
        use_flash_attn:    Whether the inner attention may use SDPA.
        graph_reasoning_config: Optional config for the dynamic layer's MLP.
    """
    def __init__(self, dim, num_heads, max_seq_len, num_layers, dropout=0.1,
                 max_meta_steps=3, gumbel_tau=1.0, forgetting_config=None,
                 saliency_decay=0.95, use_flash_attn=False,
                 graph_reasoning_config=None, force_explore_eps=0.0,
                 entropy_weight=0.0, num_region_centroids=8,
                 num_semantic_anchors=32, temporal_window=4):
        super().__init__()
        self.dim = dim
        self.max_meta_steps = max_meta_steps
        self.gumbel_tau = gumbel_tau
        # Gumbel tau annealing schedule (1.5 → 0.5 over 25k training iterations)
        # Starts soft (more exploration), ends hard (committed routing)
        # NOTE: This is in TRAINING ITERATIONS, not forward passes
        self.gumbel_tau_start = 1.5
        self.gumbel_tau_end = 0.5
        self.gumbel_tau_anneal_steps = 25000  # Extended from 10k to 25k to allow more stabilization
        # --- Geometric / Contextual Feature Config ---
        # Learnable reference points that give the router spatial awareness
        # without scanning the full vocabulary:
        #   • Region centroids: coarse "zones" in embedding space.
        #   • Semantic anchors: fine-grained triangulation "cities".
        #   • Temporal window: how many previous positions (k=1..n) to compute
        #     cosine-similarity against the current token (captures local
        #     semantic velocity).
        self.num_region_centroids = num_region_centroids
        self.num_semantic_anchors = num_semantic_anchors
        self.temporal_window = temporal_window
        # Entropy regularization weight.  When > 0, the meta phase accumulates
        # H(softmax(route_logits)) at each step and exposes it via
        # `last_entropy`.  The training loop subtracts `entropy_weight * H`
        # from the loss so the router is rewarded for NOT collapsing.
        self.entropy_weight = float(entropy_weight)
        # Per-token epsilon-greedy exploration: with probability eps a token is
        # forced to APPLY the dynamic layer at EVERY step (full rollout), with
        # router gradient cut on those tokens.  This prevents the
        # 'always-EXIT' collapse by giving the dynamic layer a clean, full-
        # strength gradient signal regardless of what the router decides.
        # eps=0 disables forcing (pure router control).  Never active at
        # inference.
        self.force_explore_eps = float(force_explore_eps)
        # Output size = 1 routing option (apply the shared dynamic layer) + 1 EXIT.
        # When extended to multiple dynamic layers this becomes
        # (num_dynamic_layers + 1).
        self.num_route_options = 2

        # --- Geometric Reference Points (learnable) ---
        # Centroids and anchors live in the same embedding space as the hidden
        # state.  The router compares the current token against them via cosine
        # similarity, producing a compact positional fingerprint.
        self.region_centroids = nn.Parameter(torch.randn(num_region_centroids, dim) * 0.02)
        self.semantic_anchors = nn.Parameter(torch.randn(num_semantic_anchors, dim) * 0.02)

        # Total geometric feature width: centroid sims + anchor sims + temporal sims.
        self.geo_feature_dim = num_region_centroids + num_semantic_anchors + temporal_window

        # --- Meta Router (single Linear, fed hidden state + geometric features) ---
        self.meta_net = nn.Linear(dim + self.geo_feature_dim, self.num_route_options)
        # Initialize with a slight bias TOWARDS APPLY (option 0) vs EXIT (option 1)
        # at the start of training.  This ensures the dynamic layer actually
        # receives gradients early on; otherwise the router could collapse to
        # "always EXIT" before the dynamic layer learns anything useful.  The
        # compute-penalty warmup (starting at 0.0) complements this.
        nn.init.normal_(self.meta_net.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.meta_net.bias)
        with torch.no_grad():
            # Bias logits: [APPLY=+0.5, EXIT=-0.5]  → softmax favours APPLY early.
            self.meta_net.bias[0] = 0.5
            self.meta_net.bias[1] = -0.5

        # --- Step embeddings (one per possible loop iteration) ---
        self.step_embeddings = nn.Embedding(max_meta_steps, dim)
        nn.init.normal_(self.step_embeddings.weight, mean=0.0, std=0.02)

        # --- Shared dynamic layer (Universal-Transformer style) ---
        # Sharing one layer keeps a single coherent KV cache and avoids the
        # cache fragmentation problem described in the design 'gotchas'.
        self.dynamic_layer = SSMAttentionBlock(
            dim, num_heads, max_seq_len, num_layers, dropout,
            forgetting_config=forgetting_config,
            use_eviction=True,
            saliency_decay=saliency_decay,
            use_flash_attn=use_flash_attn,
            graph_reasoning_config=graph_reasoning_config,
        )

        # Buffers to hold routing statistics for the most recent forward pass.
        # The training loop reads `last_compute_penalty` to regularize the loss.
        self.last_compute_penalty = None
        self.last_exit_stats = None
        # Average fraction of tokens that were force-explored this pass.
        self.last_forced_frac = 0.0
        # Mean router entropy over all routing steps (differentiable pre-detach).
        self.last_entropy = None

    # ------------------------------------------------------------------
    # Geometric / Contextual Feature Computation
    # ------------------------------------------------------------------
    def _compute_geometric_features(self, x):
        """Build a compact spatial/temporal fingerprint for the router.

        Computes three groups of features from the hidden state ``x``
        ``(B, L, D)`` and concatenates them into a single vector
        ``(B, L, geo_feature_dim)``:

        A. **Centroid similarities** ``(B, L, num_region_centroids)`` —
           cosine similarity between each token and the coarse region
           centroids.  Tells the router which "zone" of embedding space the
           token currently occupies.

        B. **Anchor similarities** ``(B, L, num_semantic_anchors)`` —
           cosine similarity against fine-grained semantic anchors acting as
           a compressed learnable vocabulary for high-resolution
           triangulation.

        C. **Temporal angles** ``(B, L, temporal_window)`` — for each offset
           k ∈ [1, temporal_window], the cosine similarity between the
           token at position *t* and the token at position *t−k*.  Positions
           without a k-th predecessor are zero-padded.  Captures local
           semantic velocity (how fast the embedding is changing).
        """
        B, L, D = x.shape
        x_norm = F.normalize(x, dim=-1)  # (B, L, D)

        # A. Centroid similarities
        centroids_norm = F.normalize(self.region_centroids, dim=-1)  # (C, D)
        centroid_sim = x_norm @ centroids_norm.T  # (B, L, C)

        # B. Anchor similarities
        anchors_norm = F.normalize(self.semantic_anchors, dim=-1)  # (A, D)
        anchor_sim = x_norm @ anchors_norm.T  # (B, L, A)

        # C. Temporal angles (cosine sim to k-th previous token)
        # Vectorized via tensor unfolding (eliminates 128 kernel launches)
        pad_x = F.pad(x_norm, (0, 0, self.temporal_window, 0))  # (B, L+W, D)
        x_prevs = pad_x.unfold(1, self.temporal_window, 1)[:, :-1, :, :]  # (B, L, D, W)
        
        # Inner product across the D dimension.
        # unfold ordering means index W-1 is t-1, W-2 is t-2. We flip to match original k=1..W order
        temporal_cat = (x_norm.unsqueeze(-1) * x_prevs).sum(dim=2)  # (B, L, W)
        temporal_cat = torch.flip(temporal_cat, dims=[-1])

        # Concatenate all features → (B, L, geo_feature_dim)
        return torch.cat([centroid_sim, anchor_sim, temporal_cat], dim=-1)

    def forward(self, x, freqs_cis_ext, abs_pos_offset=0, carry_state=None,
                past_kv=None, use_cache=False, is_training=True, global_step=None):
        """
        Args:
            x:               (B, L, D) hidden state coming out of the Fixed Phase.
            freqs_cis_ext:   Extended rotary frequencies buffer from the parent.
            abs_pos_offset:  Absolute position of the first token in x.
            carry_state:     SSM carry state (passed through the dynamic layer).
            past_kv:         KV cache for the dynamic layer (one entry).
            use_cache:       Whether to maintain the KV cache.
            is_training:     When False we use plain argmax routing and truly
                             skip compute when EXIT is chosen.
            global_step:     Global training iteration (used for tau annealing).
        Returns:
            x:                (B, L, D) refined hidden state.
            new_carry:        Updated SSM carry state.
            new_kv:           Updated KV cache (single-element list or None).
            compute_penalty:  Scalar acumulator for the loss (already averaged
                              over batch & sequence).
        """
        B, L, D = x.shape
        device = x.device

        # The KV cache for the dynamic phase is a single-element list because
        # the dynamic phase is a single shared layer.  The parent model always
        # passes `past_key_values` as a list indexed by fixed-layer index, so we
        # unwrap/rewrap here to keep the dynamic layer's API simple.
        layer_past_kv = past_kv[0] if past_kv is not None else None

        cur_carry = carry_state
        cur_kv = layer_past_kv
        total_penalty = x.new_zeros(())  # scalar on the right device/dtype
        total_entropy = x.new_zeros(())  # accumulates router entropy for loss
        exit_counts = []
        entropy_per_step = []  # Track per-step entropy for diagnostics

        # --- Per-token epsilon-greedy exploration mask (training only) ---
        # Drawn ONCE per forward pass; the same mask is reused across all
        # routing steps so that forced tokens get a full rollout through the
        # dynamic phase.  Inference is never forced.
        force_apply_mask = x.new_zeros(B, L, 1)
        unforced_mask = x.new_ones(B, L, 1)
        if is_training and self.force_explore_eps > 0.0:
            force_apply_mask = (torch.rand(B, L, 1, device=device)
                                < self.force_explore_eps).to(dtype=x.dtype)
            unforced_mask = 1.0 - force_apply_mask
        self.last_forced_frac = force_apply_mask.mean().item()

        for step_idx in range(self.max_meta_steps):
            # --- Inject step embedding so the layer knows which iteration ---
            step_id = torch.tensor(step_idx, device=device, dtype=torch.long)
            x_with_step = x + self.step_embeddings(step_id).view(1, 1, D)

            # --- Geometric / contextual features for the router ---
            # Build the spatial-temporal fingerprint from the (step-tagged)
            # hidden state and concatenate it with the raw hidden state so the
            # router sees both content and location-in-space.
            geo_features = self._compute_geometric_features(x_with_step)
            router_input = torch.cat([x_with_step, geo_features], dim=-1)

            # --- Free routing decision (router controls unforced tokens) ---
            route_logits = self.meta_net(router_input)  # (B, L, 2)
            # Entropy of the router's SOFT distribution.  This is the
            # differentiable signal used for exploration: when the router
            # collapses (one-hot), H→0 and the entropy bonus pulls it back.
            # Computed BEFORE the Gumbel hard-sample so it sees the true soft
            # probabilities, not the noisy one-hot.  Skip at inference.
            route_soft = F.softmax(route_logits, dim=-1)
            # H = -sum(p * log p), clamped for numerical stability.
            log_p = torch.log(route_soft.clamp(min=1e-8))
            step_entropy = -(route_soft * log_p).sum(dim=-1).mean()
            entropy_per_step.append(step_entropy.item())
            if is_training and self.entropy_weight > 0.0:
                total_entropy = total_entropy + step_entropy
            if is_training:
                # Use annealed tau based on global training iteration (not forward passes)
                tau = self.gumbel_tau  # fallback to fixed if annealing disabled
                if hasattr(self, 'gumbel_tau_anneal_steps') and global_step is not None:
                    # Linear anneal based on TRAINING ITERATION counter from parent model
                    progress = min(1.0, float(global_step) / self.gumbel_tau_anneal_steps)
                    tau = self.gumbel_tau_start - progress * (self.gumbel_tau_start - self.gumbel_tau_end)
                route_probs = F.gumbel_softmax(
                    route_logits, tau=tau, hard=True, dim=-1
                )
            else:
                # Hard argmax at inference, expressed as a one-hot vector so the
                # rest of the code path is identical.  We DO skip the compute for
                # tokens that pick EXIT (see below).
                route_probs = F.one_hot(
                    route_logits.argmax(dim=-1), num_classes=self.num_route_options
                ).to(dtype=x.dtype)

            apply_prob = route_probs[..., 0:1]   # apply the dynamic layer
            exit_prob = route_probs[..., 1:2]    # EXIT the loop

            # --- Force-apply exploration: override routing for forced tokens ---
            # Forced tokens get APPLY=1, EXIT=0 with NO gradient to the router
            # (those contributions are constants).  Unforced tokens retain
            # their Gumbel-ST routing decision and gradient.
            if is_training and self.force_explore_eps > 0.0:
                apply_prob = apply_prob * unforced_mask + force_apply_mask
                exit_prob  = exit_prob  * unforced_mask

            # --- Compute penalty (skip forced tokens) ---
            # Only charge the router for compute it FREELY chose, not for the
            # tokens we forced to explore.  Average over unforced tokens.
            n_unforced = unforced_mask.sum()
            denom = n_unforced.clamp(min=1.0)
            
            # Unforced_mask cancels out terms when n_unforced is 0; avoids .item() host syncs
            total_penalty = total_penalty + (unforced_mask * (1.0 - exit_prob)).sum() / denom
            
            # Store tensor to avoid synchronizing the GPU execution pipeline
            exit_counts.append(((unforced_mask * exit_prob).sum() / denom).detach())

            # Early termination: at inference, if ALL tokens chose EXIT, stop.
            if not is_training and exit_prob.min().item() >= 0.999:
                break

            # --- Compute the dynamic layer once for the whole batch ---
            if is_training:
                # TRAINING: always run the layer, then blend via Gumbel mask.
                layer_out, new_carry, new_kv = self.dynamic_layer(
                    x_with_step, freqs_cis_ext,
                    abs_pos_offset=abs_pos_offset,
                    carry_state=cur_carry,
                    past_kv=cur_kv,
                    use_cache=use_cache,
                    global_step=global_step,
                )
                # blended = apply_prob * layer_out + exit_prob * x
                # i.e. tokens that EXIT keep their previous state, tokens that
                # APPLY take the new layer output.
                x = apply_prob * layer_out + exit_prob * x
                cur_carry = new_carry
                cur_kv = new_kv
            else:
                # INFERENCE:
                # When use_cache=True (autoregressive generation), ALWAYS run the
                # dynamic layer even if every token chose EXIT.  Skipping it would
                # leave a positional hole in the KV cache: the next token that does
                # APPLY would attend over a cache that is missing this position's
                # K/V entry, corrupting the relative-position encoding for all
                # subsequent tokens.  The routing decision still controls the
                # hidden-state blend (EXIT tokens keep `x` unchanged), so the
                # semantic routing semantics are fully preserved; we only pay one
                # extra layer forward when all tokens exit — negligible for L=1.
                #
                # When use_cache=False (offline scoring, no generation), the skip
                # is still safe because no KV state is carried between calls.
                should_run = use_cache or (apply_prob.max().item() > 0.001)
                if should_run:
                    layer_out, new_carry, new_kv = self.dynamic_layer(
                        x_with_step, freqs_cis_ext,
                        abs_pos_offset=abs_pos_offset,
                        carry_state=cur_carry,
                        past_kv=cur_kv,
                        use_cache=use_cache,
                        global_step=global_step,
                    )
                    x = apply_prob * layer_out + exit_prob * x
                    cur_carry = new_carry
                    cur_kv = new_kv
                # else: not caching and all tokens exited → safe to skip.

        # Save stats for the training loop / logging.
        self.last_compute_penalty = total_penalty.detach()
        self.last_exit_stats = [p.item() if isinstance(p, torch.Tensor) else p for p in exit_counts]
        self.last_entropy_per_step = entropy_per_step  # Per-step entropy for diagnostics
        # Average per-step entropy (detached for logging metrics only; the
        # gradient-bearing version is returned separately).
        if total_entropy.requires_grad:
            self.last_entropy = total_entropy.detach() / max(1, self.max_meta_steps)
        else:
            self.last_entropy = None

        new_kv_list = [cur_kv] if (use_cache and cur_kv is not None) else None
        return x, cur_carry, new_kv_list, total_penalty, total_entropy
