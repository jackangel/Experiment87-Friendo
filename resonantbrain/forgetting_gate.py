"""
Cognitive Forgetting Gate — per-dimension adaptive health masking.
"""

import torch
import torch.nn as nn


# =============================================================================
# COGNITIVE FORGETTING GATE
# =============================================================================

class CognitiveForgettingGate(nn.Module):
    def __init__(self, hidden_dim, enable_ablation=False,
                 decay_factor=0.995, lock_threshold=0.99,
                 health_floor=0.2, gated_fraction=0.75):
        super().__init__()
        self.enable_ablation = enable_ablation
        self.decay_factor = decay_factor
        self.lock_threshold = lock_threshold
        self.health_floor = health_floor
        self.gated_fraction = gated_fraction
        # Health-bump magnitude on a consistent step (must match the literal
        # used in forward()).  Exposed so the logger can derive the
        # theoretically-required consistency p* = θ(1-d)/[θ(1-d)+Δ] without
        # duplicating the constant.
        self.health_bump = 0.002

        if self.enable_ablation:
            self.num_gated = int(hidden_dim * gated_fraction)
            self.num_wildcard = hidden_dim - self.num_gated

            self.register_buffer("health", torch.full((self.num_gated,), 0.5))
            self.register_buffer("firing_ema", torch.zeros(self.num_gated))
            self.register_buffer("is_locked", torch.zeros(self.num_gated, dtype=torch.bool))
            self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))
            # --- Phase 1 instrumentation: true per-step consistency rate ---
            # `is_consistent` is a per-dimension boolean (current firing rate
            # >= 0.8 * firing_ema).  Its MEAN over the gated dims is the
            # consistency probability 'p' used in the steady-state analysis.
            # Exposed for logging so we can compare observed p vs the
            # theoretically required p* to lock (validation gate).
            self.register_buffer("consistency_ema", torch.tensor(0.5))
            self.last_consistency_rate = 0.5

    @property
    def p_lock_required(self):
        """Theoretically required mean consistency p* to ever reach the lock
        threshold at steady state, given the update rule
            h = is_consistent ? h+Δ : h*d
        Solving the fixed point h*(p)=θ for p yields:
            p* = θ(1-d) / [θ(1-d) + Δ]
        Below this consistency, the gate CANNOT lock.  This is the key number
        for the Phase 1 verification gate: compare p* (this) vs the observed
        consistency_ema (runtime).  If observed < p* everywhere, the gate is
        provably dead code.
        """
        if not self.enable_ablation:
            return float('nan')
        theta = self.lock_threshold
        d = self.decay_factor
        delta = self.health_bump
        denom = theta * (1.0 - d) + delta
        if denom <= 0:
            return float('nan')
        return theta * (1.0 - d) / denom

    def forward(self, x, global_step=None):
        if not self.enable_ablation:
            return x

        x_gated = x[..., :self.num_gated]
        x_wildcard = x[..., self.num_gated:] if self.num_wildcard > 0 else None

        if self.training:
            with torch.no_grad():
                self.step_count += 1
                # Use global_step for locking if provided, otherwise fall back to internal counter
                lock_step = global_step if global_step is not None else self.step_count.item()

                fired = (x_gated > 1e-3).float()
                current_firing_rate = fired.mean(dim=(0, 1))

                self.firing_ema.copy_(self.firing_ema * 0.99 + current_firing_rate * 0.01)
                is_consistent = current_firing_rate >= (self.firing_ema * 0.8)

                # Phase 1 instrumentation: snapshot the true consistency
                # probability 'p' (fraction of gated dims deemed consistent
                # this forward).  EMA-smoothed with the same 0.99 beta as the
                # firing EMA for stability.  This is what the steady-state
                # lock analysis (h*, p*_lock) should be compared against.
                consistency_rate = is_consistent.float().mean().item()
                self.consistency_ema.mul_(0.99).add_(consistency_rate * 0.01)
                self.last_consistency_rate = consistency_rate

                health_update = torch.where(
                    is_consistent,
                    self.health + 0.002,
                    self.health * self.decay_factor
                )

                self.health.copy_(torch.clamp(health_update, self.health_floor, 1.0))

                # Use global training iteration for locking (not forward-pass count)
                if lock_step > 40000:
                    newly_locked = (self.health >= self.lock_threshold) & (self.firing_ema > 0.1) & (~self.is_locked)
                    self.is_locked = self.is_locked | newly_locked

                self.health.masked_fill_(self.is_locked, 1.0)

        # Clone the health buffer before using it in the gradient path.
        # When the Meta-Dynamic phase reuses a shared layer, this gate's forward
        # is called multiple times per outer forward pass. Each call mutates the
        # `health` buffer in-place (copy_, masked_fill_), which would bump its
        # version counter and cause "variable modified by inplace operation"
        # errors during backward(). The clone creates an independent snapshot
        # whose version counter never changes.
        x_gated = x_gated * self.health.clone().detach().view(1, 1, -1)

        if x_wildcard is not None:
            return torch.cat([x_gated, x_wildcard], dim=-1)
        else:
            return x_gated
