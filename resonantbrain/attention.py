"""
Bayesian Signal Attention — block-cumulative-mean attention mechanism.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rotary_emb


# =============================================================================
# BAYESIAN SIGNAL ATTENTION
# =============================================================================

class BayesianSignalAttention(nn.Module):
    def __init__(self, d_model, num_heads, signal_window=2):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.signal_window = signal_window
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, q, k, v, freqs_cis_q, freqs_cis_k_blocks):
        B, H, L_q, D_h = q.shape
        _, _, L_k, _ = k.shape

        # Calculate the absolute position offset for the queries
        offset = L_k - L_q

        # Pad K and V sequence to multiple of signal_window
        pad_len = (self.signal_window - (L_k % self.signal_window)) % self.signal_window
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        padded_L_k = k.size(2)
        M = padded_L_k // self.signal_window

        # Chunk K and V
        k_chunks = k.view(B, H, M, self.signal_window, D_h)
        v_chunks = v.view(B, H, M, self.signal_window, D_h)

        # Block Means (Unrotated)
        k_block_means = k_chunks.mean(dim=3) # (B, H, M, D_h)
        v_block_means = v_chunks.mean(dim=3) # (B, H, M, D_h)

        # Cumulative Means (Strictly Causal)
        window_positions = torch.arange(1, self.signal_window + 1, device=q.device).view(1, 1, 1, -1, 1)
        k_cum_means = k_chunks.cumsum(dim=3) / window_positions
        v_cum_means = v_chunks.cumsum(dim=3) / window_positions

        # Extract the exact cumulative mean for each query's position
        k_cum_flat = k_cum_means.view(B, H, padded_L_k, D_h)
        v_cum_flat = v_cum_means.view(B, H, padded_L_k, D_h)

        k_current_diag = k_cum_flat[:, :, offset:offset+L_q, :]
        v_current_diag = v_cum_flat[:, :, offset:offset+L_q, :]

        # --- Apply RoPE AFTER computing block means ---
        # Apply RoPE to q and k_current_diag using the query frequencies
        q_rope, k_current_diag_rope = apply_rotary_emb(q, k_current_diag, freqs_cis_q)

        # Apply RoPE to k_block_means using the block base frequencies
        # We pass q_dummy just to satisfy the function signature
        q_dummy = torch.zeros(B, H, M, D_h, device=q.device)
        _, k_block_means_rope = apply_rotary_emb(q_dummy, k_block_means, freqs_cis_k_blocks)

        # --- Vectorized Logit Calculation ---
        logits_past = torch.matmul(q_rope, k_block_means_rope.transpose(-1, -2)) * self.scale # (B, H, L_q, M)
        logits_current = (q_rope * k_current_diag_rope).sum(dim=-1) * self.scale # (B, H, L_q)

        # --- Create Absolute Masks ---
        q_abs_idx = torch.arange(offset, offset + L_q, device=q.device).unsqueeze(1) # (L_q, 1)
        b_idx = torch.arange(M, device=q.device).unsqueeze(0) # (1, M)

        current_mask = (q_abs_idx // self.signal_window) == b_idx # (L_q, M)
        future_mask = b_idx > (q_abs_idx // self.signal_window)   # (L_q, M)

        # Patch logits
        logits = torch.where(current_mask.unsqueeze(0).unsqueeze(0), logits_current.unsqueeze(-1), logits_past)
        logits.masked_fill_(future_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(logits, dim=-1)

        # --- Vectorized Value Calculation ---
        out_past = torch.matmul(attn, v_block_means) # (B, H, L_q, D_h)

        prob_current = (attn * current_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1, keepdim=True) # (B, H, L_q, 1)

        # Gather the specific block mean for each query to subtract it
        block_indices = (q_abs_idx // self.signal_window).view(-1) # (L_q,)
        v_block_means_gathered = v_block_means[:, :, block_indices, :] # (B, H, L_q, D_h)

        incorrect_contribution = prob_current * v_block_means_gathered

        # Subtract incorrect block mean, add correct cumulative mean
        out = out_past - incorrect_contribution + (prob_current * v_current_diag)

        return out
