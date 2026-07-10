"""
Saliency eviction / cache management utilities for the KV cache.
"""

import torch


# =============================================================================
# SALIENCY EVICTION LOGIC
# =============================================================================

def apply_saliency_eviction(k, v, k_rope, scores, num_sinks=4, max_capacity=256):
    B, H, L, D = k.shape
    if L <= max_capacity:
        return k, v, k_rope, scores

    device = k.device
    sink_indices = torch.arange(num_sinks, device=device).unsqueeze(0).expand(B, -1)

    rest_scores = scores[:, num_sinks:]
    num_to_keep = max_capacity - num_sinks

    _, top_indices = torch.topk(rest_scores, num_to_keep, dim=-1)
    top_indices = top_indices + num_sinks

    keep_indices, _ = torch.sort(torch.cat([sink_indices, top_indices], dim=-1), dim=-1)

    gather_idx_kv = keep_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D)
    new_k = torch.gather(k, 2, gather_idx_kv)
    new_v = torch.gather(v, 2, gather_idx_kv)
    new_k_rope = torch.gather(k_rope, 2, gather_idx_kv)
    new_scores = torch.gather(scores, 1, keep_indices)

    return new_k, new_v, new_k_rope, new_scores
