"""
Position encoding utilities — RoPE with position interpolation.

Three free functions that all attention layers depend on:

* ``precompute_freqs_cis``      — precompute rotary frequency tables.
* ``reshape_for_broadcast``      — shape helper for applying freqs to embeddings.
* ``apply_rotary_emb``           — apply rotary embedding to query/key tensors.
"""

import torch


# =============================================================================
# RoPE with Position Interpolation
# =============================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, max_train_len: int = 4096):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if end > max_train_len:
        scaling_factor = max_train_len / end
        t = t * scaling_factor
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    shape = [d if i == 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
