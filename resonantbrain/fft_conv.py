"""
FFT-based causal convolution with carry state (SSM in disguise).

This module provides the ``FFTCausalConv`` layer which implements an
exponential-decay causal convolution in frequency space with O(L log L)
complexity.  It maintains a carry state across chunks for sequential
processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTCausalConv(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.rand(d_model) * 0.5 + 0.01)
        self.max_seq_len = max_seq_len

    def _build_decay_filter(self, L, device):
        alpha = F.softplus(self.log_alpha)
        t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
        h = torch.exp(-alpha.unsqueeze(1) * t)
        return h, alpha

    def forward(self, x, carry_state=None):
        B, L, D = x.shape
        input_dtype = x.dtype
        x_t = x.transpose(1, 2).contiguous()

        h, alpha = self._build_decay_filter(L, x.device)

        # Cast to float32 for FFT operations (FFT doesn't support bfloat16)
        x_padded = F.pad(x_t, (0, L)).float()
        h_padded = F.pad(h, (0, L)).float()
        X_freq = torch.fft.rfft(x_padded, n=2 * L)
        H_freq = torch.fft.rfft(h_padded, n=2 * L)
        y = torch.fft.irfft(X_freq * H_freq, n=2 * L)[..., :L]

        h_norm_sq = 1.0 / (1.0 - torch.exp(-2.0 * alpha.unsqueeze(1)))
        h_norm = torch.sqrt(h_norm_sq).clamp(min=1e-6)

        y = y / h_norm

        if carry_state is not None:
            carry_state_f32 = carry_state.float() if carry_state.dtype != torch.float32 else carry_state
            t_pos = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(0)
            carry_decay = torch.exp(-alpha.unsqueeze(1) * (t_pos + 1))
            y = y + carry_state_f32.unsqueeze(2) * carry_decay.unsqueeze(0)

        new_carry = y[:, :, -1].clone()
        # Cast back to original dtype
        y = y.to(input_dtype)
        new_carry = new_carry.to(input_dtype)
        return y.transpose(1, 2).contiguous(), new_carry
