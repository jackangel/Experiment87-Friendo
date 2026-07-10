"""
SSM-Attention Block — hybrid SSM convolution + Bayesian/Flash attention + MLP/graph.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rotary_emb
from .fft_conv import FFTCausalConv
from .eviction import apply_saliency_eviction
from .forgetting_gate import CognitiveForgettingGate
from .attention import BayesianSignalAttention
from .graph_reasoning import LatentGraphReasoning


# =============================================================================
# SSM-Attention Block
# =============================================================================

class SSMAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len, num_layers, dropout=0.1,
                 forgetting_config=None, use_eviction=True, saliency_decay=0.95,
                 use_flash_attn=False, graph_reasoning_config=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len
        self.use_eviction = use_eviction
        self.saliency_decay = saliency_decay
        self.use_flash_attn = use_flash_attn

        self.norm_ssm = nn.LayerNorm(dim)
        self.fft_conv = FFTCausalConv(dim, max_seq_len)
        self.ssm_dropout = nn.Dropout(dropout)

        self.norm_attn = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.wo.weight, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        self.bayesian_attn = BayesianSignalAttention(dim, num_heads, signal_window=2)

        self.attn_dropout = nn.Dropout(dropout)

        self.norm_mlp = nn.LayerNorm(dim)

        # --- Latent Graph Reasoning (optional replacement for MLP) ---
        self.graph_reasoning = None
        if graph_reasoning_config is not None:
            self.graph_reasoning = LatentGraphReasoning(
                dim=dim,
                num_rules=graph_reasoning_config.get('num_rules', 8),
                graph_steps=graph_reasoning_config.get('graph_steps', 3),
                top_k_edges=graph_reasoning_config.get('top_k_edges', 16)
            )

        # Original MLP (used when graph_reasoning is None)
        self.mlp_fc1 = nn.Linear(dim, dim * 4)
        self.mlp_act = nn.GELU()

        if forgetting_config is not None:
            self.mlp_forget_gate = CognitiveForgettingGate(
                dim * 4,
                enable_ablation=True,
                decay_factor=forgetting_config.get('decay_factor', 0.995),
                lock_threshold=forgetting_config.get('lock_threshold', 0.99),
                health_floor=forgetting_config.get('health_floor', 0.2),
                gated_fraction=forgetting_config.get('gated_fraction', 0.75)
            )
            # Separate gate for graph reasoning path (operates on dim, not dim*4)
            self.mlp_forget_gate_layer = CognitiveForgettingGate(
                dim,
                enable_ablation=True,
                decay_factor=forgetting_config.get('decay_factor', 0.995),
                lock_threshold=forgetting_config.get('lock_threshold', 0.99),
                health_floor=forgetting_config.get('health_floor', 0.2),
                gated_fraction=forgetting_config.get('gated_fraction', 0.75)
            )
        else:
            self.mlp_forget_gate = CognitiveForgettingGate(dim * 4, enable_ablation=False)
            self.mlp_forget_gate_layer = CognitiveForgettingGate(dim, enable_ablation=False)

        self.mlp_drop1 = nn.Dropout(dropout)
        self.mlp_fc2 = nn.Linear(dim * 4, dim)
        self.mlp_drop2 = nn.Dropout(dropout)

    def forward(self, x, freqs_cis_ext, abs_pos_offset=0, carry_state=None, past_kv=None, use_cache=False, global_step=None):
        B, L, D = x.shape

        ssm_out, new_carry = self.fft_conv(self.norm_ssm(x), carry_state)
        x = x + self.ssm_dropout(ssm_out)

        h = self.norm_attn(x)
        q = self.wq(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        freqs_cis_q = freqs_cis_ext[abs_pos_offset : abs_pos_offset + L]

        q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis_q)

        if past_kv is not None:
            past_k, past_v, past_scores, past_k_rope = past_kv
            k_full = torch.cat([past_k, k], dim=2)
            v_full = torch.cat([past_v, v], dim=2)
            k_rope_full = torch.cat([past_k_rope, k_rope], dim=2)
        else:
            k_full = k
            v_full = v
            k_rope_full = k_rope
            past_scores = torch.zeros((B, 0), device=x.device)

        # Only compute saliency if we need it for eviction or caching
        if use_cache and self.use_eviction:
            with torch.no_grad():
                q_proxy = q_rope[:, 0, :, :]
                k_proxy = k_rope_full[:, 0, :, :]
                proxy_scores = torch.matmul(q_proxy, k_proxy.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if L > 1:
                    mask = torch.triu(torch.ones(L, k_proxy.size(1), device=x.device), diagonal=k_proxy.size(1) - L + 1).bool()
                    proxy_scores.masked_fill_(mask, float('-inf'))
                proxy_weights = F.softmax(proxy_scores, dim=-1)
                current_saliency = proxy_weights.sum(dim=1)

            updated_scores = torch.cat([past_scores * self.saliency_decay, torch.zeros((B, L), device=x.device)], dim=1)
            updated_scores += current_saliency
        else:
            # Skip saliency computation when not needed (training or no eviction)
            updated_scores = torch.cat([past_scores, torch.zeros((B, L), device=x.device)], dim=1)

        if use_cache:
            if self.use_eviction:
                k_full, v_full, k_rope_full, updated_scores = apply_saliency_eviction(
                    k_full, v_full, k_rope_full, updated_scores, num_sinks=4, max_capacity=self.max_seq_len
                )
            new_kv = (k_full, v_full, updated_scores, k_rope_full)
        else:
            new_kv = None

        padded_L_k = k_full.size(2)
        pad_len = (self.bayesian_attn.signal_window - (padded_L_k % self.bayesian_attn.signal_window)) % self.bayesian_attn.signal_window
        total_L_k = padded_L_k + pad_len
        M = total_L_k // self.bayesian_attn.signal_window

        block_base_positions = torch.arange(0, M * self.bayesian_attn.signal_window, self.bayesian_attn.signal_window, device=x.device)
        freqs_cis_k_blocks = freqs_cis_ext[block_base_positions]

        # Use Flash Attention if enabled and available
        use_flash = self.use_flash_attn and hasattr(F, 'scaled_dot_product_attention') and k_full.size(2) <= 8192

        if use_flash:
            # Flash Attention path (3-5x faster)
            # Need to handle causal masking for cached KV
            attn_mask = None
            if k_full.size(2) > L:
                # We have cached KV, so we need to allow attention to all past tokens
                attn_mask = torch.zeros((L, k_full.size(2)), dtype=torch.bool, device=x.device)

            attn_out = F.scaled_dot_product_attention(
                q_rope, k_rope_full, v_full,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=(attn_mask is None)
            )
        else:
            # Bayesian Signal Attention path (custom implementation)
            attn_out = self.bayesian_attn(q, k_full, v_full, freqs_cis_q, freqs_cis_k_blocks)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.attn_dropout(self.wo(attn_out))

        m = self.norm_mlp(x)

        if self.graph_reasoning is not None:
            # --- Latent Graph Reasoning Path (replaces MLP) ---
            # Graph reasoning provides multi-relational message passing instead of
            # the standard expand->activate->contract MLP. The forgetting gate still
            # applies on top for cognitive memory management.
            m = self.graph_reasoning(m)
            m = self.mlp_forget_gate_layer(m)  # Separate gate for graph path
            m = self.mlp_drop2(m)
        else:
            # --- Standard MLP Path (original) ---
            m = self.mlp_fc1(m)
            m = self.mlp_act(m)
            m = self.mlp_forget_gate(m)
            m = self.mlp_drop1(m)
            m = self.mlp_fc2(m)
            m = self.mlp_drop2(m)

        x = x + m

        return x, new_carry, new_kv
