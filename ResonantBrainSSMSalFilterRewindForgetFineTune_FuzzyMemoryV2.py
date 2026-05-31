import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import random
import pyarrow.parquet as pq
import tiktoken
import math
import json
from datetime import datetime
from typing import Optional, Tuple, List

# Optional: for streaming large JSON files
try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False
    print("[INFO] ijson not installed. Large JSON files will be loaded entirely into memory.")
    print("[INFO] Install with: pip install ijson")

# Saliency-Guided State Memory System
try:
    from memory import SSMStateMemoryBank, MemoryEntry, MemoryConsolidation, MemoryRouter
    from memory.router import MemoryAugmentedGenerator
    HAS_MEMORY_SYSTEM = True
    print("[INFO] ✓ Saliency-Guided State Memory System loaded successfully")
except ImportError as e:
    HAS_MEMORY_SYSTEM = False
    print(f"[INFO] Memory system not available: {e}")
    print("[INFO] The memory/ directory should be in the same folder as this script")

# AGI-Like Reasoning Systems
try:
    from reasoning import (ThinkingController, LatentThinkingWrapper, 
                          ReasoningPattern, PatternMemoryBank, PatternDetector,
                          MetaCognitiveController)
    from reasoning.controller import generate_with_metacognition
    HAS_REASONING_SYSTEM = True
    print("[INFO] ✓ Latent Thinking & Meta-Cognition Systems loaded successfully")
except ImportError as e:
    HAS_REASONING_SYSTEM = False
    print(f"[INFO] Reasoning systems not available: {e}")
    print("[INFO] The reasoning/ directory should be in the same folder as this script")

# ==========================================
# 0. TIKTOKEN TOKENIZER & CHATML CONSTANTS
# ==========================================

# Using these variants prevents tiktoken's strict `<|...|>` regex from failing
CHAT_START = "<im_start>"
CHAT_END = "<im_end>"

class TiktokenTokenizer:
    def __init__(self, encoding_name="gpt2"):
        print(f"Loading tiktoken encoding: '{encoding_name}'...")
        base_tokenizer = tiktoken.get_encoding(encoding_name)
        
        # Explicitly register special tokens so they aren't split into characters
        special_tokens = {
            CHAT_START: base_tokenizer.n_vocab,
            CHAT_END: base_tokenizer.n_vocab + 1
        }
        
        self.tokenizer = tiktoken.Encoding(
            name="custom_chatml",
            pat_str=base_tokenizer._pat_str,
            mergeable_ranks=base_tokenizer._mergeable_ranks,
            special_tokens={**base_tokenizer._special_tokens, **special_tokens}
        )
        self.vocab_size = self.tokenizer.n_vocab

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, ids):
        return self.tokenizer.decode(ids)

# =============================================================================
# 1. RoPE with Position Interpolation
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

# =============================================================================
# 2. FFT Causal Conv with Carry State (SSM in disguise)
# =============================================================================

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

# =============================================================================
# 3. SALIENCY EVICTION LOGIC
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

# =============================================================================
# 3.5. COGNITIVE FORGETTING GATE
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
        
        if self.enable_ablation:
            self.num_gated = int(hidden_dim * gated_fraction)
            self.num_wildcard = hidden_dim - self.num_gated
            
            self.register_buffer("health", torch.full((self.num_gated,), 0.5))
            self.register_buffer("firing_ema", torch.zeros(self.num_gated))
            self.register_buffer("is_locked", torch.zeros(self.num_gated, dtype=torch.bool))
            self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if not self.enable_ablation:
            return x
        
        x_gated = x[..., :self.num_gated]
        x_wildcard = x[..., self.num_gated:] if self.num_wildcard > 0 else None
            
        if self.training:
            with torch.no_grad():
                self.step_count += 1
                
                fired = (x_gated > 1e-3).float()
                current_firing_rate = fired.mean(dim=(0, 1))
                
                self.firing_ema.copy_(self.firing_ema * 0.99 + current_firing_rate * 0.01)
                is_consistent = current_firing_rate >= (self.firing_ema * 0.8)
                
                health_update = torch.where(
                    is_consistent, 
                    self.health + 0.002,
                    self.health * self.decay_factor
                )
                
                self.health.copy_(torch.clamp(health_update, self.health_floor, 1.0))
                
                if self.step_count > 8000:
                    newly_locked = (self.health >= self.lock_threshold) & (self.firing_ema > 0.1) & (~self.is_locked)
                    self.is_locked = self.is_locked | newly_locked
                
                self.health.masked_fill_(self.is_locked, 1.0)
                
        x_gated = x_gated * self.health.view(1, 1, -1)
        
        if x_wildcard is not None:
            return torch.cat([x_gated, x_wildcard], dim=-1)
        else:
            return x_gated

# =============================================================================
# 3.75. BAYESIAN SIGNAL ATTENTION
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

# =============================================================================
# 4. SSM-Attention Block with GQA and MoD
# =============================================================================

class SSMAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len, num_layers, dropout=0.1, 
                 forgetting_config=None, use_eviction=True, saliency_decay=0.95,
                 num_kv_heads=None, enable_mod=False, mod_top_k_ratio=0.75):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len
        self.use_eviction = use_eviction
        self.saliency_decay = saliency_decay
        
        # === GQA (Grouped-Query Attention) ===
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_q_groups = num_heads // self.num_kv_heads
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        # === MoD (Mixture of Depths) ===
        self.enable_mod = enable_mod
        self.mod_top_k_ratio = mod_top_k_ratio
        if enable_mod:
            # Lightweight router: token importance scorer
            self.mod_router = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 1)
            )
            self.register_buffer('mod_step_count', torch.tensor(0, dtype=torch.long))

        self.norm_ssm = nn.LayerNorm(dim)
        self.fft_conv = FFTCausalConv(dim, max_seq_len)
        self.ssm_dropout = nn.Dropout(dropout)

        self.norm_attn = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        # GQA: K and V projections use fewer heads
        self.wk = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.wo.weight, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        
        self.bayesian_attn = BayesianSignalAttention(dim, num_heads, signal_window=2)
        
        self.attn_dropout = nn.Dropout(dropout)

        self.norm_mlp = nn.LayerNorm(dim)
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
        else:
            self.mlp_forget_gate = CognitiveForgettingGate(dim * 4, enable_ablation=False)
        
        self.mlp_drop1 = nn.Dropout(dropout)
        self.mlp_fc2 = nn.Linear(dim * 4, dim)
        self.mlp_drop2 = nn.Dropout(dropout)

    def forward(self, x, freqs_cis_ext, abs_pos_offset=0, carry_state=None, past_kv=None, use_cache=False):
        B, L, D = x.shape
        
        # === MoD: Token Selection (only during training) ===
        mod_router_logits = None
        mod_selected_mask = None
        
        if self.enable_mod and self.training:
            with torch.no_grad() if not self.training else torch.enable_grad():
                router_scores = self.mod_router(x).squeeze(-1)  # (B, L)
                mod_router_logits = router_scores  # Save for load balancing loss
                
                # Select top-k tokens per batch item
                k = max(1, int(L * self.mod_top_k_ratio))
                _, top_indices = torch.topk(router_scores, k, dim=-1)
                
                # Create selection mask
                mod_selected_mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
                mod_selected_mask.scatter_(1, top_indices, True)
                
                self.mod_step_count += 1

        ssm_out, new_carry = self.fft_conv(self.norm_ssm(x), carry_state)
        x = x + self.ssm_dropout(ssm_out)

        h = self.norm_attn(x)
        q = self.wq(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # === GQA: K and V use fewer heads ===
        k = self.wk(h).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # === GQA: Expand KV heads to match Q heads ===
        if self.num_kv_heads < self.num_heads:
            # Repeat each KV head num_q_groups times
            k = k.repeat_interleave(self.num_q_groups, dim=1)
            v = v.repeat_interleave(self.num_q_groups, dim=1)
        
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

        # Try to use Flash Attention if available for speedup
        use_flash = hasattr(F, 'scaled_dot_product_attention') and k_full.size(2) <= 8192
        
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

        # === MLP with optional MoD skipping ===
        if self.enable_mod and self.training and mod_selected_mask is not None:
            # Only process selected tokens through MLP
            selected_indices = mod_selected_mask.nonzero(as_tuple=True)
            if len(selected_indices[0]) > 0:
                x_selected = x[selected_indices[0], selected_indices[1], :]
                
                m = self.norm_mlp(x_selected.unsqueeze(0))
                m = self.mlp_fc1(m)
                m = self.mlp_act(m)
                m = self.mlp_forget_gate(m)
                m = self.mlp_drop1(m)
                m = self.mlp_fc2(m)
                m = self.mlp_drop2(m)
                
                # Add MLP output to selected tokens
                x_mlp = x.clone()
                x_mlp[selected_indices[0], selected_indices[1], :] = x_selected + m.squeeze(0)
                x = x_mlp
        else:
            # Standard MLP path (inference or MoD disabled)
            m = self.norm_mlp(x)
            m = self.mlp_fc1(m)
            m = self.mlp_act(m)
            m = self.mlp_forget_gate(m)
            m = self.mlp_drop1(m)
            m = self.mlp_fc2(m)
            m = self.mlp_drop2(m)
            x = x + m

        # Return router logits for load balancing loss if MoD is enabled
        return x, new_carry, new_kv, mod_router_logits

# =============================================================================
# 5. SSM Transformer
# =============================================================================

def get_forgetting_config(layer_idx, num_layers, enable_forgetting):
    if not enable_forgetting:
        return None
    depth_ratio = layer_idx / max(1, num_layers - 1)
    return {
        # Decreased from 0.980 to 0.950 (decays faster during grad accum steps)
        'decay_factor': 0.975 + (depth_ratio * 0.018),      
        
        # Increased from 0.90 to 0.95 (requires higher sustained health to lock)
        'lock_threshold': 0.90 + (depth_ratio * 0.04),      
        
        'health_floor': 0.1 + (depth_ratio * 0.3),          
        'gated_fraction': 0.9 - (depth_ratio * 0.6),        
    }

class SSMTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, max_seq_len=512, 
                 dropout=0.1, enable_forgetting=False, saliency_decay=0.95,
                 num_kv_heads=None, enable_mod=False, mod_top_k_ratio=0.75):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.head_dim = dim // num_heads
        self.saliency_decay = saliency_decay
        self.enable_mod = enable_mod
        
        # GQA: Default to 1/4 of num_heads for KV heads if not specified
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else max(1, num_heads // 4)

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            SSMAttentionBlock(
                dim, num_heads, max_seq_len, num_layers, dropout,
                forgetting_config=get_forgetting_config(i, num_layers, enable_forgetting),
                use_eviction=(i >= 1),
                saliency_decay=saliency_decay,
                num_kv_heads=self.num_kv_heads,
                enable_mod=enable_mod,
                mod_top_k_ratio=mod_top_k_ratio
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len, max_train_len=max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)

        freqs_cis_ext = precompute_freqs_cis(self.head_dim, max_seq_len * 8, max_train_len=max_seq_len)
        self.register_buffer("freqs_cis_ext", freqs_cis_ext)
        
        # Print configurations
        print(f"\n{'='*60}")
        print(f"Model Configuration")
        print(f"{'='*60}")
        print(f"  Dimensions: {dim}, Heads: {num_heads}, Layers: {num_layers}")
        print(f"  GQA Enabled: num_kv_heads={self.num_kv_heads} (KV cache reduction: {num_heads//self.num_kv_heads}x)")
        print(f"  MoD Enabled: {enable_mod}" + (f" (top-k ratio: {mod_top_k_ratio:.2%})" if enable_mod else ""))
        print(f"{'='*60}")
        
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

    def forward(self, x=None, inputs_embeds=None, carry_states=None, is_training=True, past_key_values=None, use_cache=False, abs_pos_offset=0):
        # Support for inputs_embeds to enable Fuzzy Training
        if inputs_embeds is not None:
            h = self.embed_dropout(inputs_embeds)
        elif x is not None:
            h = self.embed_dropout(self.tok_embeddings(x))
        else:
            raise ValueError("You must specify either x or inputs_embeds")

        if carry_states is None:
            carry_states = [None] * self.num_layers

        new_carry_states = []
        new_key_values = []
        mod_router_logits_all = []

        for i, layer in enumerate(self.layers):
            layer_past_kv = past_key_values[i] if past_key_values is not None else None
            h, new_carry, new_kv, mod_router_logits = layer(
                h, self.freqs_cis_ext, abs_pos_offset=abs_pos_offset,
                carry_state=carry_states[i],
                past_kv=layer_past_kv,
                use_cache=use_cache
            )
            new_carry_states.append(new_carry)
            new_key_values.append(new_kv)
            if mod_router_logits is not None:
                mod_router_logits_all.append(mod_router_logits)

        h = self.norm(h)
        logits = self.output(h)
        
        # Calculate MoD load balancing loss if enabled
        mod_load_balance_loss = None
        if self.enable_mod and is_training and len(mod_router_logits_all) > 0:
            # Stabilized KL divergence using log_softmax (avoids log(0) = -inf → NaN)
            all_router_scores = torch.stack(mod_router_logits_all, dim=0)  # (num_layers, B, L)
            
            # Use log_softmax for numerical stability
            log_probs = F.log_softmax(all_router_scores, dim=-1)
            uniform_log_probs = -torch.log(torch.tensor(log_probs.size(-1), dtype=log_probs.dtype, device=log_probs.device))
            
            # KL(uniform || model) with both inputs in log space
            mod_load_balance_loss = F.kl_div(
                log_probs, 
                uniform_log_probs.expand_as(log_probs), 
                reduction='batchmean',
                log_target=True  # Both inputs are in log space
            ) * 0.01
            
            # Clamp to prevent extreme values
            mod_load_balance_loss = torch.clamp(mod_load_balance_loss, max=10.0)

        return logits, new_carry_states, new_key_values, mod_load_balance_loss

# =============================================================================
# 6. DATA UTILITIES (Pre-training & Fine-tuning streams)
# =============================================================================

def apply_sampling_penalties(logits, generated_ids, repetition_penalty=1.2, top_k=50, top_p=0.9):
    if repetition_penalty != 1.0:
        for token in set(generated_ids):
            if logits[token] < 0:
                logits[token] *= repetition_penalty
            else:
                logits[token] /= repetition_penalty
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
    return logits

def validate_vocab_size(model, tokenizer):
    model_vocab = model.tok_embeddings.num_embeddings
    tokenizer_vocab = tokenizer.vocab_size
    if model_vocab != tokenizer_vocab:
        raise ValueError(f"CRITICAL: Model vocab_size ({model_vocab}) != Tokenizer vocab_size ({tokenizer_vocab}).")

def load_checkpoint_with_filter(model, checkpoint_state_dict):
    """
    Load checkpoint while filtering out keys with size mismatches (e.g., forgetting gate buffers).
    This allows resuming training after changing forgetting hyperparameters.
    """
    model_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []
    
    for key, value in checkpoint_state_dict.items():
        if key in model_state:
            if value.shape == model_state[key].shape:
                filtered_state[key] = value
            else:
                skipped_keys.append(f"{key} (ckpt: {value.shape}, model: {model_state[key].shape})")
        else:
            skipped_keys.append(f"{key} (not in current model)")
    
    if skipped_keys:
        print(f"[INFO] Skipped {len(skipped_keys)} mismatched keys (forgetting gate buffers):")
        for key in skipped_keys[:5]:  # Show first 5
            print(f"  - {key}")
        if len(skipped_keys) > 5:
            print(f"  ... and {len(skipped_keys) - 5} more")
    
    model.load_state_dict(filtered_state, strict=False)
    return len(skipped_keys)

# --- Stream 1: Plain Text Parquet (Pre-training) - Flat Generator for Fuzzy Training ---
def token_generator_from_parquet(files, text_column, tokenizer):
    for file in files:
        try:
            parquet_file = pq.ParquetFile(file)
            for batch in parquet_file.iter_batches(batch_size=500, columns=[text_column]):
                df = batch.to_pandas()
                for text in df[text_column].dropna():
                    yield from tokenizer.encode(str(text))
        except Exception as e:
            print(f"[WARNING] Failed to read parquet file {file}: {e}")
            continue

# --- Stream 2: OpenHermes JSON / ChatML format (Fine-Tuning) ---
def stream_chatml_from_json(json_file, tokenizer, seq_len, device, batch_size=4, use_streaming=True):
    """
    Stream ChatML data from JSON file.
    If use_streaming=True and ijson available, uses streaming (memory efficient for large files).
    If use_streaming=False, loads entire file (faster but uses more RAM).
    """
    if use_streaming and HAS_IJSON:
        # Memory-efficient streaming mode for huge datasets
        print(f"\n[Dataset] Streaming JSON dataset from {json_file} (memory-efficient mode)...\n")
        with open(json_file, 'r', encoding='utf-8') as f:
            data_stream = ijson.items(f, 'item')
    else:
        # Original mode: load entire file
        if use_streaming and not HAS_IJSON:
            print(f"[WARNING] ijson not available, falling back to loading entire file.")
        print(f"\n[Dataset] Loading entire JSON dataset from {json_file}. This might take a minute...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[Dataset] Successfully loaded {len(data)} conversations. Generating ChatML masks...\n")
        
        random.shuffle(data)
        data_stream = iter(data)
    
    buffer_ids = []
    buffer_mask = []
    
    batch_ids = []
    batch_masks = []
    
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    
    for item in data_stream:
        conversations = item.get("conversations", [])
        if not conversations:
            continue
            
        for msg in conversations:
            role = role_map.get(msg.get("from", ""), msg.get("from", ""))
            content = msg.get("value", "")
            
            if role == "assistant":
                prefix_tokens = tokenizer.encode(f"{CHAT_START}{role}\n")
                content_tokens = tokenizer.encode(f"{content}{CHAT_END}\n")
                
                buffer_ids.extend(prefix_tokens + content_tokens)
                buffer_mask.extend([0]*len(prefix_tokens) + [1]*len(content_tokens)) 
            else:
                text_str = f"{CHAT_START}{role}\n{content}{CHAT_END}\n"
                tokens = tokenizer.encode(text_str)
                buffer_ids.extend(tokens)
                buffer_mask.extend([0]*len(tokens))
                
        while len(buffer_ids) >= seq_len + 1:
            chunk_ids = buffer_ids[:seq_len + 1]
            chunk_mask = buffer_mask[:seq_len + 1]
            
            buffer_ids = buffer_ids[seq_len:]
            buffer_mask = buffer_mask[seq_len:]
            
            batch_ids.append(chunk_ids)
            batch_masks.append(chunk_mask)
            
            if len(batch_ids) == batch_size:
                yield (
                    torch.tensor(batch_ids, dtype=torch.long, device=device),
                    torch.tensor(batch_masks, dtype=torch.float32, device=device)
                )
                batch_ids, batch_masks = [], []
                
    if batch_ids: 
        yield (
            torch.tensor(batch_ids, dtype=torch.long, device=device),
            torch.tensor(batch_masks, dtype=torch.float32, device=device)
        )

# =============================================================================
# 7. COGNITIVE MEMORY MANAGER
# =============================================================================

class CognitiveMemoryManager:
    def __init__(self, device, max_paragraphs=50):
        self.device = device
        self.max_paragraphs = max_paragraphs  # Prevent unbounded growth
        self.paragraph_states = []
        self.paragraph_tokens = []

    def save_paragraph_state(self, carry_states, past_key_values, tokens):
        # Keep states on GPU to avoid CPU-GPU transfers (much faster)
        # Only move to CPU if approaching max_paragraphs limit
        should_cpu = len(self.paragraph_states) >= self.max_paragraphs * 0.8
        
        if should_cpu:
            cpu_carry = [c.detach().cpu().clone() if c is not None else None for c in carry_states] if carry_states else None
            cpu_kv = []
            if past_key_values:
                for k, v, s, kr in past_key_values:
                    cpu_kv.append((k.detach().cpu().clone(), v.detach().cpu().clone(), s.detach().cpu().clone(), kr.detach().cpu().clone()))
            else:
                cpu_kv = None
        else:
            # Keep on GPU for faster access
            cpu_carry = [c.detach().clone() if c is not None else None for c in carry_states] if carry_states else None
            cpu_kv = []
            if past_key_values:
                for k, v, s, kr in past_key_values:
                    cpu_kv.append((k.detach().clone(), v.detach().clone(), s.detach().clone(), kr.detach().clone()))
            else:
                cpu_kv = None
            
        self.paragraph_states.append({
            'carry_states': cpu_carry,
            'past_key_values': cpu_kv
        })
        self.paragraph_tokens.append(tokens)
        
        # Enforce max paragraphs limit (keep most recent)
        if len(self.paragraph_states) > self.max_paragraphs:
            self.paragraph_states = self.paragraph_states[-self.max_paragraphs:]
            self.paragraph_tokens = self.paragraph_tokens[-self.max_paragraphs:]

    def get_paragraph_state(self, idx):
        snap = self.paragraph_states[idx]
        dev_carry = [c.to(self.device) if c is not None else None for c in snap['carry_states']] if snap['carry_states'] else None
        dev_kv = []
        if snap['past_key_values']:
            for k, v, s, kr in snap['past_key_values']:
                dev_kv.append((k.to(self.device), v.to(self.device), s.to(self.device), kr.to(self.device)))
        else:
            dev_kv = None
        return dev_carry, dev_kv

# =============================================================================
# 8. GENERATION
# =============================================================================

def generate_block_recurrent(model, context_ids, tokenizer, device,
                             max_new_tokens=256, chunk_size=512,
                             temperature=0.8, repetition_penalty=1.2,
                             top_k=50, top_p=0.9, enable_rewind=True,
                             stop_sequence=None, max_paragraph_cache=50):
    model.eval()
    memory_manager = CognitiveMemoryManager(device, max_paragraphs=max_paragraph_cache)

    with torch.inference_mode():
        generated_ids = context_ids.copy()
        
        paragraphs = [context_ids[i:i + chunk_size] for i in range(0, len(context_ids), chunk_size)]
        
        # Process context chunks with proper position tracking
        carry_states = None
        past_key_values = None
        cumulative_pos = 0
        for chunk in paragraphs:
            if len(chunk) == 0: 
                continue
            chunk_tensor = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(device)
            _, carry_states, past_key_values, _ = model(
                x=chunk_tensor, carry_states=carry_states, is_training=False, 
                past_key_values=past_key_values, use_cache=True, abs_pos_offset=cumulative_pos
            )
            cumulative_pos += len(chunk)
            memory_manager.save_paragraph_state(carry_states, past_key_values, chunk)

        tokens_generated = 0
        context_length = len(generated_ids)  # Track where new generation starts
        
        # Initialize with the last paragraph state and correct position
        if len(memory_manager.paragraph_tokens) > 0:
            active_carry, active_kv = memory_manager.get_paragraph_state(len(paragraphs) - 1)
            abs_pos_offset = cumulative_pos
        else:
            active_carry, active_kv = None, None
            abs_pos_offset = 0
        
        while tokens_generated < max_new_tokens:
            last_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)
            logits, active_carry, active_kv, _ = model(
                x=last_token, carry_states=active_carry, is_training=False, 
                past_key_values=active_kv, use_cache=True, abs_pos_offset=abs_pos_offset
            )
            
            # Increment position for next iteration
            abs_pos_offset += 1

            # Convert to float32 for numerical stability during sampling
            next_token_logits = logits[0, -1].float().clone()
            next_token_logits = apply_sampling_penalties(
                next_token_logits, generated_ids, repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p
            )
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            
            if torch.isnan(probs).any():
                print(f"\n[ERROR] NaN detected in sampling probabilities at token {tokens_generated}!")
                print(f"[ERROR] Last logits stats: min={next_token_logits.min():.2f}, max={next_token_logits.max():.2f}, mean={next_token_logits.mean():.2f}")
                print(f"[ERROR] Temperature: {temperature}, Last token: {generated_ids[-1]}")
                next_token = tokenizer.tokenizer.eot_token
            else:
                next_token = torch.multinomial(probs, 1).item()
            
            generated_ids.append(next_token)
            tokens_generated += 1
            
            #print(f"\n[DEBUG] Token {tokens_generated}: {next_token} -> '{tokenizer.decode([next_token])}'")
            
            if next_token == tokenizer.tokenizer.eot_token:
                #print(f"[DEBUG] EOT token detected. Breaking. EOT={tokenizer.tokenizer.eot_token}")
                break
            
            if stop_sequence:
                # Only check newly generated tokens, not the context
                new_tokens_only = generated_ids[context_length:]
                check_len = min(len(new_tokens_only), 10)
                recent_text = tokenizer.decode(new_tokens_only[-check_len:])
                if stop_sequence in recent_text:
                    print(f"[DEBUG] Stop sequence '{stop_sequence}' found in '{recent_text}'. Breaking.")
                    break

    return generated_ids

# =============================================================================
# 9. TRAINING & FINE-TUNING LOOPS
# =============================================================================

def print_gate_stats(model, iteration, running_loss, train_steps, scheduler, step_type="CLEAR"):
    current_lr = scheduler.get_last_lr()[0]
    log_str = f"[Step {iteration}] Type: {step_type:5s} | Loss: {running_loss / max(1, train_steps):.4f} | LR: {current_lr:.2e}"
    
    # Handle wrapped models (get base_model if needed)
    base_model = model.base_model if hasattr(model, 'base_model') else model
    
    total_locked, total_health, total_gated, total_wildcard = 0, 0, 0, 0
    for layer in base_model.layers:
        gate = layer.mlp_forget_gate
        if gate.enable_ablation:
            total_locked += gate.is_locked.sum().item()
            total_health += gate.health.sum().item()
            total_gated += gate.num_gated
            total_wildcard += gate.num_wildcard
            
    if total_gated > 0:
        locked_pct = (total_locked / total_gated) * 100
        health_avg = (total_health / total_gated) * 100
        log_str += f" | Gated: {total_gated} ({locked_pct:.1f}% locked, {health_avg:.1f}% health)"
    print(log_str)


def compute_thinking_loss(model, hidden_state, target_confidence=0.8):
    """
    Compute loss for training thinking system
    
    Encourages model to:
    1. Be confident when predictions are correct
    2. Be uncertain when predictions are uncertain
    3. Learn appropriate thinking depth
    
    Args:
        model: LatentThinkingWrapper instance
        hidden_state: Final hidden state from forward pass (B, L, D)
        target_confidence: Target confidence for correct predictions
    
    Returns:
        thinking_loss: Combined loss for thinking components
    """
    if not hasattr(model, 'thinking_controller'):
        return torch.tensor(0.0, device=hidden_state.device)
    
    # Get thinking controller predictions
    confidence, should_think, predicted_depth, uncertainty = model.thinking_controller(hidden_state)
    
    # Loss: Encourage higher confidence (regularization)
    # In practice, this should be supervised with actual correctness feedback
    confidence_loss = F.mse_loss(confidence, torch.full_like(confidence, target_confidence))
    
    # Regularize thinking depth (prefer shallower thinking to save compute)
    # predicted_depth is Long tensor (argmax result), convert to float for mean
    depth_reg = predicted_depth.float().mean()
    
    thinking_loss = confidence_loss + 0.01 * depth_reg
    
    return thinking_loss


# --- 9A. Pre-training (Causal LM on Plain Text with Fuzzy Training & Grad Accum) ---
def run_pretraining(model, parquet_files, text_column, tokenizer, optimizer, device,
                    vocab_size, start_iteration=0, chunk_size=512, enable_forgetting=False, 
                    batch_size=4, grad_accum_steps=4, fuzzy_steps=1, clear_steps=3, bag_size=2,
                    thinking_loss_weight=0.0):
    
    print(f"\n--- Starting Pre-training (Parquet) | Device: {device} | Batch Size: {batch_size} | Grad Accum: {grad_accum_steps} ---")
    print(f"--- Fuzzy Training Enabled: {fuzzy_steps} Fuzzy / {clear_steps} Clear | Bag Size: {bag_size} ---")
    print(f"--- Flash Attention: {'Available' if hasattr(F, 'scaled_dot_product_attention') else 'Not Available'} ---")
    
    # Check if model is wrapped with thinking system
    is_thinking_wrapper = hasattr(model, 'thinking_controller')
    base_model = model.base_model if is_thinking_wrapper else model
    if is_thinking_wrapper:
        print(f"--- Thinking System Training Enabled (loss weight: {thinking_loss_weight}) ---")
    
    iteration = start_iteration
    random.shuffle(parquet_files)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    
    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(ptdtype == torch.float16))
    
    cycle_length = fuzzy_steps + clear_steps
    token_stream = token_generator_from_parquet(parquet_files, text_column, tokenizer)
    buffer = []
    
    carry_states, past_key_values, abs_pos_offset = None, None, 0
    running_train_loss, train_steps = 0.0, 0

    optimizer.zero_grad(set_to_none=True)

    while True:
        is_fuzzy_step = (fuzzy_steps > 0) and ((iteration % cycle_length) < fuzzy_steps)
        
        required_len = (chunk_size + 1) * bag_size if is_fuzzy_step else chunk_size + 1
        advance_len = chunk_size * bag_size if is_fuzzy_step else chunk_size
        
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

        if abs_pos_offset + chunk_size > base_model.freqs_cis_ext.size(0):
            carry_states, past_key_values, abs_pos_offset = None, None, 0

        detached_carry = [c.detach() for c in carry_states] if carry_states else None
        detached_kv = [(k.detach(), v.detach(), s.detach(), kr.detach()) for k, v, s, kr in past_key_values] if past_key_values else None

        with torch.autocast(device_type=device, dtype=ptdtype):
            if is_fuzzy_step:
                inputs_flat = chunk[:, :-bag_size]
                targets_flat = chunk[:, bag_size:]
                
                embs = base_model.tok_embeddings(inputs_flat)
                embs = embs.view(batch_size, chunk_size, bag_size, -1).mean(dim=2)
                
                y_bags = targets_flat.view(batch_size, chunk_size, bag_size)
                
                logits, carry_states, past_key_values, mod_lb_loss = model(
                    inputs_embeds=embs, carry_states=detached_carry, past_key_values=detached_kv, 
                    is_training=True, use_cache=True, abs_pos_offset=abs_pos_offset
                )
                
                # Get actual vocab size from logits (may be extended by thinking wrapper)
                actual_vocab_size = logits.size(-1)
                
                soft_targets = torch.zeros_like(logits).scatter_add_(
                    -1, y_bags, torch.ones_like(y_bags, dtype=logits.dtype) / bag_size
                )
                
                loss = F.cross_entropy(logits.view(-1, actual_vocab_size), soft_targets.view(-1, actual_vocab_size))
                if mod_lb_loss is not None:
                    loss = loss + mod_lb_loss
            else:
                x, y = chunk[:, :-1], chunk[:, 1:]
                
                logits, carry_states, past_key_values, mod_lb_loss = model(
                    x=x, carry_states=detached_carry, past_key_values=detached_kv, 
                    is_training=True, use_cache=True, abs_pos_offset=abs_pos_offset
                )
                
                # Get actual vocab size from logits (may be extended by thinking wrapper)
                actual_vocab_size = logits.size(-1)
                
                loss = F.cross_entropy(logits.view(-1, actual_vocab_size), y.reshape(-1))
                if mod_lb_loss is not None:
                    loss = loss + mod_lb_loss
            
            # Add thinking loss if model is wrapped
            if is_thinking_wrapper and thinking_loss_weight > 0:
                # Get hidden state for thinking loss
                # Note: We need gradients to flow, so don't use no_grad here
                if is_fuzzy_step:
                    hidden_state = base_model.norm(embs)
                else:
                    h = base_model.tok_embeddings(x)
                    hidden_state = base_model.norm(h)
                
                think_loss = compute_thinking_loss(model, hidden_state)
                loss = loss + thinking_loss_weight * think_loss
                
            abs_pos_offset += chunk_size
            loss = loss / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        running_train_loss += loss.item() * grad_accum_steps
        train_steps += 1
        
        if train_steps % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
            iteration += 1

            if iteration % 100 == 0:
                step_type = "FUZZY" if is_fuzzy_step else "CLEAR"
                print_gate_stats(model, iteration, running_train_loss, grad_accum_steps * 100, scheduler, step_type)
                running_train_loss = 0.0

            if iteration % 10000 == 0:
                model.eval()
                print(f"\n{'='*60}\n[GENERATION SAMPLE (Pre-training Coherence)]\n{'='*60}")
                test_prompt = "The rapid advancement of artificial intelligence has led to"
                # Use base model for generation test
                test_model = base_model if is_thinking_wrapper else model
                gen_ids = generate_block_recurrent(
                    test_model, tokenizer.encode(test_prompt), tokenizer, device, max_new_tokens=150, 
                    chunk_size=chunk_size, temperature=0.7
                )
                print(f"{tokenizer.decode(gen_ids)}\n")
                model.train()

                torch.save({
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iteration, 'chunk_size': chunk_size,
                }, 'checkpoint_ssm_pretrain.pth')

# --- 9B. Fine-tuning (Masked Instruction Tuning on ChatML JSON with Grad Accum) ---
def run_finetuning(model, json_file, tokenizer, optimizer, device,
                   vocab_size, start_iteration=0, chunk_size=512, enable_forgetting=False, 
                   batch_size=2, grad_accum_steps=4, use_streaming=True, thinking_loss_weight=0.0):
    
    print(f"\n--- Starting ChatML Fine-tuning (OpenHermes) | Device: {device} | Batch Size: {batch_size} | Grad Accum: {grad_accum_steps} ---")
    print(f"--- Flash Attention: {'Available' if hasattr(F, 'scaled_dot_product_attention') else 'Not Available'} ---")
    
    # Check if model is wrapped with thinking system
    is_thinking_wrapper = hasattr(model, 'thinking_controller')
    base_model = model.base_model if is_thinking_wrapper else model
    if is_thinking_wrapper:
        print(f"--- Thinking System Training Enabled (loss weight: {thinking_loss_weight}) ---")
    
    iteration = start_iteration
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)

    token_stream = stream_chatml_from_json(json_file, tokenizer, chunk_size, device, batch_size, use_streaming)
    
    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(ptdtype == torch.float16))
    
    carry_states, past_key_values, abs_pos_offset = None, None, 0
    running_train_loss, train_steps = 0.0, 0
    
    optimizer.zero_grad(set_to_none=True)
    
    for x_chunk, mask_chunk in token_stream:
        x, y = x_chunk[:, :-1], x_chunk[:, 1:]
        m = mask_chunk[:, 1:] 

        if abs_pos_offset + x.size(1) > base_model.freqs_cis_ext.size(0):
            carry_states, past_key_values, abs_pos_offset = None, None, 0

        detached_carry = [c.detach() for c in carry_states] if carry_states else None
        detached_kv = [(k.detach(), v.detach(), s.detach(), kr.detach()) for k, v, s, kr in past_key_values] if past_key_values else None

        with torch.autocast(device_type=device, dtype=ptdtype):
            logits, carry_states, past_key_values, mod_lb_loss = model(
                x=x, carry_states=detached_carry, past_key_values=detached_kv, 
                is_training=True, use_cache=True, abs_pos_offset=abs_pos_offset
            )
            abs_pos_offset += x.size(1)
            
            # Get actual vocab size from logits (may be extended by thinking wrapper)
            actual_vocab_size = logits.size(-1)
            
            loss = F.cross_entropy(logits.view(-1, actual_vocab_size), y.view(-1), reduction='none')
            loss = (loss * m.view(-1)).sum() / max(m.sum().item(), 1.0)
            
            # Add thinking loss if model is wrapped
            if is_thinking_wrapper and thinking_loss_weight > 0:
                # Get hidden state for thinking loss
                # Note: We need gradients to flow, so don't use no_grad here
                h = base_model.tok_embeddings(x)
                hidden_state = base_model.norm(h)
                
                think_loss = compute_thinking_loss(model, hidden_state)
                loss = loss + thinking_loss_weight * think_loss
            if mod_lb_loss is not None:
                loss = loss + mod_lb_loss
            loss = loss / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        running_train_loss += loss.item() * grad_accum_steps
        train_steps += 1
        
        if train_steps % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
            iteration += 1

            if iteration % 100 == 0:
                print_gate_stats(model, iteration, running_train_loss, grad_accum_steps * 100, scheduler, "CLEAR")
                running_train_loss = 0.0

            if iteration % 2000 == 0:
                model.eval()
                print(f"\n{'='*60}\n[GENERATION SAMPLE (Instruction Following)]\n{'='*60}")
                test_prompt = f"{CHAT_START}user\nWhat is the purpose of AI fine-tuning?{CHAT_END}\n{CHAT_START}assistant\n"
                # Use base model for generation test
                test_model = base_model if is_thinking_wrapper else model
                gen_ids = generate_block_recurrent(
                    test_model, tokenizer.encode(test_prompt), tokenizer, device, max_new_tokens=150, 
                    chunk_size=chunk_size, temperature=0.7, stop_sequence=CHAT_END
                )
                print(f"{tokenizer.decode(gen_ids)}\n")
                model.train()

                torch.save({
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iteration, 'chunk_size': chunk_size,
                }, 'checkpoint_ssm_finetune.pth')

# =============================================================================
# 9.5. MEMORY-AUGMENTED GENERATION
# =============================================================================

def initialize_memory_system(device='cuda', 
                            episodic_capacity=50,
                            consolidation_threshold=0.7,
                            injection_mode='state_fusion'):
    """
    Initialize the Saliency-Guided State Memory System
    
    Args:
        device: Device for memory operations
        episodic_capacity: Max episodic memories before consolidation
        consolidation_threshold: Min importance for semantic promotion
        injection_mode: 'state_fusion', 'kv_injection', or 'context_prepend'
    
    Returns:
        Tuple of (MemoryConsolidation, MemoryRouter)
    """
    if not HAS_MEMORY_SYSTEM:
        raise RuntimeError("Memory system not available. Ensure memory/ directory exists.")
    
    print(f"\n{'='*60}")
    print(f"Initializing Saliency-Guided State Memory System")
    print(f"{'='*60}")
    
    # Create consolidation engine
    consolidation = MemoryConsolidation(
        device=device,
        episodic_capacity=episodic_capacity,
        consolidation_threshold=consolidation_threshold,
        decay_rate=0.95,
        access_threshold=3,
        merge_similarity_threshold=0.9
    )
    
    # Create memory router
    router = MemoryRouter(
        consolidation=consolidation,
        device=device,
        default_injection_mode=injection_mode
    )
    
    print(f"{'='*60}\n")
    
    return consolidation, router


def store_conversation_memory(model, tokenizer, text, consolidation, device, 
                              metadata=None, chunk_size=512):
    """
    Process text through the model and store as memory
    
    Args:
        model: SSMTransformer model
        tokenizer: Tokenizer instance
        text: Text to store as memory
        consolidation: MemoryConsolidation instance
        device: Device
        metadata: Optional metadata dict
        chunk_size: Chunk size for processing
    
    Returns:
        memory_id: ID of stored memory
    """
    model.eval()
    
    with torch.inference_mode():
        # Tokenize
        tokens = tokenizer.encode(text)
        
        # Process through model to get carry states and KV cache
        token_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        _, carry_states, past_kv, _ = model(
            x=token_tensor,
            carry_states=None,
            is_training=False,
            past_key_values=None,
            use_cache=True,
            abs_pos_offset=0
        )
        
        # Create memory entry
        from memory.core import MemoryEntry
        import time
        import uuid
        
        # Extract saliency scores and create embedding
        avg_saliency = torch.stack([s for _, _, s, _ in past_kv], dim=0).mean(dim=0).mean(dim=0)
        memory_embedding = carry_states[-1].mean(dim=0) if carry_states[-1].dim() == 2 else carry_states[-1].view(-1)
        
        memory_entry = MemoryEntry(
            id=str(uuid.uuid4()),
            carry_states=[c.detach().clone() for c in carry_states],
            kv_cache=[(k.detach().clone(), v.detach().clone(), s.detach().clone(), kr.detach().clone())
                     for k, v, s, kr in past_kv],
            saliency_map=avg_saliency.detach().clone(),
            token_indices=torch.arange(len(tokens), device=device),
            tokens=tokens,
            embedding=memory_embedding.detach().clone(),
            timestamp=time.time(),
            access_count=0,
            importance_score=avg_saliency.mean().item(),
            metadata=metadata or {}
        )
        
        # Add to episodic memory
        memory_id = consolidation.add_episodic_memory(memory_entry)
        
        print(f"[Memory] Stored: '{text[:50]}...' (importance={memory_entry.importance_score:.3f})")
        
        return memory_id


def generate_with_memory(model, context_ids, tokenizer, device,
                        memory_router,
                        max_new_tokens=256,
                        chunk_size=512,
                        temperature=0.8,
                        repetition_penalty=1.2,
                        top_k=50,
                        top_p=0.9,
                        memory_retrieval_interval=50,
                        max_memories_per_retrieval=3,
                        fusion_weight=0.3):
    """
    Generate text with memory augmentation
    
    Wrapper around MemoryAugmentedGenerator that's compatible with existing code.
    
    Args:
        model: SSMTransformer model
        context_ids: Input token IDs
        tokenizer: Tokenizer instance
        device: Device
        memory_router: MemoryRouter instance
        max_new_tokens: Max tokens to generate
        chunk_size: Context chunk size
        temperature: Sampling temperature
        repetition_penalty: Repetition penalty
        top_k: Top-k sampling
        top_p: Nucleus sampling
        memory_retrieval_interval: Retrieve memories every N tokens
        max_memories_per_retrieval: Max memories per retrieval
        fusion_weight: Weight for memory state fusion (0.2-0.4 typically)
    
    Returns:
        generated_ids: Complete token sequence
    """
    if not HAS_MEMORY_SYSTEM:
        print("[Warning] Memory system not available, falling back to standard generation")
        return generate_block_recurrent(
            model, context_ids, tokenizer, device,
            max_new_tokens=max_new_tokens,
            chunk_size=chunk_size,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p
        )
    
    # Use memory-augmented generator
    generator = MemoryAugmentedGenerator(
        model=model,
        tokenizer=tokenizer,
        memory_router=memory_router,
        device=device
    )
    
    generated_ids, gen_info = generator.generate_with_memory(
        context_ids=context_ids,
        max_new_tokens=max_new_tokens,
        chunk_size=chunk_size,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        memory_retrieval_interval=memory_retrieval_interval,
        max_memories_per_retrieval=max_memories_per_retrieval,
        fusion_weight=fusion_weight
    )
    
    print(f"\n[Memory Stats] Generated {gen_info['tokens_generated']} tokens, "
          f"retrieved memories {gen_info['memory_retrievals']} times")
    print(f"[Memory Stats] {gen_info['memory_stats']}")
    
    return generated_ids

# =============================================================================
# 9.75. LATENT THINKING & META-COGNITION SYSTEMS
# =============================================================================

def initialize_latent_thinking(model, tokenizer, max_thinking_depth=5, thinking_threshold=0.7):
    """
    Wrap model with latent thinking capability
    
    Args:
        model: Base SSMTransformer model
        tokenizer: Tokenizer instance
        max_thinking_depth: Max internal reasoning loops
        thinking_threshold: Confidence threshold to trigger thinking
    
    Returns:
        wrapped_model: Model with thinking capability
    """
    if not HAS_REASONING_SYSTEM:
        raise RuntimeError("Reasoning system not available. Ensure reasoning/ directory exists.")
    
    print(f"\n{'='*60}")
    print(f"Initializing Latent Thinking System")
    print(f"{'='*60}")
    print(f"\n⚠️  WARNING: Thinking components are UNTRAINED!")
    print(f"   The ThinkingController and refiners have random weights.")
    print(f"   For best results, train the model first with thinking enabled.")
    print(f"   For now, using high threshold (0.99) to minimize thinking.")
    
    # Use very high threshold for untrained models to avoid infinite loops
    safe_threshold = 0.99 if thinking_threshold < 0.99 else thinking_threshold
    
    wrapped_model = LatentThinkingWrapper(
        base_model=model,
        tokenizer=tokenizer,
        max_thinking_depth=max_thinking_depth,
        thinking_threshold=safe_threshold
    )
    
    print(f"   Thinking threshold set to: {safe_threshold}")
    print(f"{'='*60}\n")
    
    return wrapped_model


def initialize_metacognition(model, max_patterns=500, trajectory_length=5, num_domains=10):
    """
    Initialize meta-cognitive pattern system
    
    Args:
        model: Base SSMTransformer model
        max_patterns: Maximum patterns to store
        trajectory_length: Number of states per pattern
        num_domains: Number of reasoning domains
    
    Returns:
        Tuple of (pattern_bank, meta_controller)
    """
    if not HAS_REASONING_SYSTEM:
        raise RuntimeError("Reasoning system not available. Ensure reasoning/ directory exists.")
    
    print(f"\n{'='*60}")
    print(f"Initializing Meta-Cognition System")
    print(f"{'='*60}")
    print(f"\n⚠️  WARNING: Meta-cognitive components are UNTRAINED!")
    print(f"   PatternDetector and domain classifier have random weights.")
    print(f"   Pattern extraction/transfer will not work properly until trained.")
    
    # Create pattern memory bank
    pattern_bank = PatternMemoryBank(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_patterns=max_patterns,
        trajectory_length=trajectory_length
    )
    
    # Create meta-cognitive controller
    meta_controller = MetaCognitiveController(
        base_model=model,
        pattern_bank=pattern_bank,
        num_domains=num_domains
    )
    
    print(f"{'='*60}\n")
    
    return pattern_bank, meta_controller


def initialize_full_agi_stack(model, tokenizer, device='cuda'):
    """
    Initialize all AGI-like capabilities together
    
    Combines:
    - Latent Thinking
    - Meta-Cognition
    - Memory System
    
    Args:
        model: Base SSMTransformer model
        tokenizer: Tokenizer instance
        device: Device to use
    
    Returns:
        Dict with all initialized systems
    """
    print(f"\n{'='*60}")
    print(f"🧠 Initializing Full AGI Stack")
    print(f"{'='*60}\n")
    
    systems = {}
    
    # Initialize Latent Thinking
    if HAS_REASONING_SYSTEM:
        systems['thinking_model'] = initialize_latent_thinking(
            model, tokenizer,
            max_thinking_depth=5,
            thinking_threshold=0.7
        )
        
        # Initialize Meta-Cognition
        systems['pattern_bank'], systems['meta_controller'] = initialize_metacognition(
            systems['thinking_model'],
            max_patterns=500,
            trajectory_length=5,
            num_domains=10
        )
    else:
        systems['thinking_model'] = model
        systems['pattern_bank'] = None
        systems['meta_controller'] = None
        print("[Warning] Reasoning systems not available, using base model")
    
    # Initialize Memory System
    if HAS_MEMORY_SYSTEM:
        systems['consolidation'], systems['router'] = initialize_memory_system(
            device=device,
            episodic_capacity=50,
            consolidation_threshold=0.7,
            injection_mode='state_fusion'
        )
    else:
        systems['consolidation'] = None
        systems['router'] = None
        print("[Warning] Memory system not available")
    
    print(f"\n{'='*60}")
    print(f"✓ AGI Stack Initialized")
    print(f"  - Latent Thinking: {'✓' if HAS_REASONING_SYSTEM else '✗'}")
    print(f"  - Meta-Cognition: {'✓' if HAS_REASONING_SYSTEM else '✗'}")
    print(f"  - Memory System: {'✓' if HAS_MEMORY_SYSTEM else '✗'}")
    print(f"{'='*60}\n")
    
    return systems

# =============================================================================
# 10. CHAT MODE
# =============================================================================

def chat_mode(model, tokenizer, device, chunk_size=512):
    print("\n" + "="*60 + "\n💬 ENTERING CHAT MODE (ChatML Enhanced)\n" + "="*60)
    
    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model.to(dtype=ptdtype)
    model.eval()
    
    system_msg = {"role": "system", "content": "You are ResonantBrain, a highly intelligent and helpful AI assistant."}
    conversation_history = [system_msg]
    
    with torch.inference_mode():
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit']: 
                break
            if user_input.lower() == 'reset': 
                conversation_history = [system_msg]
                print("Conversation reset.")
                continue
            if not user_input: 
                continue

            conversation_history.append({"role": "user", "content": user_input})
            
            full_context = ""
            for msg in conversation_history:
                full_context += f"{CHAT_START}{msg['role']}\n{msg['content']}{CHAT_END}\n"
            
            full_context += f"{CHAT_START}assistant\n"
            context_ids = tokenizer.encode(full_context)

            print("Assistant: ", end="", flush=True)
            generated_ids = generate_block_recurrent(
                model, context_ids, tokenizer, device,
                max_new_tokens=256, chunk_size=chunk_size,
                temperature=0.7, repetition_penalty=1.15, top_k=20, top_p=0.95, 
                enable_rewind=True, stop_sequence=CHAT_END, max_paragraph_cache=50
            )

            response_text = tokenizer.decode(generated_ids[len(context_ids):])
            if CHAT_END in response_text:
                response_text = response_text[:response_text.index(CHAT_END)].strip()

            print(response_text)
            conversation_history.append({"role": "assistant", "content": response_text})

def chat_mode_with_memory(model, tokenizer, device, consolidation, router, chunk_size=512):
    """
    Memory-enhanced chat mode
    
    This version stores conversations in memory and retrieves relevant context
    during generation, enabling long-term memory across conversations.
    
    Special commands:
        - 'quit'/'exit': Exit chat mode
        - 'reset': Clear conversation history (keeps memory)
        - 'memory': Show memory statistics
        - 'consolidate': Manually trigger memory consolidation
        - 'clear_memory': Clear all memories
        - 'save_memory <path>': Save memories to disk
        - 'load_memory <path>': Load memories from disk
    """
    print("\n" + "="*60 + "\n💬 ENTERING MEMORY-ENHANCED CHAT MODE\n" + "="*60)
    print("Special commands: quit, reset, memory, consolidate, clear_memory, save_memory, load_memory")
    
    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model.to(dtype=ptdtype)
    model.eval()
    
    system_msg = {"role": "system", "content": "You are ResonantBrain, a highly intelligent AI assistant with long-term memory."}
    conversation_history = [system_msg]
    
    with torch.inference_mode():
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit']: 
                break
            
            # Handle special commands
            if user_input.lower() == 'reset': 
                conversation_history = [system_msg]
                print("Conversation reset (memories preserved).")
                continue
            
            if user_input.lower() == 'memory':
                stats = consolidation.get_stats()
                print(f"\n[Memory Statistics]")
                print(f"  Episodic memories: {stats['episodic_count']}")
                print(f"  Semantic memories: {stats['semantic_count']}")
                print(f"  Total memories: {stats['total_count']}")
                print(f"  Avg importance: {stats['semantic_avg_importance']:.3f}")
                print(f"  Total tokens stored: {stats['total_tokens']}")
                continue
            
            if user_input.lower() == 'consolidate':
                print("Triggering memory consolidation...")
                consolidation.consolidate(force=False)
                continue
            
            if user_input.lower() == 'clear_memory':
                confirm = input("Clear ALL memories? (yes/no): ")
                if confirm.lower() == 'yes':
                    consolidation.episodic_buffer.clear()
                    consolidation.semantic_memory.clear()
                    print("All memories cleared.")
                continue
            
            if user_input.lower().startswith('save_memory '):
                path = user_input[12:].strip()
                consolidation.save_to_disk(path)
                continue
            
            if user_input.lower().startswith('load_memory '):
                path = user_input[12:].strip()
                consolidation.load_from_disk(path)
                continue
            
            if not user_input: 
                continue

            # Add user message to conversation
            conversation_history.append({"role": "user", "content": user_input})
            
            # Build context
            full_context = ""
            for msg in conversation_history:
                full_context += f"{CHAT_START}{msg['role']}\n{msg['content']}{CHAT_END}\n"
            
            full_context += f"{CHAT_START}assistant\n"
            context_ids = tokenizer.encode(full_context)

            print("Assistant: ", end="", flush=True)
            
            # Generate with memory augmentation
            generated_ids = generate_with_memory(
                model, context_ids, tokenizer, device,
                memory_router=router,
                max_new_tokens=256,
                chunk_size=chunk_size,
                temperature=0.7,
                repetition_penalty=1.15,
                top_k=20,
                top_p=0.95,
                memory_retrieval_interval=50,  # Retrieve every 50 tokens
                max_memories_per_retrieval=3,
                fusion_weight=0.3
            )

            response_text = tokenizer.decode(generated_ids[len(context_ids):])
            if CHAT_END in response_text:
                response_text = response_text[:response_text.index(CHAT_END)].strip()

            print(response_text)
            
            # Store assistant response in memory
            assistant_msg = {"role": "assistant", "content": response_text}
            conversation_history.append(assistant_msg)
            
            # Store recent user-assistant exchange in memory
            exchange_text = f"User: {user_input}\nAssistant: {response_text}"
            store_conversation_memory(
                model, tokenizer, exchange_text, consolidation, device,
                metadata={
                    'type': 'conversation',
                    'user_msg': user_input,
                    'assistant_msg': response_text,
                    'turn': len(conversation_history) // 2
                }
            )

def chat_mode_with_thinking(thinking_model, tokenizer, device, chunk_size=512):
    """
    Chat mode with latent thinking capability
    
    The model engages in internal reasoning loops when uncertain,
    allowing deeper thought before responding.
    
    Special commands:
        - 'quit'/'exit': Exit chat mode
        - 'reset': Clear conversation history
        - 'thinking <on/off>': Enable/disable thinking
        - 'threshold <value>': Set thinking threshold (0.0-1.0)
    """
    print("\n" + "="*60 + "\n🧠 ENTERING LATENT THINKING CHAT MODE\n" + "="*60)
    print("Special commands: quit, reset, thinking <on/off>, threshold <value>")
    
    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    thinking_model.to(dtype=ptdtype)
    thinking_model.eval()
    
    system_msg = {"role": "system", "content": "You are ResonantBrain with latent thinking capability. You can think deeply before responding."}
    conversation_history = [system_msg]
    
    enable_thinking = True
    
    with torch.inference_mode():
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            
            if user_input.lower() == 'reset':
                conversation_history = [system_msg]
                print("Conversation reset.")
                continue
            
            if user_input.lower().startswith('thinking '):
                setting = user_input.split()[1].lower()
                if setting == 'on':
                    enable_thinking = True
                    print("Thinking enabled.")
                elif setting == 'off':
                    enable_thinking = False
                    print("Thinking disabled.")
                continue
            
            if user_input.lower().startswith('threshold '):
                try:
                    new_threshold = float(user_input.split()[1])
                    thinking_model.thinking_threshold = new_threshold
                    print(f"Thinking threshold set to {new_threshold:.2f}")
                except:
                    print("Invalid threshold. Use: threshold <0.0-1.0>")
                continue
            
            if not user_input:
                continue
            
            conversation_history.append({"role": "user", "content": user_input})
            
            # Build context
            full_context = ""
            for msg in conversation_history:
                full_context += f"{CHAT_START}{msg['role']}\n{msg['content']}{CHAT_END}\n"
            full_context += f"{CHAT_START}assistant\n"
            context_ids = tokenizer.encode(full_context)
            
            print("Assistant: ", end="", flush=True)
            
            # Generate with thinking
            generated_ids, gen_info = thinking_model.generate_with_thinking(
                context_ids=context_ids,
                device=device,
                max_new_tokens=256,
                temperature=0.7,
                top_k=20,
                top_p=0.95,
                thinking_threshold=thinking_model.thinking_threshold if enable_thinking else 1.0,
                verbose=False
            )
            
            response_text = tokenizer.decode(generated_ids[len(context_ids):])
            if CHAT_END in response_text:
                response_text = response_text[:response_text.index(CHAT_END)].strip()
            
            print(response_text)
            
            # Show thinking stats
            if gen_info['total_thinking_steps'] > 0:
                print(f"\n[Thinking] Used {gen_info['total_thinking_steps']} internal reasoning steps across {len(gen_info['thinking_events'])} events")
            
            conversation_history.append({"role": "assistant", "content": response_text})


def chat_mode_with_metacognition(meta_controller, tokenizer, device, pattern_bank, chunk_size=512):
    """
    Chat mode with meta-cognitive pattern transfer
    
    The model can recognize and apply learned reasoning patterns
    across different domains.
    
    Special commands:
        - 'quit'/'exit': Exit chat mode
        - 'reset': Clear conversation history
        - 'patterns': Show pattern statistics
        - 'save_patterns <path>': Save patterns to disk
        - 'load_patterns <path>': Load patterns from disk
        - 'domains': Show detected domains
    """
    print("\n" + "="*60 + "\n🔮 ENTERING META-COGNITIVE CHAT MODE\n" + "="*60)
    print("Special commands: quit, reset, patterns, save_patterns, load_patterns, domains")
    
    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    meta_controller.to(dtype=ptdtype)
    meta_controller.eval()
    
    system_msg = {"role": "system", "content": "You are ResonantBrain with meta-cognitive abilities. You can recognize and transfer reasoning patterns across domains."}
    conversation_history = [system_msg]
    
    with torch.inference_mode():
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            
            if user_input.lower() == 'reset':
                conversation_history = [system_msg]
                print("Conversation reset.")
                continue
            
            if user_input.lower() == 'patterns':
                stats = pattern_bank.get_stats()
                print(f"\n[Pattern Statistics]")
                print(f"  Total patterns: {stats['num_patterns']}")
                print(f"  Domains: {', '.join(stats['domains'])}")
                print(f"  Avg success rate: {stats['avg_success_rate']:.2%}")
                print(f"  Avg applicability: {stats['avg_applicability']:.2%}")
                print(f"  Total transfers: {stats['total_transfers']}")
                print(f"  Transfer success rate: {stats['transfer_success_rate']:.2%}")
                continue
            
            if user_input.lower().startswith('save_patterns '):
                path = user_input[14:].strip()
                meta_controller.save_patterns(path)
                continue
            
            if user_input.lower().startswith('load_patterns '):
                path = user_input[14:].strip()
                meta_controller.load_patterns(path)
                continue
            
            if user_input.lower() == 'domains':
                print(f"\n[Reasoning Domains]")
                for i, domain in enumerate(meta_controller.domain_names):
                    patterns = pattern_bank.get_patterns_by_domain(domain)
                    print(f"  {i+1}. {domain}: {len(patterns)} patterns")
                continue
            
            if not user_input:
                continue
            
            conversation_history.append({"role": "user", "content": user_input})
            
            # Build context
            full_context = ""
            for msg in conversation_history:
                full_context += f"{CHAT_START}{msg['role']}\n{msg['content']}{CHAT_END}\n"
            full_context += f"{CHAT_START}assistant\n"
            context_ids = tokenizer.encode(full_context)
            
            print("Assistant: ", end="", flush=True)
            
            # Generate with metacognition
            generated_ids, gen_info = generate_with_metacognition(
                model=meta_controller,
                tokenizer=tokenizer,
                context_ids=context_ids,
                device=device,
                max_new_tokens=256,
                temperature=0.7,
                top_k=20,
                top_p=0.95,
                enable_patterns=True,
                pattern_fusion_weight=0.3,
                verbose=False
            )
            
            response_text = tokenizer.decode(generated_ids[len(context_ids):])
            if CHAT_END in response_text:
                response_text = response_text[:response_text.index(CHAT_END)].strip()
            
            print(response_text)
            
            # Show metacognitive stats
            if gen_info['pattern_applications'] > 0:
                print(f"\n[MetaCognition] Applied {gen_info['pattern_applications']} patterns "
                      f"({gen_info['unique_patterns_used']} unique, "
                      f"{gen_info['cross_domain_transfers']} cross-domain)")
            
            conversation_history.append({"role": "assistant", "content": response_text})


def chat_mode_full_agi(agi_systems, tokenizer, device, chunk_size=512):
    """
    Chat mode with ALL AGI capabilities enabled
    
    Combines:
    - Latent Thinking: Internal reasoning loops
    - Meta-Cognition: Pattern recognition and transfer
    - Memory: Long-term episodic and semantic memory
    
    Special commands:
        - 'quit'/'exit': Exit chat mode
        - 'reset': Clear conversation (keeps memories/patterns)
        - 'stats': Show comprehensive statistics
        - 'thinking <on/off>': Toggle thinking
        - 'patterns': Show pattern stats
        - 'memory': Show memory stats
        - 'save_all <prefix>': Save everything
        - 'load_all <prefix>': Load everything
    """
    print("\n" + "="*60 + "\n🌟 ENTERING FULL AGI CHAT MODE\n" + "="*60)
    print("All capabilities: Latent Thinking + Meta-Cognition + Memory")
    print("Special commands: quit, reset, stats, thinking, patterns, memory, save_all, load_all")
    print("\n⚠️  NOTE: Thinking/MetaCognition disabled by default (untrained)")
    print("   Use 'thinking on' to enable if you've trained these components")
    
    # Extract systems
    thinking_model = agi_systems.get('thinking_model')
    meta_controller = agi_systems.get('meta_controller')
    pattern_bank = agi_systems.get('pattern_bank')
    consolidation = agi_systems.get('consolidation')
    router = agi_systems.get('router')
    
    if thinking_model is None:
        print("[Error] Thinking model not initialized")
        return
    
    print(f"[DEBUG] Thinking model type: {type(thinking_model).__name__}")
    print(f"[DEBUG] Base model type: {type(thinking_model.base_model).__name__}")
    
    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"[DEBUG] Converting to dtype: {ptdtype}")
    thinking_model.to(dtype=ptdtype)
    thinking_model.eval()
    
    print(f"[DEBUG] Model ready for inference")
    
    system_msg = {"role": "system", "content": "You are ResonantBrain with full AGI capabilities: latent thinking, meta-cognition, and long-term memory."}
    conversation_history = [system_msg]
    
    enable_thinking = False  # Disabled by default for untrained models
    
    with torch.inference_mode():
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            
            if user_input.lower() == 'reset':
                conversation_history = [system_msg]
                print("Conversation reset (memories and patterns preserved).")
                continue
            
            if user_input.lower() == 'stats':
                print(f"\n{'='*60}")
                print(f"📊 COMPREHENSIVE AGI STATISTICS")
                print(f"{'='*60}")
                
                if HAS_MEMORY_SYSTEM and consolidation:
                    mem_stats = consolidation.get_stats()
                    print(f"\n[Memory System]")
                    print(f"  Episodic: {mem_stats['episodic_count']}")
                    print(f"  Semantic: {mem_stats['semantic_count']}")
                    print(f"  Total tokens: {mem_stats['total_tokens']}")
                
                if HAS_REASONING_SYSTEM and pattern_bank:
                    pat_stats = pattern_bank.get_stats()
                    print(f"\n[Pattern Bank]")
                    print(f"  Patterns: {pat_stats['num_patterns']}")
                    print(f"  Domains: {len(pat_stats['domains'])}")
                    print(f"  Transfer success: {pat_stats['transfer_success_rate']:.1%}")
                
                print(f"\n[Thinking System]")
                print(f"  Max depth: {thinking_model.max_thinking_depth if hasattr(thinking_model, 'max_thinking_depth') else 'N/A'}")
                print(f"  Threshold: {thinking_model.thinking_threshold if hasattr(thinking_model, 'thinking_threshold') else 'N/A'}")
                print(f"  Enabled: {enable_thinking}")
                
                print(f"{'='*60}\n")
                continue
            
            if user_input.lower().startswith('thinking '):
                setting = user_input.split()[1].lower()
                enable_thinking = (setting == 'on')
                print(f"Thinking {'enabled' if enable_thinking else 'disabled'}.")
                continue
            
            if user_input.lower() == 'patterns' and pattern_bank:
                stats = pattern_bank.get_stats()
                print(f"\n[Pattern Statistics]")
                print(f"  Total: {stats['num_patterns']}, Domains: {', '.join(stats['domains'])}")
                print(f"  Success: {stats['avg_success_rate']:.1%}, Transfers: {stats['total_transfers']}")
                continue
            
            if user_input.lower() == 'memory' and consolidation:
                stats = consolidation.get_stats()
                print(f"\n[Memory Statistics]")
                print(f"  Episodic: {stats['episodic_count']}, Semantic: {stats['semantic_count']}")
                print(f"  Avg importance: {stats['semantic_avg_importance']:.2f}")
                continue
            
            if user_input.lower().startswith('save_all '):
                prefix = user_input[9:].strip()
                if pattern_bank:
                    meta_controller.save_patterns(f"{prefix}_patterns.pt")
                if consolidation:
                    consolidation.save_to_disk(f"{prefix}_memory.pt")
                print(f"Saved all systems to {prefix}_*")
                continue
            
            if user_input.lower().startswith('load_all '):
                prefix = user_input[9:].strip()
                if pattern_bank:
                    meta_controller.load_patterns(f"{prefix}_patterns.pt")
                if consolidation:
                    consolidation.load_from_disk(f"{prefix}_memory.pt")
                print(f"Loaded all systems from {prefix}_*")
                continue
            
            if not user_input:
                continue
            
            conversation_history.append({"role": "user", "content": user_input})
            
            # Build context
            full_context = ""
            for msg in conversation_history:
                full_context += f"{CHAT_START}{msg['role']}\n{msg['content']}{CHAT_END}\n"
            full_context += f"{CHAT_START}assistant\n"
            context_ids = tokenizer.encode(full_context)
            
            print("Assistant: ", end="", flush=True)
            
            # Generate with full AGI stack
            # When thinking is disabled, use threshold 1.0 to prevent any thinking loops
            actual_threshold = thinking_model.thinking_threshold if enable_thinking else 1.0
            
            print(f"\n[DEBUG] Generating with {len(context_ids)} context tokens...")
            print(f"[DEBUG] Thinking enabled: {enable_thinking}, threshold: {actual_threshold:.2f}")
            generated_ids, gen_info = thinking_model.generate_with_thinking(
                context_ids=context_ids,
                device=device,
                max_new_tokens=256,
                temperature=0.7,
                top_k=20,
                top_p=0.95,
                thinking_threshold=actual_threshold,
                verbose=False  # Disable verbose to reduce clutter
            )
            
            print(f"[DEBUG] Generated {len(generated_ids) - len(context_ids)} new tokens")
            response_text = tokenizer.decode(generated_ids[len(context_ids):])
            if CHAT_END in response_text:
                response_text = response_text[:response_text.index(CHAT_END)].strip()
            
            print(response_text)
            
            # Show combined stats
            if gen_info['total_thinking_steps'] > 0:
                print(f"  [Thinking] {gen_info['total_thinking_steps']} internal reasoning steps")
            
            conversation_history.append({"role": "assistant", "content": response_text})
            
            # Store in memory if available
            if consolidation:
                exchange_text = f"User: {user_input}\nAssistant: {response_text}"
                store_conversation_memory(
                    thinking_model.base_model, tokenizer, exchange_text,
                    consolidation, device,
                    metadata={'type': 'agi_conversation', 'turn': len(conversation_history) // 2}
                )

# =============================================================================
# 11. MAIN ENTRY POINT
# =============================================================================

MODEL_CONFIGS = {
    'tiny':   {'dim': 256,  'num_heads': 4,  'num_layers': 4},
    'small':  {'dim': 512,  'num_heads': 8,  'num_layers': 6},
    'medium': {'dim': 768,  'num_heads': 12, 'num_layers': 16},
}

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ENABLE_COGNITIVE_FORGETTING = True
    
    print(f"🚀 ResonantBrain SSM v5.0 - GQA + MoD Edition")
    print(f"   Device: {device}")

    # File Paths Configuration
    PARQUET_DIR = r"I:\Datasets\fineweb-edu_data_CC-MAIN-2024-18"
    JSON_DATASET_PATH = r"I:\FineTunningDatasets\OpenHermes2.5\openhermes2_5.json" 
    
    MODEL_SIZE = 'medium'
    CHUNK_SIZE = 1024
    BATCH_SIZE = 2 
    GRAD_ACCUM_STEPS = 4
    LEARNING_RATE = 4e-4
    
    # Optimization Parameters
    SALIENCY_DECAY = 0.95  # Configurable decay factor (was hardcoded to 0.9)
    USE_JSON_STREAMING = True  # Memory-efficient streaming for large datasets
    MAX_PARAGRAPH_CACHE = 50  # Limit paragraph states in generation
    
    # === GQA Configuration (Grouped-Query Attention) ===
    # Reduces KV cache VRAM by using fewer KV heads than Q heads
    # For 12-head model: num_kv_heads=3 gives 4x KV cache reduction
    ENABLE_GQA = True  # Set to False to disable GQA
    NUM_KV_HEADS = 3  # For 12-head model: 3 = 4x reduction, 4 = 3x reduction, 6 = 2x reduction
    
    # === MoD Configuration (Mixture of Depths) ===
    # Tokens are selectively processed through expensive FFN layers
    # Reduces compute (FLOPs) without reducing parameters
    ENABLE_MOD = True  # Set to False to disable MoD
    MOD_TOP_K_RATIO = 0.75  # Fraction of tokens to fully process (0.75 = 75% compute)
    
    # === Thinking System Training ===
    # Train latent thinking components (ThinkingController, refiners) during training
    # When enabled, model learns when/how to engage internal reasoning loops
    TRAIN_THINKING_SYSTEM = True  # Set to True to train thinking components from scratch
    THINKING_LOSS_WEIGHT = 0.1  # Weight for thinking loss (relative to LM loss)
    THINKING_MAX_DEPTH = 3  # Max thinking loops during training (lower to save compute)
    THINKING_THRESHOLD = 0.7  # Confidence threshold for engaging thinking
    
    # Fuzzy Training Config
    FUZZY_STEPS = 1
    CLEAR_STEPS = 3
    BAG_SIZE = 2

    tokenizer = TiktokenTokenizer("gpt2")
    vocab_size = tokenizer.vocab_size
    config = MODEL_CONFIGS[MODEL_SIZE]

    model = SSMTransformer(
        vocab_size=vocab_size, 
        dim=config['dim'], 
        num_heads=config['num_heads'],
        num_layers=config['num_layers'], 
        max_seq_len=CHUNK_SIZE,
        enable_forgetting=ENABLE_COGNITIVE_FORGETTING,
        saliency_decay=SALIENCY_DECAY,
        num_kv_heads=NUM_KV_HEADS if ENABLE_GQA else config['num_heads'],
        enable_mod=ENABLE_MOD,
        mod_top_k_ratio=MOD_TOP_K_RATIO
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Trainable parameters: {total_params:,}")
    
    # Calculate memory savings
    if ENABLE_GQA:
        kv_reduction = config['num_heads'] / (NUM_KV_HEADS if ENABLE_GQA else config['num_heads'])
        print(f"KV Cache Memory Reduction: {kv_reduction:.1f}x (GQA enabled)")
    
    if ENABLE_MOD:
        compute_reduction = 1.0 / MOD_TOP_K_RATIO
        print(f"Compute Reduction: {compute_reduction:.2f}x (MoD @ {MOD_TOP_K_RATIO:.0%} top-k)")
    
    print(f"{'='*60}\n")

    validate_vocab_size(model, tokenizer)
    
    # Optionally wrap model with thinking system for training
    training_model = model  # Default: use base model
    if TRAIN_THINKING_SYSTEM and HAS_REASONING_SYSTEM:
        print(f"\n{'='*60}")
        print(f"🧠 Wrapping model with Latent Thinking System for training")
        print(f"{'='*60}")
        print(f"  Max depth: {THINKING_MAX_DEPTH}")
        print(f"  Threshold: {THINKING_THRESHOLD}")
        print(f"  Loss weight: {THINKING_LOSS_WEIGHT}")
        
        training_model = LatentThinkingWrapper(
            base_model=model,
            tokenizer=tokenizer,
            max_thinking_depth=THINKING_MAX_DEPTH,
            thinking_threshold=THINKING_THRESHOLD
        )
        
        thinking_params = sum(p.numel() for p in training_model.thinking_controller.parameters())
        thinking_params += sum(p.numel() for p in training_model.thinking_refiners.parameters())
        print(f"  Additional thinking parameters: {thinking_params:,}")
        print(f"{'='*60}\n")
    elif TRAIN_THINKING_SYSTEM and not HAS_REASONING_SYSTEM:
        print("\n⚠️  WARNING: TRAIN_THINKING_SYSTEM=True but reasoning system not available!")
        print("   Install reasoning module or set TRAIN_THINKING_SYSTEM=False\n")
    
    use_fused = True if device == 'cuda' else False
    optimizer = torch.optim.AdamW(training_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, fused=use_fused)

    print("\nSelect an operation mode:")
    print("  [1] Pre-train on Plain Text (Parquet) [Fuzzy + Clear Alternating]")
    print("  [2] Fine-tune on OpenHermes ChatML (JSON)")
    print("  [3] Chat Mode (Standard)")
    print("  [4] Chat Mode with Memory (Saliency-Guided State Memory)")
    print("  [5] Chat Mode with Latent Thinking (Internal Reasoning Loops)")
    print("  [6] Chat Mode with Meta-Cognition (Pattern Transfer)")
    print("  [7] Chat Mode with Full AGI Stack (All Capabilities)")
    choice = input("Choice: ").strip()

    if choice == '1':
        files = glob.glob(os.path.join(PARQUET_DIR, '**', '*.parquet'), recursive=True)
        ckpt_path = 'checkpoint_ssm_pretrain.pth'
        start_it = 0
        if os.path.exists(ckpt_path) and input("Resume pre-training checkpoint? (y/n): ").strip().lower() == 'y':
            ckpt = torch.load(ckpt_path, map_location=device)
            load_checkpoint_with_filter(training_model, ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_it = ckpt.get('iteration', 0)
            print("[INFO] Loaded checkpoint (filtered mismatched keys)")
            
        run_pretraining(
            training_model, files, "text", tokenizer, optimizer, device, vocab_size, start_it, 
            CHUNK_SIZE, ENABLE_COGNITIVE_FORGETTING, BATCH_SIZE, GRAD_ACCUM_STEPS,
            FUZZY_STEPS, CLEAR_STEPS, BAG_SIZE, THINKING_LOSS_WEIGHT
        )
        
    elif choice == '2':
        ckpt_path = 'checkpoint_ssm_finetune.pth'
        start_it = 0
        if os.path.exists('checkpoint_ssm_pretrain.pth') and not os.path.exists(ckpt_path):
            if input("Load base pre-trained weights before fine-tuning? (y/n): ").strip().lower() == 'y':
                ckpt = torch.load('checkpoint_ssm_pretrain.pth', map_location=device)
                load_checkpoint_with_filter(model, ckpt['model_state_dict'])
                print("Pre-trained base weights loaded!")

        if os.path.exists(ckpt_path) and input("Resume existing fine-tuning checkpoint? (y/n): ").strip().lower() == 'y':
            ckpt = torch.load(ckpt_path, map_location=device)
            load_checkpoint_with_filter(model, ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_it = ckpt.get('iteration', 0)
            print("[INFO] Loaded checkpoint (filtered mismatched keys)")
            
        if not os.path.exists(JSON_DATASET_PATH):
            print(f"ERROR: Cannot find JSON dataset at {JSON_DATASET_PATH}. Please update the script path.")
        else:
            run_finetuning(
                training_model, JSON_DATASET_PATH, tokenizer, optimizer, device, vocab_size, start_it, 
                CHUNK_SIZE, ENABLE_COGNITIVE_FORGETTING, BATCH_SIZE, GRAD_ACCUM_STEPS, USE_JSON_STREAMING, THINKING_LOSS_WEIGHT
            )
            
    elif choice == '3':
        if os.path.exists('checkpoint_ssm_finetune.pth'):
            print("Loading fine-tuned checkpoint...")
            ckpt = torch.load('checkpoint_ssm_finetune.pth', map_location=device)
        elif os.path.exists('checkpoint_ssm_pretrain.pth'):
            print("Loading pre-trained base checkpoint...")
            ckpt = torch.load('checkpoint_ssm_pretrain.pth', map_location=device)
        else:
            print("No checkpoints found. Running with untrained random weights!")
            ckpt = {}
            
        if 'model_state_dict' in ckpt: 
            load_checkpoint_with_filter(model, ckpt['model_state_dict'])
            
        chat_mode(model, tokenizer, device, chunk_size=ckpt.get('chunk_size', CHUNK_SIZE))
    
    elif choice == '4':
        if not HAS_MEMORY_SYSTEM:
            print("\n[ERROR] Memory system not available!")
            print("Please ensure the memory/ directory exists in the same folder as this script.")
        else:
            # Load model checkpoint
            if os.path.exists('checkpoint_ssm_finetune.pth'):
                print("Loading fine-tuned checkpoint...")
                ckpt = torch.load('checkpoint_ssm_finetune.pth', map_location=device)
            elif os.path.exists('checkpoint_ssm_pretrain.pth'):
                print("Loading pre-trained base checkpoint...")
                ckpt = torch.load('checkpoint_ssm_pretrain.pth', map_location=device)
            else:
                print("No checkpoints found. Running with untrained random weights!")
                ckpt = {}
                
            if 'model_state_dict' in ckpt: 
                load_checkpoint_with_filter(model, ckpt['model_state_dict'])
            
            # Initialize memory system
            consolidation, router = initialize_memory_system(
                device=device,
                episodic_capacity=50,
                consolidation_threshold=0.7,
                injection_mode='state_fusion'
            )
            
            # Optionally load existing memories
            if os.path.exists('memories.pt'):
                load_mem = input("Load existing memories from memories.pt? (y/n): ").strip().lower()
                if load_mem == 'y':
                    consolidation.load_from_disk('memories.pt')
            
            # Run memory-enhanced chat
            chat_mode_with_memory(
                model, tokenizer, device, 
                consolidation, router,
                chunk_size=ckpt.get('chunk_size', CHUNK_SIZE)
            )
            
            # Save memories on exit
            save_mem = input("\nSave memories to disk? (y/n): ").strip().lower()
            if save_mem == 'y':
                consolidation.save_to_disk('memories.pt')
                print("Memories saved to memories.pt")
    
    elif choice == '5':
        if not HAS_REASONING_SYSTEM:
            print("\n[ERROR] Reasoning system not available!")
            print("Please ensure the reasoning/ directory exists in the same folder as this script.")
        else:
            # Load model checkpoint
            if os.path.exists('checkpoint_ssm_finetune.pth'):
                print("Loading fine-tuned checkpoint...")
                ckpt = torch.load('checkpoint_ssm_finetune.pth', map_location=device)
            elif os.path.exists('checkpoint_ssm_pretrain.pth'):
                print("Loading pre-trained base checkpoint...")
                ckpt = torch.load('checkpoint_ssm_pretrain.pth', map_location=device)
            else:
                print("No checkpoints found. Running with untrained random weights!")
                ckpt = {}
            
            if 'model_state_dict' in ckpt:
                load_checkpoint_with_filter(model, ckpt['model_state_dict'])
            
            # Initialize latent thinking
            thinking_model = initialize_latent_thinking(
                model=model,
                tokenizer=tokenizer,
                max_thinking_depth=5,
                thinking_threshold=0.7
            )
            
            # Run thinking-enhanced chat
            chat_mode_with_thinking(
                thinking_model, tokenizer, device,
                chunk_size=ckpt.get('chunk_size', CHUNK_SIZE)
            )
    
    elif choice == '6':
        if not HAS_REASONING_SYSTEM:
            print("\n[ERROR] Reasoning system not available!")
            print("Please ensure the reasoning/ directory exists in the same folder as this script.")
        else:
            # Load model checkpoint
            if os.path.exists('checkpoint_ssm_finetune.pth'):
                print("Loading fine-tuned checkpoint...")
                ckpt = torch.load('checkpoint_ssm_finetune.pth', map_location=device)
            elif os.path.exists('checkpoint_ssm_pretrain.pth'):
                print("Loading pre-trained base checkpoint...")
                ckpt = torch.load('checkpoint_ssm_pretrain.pth', map_location=device)
            else:
                print("No checkpoints found. Running with untrained random weights!")
                ckpt = {}
            
            if 'model_state_dict' in ckpt:
                load_checkpoint_with_filter(model, ckpt['model_state_dict'])
            
            # Initialize meta-cognition
            pattern_bank, meta_controller = initialize_metacognition(
                model=model,
                max_patterns=500,
                trajectory_length=5,
                num_domains=10
            )
            
            # Optionally load existing patterns
            if os.path.exists('patterns.pt'):
                load_pat = input("Load existing patterns from patterns.pt? (y/n): ").strip().lower()
                if load_pat == 'y':
                    meta_controller.load_patterns('patterns.pt')
            
            # Run meta-cognitive chat
            chat_mode_with_metacognition(
                meta_controller, tokenizer, device, pattern_bank,
                chunk_size=ckpt.get('chunk_size', CHUNK_SIZE)
            )
            
            # Save patterns on exit
            save_pat = input("\nSave patterns to disk? (y/n): ").strip().lower()
            if save_pat == 'y':
                meta_controller.save_patterns('patterns.pt')
                print("Patterns saved to patterns.pt")
    
    elif choice == '7':
        if not HAS_REASONING_SYSTEM and not HAS_MEMORY_SYSTEM:
            print("\n[ERROR] Neither reasoning nor memory systems available!")
            print("Please ensure reasoning/ and memory/ directories exist.")
        else:
            # Load model checkpoint
            if os.path.exists('checkpoint_ssm_finetune.pth'):
                print("Loading fine-tuned checkpoint...")
                ckpt = torch.load('checkpoint_ssm_finetune.pth', map_location=device)
            elif os.path.exists('checkpoint_ssm_pretrain.pth'):
                print("Loading pre-trained base checkpoint...")
                ckpt = torch.load('checkpoint_ssm_pretrain.pth', map_location=device)
            else:
                print("No checkpoints found. Running with untrained random weights!")
                ckpt = {}
            
            if 'model_state_dict' in ckpt:
                load_checkpoint_with_filter(model, ckpt['model_state_dict'])
            
            # Initialize full AGI stack
            agi_systems = initialize_full_agi_stack(
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            
            # Load existing data if available
            if os.path.exists('memories.pt') and agi_systems['consolidation']:
                load_mem = input("Load existing memories? (y/n): ").strip().lower()
                if load_mem == 'y':
                    agi_systems['consolidation'].load_from_disk('memories.pt')
            
            if os.path.exists('patterns.pt') and agi_systems['meta_controller']:
                load_pat = input("Load existing patterns? (y/n): ").strip().lower()
                if load_pat == 'y':
                    agi_systems['meta_controller'].load_patterns('patterns.pt')
            
            # Run full AGI chat
            chat_mode_full_agi(
                agi_systems, tokenizer, device,
                chunk_size=ckpt.get('chunk_size', CHUNK_SIZE)
            )
            
            # Save everything on exit
            save_all = input("\nSave all systems to disk? (y/n): ").strip().lower()
            if save_all == 'y':
                if agi_systems['consolidation']:
                    agi_systems['consolidation'].save_to_disk('memories.pt')
                if agi_systems['meta_controller']:
                    agi_systems['meta_controller'].save_patterns('patterns.pt')
                print("All systems saved!")