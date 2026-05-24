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
# 4. SSM-Attention Block
# =============================================================================

class SSMAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len, num_layers, dropout=0.1, 
                 forgetting_config=None, use_eviction=True, saliency_decay=0.95):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len
        self.use_eviction = use_eviction
        self.saliency_decay = saliency_decay  # Configurable decay factor

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

        m = self.norm_mlp(x)
        m = self.mlp_fc1(m)
        m = self.mlp_act(m)
        m = self.mlp_forget_gate(m)
        m = self.mlp_drop1(m)
        m = self.mlp_fc2(m)
        m = self.mlp_drop2(m)
        x = x + m

        return x, new_carry, new_kv

# =============================================================================
# 5. SSM Transformer
# =============================================================================

def get_forgetting_config(layer_idx, num_layers, enable_forgetting):
    if not enable_forgetting:
        return None
    depth_ratio = layer_idx / max(1, num_layers - 1)
    return {
        # Decreased from 0.980 to 0.950 (decays faster during grad accum steps)
        'decay_factor': 0.950 + (depth_ratio * 0.018),      
        
        # Increased from 0.90 to 0.95 (requires higher sustained health to lock)
        'lock_threshold': 0.95 + (depth_ratio * 0.04),      
        
        'health_floor': 0.1 + (depth_ratio * 0.3),          
        'gated_fraction': 0.9 - (depth_ratio * 0.6),        
    }

class SSMTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, max_seq_len=512, 
                 dropout=0.1, enable_forgetting=False, saliency_decay=0.95):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.head_dim = dim // num_heads
        self.saliency_decay = saliency_decay

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            SSMAttentionBlock(
                dim, num_heads, max_seq_len, num_layers, dropout,
                forgetting_config=get_forgetting_config(i, num_layers, enable_forgetting),
                use_eviction=(i >= 1),
                saliency_decay=saliency_decay
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

        for i, layer in enumerate(self.layers):
            layer_past_kv = past_key_values[i] if past_key_values is not None else None
            h, new_carry, new_kv = layer(
                h, self.freqs_cis_ext, abs_pos_offset=abs_pos_offset,
                carry_state=carry_states[i],
                past_kv=layer_past_kv,
                use_cache=use_cache
            )
            new_carry_states.append(new_carry)
            new_key_values.append(new_kv)

        h = self.norm(h)
        logits = self.output(h)

        return logits, new_carry_states, new_key_values

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
            _, carry_states, past_key_values = model(
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
            logits, active_carry, active_kv = model(
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
    
    total_locked, total_health, total_gated, total_wildcard = 0, 0, 0, 0
    for layer in model.layers:
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

# --- 9A. Pre-training (Causal LM on Plain Text with Fuzzy Training & Grad Accum) ---
def run_pretraining(model, parquet_files, text_column, tokenizer, optimizer, device,
                    vocab_size, start_iteration=0, chunk_size=512, enable_forgetting=False, 
                    batch_size=4, grad_accum_steps=4, fuzzy_steps=1, clear_steps=3, bag_size=2):
    
    print(f"\n--- Starting Pre-training (Parquet) | Device: {device} | Batch Size: {batch_size} | Grad Accum: {grad_accum_steps} ---")
    print(f"--- Fuzzy Training Enabled: {fuzzy_steps} Fuzzy / {clear_steps} Clear | Bag Size: {bag_size} ---")
    print(f"--- Flash Attention: {'Available' if hasattr(F, 'scaled_dot_product_attention') else 'Not Available'} ---")
    
    iteration = start_iteration
    random.shuffle(parquet_files)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    
    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == torch.float16))
    
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

        if abs_pos_offset + chunk_size > model.freqs_cis_ext.size(0):
            carry_states, past_key_values, abs_pos_offset = None, None, 0

        detached_carry = [c.detach() for c in carry_states] if carry_states else None
        detached_kv = [(k.detach(), v.detach(), s.detach(), kr.detach()) for k, v, s, kr in past_key_values] if past_key_values else None

        with torch.autocast(device_type=device, dtype=ptdtype):
            if is_fuzzy_step:
                inputs_flat = chunk[:, :-bag_size]
                targets_flat = chunk[:, bag_size:]
                
                embs = model.tok_embeddings(inputs_flat)
                embs = embs.view(batch_size, chunk_size, bag_size, -1).mean(dim=2)
                
                y_bags = targets_flat.view(batch_size, chunk_size, bag_size)
                
                logits, carry_states, past_key_values = model(
                    inputs_embeds=embs, carry_states=detached_carry, past_key_values=detached_kv, 
                    is_training=True, use_cache=True, abs_pos_offset=abs_pos_offset
                )
                
                soft_targets = torch.zeros_like(logits).scatter_add_(
                    -1, y_bags, torch.ones_like(y_bags, dtype=logits.dtype) / bag_size
                )
                
                loss = F.cross_entropy(logits.view(-1, vocab_size), soft_targets.view(-1, vocab_size))
            else:
                x, y = chunk[:, :-1], chunk[:, 1:]
                
                logits, carry_states, past_key_values = model(
                    x=x, carry_states=detached_carry, past_key_values=detached_kv, 
                    is_training=True, use_cache=True, abs_pos_offset=abs_pos_offset
                )
                
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.reshape(-1))
                
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

            if iteration % 20000 == 0:
                model.eval()
                print(f"\n{'='*60}\n[GENERATION SAMPLE (Pre-training Coherence)]\n{'='*60}")
                test_prompt = "The rapid advancement of artificial intelligence has led to"
                gen_ids = generate_block_recurrent(
                    model, tokenizer.encode(test_prompt), tokenizer, device, max_new_tokens=150, 
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
                   batch_size=2, grad_accum_steps=4, use_streaming=True):
    
    print(f"\n--- Starting ChatML Fine-tuning (OpenHermes) | Device: {device} | Batch Size: {batch_size} | Grad Accum: {grad_accum_steps} ---")
    print(f"--- Flash Attention: {'Available' if hasattr(F, 'scaled_dot_product_attention') else 'Not Available'} ---")
    iteration = start_iteration
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)

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
        detached_kv = [(k.detach(), v.detach(), s.detach(), kr.detach()) for k, v, s, kr in past_key_values] if past_key_values else None

        with torch.autocast(device_type=device, dtype=ptdtype):
            logits, carry_states, past_key_values = model(
                x=x, carry_states=detached_carry, past_key_values=detached_kv, 
                is_training=True, use_cache=True, abs_pos_offset=abs_pos_offset
            )
            abs_pos_offset += x.size(1)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), reduction='none')
            loss = (loss * m.view(-1)).sum() / max(m.sum().item(), 1.0)
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
    
    print(f"🚀 ResonantBrain SSM v4.0 - Hybrid Fuzzy Training Edition")
    print(f"   Device: {device}")

    # File Paths Configuration
    PARQUET_DIR = r"I:\Datasets\fineweb-edu_data_CC-MAIN-2024-10"
    JSON_DATASET_PATH = r"I:\FineTunningDatasets\OpenHermes2.5\openhermes2_5.json" 
    
    MODEL_SIZE = 'medium'
    CHUNK_SIZE = 1024
    BATCH_SIZE = 1 
    GRAD_ACCUM_STEPS = 4
    LEARNING_RATE = 4e-4
    
    # Optimization Parameters
    SALIENCY_DECAY = 0.95  # Configurable decay factor (was hardcoded to 0.9)
    USE_JSON_STREAMING = True  # Memory-efficient streaming for large datasets
    MAX_PARAGRAPH_CACHE = 50  # Limit paragraph states in generation
    
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
        saliency_decay=SALIENCY_DECAY
    ).to(device)
    
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    validate_vocab_size(model, tokenizer)
    
    use_fused = True if device == 'cuda' else False
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, fused=use_fused)

    print("\nSelect an operation mode:")
    print("  [1] Pre-train on Plain Text (Parquet) [Fuzzy + Clear Alternating]")
    print("  [2] Fine-tune on OpenHermes ChatML (JSON)")
    print("  [3] Chat Mode")
    choice = input("Choice: ").strip()

    if choice == '1':
        files = glob.glob(os.path.join(PARQUET_DIR, '**', '*.parquet'), recursive=True)
        ckpt_path = 'checkpoint_ssm_pretrain.pth'
        start_it = 0
        if os.path.exists(ckpt_path) and input("Resume pre-training checkpoint? (y/n): ").strip().lower() == 'y':
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_it = ckpt.get('iteration', 0)
            
        run_pretraining(
            model, files, "text", tokenizer, optimizer, device, vocab_size, start_it, 
            CHUNK_SIZE, ENABLE_COGNITIVE_FORGETTING, BATCH_SIZE, GRAD_ACCUM_STEPS,
            FUZZY_STEPS, CLEAR_STEPS, BAG_SIZE
        )
        
    elif choice == '2':
        ckpt_path = 'checkpoint_ssm_finetune.pth'
        start_it = 0
        if os.path.exists('checkpoint_ssm_pretrain.pth') and not os.path.exists(ckpt_path):
            if input("Load base pre-trained weights before fine-tuning? (y/n): ").strip().lower() == 'y':
                ckpt = torch.load('checkpoint_ssm_pretrain.pth', map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                print("Pre-trained base weights loaded!")

        if os.path.exists(ckpt_path) and input("Resume existing fine-tuning checkpoint? (y/n): ").strip().lower() == 'y':
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_it = ckpt.get('iteration', 0)
            
        if not os.path.exists(JSON_DATASET_PATH):
            print(f"ERROR: Cannot find JSON dataset at {JSON_DATASET_PATH}. Please update the script path.")
        else:
            run_finetuning(
                model, JSON_DATASET_PATH, tokenizer, optimizer, device, vocab_size, start_it, 
                CHUNK_SIZE, ENABLE_COGNITIVE_FORGETTING, BATCH_SIZE, GRAD_ACCUM_STEPS, USE_JSON_STREAMING
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
            model.load_state_dict(ckpt['model_state_dict'])
            
        chat_mode(model, tokenizer, device, chunk_size=ckpt.get('chunk_size', CHUNK_SIZE))