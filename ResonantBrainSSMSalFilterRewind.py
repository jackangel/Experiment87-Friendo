import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import random
import pyarrow.parquet as pq
import tiktoken
import math
from datetime import datetime
from typing import Optional, Tuple, List

# ==========================================
# 0. TIKTOKEN TOKENIZER
# ==========================================

class TiktokenTokenizer:
    def __init__(self, encoding_name="gpt2"):
        print(f"Loading tiktoken encoding: '{encoding_name}'...")
        self.tokenizer = tiktoken.get_encoding(encoding_name)
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
        x_t = x.transpose(1, 2).contiguous()

        h, alpha = self._build_decay_filter(L, x.device)

        x_padded = F.pad(x_t, (0, L))
        h_padded = F.pad(h, (0, L))
        X_freq = torch.fft.rfft(x_padded, n=2 * L)
        H_freq = torch.fft.rfft(h_padded, n=2 * L)
        y = torch.fft.irfft(X_freq * H_freq, n=2 * L)[..., :L]

        h_norm = torch.norm(h, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        y = y / h_norm

        if carry_state is not None:
            t_pos = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(0)
            carry_decay = torch.exp(-alpha.unsqueeze(1) * (t_pos + 1))
            y = y + carry_state.unsqueeze(2) * carry_decay.unsqueeze(0)

        new_carry = y[:, :, -1].clone()
        return y.transpose(1, 2).contiguous(), new_carry

    def step(self, x_t, carry_state):
        alpha = F.softplus(self.log_alpha)
        decay = torch.exp(-alpha)
        y_t = x_t + decay * carry_state
        return y_t, y_t

# =============================================================================
# 3. SALIENCY EVICTION LOGIC
# =============================================================================

def apply_saliency_eviction(k, v, scores, num_sinks=4, max_capacity=256):
    """
    Evicts tokens from KV cache based on saliency scores, preserving sinks and heavy-hitters.
    k, v: [B, H, L, D]
    scores: [B, L]
    """
    B, H, L, D = k.shape
    if L <= max_capacity:
        return k, v, scores

    device = k.device
    sink_indices = torch.arange(num_sinks, device=device).unsqueeze(0).expand(B, -1)
    
    rest_scores = scores[:, num_sinks:]
    num_to_keep = max_capacity - num_sinks
    
    _, top_indices = torch.topk(rest_scores, num_to_keep, dim=-1)
    top_indices = top_indices + num_sinks
    
    # Sort to maintain relative positional order for RoPE integrity
    keep_indices, _ = torch.sort(torch.cat([sink_indices, top_indices], dim=-1), dim=-1)
    
    gather_idx_kv = keep_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D)
    new_k = torch.gather(k, 2, gather_idx_kv)
    new_v = torch.gather(v, 2, gather_idx_kv)
    new_scores = torch.gather(scores, 1, keep_indices)
    
    return new_k, new_v, new_scores

# =============================================================================
# 4. SSM-Attention Block (with Saliency Tracking)
# =============================================================================

class SSMAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len, num_layers, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len

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
        self.attn_dropout = nn.Dropout(dropout)

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, freqs_cis, carry_state=None, past_kv=None, use_cache=False):
        B, L, D = x.shape

        # SSM Branch
        ssm_out, new_carry = self.fft_conv(self.norm_ssm(x), carry_state)
        x = x + self.ssm_dropout(ssm_out)

        # Attention Branch
        h = self.norm_attn(x)
        q = self.wq(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # RoPE application
        q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis)

        # Cache Management & Saliency
        if past_kv is not None:
            past_k, past_v, past_scores = past_kv
            k_rope_full = torch.cat([past_k, k_rope], dim=2)
            v_full = torch.cat([past_v, v], dim=2)
        else:
            k_rope_full = k_rope
            v_full = v
            past_scores = torch.zeros((B, 0), device=x.device)

        # Lightweight Proxy Saliency (using Head 0 only to save compute)
        with torch.no_grad():
            q_proxy = q_rope[:, 0, :, :]  # [B, L_q, D_head]
            k_proxy = k_rope_full[:, 0, :, :]  # [B, L_k, D_head]
            proxy_scores = torch.matmul(q_proxy, k_proxy.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if L > 1:
                mask = torch.triu(torch.ones(L, k_proxy.size(1), device=x.device), diagonal=k_proxy.size(1) - L + 1).bool()
                proxy_scores.masked_fill_(mask, float('-inf'))
            proxy_weights = F.softmax(proxy_scores, dim=-1)
            current_saliency = proxy_weights.sum(dim=1) # [B, L_k]

        # Exponential Moving Average (LTD) for Saliency
        decay_factor = 0.9
        updated_scores = torch.cat([past_scores * decay_factor, torch.zeros((B, L), device=x.device)], dim=1)
        updated_scores += current_saliency

        # Evict if over capacity
        if use_cache:
            k_rope_full, v_full, updated_scores = apply_saliency_eviction(
                k_rope_full, v_full, updated_scores, num_sinks=4, max_capacity=self.max_seq_len
            )
            new_kv = (k_rope_full, v_full, updated_scores)
        else:
            new_kv = None

        # Hardware-accelerated FlashAttention
        is_causal = (past_kv is None or L > 1)
        attn_out = F.scaled_dot_product_attention(q_rope, k_rope_full, v_full, is_causal=is_causal)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.attn_dropout(self.wo(attn_out))

        # MLP
        x = x + self.mlp(self.norm_mlp(x))

        return x, new_carry, new_kv

# =============================================================================
# 5. SSM Transformer
# =============================================================================

class SSMTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.head_dim = dim // num_heads

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            SSMAttentionBlock(dim, num_heads, max_seq_len, num_layers, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len, max_train_len=max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)

        freqs_cis_ext = precompute_freqs_cis(self.head_dim, max_seq_len * 8, max_train_len=max_seq_len)
        self.register_buffer("freqs_cis_ext", freqs_cis_ext)

    def forward(self, x, carry_states=None, is_training=True, past_key_values=None, use_cache=False, abs_pos_offset=0):
        B, L = x.shape
        h = self.embed_dropout(self.tok_embeddings(x))

        # Absolute position tracking for RoPE to survive eviction
        if use_cache and past_key_values is not None:
            freqs = self.freqs_cis_ext[abs_pos_offset : abs_pos_offset + L]
        else:
            freqs = self.freqs_cis[:L]

        if carry_states is None:
            carry_states = [None] * self.num_layers

        new_carry_states = []
        new_key_values = []

        for i, layer in enumerate(self.layers):
            layer_past_kv = past_key_values[i] if past_key_values is not None else None
            h, new_carry, new_kv = layer(
                h, freqs,
                carry_state=carry_states[i],
                past_kv=layer_past_kv,
                use_cache=use_cache
            )
            new_carry_states.append(new_carry)
            new_key_values.append(new_kv)

        h = self.norm(h)
        logits = self.output(h)

        if use_cache:
            return logits, new_carry_states, new_key_values
        return logits, new_carry_states

# =============================================================================
# 6. UTILITIES & DATA
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
    print(f"\u2713 Vocab size validated: {model_vocab}")

def get_parquet_files(directory):
    return glob.glob(os.path.join(directory, '**', '*.parquet'), recursive=True)

def setup_tokenizer(parquet_dir, encoding_name="gpt2"):
    files = get_parquet_files(parquet_dir)
    tokenizer = TiktokenTokenizer(encoding_name=encoding_name)
    return tokenizer, files

def stream_tokens_from_parquet(file, text_column, tokenizer, seq_len, device):
    buffer = []
    try:
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(batch_size=500, columns=[text_column]):
            df = batch.to_pandas()
            for text in df[text_column].dropna():
                tokens = tokenizer.encode(str(text))
                buffer.extend(tokens)
                while len(buffer) >= seq_len + 1:
                    chunk = buffer[:seq_len + 1]
                    buffer = buffer[seq_len:]
                    yield torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error streaming tokens from {file}: {e}")

# =============================================================================
# 7. COGNITIVE MEMORY MANAGER (Rewinding System)
# =============================================================================

class CognitiveMemoryManager:
    """
    Manages bounded O(1) cognitive states by offloading snapshots to CPU RAM.
    Allows for instant rewinding, self-correction, and memory paging.
    """
    def __init__(self, device):
        self.device = device
        self.snapshots = {}
        self.history_chunks = [] # Tape Drive storage

    def save_snapshot(self, step_idx, carry_states, past_key_values, abs_pos_offset, generated_ids):
        """Saves the current cognitive state to CPU memory."""
        # Detach and move to CPU to prevent VRAM accumulation
        cpu_carry = [c.detach().cpu().clone() if c is not None else None for c in carry_states] if carry_states else None
        
        cpu_kv = []
        if past_key_values:
            for k, v, s in past_key_values:
                cpu_kv.append((k.detach().cpu().clone(), v.detach().cpu().clone(), s.detach().cpu().clone()))
        else:
            cpu_kv = None

        self.snapshots[step_idx] = {
            'carry_states': cpu_carry,
            'past_key_values': cpu_kv,
            'abs_pos_offset': abs_pos_offset,
            'generated_ids': generated_ids.copy()
        }

    def load_snapshot(self, step_idx):
        """Restores a cognitive state back to the GPU."""
        if step_idx not in self.snapshots:
            return None
        
        snap = self.snapshots[step_idx]
        
        dev_carry = [c.to(self.device) if c is not None else None for c in snap['carry_states']] if snap['carry_states'] else None
        
        dev_kv = []
        if snap['past_key_values']:
            for k, v, s in snap['past_key_values']:
                dev_kv.append((k.to(self.device), v.to(self.device), s.to(self.device)))
        else:
            dev_kv = None

        return dev_carry, dev_kv, snap['abs_pos_offset'], snap['generated_ids'].copy()

    def add_history_chunk(self, chunk_ids):
        """Stores raw text chunks for the Tape Drive rewind."""
        self.history_chunks.append(chunk_ids)

# =============================================================================
# 8. GENERATION (Infinite Context via Saliency, SSM, and Rewinding)
# =============================================================================

def generate_block_recurrent(model, context_ids, tokenizer, device,
                             max_new_tokens=256, chunk_size=512,
                             temperature=0.8, repetition_penalty=1.2,
                             top_k=50, top_p=0.9, enable_rewind=True):
    model.eval()
    memory_manager = CognitiveMemoryManager(device)

    with torch.no_grad():
        generated_ids = context_ids.copy()
        carry_states = None
        past_key_values = None
        abs_pos_offset = 0

        context_tensor = torch.tensor(context_ids, dtype=torch.long).unsqueeze(0).to(device)

        # Prefill: process context in chunks and populate Tape Drive
        for i in range(0, len(context_ids), chunk_size):
            chunk = context_tensor[:, i:i + chunk_size]
            if chunk.size(1) == 0:
                continue
            
            memory_manager.add_history_chunk(chunk[0].tolist())
            
            logits, carry_states, past_key_values = model(
                chunk, carry_states=carry_states,
                is_training=False, past_key_values=past_key_values, use_cache=True,
                abs_pos_offset=abs_pos_offset
            )
            abs_pos_offset += chunk.size(1)

        # Generate tokens
        tokens_generated = 0
        low_confidence_streak = 0
        snapshot_interval = 20 # Save state every 20 tokens
        
        while tokens_generated < max_new_tokens:
            # 1. Save State (Snapshot)
            if enable_rewind and tokens_generated % snapshot_interval == 0:
                memory_manager.save_snapshot(tokens_generated, carry_states, past_key_values, abs_pos_offset, generated_ids)

            last_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)

            logits, carry_states, past_key_values = model(
                last_token, carry_states=carry_states,
                is_training=False, past_key_values=past_key_values, use_cache=True,
                abs_pos_offset=abs_pos_offset
            )
            abs_pos_offset += 1

            next_token_logits = logits[0, -1].clone()
            next_token_logits = apply_sampling_penalties(
                next_token_logits, generated_ids,
                repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p
            )

            probs = F.softmax(next_token_logits / temperature, dim=-1)
            
            # 2. Self-Correction Rewind Logic
            if enable_rewind:
                top_prob = probs.max().item()
                if top_prob < 0.15: # Confidence threshold
                    low_confidence_streak += 1
                else:
                    low_confidence_streak = 0

                if low_confidence_streak >= 4: # Model is confused/hallucinating
                    rollback_step = max(0, tokens_generated - snapshot_interval)
                    # print(f"\n[Self-Correction] Confidence dropped. Rewinding to step {rollback_step}...")
                    
                    # Restore O(1) state instantly
                    carry_states, past_key_values, abs_pos_offset, generated_ids = memory_manager.load_snapshot(rollback_step)
                    tokens_generated = rollback_step
                    low_confidence_streak = 0
                    temperature = min(1.5, temperature + 0.1) # Bump temp to force new path
                    continue

            if torch.isnan(probs).any():
                next_token = tokenizer.tokenizer.eot_token
            else:
                next_token = torch.multinomial(probs, 1).item()

            # 3. Tape Drive Memory Paging (Simulated Trigger)
            # If the model outputs a specific token (e.g., a special <SEARCH> token), 
            # we can load a past chunk to repopulate the KV cache.
            # Here we simulate it by checking if the model needs a specific fact.
            # if next_token == SPECIAL_SEARCH_TOKEN:
            #     past_chunk = memory_manager.history_chunks[0] # Retrieve relevant chunk
            #     chunk_tensor = torch.tensor(past_chunk, dtype=torch.long).unsqueeze(0).to(device)
            #     _, _, past_key_values = model(chunk_tensor, carry_states=carry_states, past_key_values=past_key_values, use_cache=True, abs_pos_offset=abs_pos_offset)
            #     continue

            generated_ids.append(next_token)
            tokens_generated += 1

            if next_token == tokenizer.tokenizer.eot_token:
                break

    return generated_ids

# =============================================================================
# 9. TRAINING LOOP
# =============================================================================

def run_training(model, parquet_files, text_column, tokenizer, optimizer, device,
                 vocab_size, start_iteration=0, chunk_size=512):
    print(f"\n--- Starting Training | Device: {device} ---")
    iteration = start_iteration
    random.shuffle(parquet_files)

    warmup_steps = 2000
    lr_base = optimizer.param_groups[0]['lr']
    lr_min = lr_base / 10
    estimated_total_steps = len(parquet_files) * 50000

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, estimated_total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return max(lr_min / lr_base, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_iteration - 1 if start_iteration > 0 else -1
    )

    for epoch, file in enumerate(parquet_files, start=1):
        print(f"\n{'='*60}\nEpoch {epoch}/{len(parquet_files)} | File: {os.path.basename(file)}\n{'='*60}")

        carry_states = None
        past_key_values = None
        abs_pos_offset = 0
        
        token_stream = stream_tokens_from_parquet(file, text_column, tokenizer, chunk_size, device)
        running_train_loss = 0.0
        train_steps = 0

        for data_chunk in token_stream:
            x = data_chunk[:, :-1]
            y = data_chunk[:, 1:]

            if abs_pos_offset + x.size(1) > model.freqs_cis_ext.size(0):
                carry_states = None
                past_key_values = None
                abs_pos_offset = 0

            detached_carry = None
            if carry_states is not None:
                detached_carry = [c.detach() for c in carry_states]
                
            detached_kv = None
            if past_key_values is not None:
                detached_kv = [(k.detach(), v.detach(), s.detach()) for k, v, s in past_key_values]

            logits, carry_states, past_key_values = model(
                x, carry_states=detached_carry, past_key_values=detached_kv, 
                is_training=True, use_cache=True, abs_pos_offset=abs_pos_offset
            )
            abs_pos_offset += x.size(1)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_train_loss += loss.item()
            train_steps += 1
            iteration += 1

            del logits, loss

            if iteration % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"[Step {iteration}] Loss: {running_train_loss / train_steps:.4f} | LR: {current_lr:.2e}")
                running_train_loss = 0.0
                train_steps = 0

            if iteration % 5000 == 0:
                model.eval()
                context_ids = x[0, :256].tolist()
                gen_ids = generate_block_recurrent(
                    model, context_ids, tokenizer, device,
                    max_new_tokens=200, chunk_size=chunk_size,
                    temperature=0.8, repetition_penalty=1.2, enable_rewind=False
                )
                print(f"\n{tokenizer.decode(gen_ids)}\n")
                model.train()

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'iteration': iteration,
            'epoch': epoch,
            'chunk_size': chunk_size,
        }
        torch.save(checkpoint, 'checkpoint_ssm.pth')

# =============================================================================
# 10. CHAT MODE
# =============================================================================

def chat_mode(model, tokenizer, device, chunk_size=512):
    print("\n" + "="*60 + "\n\U0001f4ac ENTERING CHAT MODE (With Rewind Enabled)\n" + "="*60)
    model.eval()
    conversation_history = []

    with torch.no_grad():
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit']: break
            if user_input.lower() == 'reset':
                conversation_history = []
                continue
            if not user_input: continue

            conversation_history.append(f"User: {user_input}")
            full_context = "\n".join(conversation_history) + "\nAssistant:"
            context_ids = tokenizer.encode(full_context)

            print("Assistant: ", end="", flush=True)
            generated_ids = generate_block_recurrent(
                model, context_ids, tokenizer, device,
                max_new_tokens=150, chunk_size=chunk_size,
                temperature=0.7, repetition_penalty=1.2, top_k=40, top_p=0.9,
                enable_rewind=True # Rewind logic is active here
            )

            response_ids = generated_ids[len(context_ids):]
            response_text = tokenizer.decode(response_ids)

            for stop_char in ['.', '!', '?', '\n']:
                if stop_char in response_text and len(response_text) > 20:
                    response_text = response_text[:response_text.index(stop_char) + 1]
                    break

            print(response_text)
            conversation_history.append(f"Assistant: {response_text}")

# =============================================================================
# 11. MAIN ENTRY POINT
# =============================================================================

MODEL_CONFIGS = {
    'tiny':   {'dim': 256,  'num_heads': 4,  'num_layers': 4},
    'small':  {'dim': 512,  'num_heads': 8,  'num_layers': 6},
    'medium': {'dim': 768,  'num_heads': 12, 'num_layers': 8},
}

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\U0001f680 ResonantBrain SSM v2.1 (Saliency Eviction + Rewinding System)")

    PARQUET_DIR = r"I:\Datasets\Gutenberg-BookCorpus-Cleaned-Data-English_data"
    TEXT_COLUMN = "text"
    TIKTOKEN_ENCODING = "gpt2"
    MODEL_SIZE = 'medium'
    CHUNK_SIZE = 512
    LEARNING_RATE = 4e-4
    WEIGHT_DECAY = 0.01

    tokenizer, parquet_files = setup_tokenizer(PARQUET_DIR, encoding_name=TIKTOKEN_ENCODING)
    vocab_size = tokenizer.vocab_size
    config = MODEL_CONFIGS[MODEL_SIZE]

    model = SSMTransformer(
        vocab_size=vocab_size, dim=config['dim'], num_heads=config['num_heads'],
        num_layers=config['num_layers'], max_seq_len=CHUNK_SIZE
    ).to(device)

    validate_vocab_size(model, tokenizer)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    checkpoint_path = 'checkpoint_ssm.pth'
    start_iteration = 0

    if os.path.exists(checkpoint_path):
        choice = input("Options:\n  [c] Continue training\n  [s] Train from scratch\n  [chat] Chat mode\nChoice: ").strip().lower()
        if choice == 'c':
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_iteration = ckpt.get('iteration', 0)
            run_training(model, parquet_files, TEXT_COLUMN, tokenizer, optimizer, device, vocab_size, start_iteration, CHUNK_SIZE)
        elif choice == 'chat':
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            chat_mode(model, tokenizer, device, chunk_size=ckpt.get('chunk_size', CHUNK_SIZE))
        else:
            os.remove(checkpoint_path)
            run_training(model, parquet_files, TEXT_COLUMN, tokenizer, optimizer, device, vocab_size, start_iteration, CHUNK_SIZE)
    else:
        run_training(model, parquet_files, TEXT_COLUMN, tokenizer, optimizer, device, vocab_size, start_iteration, CHUNK_SIZE)