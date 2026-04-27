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

# ==========================================
# 0. TIKTOKEN TOKENIZER
# ==========================================

class TiktokenTokenizer:
    """
    An efficient tokenizer powered by OpenAI's tiktoken.
    Uses pre-trained encodings (e.g., 'gpt2', 'cl100k_base').
    """
    def __init__(self, encoding_name="gpt2"):
        print(f"Loading tiktoken encoding: '{encoding_name}'...")
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.tokenizer.n_vocab

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special="all")
        
    def decode(self, ids):
        return self.tokenizer.decode(ids)

# =============================================================================
# 1. RoPE with Position Interpolation (Fixing the Extrapolation Reality)
# =============================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, max_train_len: int = 4096):
    """
    Precomputes RoPE frequencies. If 'end' exceeds 'max_train_len', it applies 
    Linear Position Interpolation (PI) to compress the sequence into the trained domain.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    
    # Position Interpolation: Scale positions down if exceeding trained length
    if end > max_train_len:
        scaling_factor = max_train_len / end
        t = t * scaling_factor
        
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Complex tensor
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Fixed: x has shape (B, H, L, D). The sequence length L is at index 2.
    """
    ndim = x.ndim
    # i == 2 corresponds to the Sequence Length (L) dimension in (B, H, L, D)
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
# 2. The Core Architecture: FoveaCortex + LandmarkMemory (Associative Matrix)
# =============================================================================

class LongContextAttention(nn.Module):
    def __init__(self, dim, num_heads, num_landmarks_per_chunk=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_landmarks = num_landmarks_per_chunk
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        # Learned gate to balance local (Fovea) and global (Landmark) attention
        self.memory_gate = nn.Parameter(torch.zeros(1))
        
        # PRODUCTION FIX 1: Initialize decay to 0.95 in sigmoid space
        # sigmoid(2.944) ≈ 0.95, giving 95% retention rate
        self.memory_decay = nn.Parameter(torch.tensor(2.944))

    def compress_to_landmarks(self, tensor):
        """
        Compresses a chunk of L tokens into M landmark tokens using adaptive pooling.
        Input: (B, H, L, D) -> Output: (B, H, M, D)
        """
        B, H, L, D = tensor.shape
        # Reshape for 1D pooling: (B*H, D, L)
        x = tensor.transpose(2, 3).reshape(B * H, D, L)
        # Pool along the sequence length
        x = F.adaptive_avg_pool1d(x, self.num_landmarks)
        # Reshape back: (B, H, M, D)
        return x.reshape(B, H, D, self.num_landmarks).transpose(2, 3)

    def forward(self, x, freqs_cis, memory_state=None):
        B, L, D = x.shape
        
        # Create base (unrotated) projections for memory operations
        q_base = self.wq(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_base = self.wk(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE ONLY for local attention
        q_rope, k_rope = apply_rotary_emb(q_base, k_base, freqs_cis)
        
        # --- FOVEA CORTEX (Local Attention) ---
        # Use RoPE-rotated queries and keys for local attention
        local_out = F.scaled_dot_product_attention(q_rope, k_rope, v, is_causal=True)
        
        # --- LANDMARK MEMORY (Global Attention via Associative Matrix) ---
        if memory_state is not None:
            M_old, Z_old = memory_state
            
            # PRODUCTION FIX 2: Monitor memory matrix norms BEFORE update
            M_norm = M_old.norm().item()
            Z_norm = Z_old.norm().item()
            
            # Emergency stop if values explode
            if not (math.isfinite(M_norm) and math.isfinite(Z_norm)):
                raise RuntimeError(
                    f"Memory matrix explosion detected! M_norm={M_norm}, Z_norm={Z_norm}. "
                    f"Current decay value: {torch.sigmoid(self.memory_decay).item():.4f}"
                )
            
            # Use UNROTATED q_base for memory read (content-based retrieval)
            phi_Q = F.elu(q_base) + 1.0
            
            # Read from the infinite past
            num = torch.matmul(phi_Q, M_old)
            den = (phi_Q * Z_old.unsqueeze(-2)).sum(dim=-1, keepdim=True) + 1e-6
            
            global_out = num / den
            
            # Combine using a learned gate
            gate = torch.sigmoid(self.memory_gate)
            out = local_out + gate * global_out
        else:
            out = local_out
            
        # --- MEMORY UPDATE (Write Operation) ---
        # Compress UNROTATED k_base and v into landmarks
        landmark_k = self.compress_to_landmarks(k_base)
        landmark_v = self.compress_to_landmarks(v)
        
        # Apply ELU + 1 activation to Landmark Keys
        phi_K = F.elu(landmark_k) + 1.0
        
        # Compute the update terms
        delta_M = torch.matmul(phi_K.transpose(-2, -1), landmark_v)
        delta_Z = phi_K.sum(dim=-2)
        
        if memory_state is not None:
            M_old, Z_old = memory_state
            
            # PRODUCTION FIX 3: Apply learned decay to prevent unbounded growth
            # Sigmoid ensures decay is strictly between 0 and 1 (e.g., 0.95)
            decay = torch.sigmoid(self.memory_decay)
            
            M_new = (M_old * decay) + delta_M
            Z_new = (Z_old * decay) + delta_Z
        else:
            # Initialize if this is the first chunk
            M_new = delta_M
            Z_new = delta_Z
            
        new_memory_state = (M_new, Z_new)
        
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(out), new_memory_state

class LandmarkBlock(nn.Module):
    def __init__(self, dim, num_heads, num_landmarks_per_chunk=32):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LongContextAttention(dim, num_heads, num_landmarks_per_chunk)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, freqs_cis, memory_state=None):
        attn_out, new_memory_state = self.attn(self.norm1(x), freqs_cis, memory_state)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_memory_state

class LandmarkTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, max_seq_len=4096):
        super().__init__()
        
        # PRODUCTION FIX 4: Warn if layer count is too low for production
        if num_layers < 6:
            print(f"⚠️  WARNING: num_layers={num_layers} is shallow. "
                  f"For coherent chat responses, use 6+ layers.")
        
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        self.layers = nn.ModuleList([
            LandmarkBlock(dim, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight
        
        self.head_dim = dim // num_heads
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len * 2, max_train_len=max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x, start_pos=0, memory_states=None):
        h = self.tok_embeddings(x)
        seq_len = x.shape[1]
        
        if start_pos + seq_len > self.freqs_cis.shape[0]:
            new_length = start_pos + seq_len + 1000
            new_freqs = precompute_freqs_cis(self.head_dim, new_length).to(x.device)
            self.register_buffer("freqs_cis", new_freqs, persistent=False)
            
        chunk_freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
        
        new_memory_states = []
        if memory_states is None:
            memory_states = [None] * len(self.layers)
            
        for layer, mem_state in zip(self.layers, memory_states):
            h, new_mem = layer(h, chunk_freqs_cis, mem_state)
            new_memory_states.append(new_mem)
            
        h = self.norm(h)
        return self.output(h), new_memory_states

# ==========================================
# 3. UTILITIES & GENERATION
# ==========================================

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

def detach_memory(mem_states):
    """Helper to detach memory states to prevent backprop through previous steps."""
    if mem_states is None: 
        return None
    # mem_states is a list of (M, Z) tuples
    return [(M.detach(), Z.detach()) for M, Z in mem_states]

def reset_memory_if_exploded(memory_states):
    """
    Checks if memory has exploded and resets it if necessary.
    Returns (cleaned_memory_states, was_reset)
    """
    if memory_states is None:
        return None, False
    
    for M, Z in memory_states:
        if not (torch.isfinite(M).all() and torch.isfinite(Z).all()):
            print("🔥 Memory explosion detected - resetting to None")
            return None, True
    
    return memory_states, False

def validate_vocab_size(model, tokenizer):
    """Ensures model vocab size matches tokenizer vocab size."""
    model_vocab = model.tok_embeddings.num_embeddings
    tokenizer_vocab = tokenizer.vocab_size
    
    if model_vocab != tokenizer_vocab:
        raise ValueError(
            f"CRITICAL: Model vocab_size ({model_vocab}) != "
            f"Tokenizer vocab_size ({tokenizer_vocab}). "
            f"This will cause IndexError during training!"
        )
    print(f"✓ Vocab size validated: {model_vocab}")

# ==========================================
# 4. PARQUET DATA HANDLING
# ==========================================

def get_parquet_files(directory):
    search_pattern = os.path.join(directory, '**', '*.parquet')
    files = glob.glob(search_pattern, recursive=True)
    return files

def setup_tokenizer(parquet_dir, encoding_name="gpt2"):
    files = get_parquet_files(parquet_dir)
    if not files:
        print(f"Warning: No parquet files found in {parquet_dir}. Training will not be possible.")
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
                    tensor_chunk = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(device)
                    yield tensor_chunk
    except Exception as e:
        print(f"Error streaming tokens from {file}: {e}")

# =============================================================================
# 5. TRAINING AND CHAT LOOPS
# =============================================================================

def run_training(model, parquet_files, text_column, tokenizer, optimizer, device, vocab_size, 
                 start_iteration=0):
    seq_len = 512 
    print(f"\n--- Starting Training | Device: {device} ---")
    print(f"Starting from iteration: {start_iteration}")
    
    iteration = start_iteration
    random.shuffle(parquet_files)
    
    for epoch, file in enumerate(parquet_files, start=1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{len(parquet_files)} | File: {os.path.basename(file)}")
        print(f"{'='*60}")
        memory_states = None 
        
        token_stream = stream_tokens_from_parquet(file, text_column, tokenizer, seq_len, device)
        running_train_loss = 0.0
        train_steps = 0
        
        for data_chunk in token_stream:
            x = data_chunk[:, :-1]
            y = data_chunk[:, 1:]
            
            # Forward pass
            logits, memory_states = model(x, start_pos=0, memory_states=memory_states)
            
            # PRODUCTION FIX 5: Check for memory explosion and reset if needed
            memory_states, was_reset = reset_memory_if_exploded(memory_states)
            if was_reset:
                print("⚠️  Continuing training with fresh memory...")
            
            # Detach memory states for the next iteration
            memory_states = detach_memory(memory_states)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            
            # PRODUCTION FIX 6: Clip memory_decay gradients separately to prevent instability
            for layer in model.layers:
                if hasattr(layer.attn, 'memory_decay'):
                    if layer.attn.memory_decay.grad is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [layer.attn.memory_decay], max_norm=0.1
                        )
            
            # Regular gradient clipping for other parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_train_loss += loss.item()
            train_steps += 1
            iteration += 1
            
            if iteration % 100 == 0:
                avg_train_loss = running_train_loss / train_steps
                
                # Get average decay across all layers
                decay_vals = []
                gate_vals = []
                for layer in model.layers:
                    if hasattr(layer.attn, 'memory_decay'):
                        decay_vals.append(
                            torch.sigmoid(layer.attn.memory_decay).item()
                        )
                    if hasattr(layer.attn, 'memory_gate'):
                        gate_vals.append(
                            torch.sigmoid(layer.attn.memory_gate).item()
                        )
                
                avg_decay = sum(decay_vals) / len(decay_vals) if decay_vals else 0.0
                avg_gate = sum(gate_vals) / len(gate_vals) if gate_vals else 0.0
                
                print(f"[Step {iteration}] Loss: {avg_train_loss:.4f} | "
                      f"Memory Decay: {avg_decay:.4f} | Gate: {avg_gate:.4f}")
                running_train_loss = 0.0
                train_steps = 0
                
            if iteration % 5000 == 0:
                print("\n" + "="*60 + "\n🧠 INITIATING GENERATION (PREDICTION)...\n" + "="*60)
                model.eval()
                with torch.no_grad():
                    gen_mem = memory_states 
                    context = x[:, :256] 
                    generated_ids = context[0].tolist()
                    
                    for step in range(200):
                        rolling_fovea = context[:, -256:] 
                        gen_logits, gen_mem = model(rolling_fovea, start_pos=step, memory_states=gen_mem)
                        
                        next_token_logits = gen_logits[0, -1]
                        next_token_logits = apply_sampling_penalties(
                            next_token_logits, generated_ids, repetition_penalty=1.2, top_k=50, top_p=0.9
                        )
                        
                        probs = F.softmax(next_token_logits / 0.8, dim=-1)
                        idx = torch.multinomial(probs, 1)
                        
                        generated_ids.append(idx.item())
                        context = torch.cat([context, idx.unsqueeze(0)], dim=1)
                        
                    print(f"{tokenizer.decode(generated_ids)}\n" + "="*60 + "\n")
                model.train()
                
        # Save checkpoint after each epoch
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration,
            'epoch': epoch
        }
        torch.save(checkpoint, 'checkpoint.pth')
        print(f"✓ Checkpoint saved (iteration {iteration})")

def chat_mode(model, tokenizer, device):
    print("\n" + "="*60 + "\n💬 ENTERING CHAT MODE\n" + "="*60)
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'reset' to clear conversation memory.\n")
    
    model.eval()
    memory_states = None
    conversation_history = []
    
    with torch.no_grad():
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                memory_states = None
                conversation_history = []
                print("✓ Conversation memory cleared.")
                continue
            
            if not user_input:
                continue
            
            # Add to conversation history
            conversation_history.append(f"User: {user_input}")
            full_context = "\n".join(conversation_history) + "\nAssistant:"
            
            encoded_input = tokenizer.encode(full_context)
            context = torch.tensor(encoded_input, dtype=torch.long).unsqueeze(0).to(device)
            generated_ids = context[0].tolist()
            
            print("Assistant: ", end="", flush=True)
            
            response_tokens = []
            for step in range(150): 
                rolling_fovea = context[:, -512:] if context.size(1) > 512 else context
                gen_logits, memory_states = model(rolling_fovea, start_pos=step, memory_states=memory_states)
                
                next_token_logits = gen_logits[0, -1]
                next_token_logits = apply_sampling_penalties(
                    next_token_logits, generated_ids, repetition_penalty=1.2, top_k=40, top_p=0.9
                )
                
                probs = F.softmax(next_token_logits / 0.7, dim=-1)
                idx = torch.multinomial(probs, 1)
                
                token_id = idx.item()
                generated_ids.append(token_id)
                response_tokens.append(token_id)
                context = torch.cat([context, idx.unsqueeze(0)], dim=1)
                
                word = tokenizer.decode([token_id])
                print(word, end="", flush=True)
                
                # Stop if we hit a natural break
                if word.strip() in ['.', '!', '?', '\n']:
                    # Check if we've generated enough
                    if len(response_tokens) > 20:
                        break
            
            print()  # New line after response
            
            # Add assistant response to history
            assistant_response = tokenizer.decode(response_tokens)
            conversation_history.append(f"Assistant: {assistant_response}")
            
            # Keep only last 10 exchanges to prevent context from growing too large
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

# =============================================================================
# 6. MODEL CONFIGURATIONS (PRODUCTION READY)
# =============================================================================

MODEL_CONFIGS = {
    'tiny': {
        'dim': 256, 
        'num_heads': 4, 
        'num_layers': 4,
        'description': '~10M params - Fast testing & prototyping'
    },
    'small': {
        'dim': 512, 
        'num_heads': 8, 
        'num_layers': 6,
        'description': '~40M params - Real experiments & early training'
    },
    'medium': {
        'dim': 768, 
        'num_heads': 12, 
        'num_layers': 8,
        'description': '~100M params - Better quality responses'
    },
    'large': {
        'dim': 1024, 
        'num_heads': 16, 
        'num_layers': 12,
        'description': '~300M params - Best quality (requires good GPU)'
    }
}

def print_model_info(config_name, config, vocab_size):
    """Print detailed model configuration information."""
    print(f"\n{'='*60}")
    print(f"MODEL CONFIGURATION: {config_name.upper()}")
    print(f"{'='*60}")
    print(f"Description: {config['description']}")
    print(f"Architecture:")
    print(f"  • Vocabulary Size: {vocab_size:,}")
    print(f"  • Embedding Dimension: {config['dim']}")
    print(f"  • Number of Heads: {config['num_heads']}")
    print(f"  • Number of Layers: {config['num_layers']}")
    print(f"  • Head Dimension: {config['dim'] // config['num_heads']}")
    print(f"{'='*60}\n")

# =============================================================================
# 7. MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 ResonantBrain Production v2.0")
    print(f"Device: {device}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    PARQUET_DIR = r"I:\Datasets\Gutenberg-BookCorpus-Cleaned-Data-English_data" 
    TEXT_COLUMN = "text"   
    TIKTOKEN_ENCODING = "gpt2" 
    MODEL_SIZE = 'medium'  # Options: 'tiny', 'small', 'medium', 'large'
    
    # Advanced training settings
    LEARNING_RATE = 4e-4
    WEIGHT_DECAY = 0.01
    # ---------------------------------------------------------
    
    print(f"\nInitializing tokenizer with encoding '{TIKTOKEN_ENCODING}'...")
    tokenizer, parquet_files = setup_tokenizer(PARQUET_DIR, encoding_name=TIKTOKEN_ENCODING)
    
    vocab_size = tokenizer.vocab_size
    config = MODEL_CONFIGS[MODEL_SIZE]
    
    print_model_info(MODEL_SIZE, config, vocab_size)
    
    # Create model
    model = LandmarkTransformer(
        vocab_size=vocab_size,
        dim=config['dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers']
    ).to(device)
    
    # PRODUCTION FIX 8: Validate vocab size matches
    validate_vocab_size(model, tokenizer)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized with {total_params:,} trainable parameters.")
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95)
    )
    
    checkpoint_path = 'checkpoint.pth'
    start_iteration = 0
    
    if os.path.exists(checkpoint_path):
        print(f"\n✓ Found existing checkpoint at '{checkpoint_path}'.")
        choice = input("Options:\n"
                      "  [c] Continue training from checkpoint\n"
                      "  [s] Train from scratch (delete checkpoint)\n"
                      "  [chat] Enter chat mode\n"
                      "Enter choice: ").strip().lower()
        
        if choice == 'c':
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iteration = checkpoint.get('iteration', 0)
            print(f"✓ Checkpoint loaded. Resuming from iteration {start_iteration}...")
            run_training(model, parquet_files, TEXT_COLUMN, tokenizer, optimizer, 
                        device, vocab_size, start_iteration)
        
        elif choice == 'chat':
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Checkpoint loaded.")
            chat_mode(model, tokenizer, device)
        
        else:
            print("Training from scratch...")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"✓ Deleted old checkpoint.")
            run_training(model, parquet_files, TEXT_COLUMN, tokenizer, optimizer, 
                        device, vocab_size, start_iteration)
    else:
        print("\nNo checkpoint found. Starting training from scratch...")
        run_training(model, parquet_files, TEXT_COLUMN, tokenizer, optimizer, 
                    device, vocab_size, start_iteration)