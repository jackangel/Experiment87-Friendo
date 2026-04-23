import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import random
import pyarrow.parquet as pq
import tiktoken

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
        # allowed_special="all" permits encoding of special tokens if they exist in the text
        return self.tokenizer.encode(text, allowed_special="all")
        
    def decode(self, ids):
        return self.tokenizer.decode(ids)

# ==========================================
# 1. THE ARCHITECTURE (GREENFIELD)
# ==========================================

# --- RoPE Helper Functions ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precomputes complex frequencies for Rotary Positional Embeddings (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """Applies RoPE to Query and Key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis.view(1, 1, freqs_cis.shape[0], freqs_cis.shape[1])
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# --- RMSNorm ---
class RMSNorm(nn.Module):
    """Root Mean Square Normalization for improved training stability."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class FoveaBlock(nn.Module):
    """
    A robust, Multi-Head Transformer block utilizing RMSNorm, RoPE, and Dropout.
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln1 = RMSNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, freqs_cis):
        B, L, D = x.shape
        
        # PRE-RMSNorm
        x_norm = self.ln1(x)
        
        # Multi-Head Attention reshaping
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, B, Heads, L, Head_Dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply Rotary Positional Embeddings
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, L, D) # Flatten back to [B, L, D]
        
        x = x + self.dropout(self.proj(attn))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

class FoveaCortex(nn.Module):
    def __init__(self, d_model, n_layers=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([FoveaBlock(d_model, dropout=dropout) for _ in range(n_layers)])
        
    def forward(self, x, freqs_cis):
        for layer in self.layers:
            x = layer(x, freqs_cis)
        return x

class LandmarkMemory(nn.Module):
    def __init__(self, d_model, max_landmarks=2000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_landmarks = max_landmarks
        
        self.salience_proj = nn.Linear(d_model, 1)    
        self.uncertainty_gate = nn.Linear(d_model, 1)  
        self.write_gate = nn.Linear(d_model, 1) # Write/Forget Gate
        
        nn.init.constant_(self.salience_proj.bias, -2.0)
        nn.init.constant_(self.uncertainty_gate.bias, -2.0)
        nn.init.constant_(self.write_gate.bias, 0.0)
        
        self.mem_out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.mem_out_proj.weight)
        nn.init.zeros_(self.mem_out_proj.bias)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory_state=None):
        B, L, D = x.shape 
        gate_penalty = torch.tensor(0.0, device=x.device)
        
        if memory_state is None:
            landmarks_k = torch.empty(1, 0, self.d_model, device=x.device)
            landmarks_v = torch.empty(1, 0, self.d_model, device=x.device)
        else:
            landmarks_k, landmarks_v = memory_state 

        # --- 1. NEED-DRIVEN RESONANCE ---
        need = torch.sigmoid(self.uncertainty_gate(x))
        
        if landmarks_k.size(1) > 0:
            q_norm = F.normalize(x, dim=-1)
            k_norm = F.normalize(landmarks_k, dim=-1)
            resonance = torch.bmm(q_norm, k_norm.transpose(1, 2)) 
            attn = F.softmax(resonance * 5.0, dim=-1) 
            
            retrieved = torch.bmm(attn, landmarks_v) 
            memory_injection = self.mem_out_proj(need * retrieved)
            x_out = x + self.dropout(memory_injection)
        else:
            x_out = x

        # --- 2. LATERAL INHIBITION (Anchor Drop) & GATED WRITE ---
        salience = torch.sigmoid(self.salience_proj(x)).squeeze(-1) 
        
        local_max = F.max_pool1d(
            salience.unsqueeze(1), kernel_size=31, stride=1, padding=15
        ).squeeze(1) 
        
        is_landmark = (salience == local_max) & (salience > 0.15)
        landmark_idx = is_landmark[0].nonzero(as_tuple=True)[0]
        
        if len(landmark_idx) > 0:
            new_k = x[:, landmark_idx, :]
            raw_new_v = x_out[:, landmark_idx, :] 
            
            # Apply Gating Mechanism
            gate_value = torch.sigmoid(self.write_gate(new_k))
            
            # Stabilize the Gate: Penalize the gate for closing too drastically (moving away from 1.0)
            gate_penalty = torch.mean((1.0 - gate_value) ** 2)
            
            new_v = raw_new_v * gate_value
            
            landmarks_k = torch.cat([landmarks_k, new_k], dim=1)
            landmarks_v = torch.cat([landmarks_v, new_v], dim=1)
            
            if landmarks_k.size(1) > self.max_landmarks:
                landmarks_k = landmarks_k[:, -self.max_landmarks:, :]
                landmarks_v = landmarks_v[:, -self.max_landmarks:, :]

        return x_out, (landmarks_k, landmarks_v), gate_penalty

class ResonantBrain(nn.Module):
    def __init__(self, vocab_size, d_model=256, max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
        self.head_dim = d_model // 4 
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis)
        
        self.fovea = FoveaCortex(d_model, n_layers=6, dropout=dropout) 
        self.episodic = LandmarkMemory(d_model, max_landmarks=8192, dropout=dropout)
        
        # Language modeling head without bias (standard for weight tying)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Apply Weight Tying: Share weights between input embedding and output projection
        self.head.weight = self.embed.weight

    def forward(self, x_fovea, memory_state=None, is_generating=False, start_pos=0):
        B, L = x_fovea.shape
        x = self.embed(x_fovea)
        
        # --- OPTION B: DYNAMIC ROPE EXTENSION ---
        if start_pos + L > self.freqs_cis.shape[0]:
            new_length = start_pos + L + 1000
            new_freqs = precompute_freqs_cis(self.head_dim, new_length).to(x.device)
            self.register_buffer("freqs_cis", new_freqs, persistent=False)
        # ----------------------------------------
        
        freqs_cis = self.freqs_cis[start_pos : start_pos + L]
        
        fovea_out = self.fovea(x, freqs_cis) 
        
        if is_generating:
            newest_thought = fovea_out[:, -1:]
            episodic_out, new_memory_state, gate_penalty = self.episodic(newest_thought, memory_state)
            logits = self.head(episodic_out)
        else:
            episodic_out, new_memory_state, gate_penalty = self.episodic(fovea_out, memory_state) 
            logits = self.head(episodic_out)
            
        return logits, new_memory_state, gate_penalty

# ==========================================
# 3. UTILITIES & GENERATION
# ==========================================

def apply_sampling_penalties(logits, generated_ids, repetition_penalty=1.2, top_k=50, top_p=0.9):
    """Applies repetition penalty, top-k, and top-p filtering to logits."""
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

# ==========================================
# 4. PARQUET DATA HANDLING
# ==========================================

def get_parquet_files(directory):
    """Recursively fetch all parquet files in a directory."""
    search_pattern = os.path.join(directory, '**', '*.parquet')
    files = glob.glob(search_pattern, recursive=True)
    return files

def setup_tokenizer(parquet_dir, encoding_name="gpt2"):
    """Initializes tiktoken and locates parquet data files."""
    files = get_parquet_files(parquet_dir)
    if not files:
        print(f"Warning: No parquet files found in {parquet_dir}. Training will not be possible.")
        
    tokenizer = TiktokenTokenizer(encoding_name=encoding_name)
    return tokenizer, files

def stream_tokens_from_parquet(file, text_column, tokenizer, seq_len, device):
    """Streams encoded token sequences from a single parquet file."""
    buffer = []
    try:
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(batch_size=500, columns=[text_column]):
            df = batch.to_pandas()
            for text in df[text_column].dropna():
                tokens = tokenizer.encode(str(text))
                buffer.extend(tokens)
                
                # Yield chunks of seq_len + 1 (for x and y)
                while len(buffer) >= seq_len + 1:
                    chunk = buffer[:seq_len + 1]
                    buffer = buffer[seq_len:] # overlap or shift by seq_len
                    
                    tensor_chunk = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(device)
                    yield tensor_chunk
    except Exception as e:
        print(f"Error streaming tokens from {file}: {e}")

# ==========================================
# 5. TRAINING AND CHAT LOOPS
# ==========================================

def run_training(model, parquet_files, text_column, tokenizer, optimizer, device, vocab_size):
    seq_len = 4096
    print(f"\n--- Starting Training | Device: {device} ---")
    print(f"Found {len(parquet_files)} parquet files. Each file represents 1 epoch.\n")
    
    iteration = 0
    gate_penalty_weight = 0.1 
    
    # Shuffle files for randomness
    random.shuffle(parquet_files)
    
    for epoch, file in enumerate(parquet_files, start=1):
        print(f"\n--- Epoch {epoch}/{len(parquet_files)} | File: {os.path.basename(file)} ---")
        memory_state = None 
        
        token_stream = stream_tokens_from_parquet(file, text_column, tokenizer, seq_len, device)
        
        running_train_loss = 0.0
        train_steps = 0
        
        val_buffer = [] # Store 5% of data for validation
        
        for data_chunk in token_stream:
            # 5% chance to route to validation buffer
            if random.random() < 0.05:
                val_buffer.append(data_chunk)
                if len(val_buffer) > 50: # Keep val buffer manageable
                    val_buffer.pop(0)
                continue
                
            x = data_chunk[:, :-1]
            y = data_chunk[:, 1:]
            
            logits, memory_state, gate_penalty = model(x, memory_state, start_pos=0)
            
            k, v = memory_state
            memory_state = (k.detach(), v.detach())
            
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss = ce_loss + (gate_penalty_weight * gate_penalty)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_train_loss += loss.item()
            train_steps += 1
            iteration += 1
            
            # Evaluate every 1000 steps
            if iteration % 1000 == 0:
                avg_train_loss = running_train_loss / train_steps
                
                # Calculate Validation Loss
                val_loss = 0.0
                if val_buffer:
                    model.eval()
                    with torch.no_grad():
                        val_chunk = random.choice(val_buffer)
                        v_x = val_chunk[:, :-1]
                        v_y = val_chunk[:, 1:]
                        v_logits, _, v_gate = model(v_x, memory_state, start_pos=0)
                        v_ce = F.cross_entropy(v_logits.view(-1, vocab_size), v_y.view(-1))
                        val_loss = (v_ce + (gate_penalty_weight * v_gate)).item()
                    model.train()
                
                print(f"[Step {iteration}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Landmarks: {memory_state[0].size(1)}")
                running_train_loss = 0.0
                train_steps = 0
                
            # Prediction every 5000 steps
            if iteration % 5000 == 0:
                print("\n" + "="*40 + "\n🧠 INITIATING NEED-DRIVEN GENERATION (PREDICTION)...\n" + "="*40)
                model.eval()
                with torch.no_grad():
                    gen_mem = memory_state 
                    context = x[:, :64] # Seed with current context
                    generated_ids = context[0].tolist()
                    
                    for step in range(200):
                        rolling_fovea = context[:, -64:] 
                        gen_logits, gen_mem, _ = model(rolling_fovea, gen_mem, is_generating=True, start_pos=step)
                        
                        next_token_logits = gen_logits[0, -1]
                        next_token_logits = apply_sampling_penalties(
                            next_token_logits, generated_ids, repetition_penalty=1.2, top_k=50, top_p=0.9
                        )
                        
                        probs = F.softmax(next_token_logits / 0.8, dim=-1)
                        idx = torch.multinomial(probs, 1)
                        
                        generated_ids.append(idx.item())
                        context = torch.cat([context, idx.unsqueeze(0)], dim=1)
                        
                    print(f"{tokenizer.decode(generated_ids)}\n" + "="*40 + "\n")
                model.train()
                
    print("\nTraining Complete.")
    torch.save(model.state_dict(), 'checkpoint.pth')
    print("Model checkpoint saved to 'checkpoint.pth'.")

def chat_mode(model, tokenizer, device):
    print("\n" + "="*40 + "\n💬 ENTERING CHAT MODE\n" + "="*40)
    print("Type 'quit' or 'exit' to stop.")
    model.eval()
    memory_state = None
    
    with torch.no_grad():
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            encoded_input = tokenizer.encode(user_input + " ")
            context = torch.tensor(encoded_input, dtype=torch.long).unsqueeze(0).to(device)
            generated_ids = context[0].tolist()
            
            print("Brain: ", end="", flush=True)
            
            for step in range(100): 
                rolling_fovea = context[:, -64:] if context.size(1) > 64 else context
                gen_logits, memory_state, _ = model(rolling_fovea, memory_state, is_generating=True, start_pos=step)
                
                next_token_logits = gen_logits[0, -1]
                next_token_logits = apply_sampling_penalties(
                    next_token_logits, generated_ids, repetition_penalty=1.2, top_k=40, top_p=0.9
                )
                
                probs = F.softmax(next_token_logits / 0.8, dim=-1)
                idx = torch.multinomial(probs, 1)
                
                token_id = idx.item()
                generated_ids.append(token_id)
                context = torch.cat([context, idx.unsqueeze(0)], dim=1)
                
                word = tokenizer.decode([token_id])
                print(word, end="", flush=True)
                
            print()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    PARQUET_DIR = r"D:\Datasets\Gutenberg-BookCorpus-Cleaned-Data-English_data" # Replace with your folder path
    TEXT_COLUMN = "text"   # Replace with your text column name
    TIKTOKEN_ENCODING = "gpt2" # Options: 'gpt2', 'cl100k_base', 'p50k_base', etc.
    # ---------------------------------------------------------
    
    print(f"Setting up tiktoken tokenizer using encoding '{TIKTOKEN_ENCODING}'...")
    tokenizer, parquet_files = setup_tokenizer(PARQUET_DIR, encoding_name=TIKTOKEN_ENCODING)
    
    # Dynamically set vocab size from the tiktoken encoding
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary Size: {vocab_size}")
    
    model = ResonantBrain(vocab_size=vocab_size, d_model=256, dropout=0.1).to(device)
    
    # Calculate and display the total number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {total_params:,} trainable parameters.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    
    checkpoint_path = 'checkpoint.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at '{checkpoint_path}'.")
        choice = input("Do you want to [c]ontinue training, train from [s]cratch, or enter [chat] mode? (c/s/chat): ").strip().lower()
        
        if choice == 'c':
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Checkpoint loaded. Resuming training...")
            run_training(model, parquet_files, TEXT_COLUMN, tokenizer, optimizer, device, vocab_size)
        elif choice == 'chat':
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Checkpoint loaded.")
            chat_mode(model, tokenizer, device)
        else:
            print("Training from scratch...")
            run_training(model, parquet_files, TEXT_COLUMN, tokenizer, optimizer, device, vocab_size)
    else:
        print("No checkpoint found. Training from scratch...")
        run_training(model, parquet_files, TEXT_COLUMN, tokenizer, optimizer, device, vocab_size)