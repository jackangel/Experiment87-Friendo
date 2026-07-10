"""
Data utilities — token generators, ChatML streaming, and dataset mixing.

Provides three data streams:
  1. ``token_generator_from_parquet`` — flat token generator for pretraining.
  2. ``stream_chatml_from_json``       — ChatML-masked batches for fine-tuning.
  3. ``DatasetMixer`` / ``mixed_token_generator_from_datasets`` — weighted multi-dataset mixing.
"""

import os
import glob
import json
import random

import torch
import pyarrow.parquet as pq

from .tokenizer import CHAT_START, CHAT_END

# Optional: for streaming large JSON files
try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False
    print("[INFO] ijson not installed. Large JSON files will be loaded entirely into memory.")
    print("[INFO] Install with: pip install ijson")


# =============================================================================
# DATA UTILITIES (Pre-training & Fine-tuning streams)
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
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
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


def _detach_kv_list(past_key_values):
    """Detach all levels in a KV/carry list for gradient isolation across chunks.

    Each entry is a 4-tuple (k, v, s, kr) or possibly None (meta KV during
    early-exit at inference). Returns a list in the same structure, or None
    if the input is falsy.
    """
    if not past_key_values:
        return None
    result = []
    for entry in past_key_values:
        if entry is None:
            result.append(None)
        else:
            k, v, s, kr = entry
            result.append((
                k.detach() if k is not None else None,
                v.detach() if v is not None else None,
                s.detach() if s is not None else None,
                kr.detach() if kr is not None else None,
            ))
    return result


def load_checkpoint_with_filter(model, checkpoint_state_dict):
    """
    Load checkpoint while filtering out keys with size mismatches (e.g., forgetting gate buffers).
    This allows resuming training after changing forgetting hyperparameters.

    BACKWARD COMPATIBILITY NOTE (Meta-Dynamic Routing):
    When continuing from a checkpoint that was trained BEFORE meta-routing
    was enabled, the checkpoint will simply not contain any `meta_phase.*`
    keys.  Because we use strict=False and only copy keys that exist in the
    model, this is handled transparently: the meta router / shared dynamic
    layer simply keep their freshly-initialized weights.  No error is raised.
    A one-line diagnostic is printed in that case.
    """
    model_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []
    meta_keys_loaded = 0
    meta_keys_total = sum(1 for k in model_state if k.startswith("meta_phase."))

    # Checkpoint migration: tok_embeddings.weight -> tok_embeddings_mu.weight
    # Old checkpoints trained before Gaussian embeddings used a single
    # deterministic table named 'tok_embeddings.weight'.  The model now uses
    # 'tok_embeddings_mu.weight' (+ optionally 'tok_embeddings_logvar.weight').
    # We transparently copy old embeddings into the new mu table so resumed
    # training retains learned token representations.
    migrated_gaussian = False
    if ('tok_embeddings.weight' in checkpoint_state_dict
            and 'tok_embeddings_mu.weight' not in checkpoint_state_dict
            and 'tok_embeddings_mu.weight' in model_state):
        old_w = checkpoint_state_dict.pop('tok_embeddings.weight')
        if old_w.shape == model_state['tok_embeddings_mu.weight'].shape:
            filtered_state['tok_embeddings_mu.weight'] = old_w
            migrated_gaussian = True
        else:
            skipped_keys.append(
                f"tok_embeddings.weight (ckpt: {old_w.shape}, "
                f"model mu: {model_state['tok_embeddings_mu.weight'].shape})"
            )

    # Checkpoint migration: meta_phase.meta_net geometric feature expansion
    # Old checkpoints trained before geometric features had a router input of
    # size [dim, 2].  The new router has input [dim + geo_feature_dim, 2].
    # We copy the old weights into the first 'dim' columns (preserving learned
    # routing logic for the hidden state) and keep random init for the new
    # geometric feature columns.  The bias is copied unchanged (same shape).
    migrated_meta_router = False
    meta_router_old_dim = None
    meta_router_new_dim = None
    meta_weight_key = 'meta_phase.meta_net.weight'
    meta_bias_key = 'meta_phase.meta_net.bias'
    if (meta_weight_key in checkpoint_state_dict
            and meta_weight_key in model_state):
        old_weight = checkpoint_state_dict[meta_weight_key]
        new_weight = model_state[meta_weight_key]
        # Check if checkpoint router is smaller (missing geometric features).
        # Shape: [input_dim, num_route_options], so compare dim 0.
        if (old_weight.shape[1] == new_weight.shape[1]  # same output size
                and old_weight.shape[0] < new_weight.shape[0]):  # expanded input
            # Copy old columns, keep random init for new geometric columns.
            expanded_weight = new_weight.clone()
            expanded_weight[:old_weight.shape[0], :] = old_weight
            filtered_state[meta_weight_key] = expanded_weight
            meta_router_old_dim = old_weight.shape[0]
            meta_router_new_dim = new_weight.shape[0]
            checkpoint_state_dict.pop(meta_weight_key)
            migrated_meta_router = True
            # Bias can be copied directly (shape unchanged).
            if meta_bias_key in checkpoint_state_dict:
                filtered_state[meta_bias_key] = checkpoint_state_dict.pop(meta_bias_key)

    for key, value in checkpoint_state_dict.items():
        if key in model_state:
            if value.shape == model_state[key].shape:
                filtered_state[key] = value
                if key.startswith("meta_phase."):
                    meta_keys_loaded += 1
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

    # Diagnostic for the Meta-Dynamic Routing Network.
    if meta_keys_total > 0 and meta_keys_loaded == 0:
        print(f"[INFO] Checkpoint has no Meta-Dynamic Routing weights "
              f"({meta_keys_total} new params). Meta router & shared dynamic "
              f"layer will start from fresh initialization. Compute-penalty "
              f"warmup will ramp from 0 → target to let them catch up.")
    elif meta_keys_loaded > 0:
        print(f"[INFO] Loaded {meta_keys_loaded}/{meta_keys_total} Meta-Dynamic Routing weights from checkpoint.")

    # Diagnostic for the Gaussian embedding migration.
    if migrated_gaussian:
        print(f"[INFO] Migrated tok_embeddings.weight → tok_embeddings_mu.weight "
              f"(Gaussian embeddings: mean table retained from checkpoint). "
              f"Log-variance table starts fresh from N(0, sigma^2) init.")

    # Diagnostic for the meta router geometric feature expansion.
    if migrated_meta_router:
        geo_dim = meta_router_new_dim - meta_router_old_dim
        print(f"[INFO] Migrated meta_phase.meta_net.weight: expanded input "
              f"{meta_router_old_dim} → {meta_router_new_dim} (+{geo_dim} geometric features). "
              f"Preserved learned routing weights for hidden state; new "
              f"geometric columns start from random init.")

    model.load_state_dict(filtered_state, strict=False)
    return len(skipped_keys)


def apply_meta_compute_penalty(model, loss):
    """
    Add the Meta-Dynamic Routing regularizers to the language-model loss.

    Two terms are added:
      1.  Compute penalty (+weight * penalty) — pushes the router to EXIT early
          for 'easy' tokens.  Its weight is warmed up from 0 → target by the
          model itself (see _update_compute_penalty_schedule) and KILLED (set
          to 0) while the router is collapsed (see penalty kill-switch).
      2.  Entropy bonus (-entropy_weight * H) — rewards the router for NOT
          collapsing to a degenerate always-EXIT / always-APPLY policy.  This
          is the primary exploration mechanism that lets the router discover
          that APPLY can be useful.

    Should be called once per forward pass (after CE loss is computed), BEFORE
    loss.backward(), so both regularizer gradients flow into the meta router.
    """
    if not getattr(model, "enable_meta_routing", False):
        return loss

    # 1) Compute penalty (pushes toward EXIT).
    penalty = getattr(model, "last_meta_compute_penalty", None)
    weight = getattr(model, "compute_penalty_weight", 0.0)
    if penalty is not None and weight > 0.0:
        loss = loss + weight * penalty

    # 2) Entropy bonus (pulls back from collapse).
    entropy = getattr(model, "last_meta_entropy", None)
    ew = getattr(model.meta_phase, "entropy_weight", 0.0)
    if entropy is not None and ew > 0.0:
        # Normalise by number of routing steps so total_entropy is a mean.
        M = getattr(model.meta_phase, "max_meta_steps", 1)
        loss = loss - ew * (entropy * M / max(1, M))

    return loss


def apply_gaussian_kl_penalty(model, loss):
    """
    Add the Gaussian-embedding KL divergence regularizer to the loss.

    The KL divergence is computed inside ``SSMTransformer._gaussian_embed``
    during each TRAINING forward pass (when sigma sampling occurs) and stored
    in ``model.last_kl_loss``.  This function adds `kl_weight * KL` so the
    learned token distributions stay reasonably close to N(0, 1), preventing
    variance collapse (which would revert to a deterministic embedding).

    Should be called once per forward pass (after CE loss is computed), BEFORE
    loss.backward(), so the KL gradient flows into the mu/logvar tables.

    No-op when:
      - Gaussian embeddings are disabled.
      - last_kl_loss is None (e.g. inference / deterministic forward).
    """
    if not getattr(model, "enable_gaussian_embeddings", False):
        return loss

    kl = getattr(model, "last_kl_loss", None)
    weight = getattr(model, "kl_weight", 0.0)
    if kl is not None and weight > 0.0:
        loss = loss + weight * kl

    return loss


# =============================================================================
# Stream 1: Plain Text Parquet (Pre-training) - Flat Generator
# =============================================================================

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


# =============================================================================
# Stream 2: OpenHermes JSON / ChatML format (Fine-Tuning)
# =============================================================================

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
# DATASET MIXING FOR GENERAL LANGUAGE MODELS
# =============================================================================

class DatasetMixer:
    """
    Mixes multiple datasets according to specified weights for general language model training.

    Challenges addressed:
    1. Weighted sampling across datasets
    2. Different dataset formats (Parquet with different schemas)
    3. Epoch boundaries (some datasets finish before others)
    4. Memory efficiency (streaming from all datasets)
    5. Resumability (can save/restore mixing state)
    6. Temperature scaling to control over/under-sampling
    """

    def __init__(self, dataset_config, tokenizer, text_column_map=None, temperature=1.0):
        """
        Args:
            dataset_config: Dict of {name: {"path": str, "weight": float}}
            tokenizer: Tokenizer instance
            text_column_map: Dict mapping dataset name to column name (default: "text")
            temperature: Temperature for weight scaling (lower = more uniform, higher = more skewed)
        """
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.text_column_map = text_column_map or {}

        # Compute sampling probabilities with temperature scaling
        weights = torch.tensor([cfg["weight"] for cfg in dataset_config.values()])
        weights_temp = weights ** (1.0 / temperature)
        self.sampling_probs = weights_temp / weights_temp.sum()

        self.dataset_names = list(dataset_config.keys())
        self.token_generators = {}
        self.tokens_sampled = {name: 0 for name in self.dataset_names}
        self.batches_sampled = {name: 0 for name in self.dataset_names}

        print(f"\n{'='*60}")
        print(f"Dataset Mixer Configuration (Temperature: {temperature})")
        print(f"{'='*60}")
        for name, prob in zip(self.dataset_names, self.sampling_probs):
            original_weight = dataset_config[name]["weight"]
            print(f"{name:15s}: Target={original_weight:.3f}, Sampling={prob:.3f}")
        print(f"{'='*60}\n")

    def _get_text_column(self, dataset_name):
        """Get the text column name for a dataset."""
        return self.text_column_map.get(dataset_name, "text")

    def _init_generator(self, dataset_name):
        """Initialize a token generator for a specific dataset."""
        if dataset_name in self.token_generators:
            return self.token_generators[dataset_name]

        config = self.dataset_config[dataset_name]
        path = config["path"]
        text_column = self._get_text_column(dataset_name)

        # Find all parquet files in the dataset directory
        files = glob.glob(os.path.join(path, '**', '*.parquet'), recursive=True)

        if not files:
            print(f"[WARNING] No parquet files found in {path}")
            return None

        print(f"[Dataset Mixer] Loading {dataset_name}: {len(files)} files from {path}")
        random.shuffle(files)

        # Create generator
        generator = token_generator_from_parquet(files, text_column, self.tokenizer)
        self.token_generators[dataset_name] = generator
        return generator

    def mixed_token_stream(self, buffer_size=10000, chunk_size=2048):
        """
        Generate a mixed stream of tokens from all datasets.

        Strategy: Sample at CHUNK level (not token level) to maintain coherent sequences.
        Each yielded token comes from a contiguous chunk of the selected dataset, preventing
        incoherent mixing of encyclopedia text, code, math, and literature within a single sequence.

        Args:
            buffer_size: Number of tokens to buffer per dataset
            chunk_size: Number of contiguous tokens to yield from same dataset before resampling

        Yields:
            tokens from the mixed dataset stream
        """
        # Initialize all generators
        for name in self.dataset_names:
            self._init_generator(name)

        # Maintain buffers for each dataset
        buffers = {name: [] for name in self.dataset_names}
        exhausted = set()
        current_chunk = []
        current_dataset = None
        tokens_from_current = 0

        while True:
            # Fill buffers
            for name in self.dataset_names:
                if name in exhausted:
                    continue

                generator = self.token_generators.get(name)
                if generator is None:
                    exhausted.add(name)
                    continue

                # Fill buffer to target size
                while len(buffers[name]) < buffer_size:
                    try:
                        token = next(generator)
                        buffers[name].append(token)
                    except StopIteration:
                        # Dataset exhausted, reinitialize for next epoch
                        print(f"[Dataset Mixer] {name} epoch complete, restarting...")
                        self._init_generator(name)
                        break

            # Check if all datasets exhausted (shouldn't happen with reinitialization)
            if len(exhausted) == len(self.dataset_names):
                print("[Dataset Mixer] All datasets exhausted.")
                break

            # Sample from buffers according to probabilities
            available_datasets = [name for name in self.dataset_names
                                  if name not in exhausted and len(buffers[name]) > 0]

            if not available_datasets:
                continue

            # Filter probabilities for available datasets
            available_indices = [self.dataset_names.index(name) for name in available_datasets]
            available_probs = self.sampling_probs[available_indices]
            available_probs = available_probs / available_probs.sum()

            # If we need to select a new dataset (finished previous chunk or first time)
            if current_dataset is None or tokens_from_current >= chunk_size:
                # Sample new dataset for next chunk
                selected_idx = torch.multinomial(available_probs, 1).item()
                current_dataset = available_datasets[selected_idx]
                tokens_from_current = 0

            # Yield contiguous tokens from the same dataset
            if buffers[current_dataset]:
                token = buffers[current_dataset].pop(0)
                self.tokens_sampled[current_dataset] += 1
                tokens_from_current += 1
                yield token

    def get_statistics(self):
        """Return mixing statistics for monitoring."""
        total_tokens = sum(self.tokens_sampled.values())
        total_batches = sum(self.batches_sampled.values())

        stats = {
            "total_tokens": total_tokens,
            "total_batches": total_batches,
            "per_dataset": {}
        }

        for name in self.dataset_names:
            tokens = self.tokens_sampled[name]
            batches = self.batches_sampled[name]
            target_weight = self.dataset_config[name]["weight"]
            actual_weight = tokens / total_tokens if total_tokens > 0 else 0

            stats["per_dataset"][name] = {
                "tokens": tokens,
                "batches": batches,
                "target_weight": target_weight,
                "actual_weight": actual_weight,
                "drift": actual_weight - target_weight
            }

        return stats

    def print_statistics(self):
        """Print mixing statistics in a readable format."""
        stats = self.get_statistics()

        print(f"\n{'='*70}")
        print(f"Dataset Mixing Statistics (Total Tokens: {stats['total_tokens']:,})")
        print(f"{'='*70}")
        print(f"{'Dataset':<15} {'Target':>8} {'Actual':>8} {'Drift':>8} {'Tokens':>12}")
        print(f"{'-'*70}")

        for name, ds_stats in stats["per_dataset"].items():
            print(f"{name:<15} {ds_stats['target_weight']:>7.1%} {ds_stats['actual_weight']:>7.1%} "
                  f"{ds_stats['drift']:>+7.1%} {ds_stats['tokens']:>12,}")

        print(f"{'='*70}\n")


def mixed_token_generator_from_datasets(dataset_config, tokenizer, text_column_map=None, temperature=1.0, chunk_size=2048):
    """
    Convenience function to create a mixed token generator.

    Args:
        dataset_config: Dict of {name: {"path": str, "weight": float}}
        tokenizer: Tokenizer instance
        text_column_map: Dict mapping dataset name to column name
        temperature: Temperature for weight scaling
        chunk_size: Number of contiguous tokens from same dataset before resampling

    Returns:
        (generator, mixer): Generator yielding tokens and DatasetMixer instance for stats

    Example:
        dataset_config = {
            "FineWeb": {"path": "path/to/fineweb", "weight": 0.60},
            "Wikipedia": {"path": "path/to/wiki", "weight": 0.20},
            "GitHub": {"path": "path/to/github", "weight": 0.20}
        }

        token_gen, mixer = mixed_token_generator_from_datasets(dataset_config, tokenizer)
        for token in token_gen:
            # Use token for training
    """
    mixer = DatasetMixer(dataset_config, tokenizer, text_column_map, temperature)
    return mixer.mixed_token_stream(chunk_size=chunk_size), mixer
