"""
Main entry point — interactive menu for pretraining, fine-tuning, and chat.

Run with::

    python -m resonantbrain          # module execution
    python -m resonantbrain.main     # explicit module path
    python run_resonantbrain.py      # convenience wrapper in the repo root

This module mirrors the original monolith's ``if __name__ == "__main__"`` block
of :mod:`ResonantBrainSSMSalFilterRewindForgetFineTune`.
"""

from __future__ import annotations

import glob
import os

import torch

from .tokenizer import TiktokenTokenizer
from .model import SSMTransformer
from .data import validate_vocab_size, load_checkpoint_with_filter
from .training import run_pretraining, run_pretraining_mixed, run_finetuning
from .chat import chat_mode


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS = {
    'tiny':   {'dim': 256,  'num_heads': 4,  'num_layers': 4},
    'small':  {'dim': 512,  'num_heads': 8,  'num_layers': 6},
    'medium': {'dim': 768,  'num_heads': 12, 'num_layers': 16},
}


# =============================================================================
# MAIN
# =============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ENABLE_COGNITIVE_FORGETTING = True
    ENABLE_GRAPH_REASONING = True   # Enable Latent Graph Reasoning in deep layers
    ENABLE_META_ROUTING = True      # Enable Meta-Dynamic Routing Network (per-token compute allocation)
    META_MAX_STEPS = 3             # Max routing loop iterations (M in the design doc)
    META_GUMBEL_TAU = 1.0          # Gumbel-Softmax temperature (anneal lower for harder routing)
    # Per-token epsilon-greedy exploration in the Meta-Dynamic Phase:
    # with this probability a token is FORCED to APPLY the dynamic layer at
    # every step (full rollout), with router gradient cut on it.  Prevents
    # the 'always-EXIT' collapse by giving the layer a steady bootstrap signal.
    # Suggested range: 0.05–0.15.  Set to 0.0 to disable.
    META_FORCE_EXPLORE_EPS = 0.30
    # Entropy bonus weight β.  Subtract β·H(p) from the loss so the router is
    # rewarded for keeping its soft distribution non-degenerate (exploration).
    # With 2 route options, max H = ln(2) ≈ 0.693; a β of 0.05 makes a fully-
    # collapsed router pay ~0.035 vs a balanced one → strong enough to fight
    # collapse even at low tau during annealing.
    META_ENTROPY_WEIGHT = 0.05
    # Kill-switch: when the router's free exit-prob EMA exceeds this floor,
    # the compute-penalty weight is forced to 0 so the entropy bonus can
    # recover the router unopposed.  Once exit-EMA drops below, warmup resumes.
    META_PENALTY_COLLAPSE_FLOOR = 0.85

    # ── Geometric & Contextual Router Features ───────────────────────────
    # Learnable reference vectors give the meta-router spatial awareness.
    #   • Region centroids: coarse "zones" in embedding space (like continents).
    #   • Semantic anchors:  fine-grained reference points (like cities) that
    #     act as a compressed learnable vocabulary for triangulation.
    # The router computes cosine similarity between the current token and
    # each centroid/anchor, plus temporal cosine similarity against the last
    # N tokens (semantic velocity).  These are concatenated with the hidden
    # state and fed to the routing Linear.
    META_NUM_REGION_CENTROIDS = 8    # Number of coarse region centroids
    META_NUM_SEMANTIC_ANCHORS = 32   # Number of fine-grained semantic anchors
    META_TEMPORAL_WINDOW = 64         # How many previous tokens for temporal angle

    # ── Gaussian (Probabilistic) Embeddings ──────────────────────────────
    # Each token is modelled as N(mu, sigma^2) instead of a fixed point.
    # During training we sample z = mu + sigma*eps (reparameterization trick);
    # during inference we use the mean deterministically.  KL divergence keeps
    # the distributions close to N(0,1) and prevents variance collapse.
    ENABLE_GAUSSIAN_EMBEDDINGS = True
    KL_WEIGHT = 0.001   # Weight on KL divergence term added to CE loss

    print(f"🚀 ResonantBrain SSM v5.0 - Gaussian Embeddings + Graph Reasoning + Meta-Routing Edition")
    print(f"   Device: {device}")
    print(f"   Gaussian Embeddings: {'Enabled' if ENABLE_GAUSSIAN_EMBEDDINGS else 'Disabled'}"
          + (f" (kl_weight={KL_WEIGHT})" if ENABLE_GAUSSIAN_EMBEDDINGS else ""))
    print(f"   Graph Reasoning: {'Enabled' if ENABLE_GRAPH_REASONING else 'Disabled'}")
    print(f"   Meta-Routing:    {'Enabled' if ENABLE_META_ROUTING else 'Disabled'}"
          + (f" (max_steps={META_MAX_STEPS}, tau={META_GUMBEL_TAU})" if ENABLE_META_ROUTING else ""))

    # File Paths Configuration
    PARQUET_DIR = r"I:\Datasets\FineWeb\fineweb-edu_data_CC-MAIN-2024-10"
    JSON_DATASET_PATH = r"I:\FineTunningDatasets\OpenHermes2.5\openhermes2_5.json"

    MODEL_SIZE = 'medium'
    CHUNK_SIZE = 768
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 4
    LEARNING_RATE = 4e-4

    # Optimization Parameters
    SALIENCY_DECAY = 0.95  # Configurable decay factor (was hardcoded to 0.9)
    USE_FLASH_ATTENTION = False  # Use Flash Attention (True) or Bayesian Signal Attention (False)
    USE_JSON_STREAMING = True  # Memory-efficient streaming for large datasets
    MAX_PARAGRAPH_CACHE = 50  # Limit paragraph states in generation

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
        use_flash_attn=USE_FLASH_ATTENTION,
        enable_graph_reasoning=ENABLE_GRAPH_REASONING,
        enable_meta_routing=ENABLE_META_ROUTING,
        meta_max_steps=META_MAX_STEPS,
        meta_gumbel_tau=META_GUMBEL_TAU,
        meta_force_explore_eps=META_FORCE_EXPLORE_EPS,
        meta_entropy_weight=META_ENTROPY_WEIGHT,
        meta_penalty_collapse_floor=META_PENALTY_COLLAPSE_FLOOR,
        meta_num_region_centroids=META_NUM_REGION_CENTROIDS,
        meta_num_semantic_anchors=META_NUM_SEMANTIC_ANCHORS,
        meta_temporal_window=META_TEMPORAL_WINDOW,
        enable_gaussian_embeddings=ENABLE_GAUSSIAN_EMBEDDINGS,
        kl_weight=KL_WEIGHT,
    ).to(device)

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    validate_vocab_size(model, tokenizer)

    use_fused = True if device == 'cuda' else False
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, fused=use_fused)

    print("\nSelect an operation mode:")
    print("  [1] Pre-train on Plain Text (Parquet)")
    print("  [1M] Pre-train on MIXED Datasets (General Language Model)")
    print("  [2] Fine-tune on OpenHermes ChatML (JSON)")
    print("  [3] Chat Mode")
    choice = input("Choice: ").strip()

    if choice == '1':
        files = glob.glob(os.path.join(PARQUET_DIR, '**', '*.parquet'), recursive=True)
        ckpt_path = 'checkpoint_ssm_pretrain.pth'
        start_it = 0
        if os.path.exists(ckpt_path) and input("Resume pre-training checkpoint? (y/n): ").strip().lower() == 'y':
            ckpt = torch.load(ckpt_path, map_location=device)
            load_checkpoint_with_filter(model, ckpt['model_state_dict'])
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("[INFO] Loaded optimizer state from checkpoint")
            except (ValueError, KeyError) as e:
                print(f"[WARNING] Could not load optimizer state (will start fresh): {e}")
            start_it = ckpt.get('iteration', 0)
            # Sync global_training_iteration with checkpoint iteration (for tau annealing & locking)
            if hasattr(model, 'global_training_iteration'):
                model.global_training_iteration.fill_(start_it)
                print(f"[INFO] Synced global_training_iteration to {start_it}")
            print("[INFO] Loaded checkpoint (filtered mismatched keys)")

        run_pretraining(
            model, files, "text", tokenizer, optimizer, device, vocab_size, start_it,
            CHUNK_SIZE, ENABLE_COGNITIVE_FORGETTING, BATCH_SIZE, GRAD_ACCUM_STEPS,
        )

    elif choice == '1M' or choice.lower() == '1m':
        # Mixed dataset configuration - customize paths for your setup
        DATASET_CONFIG = {
            "FineWeb":      {"path": r"I:\Datasets\FineWeb\fineweb-edu_data_CC-MAIN-2024-10", "weight": 0.60},
            "Wikipedia":    {"path": r"I:\Datasets\wikipedia_20231101.en", "weight": 0.20},
            "GitHubCode":   {"path": r"I:\Datasets\github-code_data", "weight": 0.10},
            "OpenWebMath":  {"path": r"I:\Datasets\OpenWebMath", "weight": 0.07},
            "Gutenberg":    {"path": r"I:\Datasets\Gutenberg-BookCorpus-Cleaned-Data-English_data", "weight": 0.03}
        }

        # Optional: Specify custom text columns for different datasets
        TEXT_COLUMN_MAP = {
            "GitHubCode": "text",      # If GitHub dataset uses "code" column
            # "Wikipedia": "content",  # If Wikipedia uses "content" column
            # Add more mappings as needed
        }

        ckpt_path = 'checkpoint_ssm_pretrain_mixed.pth'
        start_it = 0
        if os.path.exists(ckpt_path) and input("Resume mixed pre-training checkpoint? (y/n): ").strip().lower() == 'y':
            ckpt = torch.load(ckpt_path, map_location=device)
            load_checkpoint_with_filter(model, ckpt['model_state_dict'])
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("[INFO] Loaded optimizer state from checkpoint")
            except (ValueError, KeyError) as e:
                print(f"[WARNING] Could not load optimizer state (will start fresh): {e}")
            start_it = ckpt.get('iteration', 0)
            if 'mixer_stats' in ckpt:
                print("[INFO] Previous mixing statistics:")
                for name, stats in ckpt['mixer_stats']['per_dataset'].items():
                    print(f"  {name}: {stats['tokens']:,} tokens ({stats['actual_weight']:.1%})")
            # Sync global_training_iteration with checkpoint iteration (for tau annealing & locking)
            if hasattr(model, 'global_training_iteration'):
                model.global_training_iteration.fill_(start_it)
                print(f"[INFO] Synced global_training_iteration to {start_it}")
            print("[INFO] Loaded checkpoint (filtered mismatched keys)")

        run_pretraining_mixed(
            model, DATASET_CONFIG, tokenizer, optimizer, device, vocab_size, start_it,
            CHUNK_SIZE, ENABLE_COGNITIVE_FORGETTING, BATCH_SIZE, GRAD_ACCUM_STEPS,
            text_column_map=TEXT_COLUMN_MAP,
            mix_temperature=1.0,        # Temperature for mixing (1.0 = use weights as-is)
            stats_interval=5000         # Print mixing stats every 5000 iterations
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
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("[INFO] Loaded optimizer state from checkpoint")
            except (ValueError, KeyError) as e:
                print(f"[WARNING] Could not load optimizer state (will start fresh): {e}")
            start_it = ckpt.get('iteration', 0)
            # Sync global_training_iteration with checkpoint iteration (for tau annealing & locking)
            if hasattr(model, 'global_training_iteration'):
                model.global_training_iteration.fill_(start_it)
                print(f"[INFO] Synced global_training_iteration to {start_it}")
            print("[INFO] Loaded checkpoint (filtered mismatched keys)")

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
            load_checkpoint_with_filter(model, ckpt['model_state_dict'])

        chat_mode(model, tokenizer, device, chunk_size=ckpt.get('chunk_size', CHUNK_SIZE))

    else:
        print(f"Invalid choice: {choice}")


if __name__ == "__main__":
    main()
