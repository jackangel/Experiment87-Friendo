"""
ResonantBrain SSM — modular package.

Re-exports the public API of the original monolith
``ResonantBrainSSMSalFilterRewindForgetFineTune.py`` so that callers can
do::

    python -m resonantbrain                  # launch the interactive menu
    from resonantbrain import SSMTransformer  # use the model class

Public symbols
--------------
Tokenizer
    TiktokenTokenizer, CHAT_START, CHAT_END

Model building blocks
    precompute_freqs_cis, apply_rotary_emb,
    FFTCausalConv,
    apply_saliency_eviction,
    BayesianSignalAttention,
    LatentGraphReasoning,
    CognitiveForgettingGate,
    SSMAttentionBlock,
    MetaDynamicPhase,

Top-level model
    SSMTransformer,
    get_forgetting_config, get_graph_reasoning_config,

Data / checkpoint utilities
    validate_vocab_size, load_checkpoint_with_filter, apply_sampling_penalties,
    apply_meta_compute_penalty, token_generator_from_parquet,
    stream_chatml_from_json, DatasetMixer, mixed_token_generator_from_datasets,

Generation
    generate_block_recurrent,
    CognitiveMemoryManager,

Training & chat
    run_pretraining, run_pretraining_mixed, run_finetuning, chat_mode,
"""

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
from .tokenizer import TiktokenTokenizer, CHAT_START, CHAT_END

# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
from .rope import (
    precompute_freqs_cis,
    reshape_for_broadcast,
    apply_rotary_emb,
)

# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
from .fft_conv import FFTCausalConv
from .eviction import apply_saliency_eviction
from .forgetting_gate import CognitiveForgettingGate
from .attention import BayesianSignalAttention
from .graph_reasoning import LatentGraphReasoning
from .block import SSMAttentionBlock
from .meta import MetaDynamicPhase

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
from .model import (
    get_forgetting_config,
    get_graph_reasoning_config,
    SSMTransformer,
)

# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
from .data import (
    validate_vocab_size,
    load_checkpoint_with_filter,
    apply_sampling_penalties,
    apply_meta_compute_penalty,
    apply_gaussian_kl_penalty,
    token_generator_from_parquet,
    stream_chatml_from_json,
    DatasetMixer,
    mixed_token_generator_from_datasets,
)

# ---------------------------------------------------------------------------
# Generation & memory
# ---------------------------------------------------------------------------
from .memory import CognitiveMemoryManager
from .generation import generate_block_recurrent

# ---------------------------------------------------------------------------
# Training & chat
# ---------------------------------------------------------------------------
from .training import (
    print_gate_stats,
    build_cosine_scheduler,
    run_pretraining,
    run_pretraining_mixed,
    run_finetuning,
)
from .chat import chat_mode

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
from .main import MODEL_CONFIGS, main

__all__ = [
    # tokenizer
    "TiktokenTokenizer", "CHAT_START", "CHAT_END",
    # rope
    "precompute_freqs_cis", "reshape_for_broadcast", "apply_rotary_emb",
    # layers
    "FFTCausalConv", "apply_saliency_eviction", "CognitiveForgettingGate",
    "BayesianSignalAttention", "LatentGraphReasoning", "SSMAttentionBlock",
    "MetaDynamicPhase",
    # model
    "get_forgetting_config", "get_graph_reasoning_config", "SSMTransformer",
    # data
    "validate_vocab_size", "load_checkpoint_with_filter",
    "apply_sampling_penalties", "apply_meta_compute_penalty",
    "apply_gaussian_kl_penalty",
    "token_generator_from_parquet", "stream_chatml_from_json",
    "DatasetMixer", "mixed_token_generator_from_datasets",
    # generation
    "generate_block_recurrent", "CognitiveMemoryManager",
    # training
    "print_gate_stats", "build_cosine_scheduler",
    "run_pretraining", "run_pretraining_mixed", "run_finetuning",
    # chat
    "chat_mode",
    # entry
    "MODEL_CONFIGS", "main",
]
