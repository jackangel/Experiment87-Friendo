"""
lm-evaluation-harness model wrapper for the ResonantBrain SSMTransformer.

ResonantBrain is NOT a HuggingFace ``PreTrainedModel`` -- it has its own
``forward(x, ...) -> (logits, carry_states, kv)`` API, a tiktoken tokenizer,
and a block-recurrent state design.  This module exposes it through a thin
adapter that implements the three methods `lm-eval` needs:

    * ``loglikelihood(requests)``        -- scored continuations (the workhorse
                                            for MCQ tasks: MMLU, ARC, ...).
    * ``loglikelihood_rolling(requests)``-- corpus perplexity (WikiText).
    * ``generate_until(requests)``       -- free-form generation (GSM8K, ...).

The wrapper is deliberately self-contained: it only depends on
``resonantbrain.model.SSMTransformer`` and ``resonantbrain.tokenizer``.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from ..model import SSMTransformer
from ..tokenizer import TiktokenTokenizer
from ..data import load_checkpoint_with_filter


# ---------------------------------------------------------------------------
# MODEL CONFIG MIRROR (must match resonantbrain/main.py -> MODEL_CONFIGS)
# ---------------------------------------------------------------------------
_MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "tiny":   {"dim": 256,  "num_heads": 4,  "num_layers": 4},
    "small":  {"dim": 512,  "num_heads": 8,  "num_layers": 6},
    "medium": {"dim": 768,  "num_heads": 12, "num_layers": 16},
}


# ---------------------------------------------------------------------------
# BUILD HELPER
# ---------------------------------------------------------------------------
def build_model_from_checkpoint(
    checkpoint_path: str,
    model_size: str = "medium",
    chunk_size: int = 768,
    device: str = "cuda",
    enable_forgetting: bool = True,
    saliency_decay: float = 0.95,
    use_flash_attn: bool = False,
    enable_graph_reasoning: bool = True,
    enable_meta_routing: bool = True,
    meta_max_steps: int = 3,
    meta_gumbel_tau: float = 1.0,
    meta_force_explore_eps: float = 0.30,
    meta_entropy_weight: float = 0.05,
    meta_penalty_collapse_floor: float = 0.85,
    meta_num_region_centroids: int = 8,
    meta_num_semantic_anchors: int = 32,
    meta_temporal_window: int = 64,
    enable_gaussian_embeddings: bool = True,
    kl_weight: float = 0.001,
) -> Tuple[SSMTransformer, TiktokenTokenizer]:
    """Instantiate an ``SSMTransformer`` and load weights from a checkpoint.

    The architecture flags must match whatever the checkpoint was trained
    with.  The defaults below mirror :mod:`resonantbrain.main` so a vanilla
    mixed-pretraining checkpoint (``checkpoint_ssm_pretrain_mixed.pth``)
    loads without changes.

    Returns ``(model, tokenizer)`` ready for wrapping in :class:`ResonantBrainLM`.
    """
    if model_size not in _MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_size={model_size!r}. "
            f"Choose one of {sorted(_MODEL_CONFIGS)}"
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[benchmark] Loading tokenizer (tiktoken 'gpt2')... ")
    tokenizer = TiktokenTokenizer("gpt2")
    vocab_size = tokenizer.vocab_size
    cfg = _MODEL_CONFIGS[model_size]

    print(f"[benchmark] Building SSMTransformer ({model_size}: "
          f"dim={cfg['dim']}, layers={cfg['num_layers']}, heads={cfg['num_heads']})...")
    model = SSMTransformer(
        vocab_size=vocab_size,
        dim=cfg["dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_seq_len=chunk_size,
        enable_forgetting=enable_forgetting,
        saliency_decay=saliency_decay,
        use_flash_attn=use_flash_attn,
        enable_graph_reasoning=enable_graph_reasoning,
        enable_meta_routing=enable_meta_routing,
        meta_max_steps=meta_max_steps,
        meta_gumbel_tau=meta_gumbel_tau,
        meta_force_explore_eps=meta_force_explore_eps,
        meta_entropy_weight=meta_entropy_weight,
        meta_penalty_collapse_floor=meta_penalty_collapse_floor,
        meta_num_region_centroids=meta_num_region_centroids,
        meta_num_semantic_anchors=meta_num_semantic_anchors,
        meta_temporal_window=meta_temporal_window,
        enable_gaussian_embeddings=enable_gaussian_embeddings,
        kl_weight=kl_weight,
    ).to(device)

    print(f"[benchmark] Loading weights from {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    load_checkpoint_with_filter(model, state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[benchmark] Loaded. Parameters: {n_params:,} | device: {device}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# LM WRAPPER
# ---------------------------------------------------------------------------
@register_model("resonantbrain")
class ResonantBrainLM(LM):
    """Chain-of-thought-free causal LM wrapper around ``SSMTransformer``.

    Parameters
    ----------
    model : SSMTransformer
        A *loaded, eval-mode* model.
    tokenizer : TiktokenTokenizer
    max_length : int
        Hard cap on tokens fed to the model in one forward pass.  Should equal
        -- or be <= -- the model's ``max_seq_len`` (``chunk_size`` during
        training).  Sequences are left-trimmed to this length.
    batch_size : int
        Number of (context, continuation) pairs scored per forward pass.
    device : str
    """

    # lm-eval introspects these attributes.
    AUTO_MODEL_CLASS = None  # not a HF transformers model.

    def __init__(
        self,
        model: SSMTransformer,
        tokenizer: TiktokenTokenizer,
        max_length: int = 768,
        batch_size: int = 8,
        device: str = "cuda",
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self._batch_size = int(batch_size)
        self._device = device

        # tiktoken has no native bos/pad; reuse eot as a neutral pad id for
        # batching bookkeeping (scores ignore pad positions -- see _batch_score).
        self.eot_token_id = self.tokenizer.tokenizer.eot_token  # gpt2: 50256
        self.pad_token_id = self.eot_token_id

    # -- lm-eval mandated properties ---------------------------------------
    @property
    def eot_token_id(self) -> int:  # noqa: F811  (lm-eval expects this name)
        return self._eot_id

    @eot_token_id.setter
    def eot_token_id(self, value: int) -> None:
        self._eot_id = int(value)

    @property
    def max_length(self) -> int:
        return self._max_length

    @max_length.setter
    def max_length(self, value: int) -> None:
        self._max_length = int(value)

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device

    # -- tokenization helpers ----------------------------------------------
    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    @staticmethod
    def _pair(context: str, continuation: str, encode_fn) -> Tuple[List[int], List[int]]:
        """Split an lm-eval ``(context, continuation)`` request into token lists.

        lm-eval guarantees word-boundary handling: the continuation string
        already starts with the leading space if the option is a new word.
        To honor tiktoken's BPE merges *exactly*, we tokenize the full text
        and the context separately, then take the suffix as the continuation.

        Returns ``(context_ids, continuation_ids)``.
        """
        ctx_ids = encode_fn(context) if context else []
        full_ids = encode_fn(context + continuation)
        cont_ids = full_ids[len(ctx_ids):]
        if not cont_ids:
            # Defensive fallback if a boundary merge shrank things oddly.
            cont_ids = encode_fn(continuation)
        return ctx_ids, cont_ids

    # =====================================================================
    # loglikelihood -- the core scoring path used by MCQ benchmarks
    # =====================================================================
    def loglikelihood(self, requests):
        """Score ``(context, continuation)`` pairs.

        Returns a list of tuples ``(loglike_sum, is_greedy)`` -- one per
        request -- matching lm-eval's contract.  ``is_greedy`` records
        whether the continuation equals the model's argmax (used for
        strict-match MCQ scoring in some tasks).
        """
        results: List[Tuple[float, bool]] = []
        for req in requests:
            context, continuation = req.arguments
            res = self._score_one(context, continuation)
            results.append(res)
        return results

    @torch.inference_mode()
    def _score_one(self, context: str, continuation: str) -> Tuple[float, bool]:
        ctx_ids, cont_ids = self._pair(context, continuation, self._encode)
        if len(cont_ids) == 0:
            # Nothing to score; logprob 0, vacuously greedy.
            return 0.0, True

        full = ctx_ids + cont_ids
        # Left-trim to fit max_length, keeping the continuation intact.
        if len(full) > self.max_length:
            overflow = len(full) - self.max_length
            # Prefer trimming context; only trim the continuation when
            # the continuation alone is longer than the window.
            clip_from_ctx = min(overflow, len(ctx_ids))
            ctx_ids = ctx_ids[clip_from_ctx:]
            full = ctx_ids + cont_ids
            if len(full) > self.max_length:
                # Pathological case: continuation alone exceeds window.
                # Truncate BOTH context and the FRONT of the continuation so
                # the score reflects the last `max_length` tokens.  We then
                # keep cont_ids pointing at the tail we actually scored.
                keep = self.max_length
                full = full[-keep:]
                # Recompute the (trimmed_ctx, trimmed_cont) split consistently.
                n_cont_trim = min(len(cont_ids), keep)
                cont_ids = full[-n_cont_trim:]
                ctx_ids = full[:-n_cont_trim]

        # We need logits predicting the continuation tokens.  Feed the
        # full sequence in one shot (no KV reuse; correctness > speed).
        x = torch.tensor([full], dtype=torch.long, device=self._device)
        logits, _, _ = self.model(
            x=x, carry_states=None, is_training=False,
            use_cache=False, abs_pos_offset=0,
        )  # [1, T, V]
        logits = logits.float()

        # Shift to next-token convention: logits[:, :-1] predicts ids[:, 1:].
        n_ctx = len(ctx_ids)
        n_cont = len(cont_ids)
        # Positions of the continuation tokens within `full` (0-indexed).
        # The logit predicting cont token j lives at position (n_ctx + j - 1).
        pred_logits = logits[0, n_ctx - 1: n_ctx - 1 + n_cont, :]  # [n_cont, V]
        target = torch.tensor(cont_ids, dtype=torch.long, device=self._device)

        log_probs = F.log_softmax(pred_logits, dim=-1)
        picked = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [n_cont]
        ll_sum = float(picked.sum().item())

        argmax_ids = pred_logits.argmax(dim=-1)
        is_greedy = bool(torch.equal(argmax_ids, target))
        return ll_sum, is_greedy

    # =====================================================================
    # loglikelihood_rolling -- corpus perplexity / WikiText-style
    # =====================================================================
    def loglikelihood_rolling(self, requests):
        results: List[float] = []
        for req in requests:
            (text,) = req.arguments if isinstance(req.arguments, tuple) else (req.arguments,)
            results.append(self._score_rolling(text))
        return results

    @torch.inference_mode()
    def _score_rolling(self, text: str) -> float:
        ids = self._encode(text)
        if len(ids) < 2:
            return 0.0
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]
        x = torch.tensor([ids], dtype=torch.long, device=self._device)
        logits, _, _ = self.model(
            x=x, carry_states=None, is_training=False,
            use_cache=False, abs_pos_offset=0,
        )
        logits = logits.float()
        # Standard next-token CE over positions [0 .. T-2] -> ids[1:].
        log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
        targets = torch.tensor(ids[1:], dtype=torch.long, device=self._device)
        picked = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return float(picked.sum().item())

    # =====================================================================
    # generate_until -- free-form generation (GSM8K, HumanEval, etc.)
    # =====================================================================
    def generate_until(self, requests):
        from ..generation import generate_block_recurrent

        results = []
        for req in requests:
            context, gen_kwargs = req.arguments
            until = gen_kwargs.get("until", [])
            max_gen = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            temperature = float(gen_kwargs.get("temperature", 0.8))
            top_p = float(gen_kwargs.get("top_p", 0.9))
            top_k = int(gen_kwargs.get("top_k", 50))
            rep_pen = float(gen_kwargs.get("repetition_penalty", 1.3))

            input_ids = self._encode(context)
            stop = until[0] if until else None
            gen_ids = generate_block_recurrent(
                self.model, input_ids, self.tokenizer, self._device,
                max_new_tokens=max_gen, chunk_size=self.max_length,
                temperature=max(temperature, 1e-3), repetition_penalty=rep_pen,
                top_k=top_k, top_p=top_p, stop_sequence=stop,
            )
            text = self.tokenizer.decode(gen_ids[len(input_ids):])
            # Enforce stop strings (lm-eval relies on text being truncated).
            for stop_str in until:
                if stop_str and stop_str in text:
                    text = text.split(stop_str, 1)[0]
            results.append(text)
        return results
