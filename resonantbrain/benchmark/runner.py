"""
High-level benchmark runner for ResonantBrain models.

Thin convenience layer over `lm-evaluation-harness`_ that wires up a trained
:mod:`resonantbrain` checkpoint and a standard benchmark suite, then prints &
optionally saves the results.

.. _lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness

Quick start
-----------
::

    # from the repo root (GoodForTraining/)
    python -m resonantbrain.benchmark.runner \
        --checkpoint checkpoint_ssm_pretrain_mixed.pth \
        --model-size medium --chunk-size 768 \
        --tasks hellaswag,arc_easy,arc_challenge,winogrande \
        --output results.json

Recommended "first run" tasks for a *base* (non-instruction-tuned) model:

    hellaswag arc_easy arc_challenge winogrande piqa

    # Optional, cheap subset of MMLU:
    mmlu_abstract_algebra,mmlu_high_school_biology

    # Perplexity corpus:
    wikitext

For the full >800-task catalog, see:
    https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import torch

# Importing this module registers the "resonantbrain" model with lm-eval.
from . import wrapper as _wrapper  # noqa: F401  (registration side-effect)
from .wrapper import build_model_from_checkpoint, ResonantBrainLM


# Default tasks suitable for a base (pretrained-only) causal LM.
DEFAULT_TASKS: List[str] = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "piqa",
]


def run_benchmark(
    checkpoint_path: str,
    tasks: List[str],
    model_size: str = "medium",
    chunk_size: int = 768,
    batch_size: int = 8,
    device: str = "cuda",
    num_fewshot: Optional[int] = None,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    **model_kwargs,
) -> dict:
    """Build the model, register it with lm-eval, and run the harness.

    Parameters
    ----------
    checkpoint_path : str
        Path to a ``.pth`` produced by training.  Must contain
        ``model_state_dict``.
    tasks : list[str]
        lm-eval task IDs (e.g. ``["hellaswag", "arc_easy"]``).
    model_size, chunk_size, model_kwargs:
        Forwarded to :func:`build_model_from_checkpoint`.  These must match
        the architecture the checkpoint was trained with.
    num_fewshot : int, optional
        Few-shot examples per task (``None`` = task default, usually 0).
    output_path : str, optional
        If given, write the full lm-eval JSON results here.
    limit : int, optional
        Cap docs-per-task -- useful for a quick smoke test (e.g. ``limit=100``).

    Returns
    -------
    dict
        The serializable results from ``lm_eval.simple_evaluate`` plus a
        convenience ``"summary"`` block with per-task accuracy.
    """
    # 1) Build & load the model once.
    model, tokenizer = build_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_size=model_size,
        chunk_size=chunk_size,
        device=device,
        **model_kwargs,
    )
    lm = ResonantBrainLM(model, tokenizer, max_length=chunk_size,
                         batch_size=batch_size, device=device)

    # 2) Hand off to lm-eval.
    import lm_eval
    from lm_eval import simple_evaluate

    print(f"\n[benchmark] Running tasks: {tasks}")
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        bootstrap_iters=1000,   # for stderr on accuracy
    )

    # 3) Extract a friendly summary.
    summary = _Summarize(results)
    _print_summary(summary)

    # 4) Persist full JSON if requested.
    out = results if isinstance(results, dict) else {"results": results}
    out["summary"] = summary
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n[benchmark] Full results written to: {output_path}")

    return out


# ---------------------------------------------------------------------------
# Result presentation helpers
# ---------------------------------------------------------------------------
def _Summarize(results: dict) -> dict:
    """Flatten lm-eval's per-task metrics into {task: {metric: value}}."""
    summary = {}
    for task, metrics in results.get("results", {}).items():
        entry = {}
        for k, v in metrics.items():
            # keep the canonical primary metric + stderr fields.
            if isinstance(v, (int, float)):
                entry[k] = float(v)
        summary[task] = entry
    return summary


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 64)
    print("BENCHMARK RESULTS")
    print("=" * 64)
    for task, metrics in summary.items():
        print(f"\n[{task}]")
        for k, v in metrics.items():
            # Pretty-print key metrics; pass everything else through.
            print(f"    {k:<40} {v:.4f}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m resonantbrain.benchmark.runner",
        description="Run lm-evaluation-harness tasks against a ResonantBrain checkpoint.",
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to the .pth checkpoint to evaluate.")
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                   help="Comma-separated lm-eval task IDs.")
    p.add_argument("--model-size", default="medium",
                   choices=["tiny", "small", "medium"],
                   help="Architecture (must match training).")
    p.add_argument("--chunk-size", type=int, default=768,
                   help="Model max_seq_len (must match training).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="(Context, continuation) pairs per forward pass.")
    p.add_argument("--num-fewshot", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap docs/task for a quick smoke test.")
    p.add_argument("--output", default=None,
                   help="Path to write full results JSON.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise SystemExit("No tasks specified after --tasks.")

    run_benchmark(
        checkpoint_path=args.checkpoint,
        tasks=tasks,
        model_size=args.model_size,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        device=args.device,
        num_fewshot=args.num_fewshot,
        output_path=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
