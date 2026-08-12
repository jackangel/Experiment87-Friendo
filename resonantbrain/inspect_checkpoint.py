"""
Checkpoint inspection utility for ResonantBrain.

Reads a ``.pth`` checkpoint and reports:
    * optimizer iterations stored
    * total training tokens (exact count if the checkpoint stores
      ``tokens_seen``; otherwise reconstructed from iteration + config)
    * effective batch / chunk / grad-accum configuration
    * checkpointed batch sizes for the per-component optimizer groups
    * saved dataset-mixing statistics (mixed checkpoints only)

Usage
-----
Module CLI::

    python -m resonantbrain.inspect_checkpoint checkpoint_ssm_pretrain_mixed.pth

From a Python shell / other modules::

    from resonantbrain.inspect_checkpoint import inspect_checkpoint, tokens_from_checkpoint
    info = inspect_checkpoint('checkpoint_ssm_pretrain_mixed.pth')
    print(info['summary'])
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import torch


# ---------------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------------
def inspect_checkpoint(path: str, *, verbose: bool = True) -> Dict[str, Any]:
    """Load a ResonantBrain checkpoint and return a structured report.

    Parameters
    ----------
    path : str
        Path to the ``.pth`` checkpoint.
    verbose : bool
        If True, print a human-readable summary.

    Returns
    -------
    dict with keys:
        ``iteration``       (int | None)  – optimizer steps stored in the checkpoint
        ``chunk_size``      (int)
        ``batch_size``      (int | None)
        ``grad_accum_steps``(int | None)
        ``tokens_seen``     (int)         – exact (if stored) else reconstructed
        ``tokens_exact``    (bool)        – whether tokens_seen came from the file
        ``has_optimizer``   (bool)
        ``mixer_stats``     (dict | None) – only for mixed checkpoints
        ``param_count``     (int)         – trainable params in model_state_dict
        ``keys``            (list[str])   – top-level keys in the checkpoint
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint format (not a dict): {type(ckpt)}")

    iteration = ckpt.get("iteration")
    chunk_size = ckpt.get("chunk_size")
    batch_size = ckpt.get("batch_size")
    grad_accum = ckpt.get("grad_accum_steps")
    mixer_stats = ckpt.get("mixer_stats")

    # Prefer an explicit tokens_seen field (written by newer training runs).
    if "tokens_seen" in ckpt and ckpt["tokens_seen"] is not None:
        tokens_seen = int(ckpt["tokens_seen"])
        tokens_exact = True
    else:
        # Reconstruct: tokens = iters * chunk * batch * grad_accum.
        # Fall back to sane defaults if the checkpoint predates these fields
        # (most old checkpoints were saved with batch=1, grad_accum=4, see
        # main.py defaults at the time of those runs).
        if iteration is not None and chunk_size is not None:
            _b = batch_size if batch_size is not None else 1
            _g = grad_accum if grad_accum is not None else 4
            tokens_seen = iteration * chunk_size * _b * _g
            tokens_exact = False
        else:
            tokens_seen = 0
            tokens_exact = False

    # Parameter count from the model state dict (size only — no load).
    state = ckpt.get("model_state_dict", {}) or {}
    param_count = int(sum(t.numel() for t in state.values() if torch.is_tensor(t)))

    report = {
        "path": os.path.abspath(path),
        "iteration": iteration,
        "chunk_size": chunk_size,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "tokens_seen": tokens_seen,
        "tokens_exact": tokens_exact,
        "has_optimizer": "optimizer_state_dict" in ckpt,
        "mixer_stats": mixer_stats,
        "param_count": param_count,
        "keys": sorted(ckpt.keys()),
    }

    if verbose:
        _print_report(report)
    return report


def tokens_from_checkpoint(path: str) -> int:
    """Convenience: return just the reconstructed/exact trained token count."""
    return inspect_checkpoint(path, verbose=False)["tokens_seen"]


# ---------------------------------------------------------------------------
# PRESENTATION
# ---------------------------------------------------------------------------
def _human_count(n: int) -> str:
    """e.g. 3.07B, 12.4M, 850k."""
    a = abs(int(n))
    if a >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if a >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if a >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _print_report(r: Dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print(f"CHECKPOINT REPORT: {r['path']}")
    print("=" * 64)

    def line(label: str, value) -> None:
        print(f"  {label:<22} {value}")

    line("Iteration (steps)", r["iteration"] if r["iteration"] is not None else "(unknown)")
    line("Chunk size", r["chunk_size"] if r["chunk_size"] is not None else "(unknown)")
    line("Batch size", r["batch_size"] if r["batch_size"] is not None else "(unknown)")
    line("Grad accum steps", r["grad_accum_steps"] if r["grad_accum_steps"] is not None else "(unknown)")

    tok_tag = "exact (saved)" if r["tokens_exact"] else "reconstructed (assumed b=1,g=4)"
    line("Tokens seen", f"{r['tokens_seen']:,}  ({_human_count(r['tokens_seen'])})  [{tok_tag}]")

    line("Parameters", f"{r['param_count']:,}  ({_human_count(r['param_count'])})")
    line("Has optimizer state", r["has_optimizer"])
    line("Checkpoint keys", ", ".join(r["keys"]))

    if r["mixer_stats"]:
        print("\n  Dataset mix statistics:")
        per_ds = r["mixer_stats"].get("per_dataset", {})
        if per_ds:
            for name, stats in per_ds.items():
                tokens = stats.get("tokens", 0)
                actual = stats.get("actual_weight", 0.0)
                print(f"    {name:<16} {tokens:>14,} tokens ({actual:.1%})")
        else:
            print("    (no per-dataset breakdown available)")

    print("=" * 64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m resonantbrain.inspect_checkpoint",
        description="Report iteration count, trained tokens, and config from a ResonantBrain checkpoint.",
    )
    p.add_argument("checkpoint", help="Path to the .pth checkpoint.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the human-readable report.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    inspect_checkpoint(args.checkpoint, verbose=not args.quiet)


if __name__ == "__main__":
    main()
