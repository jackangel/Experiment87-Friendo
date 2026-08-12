"""
Extract YOUR model's benchmark scores from lm-eval's ``results.json``.

lm-eval dumps a >50MB JSON because it includes every sample's full log under
``"samples"``.  This module streams just the lightweight ``results`` / 
``summary`` blocks we need, with a pure-Python parser that does NOT load the
whole file into memory.

Public API
----------
- :func:`extract_scores`  – load results.json, return ``{task: {metric: value}}``.
- :func:`choose_primary_metric` – pick acc_norm -> acc -> perplexity for a task.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple


# Metric preference per charting: higher-is-better accuracy first, then
# perplexity (which we'll invert for display).
_HIGH_BETTER = ("acc_norm", "acc")
_LOW_BETTER = ("word_perplexity", "byte_perplexity", "perplexity", "bits_per_byte")


# Metric families whose values live on a fixed [0, 1] scale; used both to
# pick the headline metric and to sanity-clamp the plot.
_ACCURACY_FAMILY = ("acc", "f1", "exact_match")


def _metric_prefix(key: str) -> str:
    """Normalize an lm-eval metric key to its plain name.

    lm-eval-harness emits composite keys like ``"acc,none"``,
    ``"acc_norm,binary"``, ``"word_perplexity,none"`` where the part after
    the comma is a filter/aggregation label, NOT part of the metric name.
    Stripping it lets us match against canonical names (``acc`` etc.).
    Without this, ``choose_primary_metric`` never finds ``acc``/``acc_norm``
    and falls back to the first numeric field — which for perplexity tasks
    yields huge raw values plotted as if they were accuracy.
    """
    return key.split(",", 1)[0].strip()


def extract_scores(path: str) -> Dict[str, Dict[str, float]]:
    """Read lm-eval output JSON and return per-task metric dict.

    Returns
    -------
    dict
        ``{task_name: {metric_name: value, ...}, ...}`` e.g.::

            {"hellaswag": {"acc": 0.273, "acc_norm": 0.310,
                           "acc_stderr": 0.0044, "acc_norm_stderr": 0.0046}}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")

    # Strategy: try ijson streaming first (fast, low-memory), but VALIDATE
    # the result.  Some ijson backends silently drop metric keys that contain
    # a comma (lm-eval writes keys like "acc,none", "acc_norm,none") while
    # keeping plain keys ("sample_len").  That dropped the accuracy values
    # entirely, leaving only counts like sample_len (~1667) — which the chart
    # then mis-inverted via 1/value into a bogus ~0.1% "accuracy".
    # If validation finds no accuracy-family metric anywhere, we fall back to
    # json.load (loads the whole file; slow but CORRECT).
    ijson_ok = False
    try:
        result = _extract_with_ijson(path)
        ijson_ok = True
    except Exception:
        # ijson is purely an optimization (it streams past the 50MB 'samples'
        # block without materialising it).  ANY failure -- ImportError, parse
        # error, a missing 'results' key, etc. -- must fall through to the
        # guaranteed-correct json.load fallback below.  We deliberately catch
        # broad Exception here because ijson correctness is best-effort only.
        result = {}

    if result and _contains_accuracy_metric(result):
        return result

    # ijson produced nothing usable (or produced counts-only output, or ijson
    # is absent).  Fall back to a full json.load.
    fallback = _extract_with_json(path)
    if ijson_ok and _contains_accuracy_metric(fallback) and not _contains_accuracy_metric(result):
        import sys
        print(f"[extract_scores] WARNING: ijson dropped comma-suffixed metric "
              f"keys (e.g. 'acc,none'); fell back to json.load. "
              f"Chart values are now correct.", file=sys.stderr)
    return fallback


def _contains_accuracy_metric(scores: Dict[str, Dict[str, float]]) -> bool:
    """True if at least one task has an acc / acc_norm / f1 / exact_match key."""
    _acc = ("acc", "acc_norm", "f1", "exact_match")
    for metrics in scores.values():
        if any(k in metrics for k in _acc):
            return True
    return False


def _extract_with_json(path: str) -> Dict[str, Dict[str, float]]:
    """Fallback: load whole JSON (works, but may use lots of RAM)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _results_block(data)


def _extract_with_ijson(path: str) -> Dict[str, Dict[str, float]]:
    """Fetch the top-level ``results`` object without parsing ``samples``.

    Uses ``next(ijson.items(f, "results"))`` to pull the *entire* ``results``
    mapping as ONE complete dict.  This is more robust than ``kvitems`` or
    an ``items('results.*')`` wildcard loop -- some ijson backends mishandle
    lm-eval's comma-bearing keys (``"acc,none"``) when reassembling
    incremental pieces, and ``results.*`` is not a valid prefix token at all.
    Building the whole object once sidesteps both issues.

    ijson still streams past the 50MB ``samples`` block: we close the file the
    moment ``results`` is yielded, so any trailing content is never parsed.
    ``extract_scores`` validates the output and transparently falls back to
    ``json.load`` if ijson produced anything unusable.
    """
    import ijson

    def _pull(prefix: str) -> dict:
        with open(path, "rb") as f:
            obj = next(ijson.items(f, prefix), None)
        return obj if isinstance(obj, dict) else {}

    # Try lm-eval's native 'results' first; fall back to our runner's summary.
    src = _pull("results") or _pull("summary")
    if not src:
        raise ValueError(
            f"No 'results' or 'summary' keys found in {path}. "
            "Confirm the file is lm-eval output."
        )

    out: Dict[str, Dict[str, float]] = {}
    for task_name, metrics in src.items():
        if not isinstance(metrics, dict):
            continue
        entry: Dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                # Strip lm-eval's ",none"/",binary" filter suffix.
                entry[_metric_prefix(k)] = float(v)
        if entry:
            out[task_name] = entry

    if not out:
        raise ValueError(
            f"No numeric metrics found under 'results'/'summary' in {path}."
        )
    return out


def _results_block(data: dict) -> Dict[str, Dict[str, float]]:
    """Pull per-task metrics from a fully-loaded lm-eval JSON dict."""
    out: Dict[str, Dict[str, float]] = {}
    # Prefer lm-eval's native "results"; fall back to our runner's "summary".
    block = data.get("results") or data.get("summary") or {}
    for task, metrics in block.items():
        entry = {_metric_prefix(k): float(v)
                 for k, v in metrics.items() if isinstance(v, (int, float))}
        if entry:
            out[task] = entry
    if not out:
        raise ValueError(
            f"No 'results' or 'summary' keys found in {data.keys()}"
        )
    return out


def choose_primary_metric(task_metrics: Dict[str, float]) -> Tuple[str, float, bool]:
    """Pick the headline metric for a task.

    Preference order: ``acc_norm`` -> ``acc`` -> any perplexity -> any other.

    Returns
    -------
    (metric_name, value, higher_is_better)
    """
    # Keys are already normalized (",none" suffix stripped by _metric_prefix).
    for m in _HIGH_BETTER:
        if m in task_metrics:
            return m, float(task_metrics[m]), True
    for m in _LOW_BETTER:
        if m in task_metrics:
            return m, float(task_metrics[m]), False
    # Fall back to the first numeric field. Detect accuracy-family metrics so
    # we don't mislabel a raw perplexity as higher-is-better accuracy.
    name = next(iter(task_metrics))
    is_acc = name.startswith(_ACCURACY_FAMILY)
    return name, float(task_metrics[name]), True if is_acc else False


def stderr_for(task_metrics: Dict[str, float], metric_name: str) -> Optional[float]:
    """Return the stderr for the given metric if present (acc -> acc_stderr)."""
    candidates = (f"{metric_name}_stderr", f"{metric_name}_se", "stderr")
    for c in candidates:
        if c in task_metrics:
            return float(task_metrics[c])
    return None
