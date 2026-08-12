"""
Build a grouped bar chart comparing your ResonantBrain benchmark scores
against the 5 closest reference models (by parameter count).

Usage
-----
::

    python -m resonantbrain.benchmark.compare \\
        --results results.json \\
        --your-params 350M \\
        --output comparison.png

Or via the API::

    from resonantbrain.benchmark.compare import compare_to_closest
    compare_to_closest(results_path="results.json", your_params=350e6,
                       output_path="comparison.png")
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

from .extract_scores import extract_scores, choose_primary_metric, stderr_for
from .reference_models import (
    REFERENCE_MODELS, select_closest_models, human_params, model_family_hue
)


# ---------------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------------
def compare_to_closest(
    results_path: str,
    your_params: int,
    your_name: str = "ResonantBrain",
    n_ref: int = 5,
    output_path: str = "benchmark_comparison.png",
    title: Optional[str] = None,
    chart_kind: str = "grouped",
) -> str:
    """Extract your scores, pick the N closest models, plot, save PNG.

    Parameters
    ----------
    results_path : str
        Path to lm-eval ``results.json`` (any size; streamed).
    your_params : int
        YOUR model's total parameter count (used only to pick the closest
        reference models — yours is always included as a bar).
    your_name : str
        Legend label for your model.
    n_ref : int
        Number of closest reference models to include.
    output_path : str
        Where to save the PNG.
    chart_kind : str
        ``"grouped"`` (default; side-by-side bars per task) or
        ``"faceted"`` (one subplot per task — nicer when there are many tasks).
    """
    # 1) Pull YOUR real scores from the lm-eval output.
    your_results = extract_scores(results_path)
    your_scores: Dict[str, Tuple[str, float, bool, Optional[float]]] = {}
    for task, metrics in your_results.items():
        # Skip aggregate "groups" that have no metrics of their own.
        if not metrics:
            continue
        metric, value, higher_better = choose_primary_metric(metrics)
        se = stderr_for(metrics, metric)
        your_scores[task] = (metric, value, higher_better, se)

    if not your_scores:
        raise ValueError(
            "No per-task metrics found in results.json. "
            "Run the benchmark first."
        )

    # 2) Pick the N closest reference models by params.
    refs = select_closest_models(your_params, n=n_ref)

    # 3) Plot.
    out = _plot(
        your_name=your_name,
        your_params=your_params,
        your_scores=your_scores,
        refs=refs,
        output_path=output_path,
        title=title,
        chart_kind=chart_kind,
    )
    return out


# ---------------------------------------------------------------------------
# PLOTTING (matplotlib, lazy-imported so the module imports without it)
# ---------------------------------------------------------------------------
def _plot(your_name, your_params, your_scores, refs, output_path, title, chart_kind) -> str:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    import numpy as np

    # Tasks your model was evaluated on.
    tasks = list(your_scores.keys())
    n_tasks = len(tasks)
    n_models = 1 + len(refs)  # you + refs

    # Column 0 = your model; 1..N = refs.
    model_labels = [f"{your_name} ({human_params(your_params)})"]
    model_labels += [f"{r.name}\n({human_params(r.params)})" for r in refs]

    family_colors = model_family_hue([REFERENCE_MODELS[0].family] + [r.family for r in refs])
    # Your model gets a distinct highlight color.
    YOUR_COLOR = "#D55E00"

    # ------------------------------------------------------------------
    # HEADLINE: bar chart where each bar = a model, height = mean accuracy
    # across the shared tasks (the simplest, most-readable comparison).
    # ------------------------------------------------------------------
    # Compute each model's mean accuracy (over tasks both models have).
    # Perplexity-style metrics are inverted via _task_to_higher_better so the
    # mean stays meaningful ("higher is better" everywhere).
    means = []
    errs = []

    # Your model (column 0): invert perplexities, then average.
    your_vals = [_task_to_higher_better(*your_scores[t][:3]) for t in tasks]
    means.append(float(np.mean(your_vals)))
    errs.append(0.0)

    # Reference models (columns 1..N): they only store acc/acc_norm, so no
    # inversion needed; average over the tasks they share with yours.
    for r in refs:
        vals = [r.acc_norm[t] for t in tasks if t in r.acc_norm]
        means.append(float(np.mean(vals)) if vals else float("nan"))
        errs.append(0.0)

    means = np.array(means, dtype=float)
    errs = np.array(errs, dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.4), 5.5))
    x = np.arange(n_models)
    bar_colors = [YOUR_COLOR] + [family_colors[r.family] for r in refs]
    # Highlight your bar with a hatch so it's distinguishable even in B/W.
    bars = ax.bar(x, means, yerr=errs, color=bar_colors,
                  edgecolor="black", linewidth=0.8, width=0.6, capsize=4)
    bars[0].set_hatch("//")
    bars[0].set_edgecolor("black")

    for i, (b, v) in enumerate(zip(bars, means)):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if i == 0 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel("Mean accuracy across shared tasks\n(higher is better)")
    ax.set_ylim(0, max(1.0, np.nanmax(means) * 1.15))
    ax.set_title(title or
                 f"{your_name} vs {len(refs)} closest models (by parameter count)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Caption with the shared task list + source disclaimer.
    shared_str = ", ".join(tasks)
    fig.text(0.5, -0.02,
             f"Tasks scored: {shared_str}\n"
             "Reference scores are publicly reported values (may differ slightly across "
             "lm-eval versions / few-shot settings).",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # SECONDARY: a per-task grouped chart (one cluster per task).
    # ------------------------------------------------------------------
    secondary_path = _swap_suffix(output_path, "_per_task")
    _plot_per_task(tasks, your_scores, your_name, your_params, refs,
                   bar_colors, secondary_path, title)

    # ------------------------------------------------------------------
    # Plain text summary alongside the chart.
    # ------------------------------------------------------------------
    txt_path = _swap_suffix(output_path, ".txt")
    _write_summary_txt(tasks, your_scores, your_name, your_params, refs, means, txt_path)

    print(f"\n[compare] Saved:")
    print(f"  - {output_path}        (headline mean-accuracy bars)")
    print(f"  - {secondary_path}     (per-task grouped bars)")
    print(f"  - {txt_path}            (text summary)")
    return output_path


def _task_to_higher_better(metric: str, value: float, higher_better: bool) -> float:
    """Convert (metric,value,dir) to a 'higher is better' scalar in [0,1].

    Hard-clamps accuracy-family metrics to [0, 1] so a mislabelled raw
    perplexity can never again blow the chart up to thousands of percent.
    """
    name = metric.split(",", 1)[0].strip().lower()
    is_accuracy_family = name.startswith(("acc", "f1", "exact_match"))
    if higher_better:
        v = float(value)
        if is_accuracy_family:
            # Physical sanity: accuracy/F1/exact-match cannot exceed 1.0.
            if v > 1.0:
                import sys
                print(f"[compare] WARNING: {metric!r}={v} > 1.0 clamped to "
                      f"1.0 (metric does not look like accuracy)",
                      file=sys.stderr)
            return min(max(v, 0.0), 1.0)
        return v
    # Perplexity-style: invert so higher = better (cap to avoid infinity).
    return float(1.0 / max(value, 1e-6))


def _plot_per_task(tasks, your_scores, your_name, your_params, refs,
                   bar_colors, output_path, title) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_tasks = len(tasks)
    n_models = 1 + len(refs)
    width = 0.8 / n_models
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 2.2), 6))

    # Your model.
    your_vals = [_task_to_higher_better(*your_scores[t][:3]) for t in tasks]
    ax.bar(x - (n_models - 1) * width / 2, your_vals, width,
           label=f"{your_name} ({human_params(your_params)})",
           color="#D55E00", edgecolor="black", linewidth=0.8, hatch="//")

    # Reference models.
    for j, r in enumerate(refs):
        vals = []
        for t in tasks:
            vals.append(r.acc_norm.get(t, float("nan")))
        offset = x - (n_models - 1) * width / 2 + (j + 1) * width
        ax.bar(offset, vals, width, label=f"{r.name} ({human_params(r.params)})",
               color=bar_colors[j + 1], edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (higher is better)")
    ax.set_ylim(0, 1.0)
    ax.set_title(title or f"Per-task accuracy: {your_name} vs closest models")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_summary_txt(tasks, your_scores, your_name, your_params, refs,
                       means, txt_path) -> None:
    lines = []
    lines.append(f"{your_name} ({human_params(your_params)}) vs closest reference models")
    lines.append("=" * 72)
    lines.append("")
    header = f"{'Model':<28}{'Params':>12}{'Mean acc':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    lines.append(f"{your_name:<28}{human_params(your_params):>12}{means[0]*100:>11.2f}%")
    for r, m in zip(refs, means[1:]):
        m_str = f"{m*100:.2f}%" if not (m != m) else "n/a"  # m != m is NaN test
        lines.append(f"{r.name:<28}{human_params(r.params):>12}{m_str:>12}")
    lines.append("")
    lines.append("Per-task scores:")
    lines.append("")
    sub = f"{'Task':<20}{'Your':>10}" + "".join(f"{r.name[:10]:>12}" for r in refs)
    lines.append(sub)
    lines.append("-" * len(sub))
    for t in tasks:
        yours = _task_to_higher_better(*your_scores[t][:3])
        row = f"{t:<20}{yours*100:>9.2f}%"
        for r in refs:
            v = r.acc_norm.get(t, float("nan"))
            if v == v:  # not NaN
                row += f"{v*100:>11.2f}%"
            else:
                row += f"{'n/a':>12}"
        lines.append(row)
    lines.append("")
    lines.append("Reference scores are publicly reported values.")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _swap_suffix(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    if suffix.endswith(".txt") or "." in suffix:
        return f"{base}{suffix}"
    return f"{base}{suffix}{ext if ext else '.png'}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_params(s: str) -> int:
    """Parse '350M', '1.1B', '70000000' to an int."""
    s = s.strip()
    if not s:
        raise ValueError("Empty parameter string.")
    mult = 1
    if s[-1].lower() == "k":
        mult, s = 1_000, s[:-1]
    elif s[-1].lower() == "m":
        mult, s = 1_000_000, s[:-1]
    elif s[-1].lower() == "b":
        mult, s = 1_000_000_000, s[:-1]
    return int(float(s) * mult)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m resonantbrain.benchmark.compare",
        description="Bar chart comparing your ResonantBrain benchmark scores "
                    "against the N closest reference models by parameter count.",
    )
    p.add_argument("--results", default="results.json",
                   help="Path to lm-eval results.json (any size; streamed).")
    p.add_argument("--your-params", required=True,
                   help="Your model's total parameter count, e.g. '350M' or '7B'.")
    p.add_argument("--your-name", default="ResonantBrain",
                   help="Legend label for your model.")
    p.add_argument("--n-ref", type=int, default=5,
                   help="Number of closest reference models to plot.")
    p.add_argument("--output", default="benchmark_comparison.png")
    p.add_argument("--title", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    compare_to_closest(
        results_path=args.results,
        your_params=_parse_params(args.your_params),
        your_name=args.your_name,
        n_ref=args.n_ref,
        output_path=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
