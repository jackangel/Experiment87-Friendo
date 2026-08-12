"""
Reference benchmark scores for selecting "closest comparable" models.

All numbers are **publicly reported values** from each model's paper or the
EleutherAI / HELM leaderboards.  They are approximate and depend on the
exact harness version, few-shot setting, and normalization (acc vs
acc_norm).  Update ``REFERENCE_MODELS`` with values from the same lm-eval
version you used for your run if you need exact head-to-head comparisons.

Parameters are TOTAL parameter counts (not "non-embedding").  The closest-N
picker in :func:`select_closest_models` ranks by absolute distance in
parameter count to your model, so the comparison is apples-to-apples by
capacity.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional


class RefModel(NamedTuple):
    name: str
    params: int               # total parameter count
    family: str               # short label: e.g. "GPT-2", "Pythia"
    acc_norm: Dict[str, float]  # task -> acc_norm (or acc where only acc exists)


# Reported accuracy/acc_norm scores.  Sources: original papers + HELM +
# EleutherAI published runs.  When only `acc` was reported (rare), the value
# sits in acc_norm too so the chart still plots a bar.
REFERENCE_MODELS: List[RefModel] = [
    RefModel(
        name="TinyLlama-1.1B",
        params=1_100_000_000,
        family="Llama2",
        acc_norm={
            "hellaswag":     0.4180,
            "arc_easy":      0.5330,
            "arc_challenge": 0.3300,
            "winogrande":    0.5300,
            "piqa":          0.7160,
        },
    ),
    RefModel(
        name="Pythia-1B",
        params=1_005_000_000,
        family="Pythia",
        acc_norm={
            "hellaswag":     0.4259,
            "arc_easy":      0.5156,
            "arc_challenge": 0.3029,
            "winogrande":    0.5222,
            "piqa":          0.7008,
        },
    ),
    RefModel(
        name="OPT-1.3B",
        params=1_300_000_000,
        family="OPT",
        acc_norm={
            "hellaswag":     0.4290,
            "arc_easy":      0.5000,
            "arc_challenge": 0.2900,
            "winogrande":    0.5300,
            "piqa":          0.7130,
        },
    ),
    RefModel(
        name="Pythia-410M",
        params=405_000_000,
        family="Pythia",
        acc_norm={
            "hellaswag":     0.3671,
            "arc_easy":      0.4742,
            "arc_challenge": 0.2704,
            "winogrande":    0.5166,
            "piqa":          0.6730,
        },
    ),
    RefModel(
        name="Pythia-160M",
        params=162_000_000,
        family="Pythia",
        acc_norm={
            "hellaswag":     0.2983,
            "arc_easy":      0.3984,
            "arc_challenge": 0.2200,
            "winogrande":    0.5035,
            "piqa":          0.6061,
        },
    ),
    RefModel(
        name="GPT-Neo-125M",
        params=125_000_000,
        family="GPT-Neo",
        acc_norm={
            "hellaswag":     0.2650,
            "arc_easy":      0.3620,
            "arc_challenge": 0.2050,
            "winogrande":    0.5012,
            "piqa":          0.5910,
        },
    ),
    RefModel(
        name="GPT-2 (117M)",
        params=124_000_000,
        family="GPT-2",
        acc_norm={
            "hellaswag":     0.2860,
            "arc_easy":      0.3880,
            "arc_challenge": 0.2150,
            "winogrande":    0.5050,
            "piqa":          0.6250,
        },
    ),
    RefModel(
        name="Pythia-70M",
        params=70_000_000,
        family="Pythia",
        acc_norm={
            "hellaswag":     0.2700,
            "arc_easy":      0.3500,
            "arc_challenge": 0.2080,
            "winogrande":    0.5010,
            "piqa":          0.5610,
        },
    ),
    RefModel(
        name="Llama-2-7B",
        params=7_000_000_000,
        family="Llama2",
        acc_norm={
            "hellaswag":     0.7290,
            "arc_easy":      0.7680,
            "arc_challenge": 0.4670,
            "winogrande":    0.6880,
            "piqa":          0.7880,
        },
    ),
    RefModel(
        name="OPT-350M",
        params=350_000_000,
        family="OPT",
        acc_norm={
            "hellaswag":     0.3350,
            "arc_easy":      0.4470,
            "arc_challenge": 0.2480,
            "winogrande":    0.5100,
            "piqa":          0.6460,
        },
    ),
]


def select_closest_models(
    your_params: int,
    n: int = 5,
    exclude: Optional[List[str]] = None,
) -> List[RefModel]:
    """Return the N reference models closest in parameter count to yours.

    Parameters
    ----------
    your_params : int
        Total parameter count of your model.
    n : int
        Number of reference models to return.
    exclude : list[str], optional
        Reference model names to skip (e.g. if you only want trained-from-
        scratch baselines).
    """
    exclude = set(exclude or [])
    ranked = sorted(REFERENCE_MODELS, key=lambda m: (abs(m.params - your_params), m.name))
    ranked = [m for m in ranked if m.name not in exclude]
    return ranked[:n]


def model_family_hue(families) -> Dict[str, str]:
    """Stable color mapping per model family for consistent chart styling.

    Parameters
    ----------
    families : iterable of str
        Family labels to map (e.g. ``["Pythia", "GPT-Neo", "OPT"]``).

    Returns
    -------
    dict
        One-to-one mapping ``{family_label: color}`` covering every distinct
        label in ``families``.  The full label is the key (no splitting), so
        families like "GPT-Neo" and "GPT-2" stay distinct.
    """
    unique = sorted(set(families))
    # Distinct, colorblind-friendly palette.
    palette = [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7",
        "#56B4E9", "#D55E00", "#F0E442", "#999999",
    ]
    return {fam: palette[i % len(palette)] for i, fam in enumerate(unique)}


def human_params(p: int) -> str:
    """e.g. 7_000_000_000 -> '7.0B', 410_000_000 -> '410.0M'."""
    if p >= 1_000_000_000:
        return f"{p / 1_000_000_000:.1f}B"
    if p >= 1_000_000:
        return f"{p / 1_000_000:.1f}M"
    if p >= 1_000:
        return f"{p / 1_000:.1f}k"
    return str(p)
