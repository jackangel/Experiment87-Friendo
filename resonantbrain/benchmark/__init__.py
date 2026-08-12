"""
Benchmark utilities for the ResonantBrain SSM-Transformer.

Exposes a custom ``lm-evaluation-harness`` model wrapper so standard LLM
benchmarks (MMLU, HellaSwag, ARC, WinoGrande, ...) can be run against a
trained checkpoint despite the model NOT being a HuggingFace
``PreTrainedModel``.

Submodules are imported LAZILY so that running a submodule directly, e.g.::

    python -m resonantbrain.benchmark.compare
    python -m resonantbrain.benchmark.runner

does not trigger ``RuntimeWarning: found in sys.modules after import of
package 'resonantbrain.benchmark', but prior to execution of ...`` (runpy
emits that warning when the target module has already been imported eagerly
via the package ``__init__``).

Available names (importable from ``resonantbrain.benchmark``):
- :class:`ResonantBrainLM`
- :func:`build_model_from_checkpoint`
- :func:`run_benchmark`
- :func:`compare_to_closest`
"""

__all__ = [
    "ResonantBrainLM",
    "build_model_from_checkpoint",
    "run_benchmark",
    "compare_to_closest",
]


def __getattr__(name):
    # PEP 562 lazy attribute access — submodules are only imported when
    # actually requested, never on package import.  This prevents runpy's
    # "found in sys.modules after import of package" RuntimeWarning that
    # occurs when running a submodule via `python -m <pkg>.<module>`.
    if name == "ResonantBrainLM":
        from .wrapper import ResonantBrainLM
        return ResonantBrainLM
    if name == "build_model_from_checkpoint":
        from .wrapper import build_model_from_checkpoint
        return build_model_from_checkpoint
    if name == "run_benchmark":
        from .runner import run_benchmark
        return run_benchmark
    if name == "compare_to_closest":
        from .compare import compare_to_closest
        return compare_to_closest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
