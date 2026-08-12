# Benchmarking ResonantBrain

Run standard LLM benchmarks (**MMLU, HellaSwag, ARC, WinoGrande, PIQA, …**)
on a trained ResonantBrain checkpoint using
[EleutherAI's `lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness).

ResonantBrain is **not** a HuggingFace `PreTrainedModel`, so `lm-eval` can't
talk to it directly. This package provides a thin adapter
(`resonantbrain/benchmark/wrapper.py`) that implements the three methods the
harness needs:

| Method | Used by |
|---|---|
| `loglikelihood`        | MCQ tasks: MMLU, ARC, HellaSwag, WinoGrande, PIQA, … |
| `loglikelihood_rolling`| Perplexity: WikiText, Lambada |
| `generate_until`       | Generation tasks: GSM8K, HumanEval, … |

---

## 1. Install the harness

```bash
pip install lm-eval
```

> Tiktoken (`tiktoken`) and `torch` are already part of your training stack.

Verify it imports cleanly:

```bash
python -c "import lm_eval; print(lm_eval.__version__)"
```

---

## 2. Run via the interactive menu

```bash
python run_resonantbrain.py
# ... choose [5] Run Benchmark
```

You'll be prompted for:

- **Checkpoint path** (default: `checkpoint_ssm_pretrain_mixed.pth`)
- **Task preset**:
  - `a` — Quick: HellaSwag + ARC-easy/challenge + WinoGrande + PIQA
  - `b` — MMLU (full 57-subject suite — slow)
  - `c` — Custom (enter your own comma-separated lm-eval task IDs)
  - `d` — Smoke test (100 docs/task: HellaSwag + ARC-easy)
- **Output JSON** (`results.json`)
- **Few-shot examples** (ENTER = task default, usually 0)

---

## 3. Run from the command line

```bash
python -m resonantbrain.benchmark.runner \
    --checkpoint checkpoint_ssm_pretrain_mixed.pth \
    --model-size medium \
    --chunk-size 768 \
    --tasks hellaswag,arc_easy,arc_challenge,winogrande,piqa \
    --num-fewshot 0 \
    --output results.json
```

Quick sanity check (100 docs only):

```bash
python -m resonantbrain.benchmark.runner \
    --checkpoint checkpoint_ssm_pretrain_mixed.pth \
    --tasks hellaswag,arc_easy \
    --limit 100 \
    --output smoke.json
```

---

## 4. Recommended tasks for a base (pretrained-only) model

A **base** model (no instruction/chat tuning) is best measured with
multiple-choice tasks scored by `loglikelihood`. Good first targets:

| Task | Metric | Notes |
|---|---|---|
| `hellaswag`       | acc / acc_norm |Sentence completion; strong general signal |
| `arc_easy`        | acc / acc_norm | Grade-school science MCQ |
| `arc_challenge`   | acc / acc_norm | Harder science MCQ |
| `winogrande`      | acc            | Coreference resolution |
| `piqa`            | acc            | Physical commonsense |
| `lambada_openai`  | perplexity/acc | Last-word prediction |
| `wikitext`        | bits/byte, ppl | Raw language-modeling quality |

Generation tasks (`gsm8k`, `humaneval`) generally *require* an
instruction-tuned checkpoint to produce usable numbers — a base model will
score near zero there.

**MMLU note:** the suite has 57 subtasks. Run it as a single group:

```bash
--tasks mmlu --num-fewshot 5
```

For the full >800-task catalog, see
<https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks>.

---

## 5. Architecture must match training

The wrapper rebuilds the `SSMTransformer` from `main.py`'s flags and reloads
weights. If you trained with non-default toggles (e.g. `enable_meta_routing=False`),
edit the call in `resonantbrain/main.py` (menu branch `[5]`) or invoke
`run_benchmark(...)` directly with matching keyword arguments via
`build_model_from_checkpoint` parameters in `resonantbrain/benchmark/wrapper.py`.

The model size **must** match the checkpoint:

| `model_size` | dim | layers | heads |
|---|---|---|---|
| `tiny`   | 256 | 4  | 4  |
| `small`  | 512 | 6  | 8  |
| `medium` | 768 | 16 | 12 |

---

## 6. Output format

`results.json` contains the full `lm-eval` payload plus a `summary` block:

```json
{
  "results":  { ...lm-eval native... },
  "configs":  { ... },
  "summary": {
    "hellaswag":     { "acc": 0.2573, "acc_norm": 0.3012, "acc_stderr": 0.0044, ... },
    "arc_easy":      { "acc": 0.3421, "acc_norm": 0.3710, ... }
  }
}
```

The console also prints a per-task table at the end of the run.

---

## 7. Compare against other models (bar chart)

After a benchmark run has written `results.json`, you can plot your model's
scores against the **5 closest public models by parameter count**:

```bash
pip install matplotlib ijson   # if not already installed

python -m resonantbrain.benchmark.compare \
    --results results.json \
    --your-params 350M \
    --your-name ResonantBrain \
    --n-ref 5 \
    --output benchmark_comparison.png
```

This produces three artifacts:

| File | Contents |
|---|---|
| `benchmark_comparison.png`        | Headline chart — one bar per model, height = mean accuracy across the tasks you ran |
| `benchmark_comparison_per_task.png` | Grouped bars: one cluster per task, your model alongside each reference |
| `benchmark_comparison.txt`        | Plain-text summary table (your scores vs each reference) |

Your model is always highlighted (orange, hatched). Reference models are
color-coded by family. The closest-N selection is by **total parameter
count**, so you're always compared to models of similar capacity.

### Editing reference scores

Reference numbers live in [`reference_models.py`](reference_models.py) and are
publicly reported values (papers / HELM / EleutherAI runs). Because benchmark
results differ slightly across `lm-eval` versions, few-shot settings, and
`acc` vs `acc_norm` normalization, **verify them against your exact harness
version** before publishing a comparison. To add a model:

```python
REFERENCE_MODELS.append(RefModel(
    name="YourBaseline-300M",
    params=300_000_000,
    family="Custom",
    acc_norm={"hellaswag": 0.32, "arc_easy": 0.41, ...},
))
```

### Programmatic API

```python
from resonantbrain.benchmark import compare_to_closest

compare_to_closest(
    results_path="results.json",
    your_params=350_000_000,
    your_name="ResonantBrain",
    n_ref=5,
    output_path="comparison.png",
)
```
