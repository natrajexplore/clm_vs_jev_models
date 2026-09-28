# CLM vs Jev

Compares a **contrastive language model** (sentence embeddings, run locally) with **Jev**
(TypeSafe AI's typed-decision API) on text classification. Both see the same label
descriptions and the same test examples, and are scored on accuracy, calibration, latency and
cost. See `CLAUDE.md` for the background and the rules.

## Setup

```bash
uv sync
uv run pytest                     # no network or API key needed
```

For Jev, get a key at https://console.typesafe.ai/keys and export it in your shell:

```bash
export TYPESAFE_API_KEY=...       # PowerShell: $env:TYPESAFE_API_KEY = "..."
```

## Run

```bash
T=configs/tasks/ag_news.yaml
uv run python -m clm_jev_model_comparison.run --task $T --method configs/methods/clm_zeroshot.yaml
uv run python -m clm_jev_model_comparison.run --task $T --method configs/methods/clm_probe.yaml
uv run python -m clm_jev_model_comparison.run --task $T --method configs/methods/jev.yaml

uv run python -m clm_jev_model_comparison.eval.compare      # -> results/summary.md
```

Override any config key with `--set`, e.g. `--set task.test_size=50 method.shots_per_class=64`.
Try a small `test_size` with Jev first. Responses are cached in `results/cache/`, so reruns are free.

## Methods

| Method | Labeled data used | Probabilities come from |
|---|---|---|
| `clm_zeroshot` | `calibration_size` train examples (default 500), used only to fit a softmax temperature. Set to 0 for pure zero-shot. | softmax(cosine similarity to label descriptions / T) |
| `clm_probe` | `shots_per_class` train examples per class | logistic regression on embeddings |
| `jev` | none | Jev `choice` question: `probabilities` field |

## Output per run (`results/runs/<task>_<method>_<timestamp>/`)

- `config.yaml`: the full task + method config
- `predictions.jsonl`: label, prediction and probabilities per example (plus Jev confidence and latency)
- `metrics.json`: accuracy, macro-F1, ECE, Brier, NLL, labeled examples used, latency, cost and the Jev model version

## Caveats

- Jev latency is network round-trip time; CLM latency is amortized local CPU time. They are
  different measurements.
- Jev's `usage.input_tokens` location in the response isn't pinned down in the docs. The
  client accepts it per answer or top-level; check `input_tokens_missing` in `metrics.json`.
- Banking77 (77 intents) was planned as a second task, but its Hugging Face version is
  script-based and won't load with `datasets` 5.x.
