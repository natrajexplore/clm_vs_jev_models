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

## Comparison app (dashboard + live playground)

```bash
uv run --env-file .env uvicorn clm_jev_model_comparison.app.server:app --port 8006
# open http://127.0.0.1:8006
```

- **Dashboard:** results table (mean ± std over seeds), accuracy-vs-calibration scatter,
  reliability diagrams, and corrective actions with before/after numbers. It reads
  `results/runs/` on every page load, so new runs show up after a refresh.
- **Playground:** type any text and see Jev and both CLM methods side by side. The first
  request loads the models (about 30 s). Each playground request makes one real Jev call.
- The backend holds the Jev key. It never reaches the browser.

## Methods (each config file is one variant)

| Config | Labeled data used | Probabilities come from |
|---|---|---|
| `jev.yaml` | none | Jev `choice` question: `probabilities` field |
| `clm_zeroshot_nolabels.yaml` | none (fixed T=0.05) | softmax(cosine similarity to label descriptions / T) |
| `clm_zeroshot.yaml` | 500 train examples, used only to fit T | same, with a fitted temperature |
| `clm_probe_fixedC.yaml` | 16 per class | logistic regression, C=1.0 (the "before" config) |
| `clm_probe.yaml` | 16 per class | logistic regression, C chosen by CV log-loss on those 16 shots |

`task.seed` fixes the test subset for all methods. `method.seed` drives each method's own
sampling; vary it for error bars, e.g. `--set method.seed=1`.

## Output per run (`results/runs/<task>_<method>_<timestamp>/`)

- `config.yaml`: the full task + method config
- `predictions.jsonl`: label, prediction and probabilities per example (plus Jev confidence and latency)
- `metrics.json`: accuracy (with 95% bootstrap CI), macro-F1, ECE, Brier, NLL, labeled examples used, latency, cost and the Jev model version

## Caveats

- Jev latency is network round-trip time; CLM latency is amortized local CPU time. They are
  different measurements.
- Jev can return probability exactly 0.0 for the true label. NLL clips at 1e-12, so a few
  such answers dominate Jev's NLL. See `results/NOTES.md`.
- Banking77 (77 intents) was planned as a second task, but its Hugging Face version is
  script-based and won't load with `datasets` 5.x.
