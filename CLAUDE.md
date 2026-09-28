# CLAUDE.md — Contrastive Language Model (CLM) vs Jev

## Project Purpose

A hands-on Python comparison of two ways to make typed decisions about text:

- **CLM (contrastive language model)**: a contrastively trained sentence encoder
  (default `sentence-transformers/all-MiniLM-L6-v2`). It runs locally, and its embeddings feed
  either zero-shot label matching or a small linear probe.
- **Jev** (TypeSafe AI, https://typesafe.ai): a proprietary "System One" model reached through an
  API. It answers typed questions (Choice / Score / Noul) with a probability for every option
  instead of generating text.

Goal: measure which is better for which situation, looking at accuracy, calibration, latency,
cost and labeled-data needs, and document the advantages, disadvantages, do's and don'ts.

This is a learning and evaluation project, not production code. Favor clarity, reproducibility
and honest measurement.

## Core Concepts (keep answers consistent with these)

### CLM
- Trained with InfoNCE-style objectives on positive/negative text pairs, so similar meanings
  land close together in embedding space.
- Zero-shot classification: cosine similarity between the text embedding and each label
  description's embedding, then a softmax with a temperature.
- Raw similarities are **not probabilities**. Calibration needs a temperature fitted on a few
  labeled examples (temperature scaling).
- Linear probe: logistic regression on frozen embeddings using k labeled examples per class.

### Jev
- Request = `state` (text or JSON) + typed `questions`. A Choice question has
  `instructions` and `criteria` (option -> description), with up to 255 options.
- Response = `choice`, `probabilities` over every option, and `confidence` (a concentration
  statistic derived from the probabilities, **not** a separate calibration estimate).
- Trained with "RLCD" (Reinforcement Learning for Calibrated Decisions), per TypeSafe.
  **Calibration is a vendor claim and needs testing, not assuming.**
- Text only. Priced per input token (output free). No fine-tuning: all customization goes
  through state, instructions and criteria.
- API: `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer $TYPESAFE_API_KEY`.
  Docs: https://docs.typesafe.ai (index at `/llms.txt`).

## Comparison Summary (hypotheses to test, not conclusions)

| Aspect | CLM | Jev |
|---|---|---|
| Where it runs | Local (CPU OK) | Remote API |
| Output | Embeddings -> you build the classifier | Typed answer + probabilities |
| Labeled data | Zero-shot possible; improves with k-shot probe | Zero-shot only (via instructions/criteria) |
| Calibration | Needs temperature scaling / probe tuning | Claimed calibrated by training |
| Latency | ms per example locally, batchable | Network round-trip, ~70–500 ms claimed |
| Cost | Compute only | Per input token |
| Reuse | Same embeddings serve search, clustering, retrieval | One answer per question |
| Changing labels | Re-embed descriptions (zero-shot) or relabel (probe) | Edit criteria text |
| Privacy / offline | Data never leaves the machine | Data sent to vendor |

State every conclusion relative to the dataset, label set, prompt/criteria wording and sample
size that were actually tested.

## Do's
- Do give both methods **identical label descriptions**: Jev `criteria` = CLM label texts.
- Do score every method on the **same fixed test subset** (the task `seed` and `test_size`).
- Do report `labeled_examples_used` for every run. Zero-shot, temperature-calibrated and k-shot
  are different regimes.
- Do report calibration (ECE, Brier, NLL) alongside accuracy. Calibration is Jev's main claim.
- Do log the exact Jev model version the API returns (`jev-latest` is an alias that moves).
- Do cache Jev responses (`results/cache/`) so reruns and re-analysis don't re-bill.
- Do fix seeds, and use more than one seed / test subset before trusting small differences.
- Do write a short entry in `results/NOTES.md` after each experiment.

## Don'ts
- Don't tune prompts, criteria, temperatures or probe hyperparameters on the test set. Use
  labeled TRAIN data only.
- Don't compare Jev network latency with CLM local compute time as if they were the same thing.
- Don't declare a winner from one dataset or one run.
- Don't treat Jev's `confidence` field as a calibrated probability. Use `probabilities`.
- Don't commit API keys. `TYPESAFE_API_KEY` comes from the environment; never touch `.env`
  without asking.

## Tech Stack
- Python 3.11+, managed with uv
- sentence-transformers (CLM), scikit-learn (probe, metrics), scipy (temperature fitting)
- httpx (Jev HTTP client with retries and cache), Hugging Face `datasets`
- PyYAML configs

Always check current package versions and the Jev API docs before writing code. Both change often.

## Project Structure
```
.
├── CLAUDE.md
├── README.md
├── configs/
│   ├── tasks/        # dataset, test subset, instructions, label keys + descriptions
│   └── methods/      # jev.yaml, clm_zeroshot.yaml, clm_probe.yaml
├── src/clm_jev_model_comparison/
│   ├── data/tasks.py     # load HF dataset -> TaskData, k-shot sampling
│   ├── models/clm.py     # encoder, zero-shot probs, temperature fit, probe
│   ├── models/jev.py     # Jev client: async, retries, disk cache
│   ├── eval/metrics.py   # accuracy, macro-F1, ECE, Brier, NLL, latency
│   ├── eval/compare.py   # results/runs/*/metrics.json -> results/summary.md
│   ├── run.py            # one method x one task -> predictions.jsonl + metrics.json
│   └── utils/            # config overrides, seeding, run dirs
├── results/
│   ├── runs/         # per-run config, predictions, metrics
│   ├── cache/        # Jev responses (gitignored)
│   └── NOTES.md      # experiment journal
└── tests/            # metrics, CLM math, Jev client (mocked HTTP), configs
```

## Coding Conventions
- Type hints on public functions; docstrings state array shapes, e.g. probs `(N, C)`.
- Keep metric and probability functions pure and unit-tested (known inputs -> known outputs).
- Jev client tests use `httpx.MockTransport`. Tests never call the real API.
- Configs drive experiments; no hard-coded hyperparameters in code.
- Every method returns probabilities `(N, C)` in the task's label order, so all metrics are shared.
- Small, focused commits; one experiment idea per branch where practical.
