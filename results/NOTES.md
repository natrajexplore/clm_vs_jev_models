# Experiment Journal

For each entry: date, run dir, what changed, what happened, and what it suggests.
Always cite the config and seed.

---

## 2026-09-28: first CLM baselines, AG News (500 test, seed 0)

Runs: `ag_news_clm_zeroshot_20260928-125953`, `ag_news_clm_probe_20260928-130022`.
Model: all-MiniLM-L6-v2 on CPU.

| method | acc | macro-F1 | ECE | labeled |
|---|---|---|---|---|
| clm_zeroshot (T fitted on 500 train) | 0.762 | 0.760 | 0.038 | 500 |
| clm_probe (16-shot, C=1.0) | 0.822 | 0.818 | 0.405 | 64 |

- Zero-shot with one fitted temperature (T≈0.049) is well calibrated but less accurate.
- 16-shot probe is more accurate but badly **underconfident** (ECE 0.40). Default C=1.0 on
  unit-norm embeddings regularizes heavily. Next: choose C by cross-validation on the k-shot
  train sample (not test), or standardize features.
- Both run at ~10 ms/example on CPU.
- Jev not run yet: needs `TYPESAFE_API_KEY`.
