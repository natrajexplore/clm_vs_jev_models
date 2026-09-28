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

(These two runs were superseded by the multi-seed reruns below; they remain in git history at ce7e310.)

## 2026-09-28: Jev + corrective actions, AG News (500 test, task seed 0)

All runs in `results/runs/*_s{0,1,2}_*`; CLM variants with method seeds 0, 1 and 2. Jev ran once
(`jev-1.13.0`, 500 requests, $0.0091, p50 337 ms).

| variant | labels | acc | ECE | NLL |
|---|---|---|---|---|
| Jev zero-shot | 0 | 0.880 [0.852–0.906] | 0.081 | 1.455 |
| CLM zero-shot, no labels (T=0.05) | 0 | 0.762 | 0.042 | 0.647 |
| CLM zero-shot, temp-scaled | 500 | 0.762 ± 0.000 | 0.047 | 0.644 |
| CLM probe, C=1 (before) | 64 | 0.812 ± 0.007 | 0.394 | 0.948 |
| CLM probe, CV-chosen C (after) | 64 | 0.794 ± 0.009 | 0.052 | 0.554 |

- **Jev is the most accurate with zero labels** (+6.8 pts over the best 64-label CLM probe,
  +11.8 over label-free CLM zero-shot). The Jev CI and CLM spreads don't overlap.
- **Jev probabilities are mildly overconfident** (mean top prob 0.955 vs acc 0.88). It gave
  exactly 0.0 to the true label in 23/500 cases. Those are 87% of its NLL (0.19 without them).
  Don't treat a Jev 0.0 as impossible.
- **Probe fix is a trade-off:** CV-chosen C cut ECE 0.39 → 0.05 and NLL 0.95 → 0.55 but lost
  ~2 accuracy points. CV picked C=1000 (the top of the search range) on seeds 0 and 1, so widen the range next.
- **Zero-shot temperature:** the T=0.05 guess was already close to the fitted T≈0.049, so the
  500 calibration labels bought almost nothing here. That's luck, not a rule.
- Next: widen the probe's C range; a many-label task; Jev run-to-run variation without the cache.
