| task | method | n | accuracy | macro_f1 | ece | brier | nll | labeled_examples_used | latency_ms | cost_usd | run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ag_news | clm_probe | 500 | 0.8220 | 0.8183 | 0.4045 | 0.4997 | 0.9489 | 64 | 11.1156 | 0.0000 | ag_news_clm_probe_20260928-130022 |
| ag_news | clm_zeroshot | 500 | 0.7620 | 0.7600 | 0.0376 | 0.3513 | 0.6456 | 500 | 9.6733 | 0.0000 | ag_news_clm_zeroshot_20260928-125953 |

latency_ms: Jev = p50 per-request network latency; CLM = amortized local CPU embedding time per example. Not the same measurement; compare with care.
