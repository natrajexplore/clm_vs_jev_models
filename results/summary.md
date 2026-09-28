| task | method | variant | seed | n | accuracy | macro_f1 | ece | brier | nll | labeled_examples_used | latency_ms | cost_usd | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ag_news | clm_probe | cv_C | 0 | 500 | 0.8020 | 0.7988 | 0.0528 | 0.2759 | 0.5293 | 64 | 7.6622 | 0.0000 | ag_news_clm_probe_cv_C_s0_20260928-134937 |
| ag_news | clm_probe | cv_C | 1 | 500 | 0.7980 | 0.7976 | 0.0660 | 0.2938 | 0.5436 | 64 | 9.2465 | 0.0000 | ag_news_clm_probe_cv_C_s1_20260928-135049 |
| ag_news | clm_probe | cv_C | 2 | 500 | 0.7820 | 0.7767 | 0.0377 | 0.3119 | 0.5890 | 64 | 8.6077 | 0.0000 | ag_news_clm_probe_cv_C_s2_20260928-135210 |
| ag_news | clm_probe | fixed_C | 0 | 500 | 0.8220 | 0.8183 | 0.4045 | 0.4997 | 0.9489 | 64 | 8.1162 | 0.0000 | ag_news_clm_probe_fixed_C_s0_20260928-134915 |
| ag_news | clm_probe | fixed_C | 1 | 500 | 0.8100 | 0.8096 | 0.3951 | 0.5029 | 0.9501 | 64 | 8.7528 | 0.0000 | ag_news_clm_probe_fixed_C_s1_20260928-135026 |
| ag_news | clm_probe | fixed_C | 2 | 500 | 0.8040 | 0.7991 | 0.3828 | 0.4986 | 0.9449 | 64 | 9.8923 | 0.0000 | ag_news_clm_probe_fixed_C_s2_20260928-135147 |
| ag_news | clm_zeroshot | no_labels | 0 | 500 | 0.7620 | 0.7600 | 0.0423 | 0.3519 | 0.6474 | 0 | 7.0204 | 0.0000 | ag_news_clm_zeroshot_no_labels_s0_20260928-134828 |
| ag_news | clm_zeroshot | temp_scaled | 0 | 500 | 0.7620 | 0.7600 | 0.0376 | 0.3513 | 0.6456 | 500 | 8.2971 | 0.0000 | ag_news_clm_zeroshot_temp_scaled_s0_20260928-134853 |
| ag_news | clm_zeroshot | temp_scaled | 1 | 500 | 0.7620 | 0.7600 | 0.0464 | 0.3510 | 0.6430 | 500 | 8.1081 | 0.0000 | ag_news_clm_zeroshot_temp_scaled_s1_20260928-135002 |
| ag_news | clm_zeroshot | temp_scaled | 2 | 500 | 0.7620 | 0.7600 | 0.0572 | 0.3509 | 0.6428 | 500 | 10.5976 | 0.0000 | ag_news_clm_zeroshot_temp_scaled_s2_20260928-135124 |
| ag_news | jev | zero_shot | 0 | 500 | 0.8800 | 0.8769 | 0.0807 | 0.2016 | 1.4545 | 0 | 337 | 0.0091 | ag_news_jev_zero_shot_s0_20260928-134808 |

latency_ms: Jev = p50 per-request network latency; CLM = amortized local CPU embedding time per example. Not the same measurement; compare with care.
