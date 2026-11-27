# Chapter XXIX — Causal multi-timeframe feature engineering

**27 November 2026.** Synthetic-only M3-TF ablation. The data permission gate remains active. No real prices, final holdout, trading or live orders.

## 1. Research question

Do longer-timeframe features add predictive information beyond 5m features? Compare cumulative feature groups using the same origins, targets, training/validation dates and fixed XGBoost settings. This tests software, not BTC skill, because the current synthetic generator has independent minute return innovations.

| Variant | Input frames | Features |
|---|---|---:|
| 5m_only | 5m | 12 |
| plus_1h | 5m + 1h | 15 |
| plus_4h | 5m + 1h + 4h | 18 |
| plus_1d | 5m + 1h + 4h + 1D | 21 |

Keep the existing 5m and 1h forecast horizons. No 1m inputs: the source export contains only 5m candles. No RSI, EMA, ATR, volume or GARCH additions in this ablation.

## 2. Complete UTC-aligned buckets

A higher interval of length D seconds needs m=D/300 complete 5m bars. Let t denote a UTC Unix-second opening timestamp. The bucket open and closing timestamp are:


```math
\begin{aligned}
m&=\frac{D}{300}\\
b(t)&=D\left\lfloor\frac{t}{D}\right\rfloor\\
T_{\mathrm{close}}&=b(t)+D
\end{aligned}
```

For 1h, 4h and 1D, m is 12, 48 and 288 respectively. Only a complete bucket is eligible. Its close is the final constituent 5m close. Use UTC midnight for daily boundaries, not local midnight. Omit incomplete leading and trailing buckets. The upstream Day 10 loader audits OHLCV and splits at gaps; each segment is aggregated independently. No fill, interpolation or carry-forward across gaps or blocks.

## 3. Higher-timeframe returns and rolling features

For completed closes C_k at a single higher timeframe:


```math
\begin{aligned}
r_k^{(D)}&=\ln\left(\frac{C_k^{(D)}}{C_{k-1}^{(D)}}\right)\\
\bar r_{k,3}^{(D)}&=\frac13\sum_{i=0}^{2}r_{k-i}^{(D)}\\
s_{k,3}^{(D)}&=\sqrt{\frac12\sum_{i=0}^{2}\left(r_{k-i}^{(D)}-\bar r_{k,3}^{(D)}\right)^2}
\end{aligned}
```

Four consecutive complete higher-frame closes are required for three returns. For an illustrative sequence 100, 102, 101, 103, the last return is ln(103/101), about 0.01961. The preceding returns are ln(102/100) and ln(101/102); compute their three-return mean and sample standard deviation. These are returns, not future targets, and no prices from a still-forming bucket enter a feature.

## 4. Causal as-of alignment

Define a decision origin a as the close of the observed base candle. Let delta be the common assumed publication delay in seconds.


```math
\begin{aligned}
t_{\mathrm{decision}}&=a+\delta\\
t_{\mathrm{available},k}^{(D)}&=T_{\mathrm{close},k}^{(D)}+\delta\\
k^*(a,D)&=\max\{k:t_{\mathrm{available},k}^{(D)}\le t_{\mathrm{decision}}\}
\end{aligned}
```

The latest eligible higher-timeframe feature vector is joined using binary search. At 10:35:02 UTC with delta=2, the hourly bar ending 10:00 is available, the 11:00 close is not. At 11:00:02 the completed 11:00 bar becomes eligible under this common-delay assumption. In real data, higher-timeframe publication/computation delays may differ; measured availability must replace this assumption before real-time claims. No historical latency verified.

## 5. Matched origins and boundary purge

Construct Day 10 base features and both targets separately within each segment. Then join all three higher-frame vectors. Exclude any origin without all 21 finite features, including for the 5m-only comparison. All four variants see identical training and validation anchors. Preserve strict horizon-label availability before each block's end:


```math
a+3600+\delta<T_{\mathrm{end\ exclusive}}
```

This common 1h constraint also restricts the 5m task. The maximum warmup is four completed daily candles. On complete UTC-aligned blocks, the first eligible anchor is four days after the start; the final common anchor obeys the strict cutoff. Expected counts at a two-second delay:


```math
\begin{aligned}
N_{\mathrm{train}}&=69984-1164=68820\\
N_{\mathrm{validation}}&=17568-1164=16404
\end{aligned}
```

These are expected for the user's reported complete blocks, not an observed full run. Resetting daily features at the validation start deliberately discards its first four days, instead of carrying previous-block context. This is conservative and can be revisited with explicitly causal carry-in later. The numbers differ from Chapter XXVIII; compare only within this experiment, not against unmatched earlier scores.

## 6. Fixed evaluation

Reuse Chapter XXVIII's 100 depth-two trees, learning rate 0.05, child Hessian minimum 20, L2=5, seed 42, two workers. Each variant fits a binary classifier and return regressor at each horizon (16 fits). Training only; no early stopping or tuning on validation. Classification baseline is training prevalence, regression baselines are zero and training mean. Use accuracy, AUC, Brier, log loss, MAE, RMSE and sign accuracy as in the prior chapter.



```math
\Delta L_D=L_{\mathrm{variant}\ D}-L_{\mathrm{5m\ only}}
```

For loss metrics, a negative difference favors the variant on these matched origins. Accuracy/AUC use the opposite direction. Do not claim that a synthetic improvement transfers to real data; dependent overlapping hourly outcomes preclude naive independent-trial inference. This is a cumulative ablation, not all possible combinations or an isolated measure of each frame independent of previous additions.

## 7. Implementation and run

Reuses the Day 10 audited synthetic loader, adding an optional builder argument with the original builder as default. Day 10 CLI behavior is unchanged. New module builds arrays in memory and exports report.json and predictions.csv. No new dependencies; use requirements-xgboost.txt. Run from the repo root:

```bash
git pull origin main
python3 -m unittest tests.test_day11_multitimeframe -v
python3 -m examples.day11_multitimeframe data/history/synthetic_1y_plan --output data/multitimeframe/synthetic_v1 --replay-delay-seconds 2
cat data/multitimeframe/synthetic_v1/report.json
```

If output exists, read its report or choose a different directory, not delete existing results. The loader still rejects non-synthetic manifests and enforces fixed dates, checksums and no holdout export. Output paths under data/ are gitignored.

Five new tests passed in an isolated container (complete aligned buckets, as-of boundary, gap reset, future perturbation and cutoff, matched model fitting). The tests ran using copied upstream build and scoring helpers and a stub for the unused full loader; the actual repository CLI and manifest integration were not executed in the container because direct GitHub network access was unavailable. Local full history run and regression of Day 10 remain pending. Test commands are provided for the user's Mac.

## 8. Next research gate

Review local synthetic MTF results and ensure expected counts and hashes. Continue with a known-signal synthetic positive control and causal volatility feature ablation (historically fitted GARCH, not a full-sample fitted training path). No market model promotion or feature selection based on synthetic scores. Authorized real-data coverage, untouched real holdout, publication latency, costs, XAU coverage and prospective paper evaluation remain unresolved.

[Chapter XXVIII](28_xgboost.md) | [Chapter XXVII](27_historical_data_expansion.md)


## 9. User-reported synthetic MTF results — 27 November 2026

The local report contains the expected 68,820 training and 16,404 validation pairs, 21 features, four cumulative variants and unchanged XGBoost settings. Both source block hashes match Chapter XXVII. Holdout_exported is false. This is user-reported evidence of successful full CLI execution; raw CSVs, individual predictions and local test output were not independently inspected.

### Classification

| Horizon | Variant | Accuracy | Brier | Log loss | ROC-AUC |
|---|---|---:|---:|---:|---:|
| 5m | Prevalence | 0.495001 | 0.250003 | 0.693153 | 0.500000 |
| 5m | 5m_only | 0.494270 | 0.250127 | 0.693401 | 0.496105 |
| 5m | plus_1h | 0.497744 | 0.250103 | 0.693352 | 0.497445 |
| 5m | plus_4h | 0.497927 | 0.250106 | 0.693359 | 0.500862 |
| 5m | plus_1d | 0.498537 | 0.250126 | 0.693401 | 0.500125 |
| 1h | Prevalence | 0.484760 | 0.250060 | 0.693266 | 0.500000 |
| 1h | 5m_only | 0.491344 | 0.250907 | 0.694973 | 0.486650 |
| 1h | plus_1h | 0.486528 | 0.250799 | 0.694758 | 0.492064 |
| 1h | plus_4h | 0.510059 | 0.249720 | 0.692585 | 0.519880 |
| 1h | plus_1d | 0.513655 | 0.249618 | 0.692383 | 0.527637 |

Training UP rates: 0.5002615519 (5m), 0.5018453938 (1h). Both prevalence forecasts predict UP at the fixed threshold. All learned variants have worse 5m probability losses than prevalence. The full MTF 1h variant improves accuracy by about 2.23 percentage points over matched 5m-only XGBoost and 2.89 points over prevalence; Brier, log loss and AUC also improve on this particular validation block.

### Regression

| Horizon | Variant | MAE | RMSE | Sign accuracy |
|---|---|---:|---:|---:|
| 5m | Zero | 0.000899706 | 0.001129797 | 0.504999 |
| 5m | Train mean | 0.000899708 | 0.001129800 | 0.495001 |
| 5m | 5m_only | 0.000900026 | 0.001129968 | 0.502926 |
| 5m | plus_1h | 0.000900032 | 0.001129960 | 0.498232 |
| 5m | plus_4h | 0.000899978 | 0.001129934 | 0.498842 |
| 5m | plus_1d | 0.000900021 | 0.001129944 | 0.499329 |
| 1h | Zero | 0.003151525 | 0.003947176 | 0.515240 |
| 1h | Train mean | 0.003151590 | 0.003947291 | 0.484760 |
| 1h | 5m_only | 0.003157916 | 0.003953843 | 0.490734 |
| 1h | plus_1h | 0.003161744 | 0.003957595 | 0.493233 |
| 1h | plus_4h | 0.003148047 | 0.003943447 | 0.503597 |
| 1h | plus_1d | 0.003149267 | 0.003947232 | 0.513655 |

Units are unscaled log returns. Zero return has the lowest 5m errors and highest sign accuracy at both horizons. The 18-feature variant has the lowest 1h MAE and RMSE; its improvement versus zero-return RMSE is only about 0.0945%. The full 21-feature variant has slightly higher 1h RMSE than zero return.

### Interpretation and next research gate

The generator uses independent Gaussian minute return innovations. Therefore, the observed 1h classification gains are not evidence that MTF predicts real markets or that the simulated process has a durable conditional mean signal. Finite-sample variation, overlapping outcomes and multiple comparisons can explain apparent gains; an unexpected result should also motivate further leakage checks, not automatic promotion. Do not select a production feature group from these results. The 5m-only comparison here uses matched origins, not Chapter XXVIII's larger sample.

Next: Chapter XXX, causal GARCH + XGBoost feature ablation. Keep fixed MTF configurations for comparison, not a winner chosen from this synthetic run. Generate one-step variance forecasts using parameters estimated strictly from historical prefixes, with a prespecified refit cadence and causal recursive updates; reset at gaps, document failure and warmup, and restrict every comparison to matched eligible anchors. Compare base, base + rolling-volatility benchmark, and base + GARCH, with MTF variants where appropriate. Do not use a full-training fitted variance series as if its parameters were known at past origins. Include a separately designed synthetic positive-control fixture with a known predictable variance process; independent-return synthetic data remain a negative control. No market skill, validated real-time availability, authorized real-data coverage, final holdout evaluation, transaction-cost assessment or trading performance established.
