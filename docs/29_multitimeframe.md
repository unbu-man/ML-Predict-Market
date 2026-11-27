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
