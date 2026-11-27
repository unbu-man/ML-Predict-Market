# Chapter XXVIII — XGBoost and gap-aware feature engineering

**27 November 2026.** Synthetic-only M2/M3 software experiment. The real-data licensing gate in Chapter XXVII remains active. Do not use the synthetic results as evidence of BTC prediction skill. The reserved June–July 2026 holdout remains unexported.

## 1. Input contract and causal windows

Audit the two CSV files separately against the manifest: exact headers, SHA256, date bounds, counts, longest contiguous run, UTC time grid, OHLCV and ordering. Reject non-synthetic manifests in this implementation. Split each block into contiguous runs; skip runs with fewer than 37 candles. Do not bridge a gap or carry windows across the block boundary. Generate features and targets in separate arrays in memory; output predictions only. The 12 inputs match Chapter XXVI (six return lags and three each of rolling means and standard deviations). No GARCH, target or timestamp inputs.



```math
\begin{aligned}
r_t&=\ln(C_t/C_{t-1})\\
\bar r_{t,w}&=\frac1w\sum_{i=0}^{w-1}r_{t-i}\\
s_{t,w}&=\sqrt{\frac1{w-1}\sum_{i=0}^{w-1}(r_{t-i}-\bar r_{t,w})^2}\\
Y_{t,h}&=\ln(C_{t+h}/C_t),\qquad h\in\{1,12\}
\end{aligned}
```

Use a common set of eligible anchors for both horizons, with 24 past returns and 12 future bars inside each contiguous run. Strict cutoff: anchor close + 3600 seconds + assumed delay must be earlier than the block end. The last otherwise eligible anchor is dropped when the segment reaches the boundary exactly. The two-second delay is an assumption. No actual publication times verified.

Expected complete-block feature counts (not yet observed on the user's machine):


```math
\begin{aligned}N_{\mathrm{train}}&=69984-24-12-1=69947\\N_{\mathrm{val}}&=17568-24-12-1=17531\end{aligned}
```

## 2. Decision trees

A regression tree partitions input space into leaves. Each leaf has a constant score. Example: split on the previous return, then rolling volatility. Splits must be learned from training observations, not validation. A depth-two tree can learn simple feature interactions without fitting a deep tree.

## 3. Gradient boosting

Start with a base score and add trees sequentially. Let eta be learning rate and f_k be the kth tree:


```math
\hat y_i^{(K)}=\hat y_i^{(0)}+\eta\sum_{k=1}^K f_k(x_i)
```

At each iteration fit the next tree to improve the objective, rather than train independent trees like a random forest.

For regression, train on percent-unit log returns and convert output back to unscaled log returns. For binary classification, the additive score is a logit:


```math
p_i=\frac{1}{1+e^{-\hat y_i^{(K)}}}
```

## 4. Objective and second-order approximation

For one tree, ignoring constants independent of the new tree:


```math
\begin{aligned}
\mathcal L&=\sum_i \ell(y_i,\hat y_i)+\sum_k\Omega(f_k)\\
\Omega(f)&=\gamma T+\frac{\lambda}{2}\sum_{j=1}^T w_j^2\\
g_i&=\frac{\partial\ell(y_i,\hat y_i)}{\partial\hat y_i}\\
h_i&=\frac{\partial^2\ell(y_i,\hat y_i)}{\partial\hat y_i^2}\\
\widetilde{\mathcal L}^{(k)}&=\sum_i\left[g_if_k(x_i)+\frac12h_if_k(x_i)^2\right]+\Omega(f_k)
\end{aligned}
```

Here T is the leaf count, w_j is a leaf score, gamma is split complexity penalty and lambda is L2 penalty. The Taylor approximation is at the previous raw prediction score; learning-rate shrinkage is applied to the resulting tree. This lesson uses zero L1 penalty.

Squared-error loss with the half-square convention:


```math
\ell=\frac12(y-\hat y)^2,\qquad g=\hat y-y,\qquad h=1
```

Binary cross-entropy with p equal to sigmoid(raw score):


```math
\ell=-y\ln p-(1-y)\ln(1-p),\qquad g=p-y,\qquad h=p(1-p)
```

## 5. Optimal leaf weight and split gain

For the observations I_j routed to a leaf, sum gradients and Hessians:


```math
\begin{aligned}G_j&=\sum_{i\in I_j}g_i,\quad H_j=\sum_{i\in I_j}h_i\\
J_j(w)&=G_jw+\tfrac12(H_j+\lambda)w^2\\
\frac{\partial J_j}{\partial w}&=G_j+(H_j+\lambda)w=0\\
w_j^*&=-\frac{G_j}{H_j+\lambda}
\end{aligned}
```

The theoretical gain for dividing a parent into left/right leaves is:


```math
\mathrm{Gain}=\frac12\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma
```

Worked synthetic arithmetic: G_L=-12, H_L=30, G_R=8, H_R=30, lambda=5, gamma=0. Then left weight=12/35≈0.34286 and right weight=-8/35≈-0.22857; gain≈2.84835. Both leaves satisfy the chosen minimum child Hessian of 20. At eta=0.05 the left score increment is about 0.01714. This is a hand example, not an actual fitted split.

## 6. Fixed model comparison

| Parameter | Initial value |
|---|---:|
| n_estimators | 100 |
| max_depth | 2 |
| learning_rate | 0.05 |
| min_child_weight | 20 |
| reg_lambda | 5 |
| reg_alpha | 0 |
| subsample / colsample_bytree | 1 / 1 |
| tree_method | hist |
| random_state | 42 |
| n_jobs | 2 |

min_child_weight is a minimum Hessian sum, not always a minimum number of rows. Shallow trees and regularization are conservative engineering defaults, not proven optimal parameters. No validation-driven tuning or early stopping. For each horizon fit Logistic/Ridge and XGBoost on identical training anchors, evaluate on identical validation anchors. Logistic and Ridge scale on training only; tree models receive raw features. Compare classification accuracy, Brier, log loss and AUC to training prevalence, and regression MAE/RMSE/sign accuracy to zero return and training mean. All four XGBoost fits are independent. No trading scores, saved deployable model or probability calibration. Synthetic generator creates independent Gaussian minute log-return innovations; no durable predictive signal is intentionally injected. A model may underperform simple baselines. One-hour outcomes overlap; accuracy is not a trade win rate.

## 7. Run on the Mac

From repo root:
```bash
git pull origin main
python3 -m pip install -r requirements-xgboost.txt
python3 -m unittest tests.test_day10_xgboost -v
python3 -m examples.day10_xgboost data/history/synthetic_1y_plan --output data/xgboost/synthetic_v1 --replay-delay-seconds 2
cat data/xgboost/synthetic_v1/report.json
```

Five synthetic fixture tests executed and passed in the assistant container: feature alignment and strict boundary, gaps, future-perturbation invariance of features, checksum rejection, matched models. The full local history CLI has not been executed by the assistant for this chapter. Existing output directories are rejected before model fitting; read existing report or use a new output name. Output report.json and predictions.csv under gitignored data/. The report records source block hashes and settings, not a fitted model artifact. Dependency versions are not pinned yet.

## 8. Next gate

Review the local synthetic report as a pipeline sanity check. Obtain appropriate authorization and longer real-data coverage before implementing a separately reviewed real-data mode. Do not infer market skill or choose features from synthetic metrics. Later compare all models on identical real-data purged walk-forward folds; add multi-timeframe features and causal GARCH forecasts in controlled ablations, and retain a fresh untouched holdout. No live orders, no market-data downloads.

[Chapter XXVII](27_historical_data_expansion.md) | [Chapter XXVI](26_supervised_baselines.md)


## 9. User-reported synthetic XGBoost results — 27 November 2026

The user pasted the local report for the full synthetic development dataset. The reported source-block checksums match Chapter XXVII, settings match the fixed experiment, and the common eligible counts match the design (69,947 train and 17,531 validation). Holdout_exported is false. These are user-reported software-test results; raw files, individual predictions and local unit-test output were not independently inspected.

### Classification metrics

| Horizon | Model | Accuracy | Brier | Log loss | ROC-AUC |
|---|---|---:|---:|---:|---:|
| 5m | Prevalence | 0.494495 | 0.250001 | 0.693149 | 0.500000 |
| 5m | Logistic | 0.494838 | 0.250173 | 0.693493 | 0.494095 |
| 5m | XGBoost | 0.499059 | 0.250133 | 0.693414 | 0.496692 |
| 1h | Prevalence | 0.482859 | 0.250051 | 0.693249 | 0.500000 |
| 1h | Logistic | 0.503166 | 0.250210 | 0.693567 | 0.500016 |
| 1h | XGBoost | 0.490160 | 0.250826 | 0.694813 | 0.488737 |

Training UP proportions: 0.5000643344 (5m), 0.5014225056 (1h). Fixed-threshold accuracy is near chance; the prevalence forecast predicts UP in both cases. The learned models have worse Brier and log loss at both horizons. No robust information advantage is established by the small accuracy differences.

### Regression metrics

| Horizon | Model | MAE | RMSE | Sign accuracy |
|---|---|---:|---:|---:|
| 5m | Zero | 0.000896841 | 0.001126677 | 0.505505 |
| 5m | Train mean | 0.000896839 | 0.001126675 | 0.505505 |
| 5m | Ridge | 0.000897052 | 0.001126873 | 0.496606 |
| 5m | XGBoost | 0.000897115 | 0.001126818 | 0.500713 |
| 1h | Zero | 0.003124030 | 0.003917244 | 0.517141 |
| 1h | Train mean | 0.003123983 | 0.003917172 | 0.517141 |
| 1h | Ridge | 0.003125669 | 0.003919099 | 0.505562 |
| 1h | XGBoost | 0.003129150 | 0.003922346 | 0.496207 |

Unscaled log-return units. Training mean has the lowest MAE and RMSE at both horizons. This is consistent with the synthetic generator's independent random minute log returns. Overlapping 1h targets are dependent; do not interpret the 17,531 rows as independent trials. No inference about actual BTC prediction or trading profitability is justified. A negative-control result is useful but cannot by itself prove that every possible leakage bug is absent.

### Next research gate

M3 software execution milestone complete; no model promoted. Add a causal multi-timeframe (MTF) module: aggregate 5m bars into UTC-aligned complete 1h, 4h and 1D buckets; omit incomplete buckets; reset feature windows at gaps and block boundaries; as-of join only closed and assumed-available higher-timeframe bars. Keep the 5m prediction origins and 5m/1h targets unchanged. Compare 5m-only and MTF variants on exactly the same eligible origins, with fixed settings and train-only transforms. The current export has no 1m bars; defer 1m inputs until a separate permitted source/export exists. Synthetic-only testing continues; consider a separately designed positive-control fixture with a known causal signal to test learnability, not market skill. No final holdout, live orders or authorized real-data expansion.
