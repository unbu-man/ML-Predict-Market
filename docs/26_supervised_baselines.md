# Chapter XXVI — Supervised learning: Logistic Regression and Ridge Regression

26 November 2026. M2 development replay only. Five synthetic fixture tests passed in the assistant container (scikit-learn 1.8.0, NumPy 2.3.5). Real BTC feature export and model scores are pending.

## 1. Feature and target contract

Read Chapter XXIII's features.csv, targets.csv and manifest.json. Verify both CSV checksums, exact column allowlists, row counts, chronological contiguous UTC anchors, paired keys, decision timestamps and horizon-specific label availability. Checksums confirm correspondence to the manifest, not raw source authenticity or correct feature calculations; source hash is reported as supplied. Exclude momentum columns because momentum equals window length times rolling mean. Use six lagged returns, three trailing means and three trailing sample standard deviations. No GARCH, price targets or timestamp metadata as inputs. Fit independent models per horizon.



```math
\begin{aligned}Y_{t,h}&=\ln\left(\frac{C_{t+h}}{C_t}\right),\qquad h\in\{1,12\}\\
y_{t,h}&=\begin{cases}1,&Y_{t,h}>0\\0,&Y_{t,h}\leq0\end{cases}
\end{aligned}
```

One step is five minutes. NOT_UP includes zero, not a short-trade instruction.

## 2. Standardization



```math
\begin{aligned}
\mu_j&=\frac1{N_{\mathrm{train}}}\sum_{i\in\mathrm{train}}x_{ij}\\
s_j&=\sqrt{\frac1{N_{\mathrm{train}}}\sum_{i\in\mathrm{train}}(x_{ij}-\mu_j)^2}\\
z_{ij}&=\frac{x_{ij}-\mu_j}{s_j}
\end{aligned}
```

The implementation uses StandardScaler's population standard deviation and unit scaling for constant features. Fit only on the purged training rows; apply the frozen scaler to both later blocks. Each model has its own pipeline.

## 3. Logistic Regression



```math
\begin{aligned}
\eta_i&=\beta_0+\sum_{j=1}^{p}\beta_j z_{ij}\\
p_i&=\frac{1}{1+\exp(-\eta_i)}\\
\log\frac{p_i}{1-p_i}&=\eta_i
\end{aligned}
```

Here p is the feature count (12), beta is a standardized-feature coefficient, and p_i is the estimated UP probability. Minimize binary cross-entropy with L2 regularization (unweighted data, intercept not penalized):


```math
\min_{\beta_0,\boldsymbol\beta}\left[-\sum_{i=1}^{N}\{y_i\ln p_i+(1-y_i)\ln(1-p_i)\}+\frac{1}{2C}\sum_{j=1}^{p}\beta_j^2\right]
```

Use fixed C=1, no tuning, no class weighting. For a synthetic score of 0.4 the sigmoid gives approximately 0.5987; use UP when p_i > 0.5. The estimated probabilities are not yet independently calibrated.

## 4. Ridge Regression



```math
\begin{aligned}
\widehat Y_i&=\beta_0+\sum_{j=1}^{p}\beta_jz_{ij}\\
\min_{\beta_0,\boldsymbol\beta}J&=\sum_{i=1}^{N}(Y_i-\widehat Y_i)^2+\lambda\sum_{j=1}^{p}\beta_j^2
\end{aligned}
```

For centered data, the normal equations and solution are:


```math
\begin{aligned}
(X^\top X+\lambda I)\hat{\boldsymbol\beta}&=X^\top\mathbf y\\
\hat{\boldsymbol\beta}&=(X^\top X+\lambda I)^{-1}X^\top\mathbf y
\end{aligned}
```

Use fixed alpha=1, fit on returns times 100 for percent units and convert predictions back to log-return units. The intercept is not penalized. A positive conditional mean need not imply an UP probability above 0.5. Do not interpret coefficients causally.

## 5. Chronological split and purge

Use Chapter XXIV's raw-return index boundaries (floor 60% and 80% of source_candles minus one). Feature anchors start at source candle index 24 and end at source_candles minus 13. Strictly exclude labels whose assumed availability is equal to or later than the next block's first decision. A constant replay delay cancels in the boundary inequality; for anchor index j, horizon h, boundary b:


```math
j+h<b
```

For 2,016 candles and 1,980 feature pairs, expected split counts:

| Horizon | Train | Validation | Development replay | Purged |
|---|---:|---:|---:|---:|
| 5m | 1184 | 402 | 392 | 2 |
| 1h | 1173 | 391 | 392 | 24 |

These are expected, not actual training results. The whole week has already been explored. The ML and previous TSA metrics use different eligible origins, so direct score ranking against Chapter XXIV is not valid without a matched-anchor comparison. Do not use this replay as an untouched test.

## 6. Evaluation

Compare Logistic to constant training prevalence (threshold >0.5 gives the training majority). Accuracy and ROC-AUC are higher-is-better; Brier and log loss are lower-is-better.


```math
\begin{aligned}
\mathrm{Brier}&=\frac1N\sum_i(p_i-y_i)^2\\
\mathrm{LogLoss}&=-\frac1N\sum_i[y_i\ln p_i+(1-y_i)\ln(1-p_i)]
\end{aligned}
```

Compare Ridge to zero return and training mean using MAE, RMSE and sign accuracy. AUC is null if an evaluation block contains only one class. Report both horizons separately; overlapping hourly outcomes are dependent. No costs, trades or win-rate claims.

## 7. Run

From repo root:
```bash
git pull origin main
python3 -m pip install -r requirements-ml.txt
python3 -m unittest tests.test_day08_supervised -v
```

If you have not already generated the features:
```bash
python3 -m examples.day05_features data/coinbase/BTC-USD_300_1794528000_1795132800 --output data/features/btc_5m_smoke --replay-delay-seconds 2
```

Then run:
```bash
python3 -m examples.day08_supervised data/features/btc_5m_smoke --output data/supervised/btc_5m_smoke
cat data/supervised/btc_5m_smoke/report.json
```

Use the actual feature directory, and a fresh output directory for reruns. Outputs include report.json and predictions.csv, gitignored. Five synthetic fixture tests cover split purge, feature loader, checksum tamper rejection, replay perturbation invariance and end-to-end classification/regression output. The real-data run remains pending.

## 8. Next research gate

Review baseline scores, then compare XGBoost with the exact same inputs, splits and anchors, using fixed settings or training-only tuning. Before serious model selection, collect longer authorized data and reserve a fresh chronological holdout. Later evaluate GARCH in a causal historical-fold ablation; do not copy full-sample fitted volatility paths into training. Historical availability, raw source authenticity, XAU coverage and trade results remain unverified. No live orders.

[Chapter XXIII](23_causal_feature_dataset.md) | [Chapter XXV](25_garch_volatility.md)


## 9. User-reported M2 results — 26 November 2026

Evidence: user-pasted local feature manifest and supervised report. Source hash matches previous reports. 1,980 pairs, 12 model inputs, observed split counts equal expected: 5m 1184/402/392 (2 purged), 1h 1173/391/392 (24 purged). Both horizons ran through the manifest-verifying loader. The assistant has not independently inspected raw CSVs or individual prediction rows. No final untouched holdout.

### Logistic Regression comparison

| Horizon / block | Model | Accuracy | Brier | Log loss | ROC-AUC |
|---|---|---:|---:|---:|---:|
| 5m validation | Logistic | 0.475124 | 0.252769 | 0.698777 | 0.447233 |
| 5m validation | Prevalence | 0.534826 | 0.249051 | 0.691249 | 0.500000 |
| 5m replay | Logistic | 0.510204 | 0.252050 | 0.697310 | 0.479878 |
| 5m replay | Prevalence | 0.525510 | 0.249397 | 0.691941 | 0.500000 |
| 1h validation | Logistic | 0.552430 | 0.255635 | 0.706623 | 0.479187 |
| 1h validation | Prevalence | 0.560102 | 0.246623 | 0.686381 | 0.500000 |
| 1h replay | Logistic | 0.487245 | 0.272544 | 0.741767 | 0.429625 |
| 1h replay | Prevalence | 0.553571 | 0.247208 | 0.687553 | 0.500000 |

Training UP rates: 0.4814189189 (5m), 0.4552429668 (1h). Logistic underperforms all four classification metrics in every block. Do not flip predicted classes or tune thresholds based on this inspected replay.

### Ridge Regression comparison

| Horizon / block | Model | MAE | RMSE | Sign accuracy |
|---|---|---:|---:|---:|
| 5m validation | Ridge | 0.001053991 | 0.001549562 | 0.519900 |
| 5m validation | Zero | 0.001049708 | 0.001549649 | 0.534826 |
| 5m validation | Train mean | 0.001045332 | 0.001549500 | 0.534826 |
| 5m replay | Ridge | 0.001346245 | 0.001851919 | 0.500000 |
| 5m replay | Zero | 0.001336176 | 0.001849067 | 0.525510 |
| 5m replay | Train mean | 0.001333881 | 0.001847566 | 0.525510 |
| 1h validation | Ridge | 0.003839955 | 0.005119213 | 0.544757 |
| 1h validation | Zero | 0.003304090 | 0.004551452 | 0.560102 |
| 1h validation | Train mean | 0.003283383 | 0.004543923 | 0.560102 |
| 1h replay | Ridge | 0.005178641 | 0.006432258 | 0.479592 |
| 1h replay | Zero | 0.004562075 | 0.006050837 | 0.553571 |
| 1h replay | Train mean | 0.004552407 | 0.005998628 | 0.553571 |

Training means: -0.0000721606507132 (5m), -0.000826011163926 (1h). Ridge loses to training mean across all four blocks on all reported metrics. Its marginal 5m validation RMSE improvement over zero return is not useful evidence of superiority. All returns are unscaled log returns.

### Coefficient observations and next decision

For 1h Logistic, standardized volatility coefficients are -0.278956 (std_return_6), -0.433606 (std_return_12), +0.729834 (std_return_24). For 5m the largest positive standardized coefficient is mean_return_6 (+0.149299). These are conditional fitted associations, not causal relationships or stable feature importance. Overlapping volatility windows are correlated, and the current short sample cannot distinguish changing relationships, weak signal and estimation noise. Poor out-of-sample performance alone does not prove overfitting, since training scores were not reported.

Keep M2 as negative-result baselines. GARCH remains a promising variance candidate only. Recommended next gate: extend authorized BTC history, with a fresh untouched chronological holdout and purged walk-forward research splits frozen before model selection. Keep this week for smoke testing. M3 XGBoost can be implemented and sanity-tested on this week with fixed shallow trees, then compared against Logistic/Ridge and no-skill baselines on identical anchors on longer data. Do not select a complex model merely because linear models failed. Later M4 causal GARCH and multi-timeframe ablations must earn inclusion. No directional advantage, XAU coverage, verified publication latency or trading results established.
