# Chapter XXXII — Combined hybrid factorial ablation

**27 November 2026.** Synthetic-only research; no real-data training, holdout access or trading. Data rights gate remains active.

## 1. Matched factorial feature design

Fix the original 21 MTF inputs and both forecast horizons. Volatility factor V has levels none, rolling RMS and GARCH. Regime factor G has levels none, K-Means one-hot and GMM responsibilities. Reuse the audited Day 10 loader, Day 12 gap-aware historical volatility fitting (1,440-return window, 2,880-return refit interval, 288-return RMS) and Day 13 train-only cluster fits. No validation fitting of the scaler/clusters/XGBoost. Historical GARCH state updates and scheduled fits use already observed returns, including within validation. The same assumed two-second availability delay is not measured real latency.



```math
\begin{aligned}
X_t^{\mathrm{MTF}}&=\begin{bmatrix}X_t^{5m}\\X_t^{1h}\\X_t^{4h}\\X_t^{1D}\end{bmatrix}\\
X_t^{\mathrm{GARCH,GMM}}&=\begin{bmatrix}X_t^{5m}\\X_t^{1h}\\X_t^{4h}\\X_t^{1D}\\\widehat\sigma_{t+1\mid t}\\\gamma_{t,1}\\\gamma_{t,2}\\\gamma_{t,3}\end{bmatrix}\\
d(V,G)&=21+\mathbf1\{V\ne0\}+3\mathbf1\{G\ne0\}
\end{aligned}
```

Here the GARCH sigma is a raw RMS proxy when mean is nonzero. All nine variants use identical eligible anchors and targets, restricted to full MTF and successfully fitted volatility features. This is an ablation conditional on common eligibility; a failed GARCH refit can reduce coverage for every variant, including baseline. Do not compare unmatched earlier chapter scores.

| Volatility | None | K-Means | GMM |
|---|---|---|---|
| None | none_none (21) | none_kmeans (24) | none_gmm (24) |
| Rolling | rolling_none (22) | rolling_kmeans (25) | rolling_gmm (25) |
| GARCH | garch_none (22) | garch_kmeans (25) | garch_gmm (25) |

K-Means and GMM use the same four MTF features as Chapter XXXI; they do not consume the added volatility estimate or state truth. Each variant is a separate XGBoost classifier and regressor for each horizon with the unchanged 100-tree settings. Nine variants times two tasks times two horizons equals 36 fits per control. Four controls mean 144 XGBoost fits in total. No early stopping, selection or hyperparameter tuning.

## 2. Factorial interaction contrasts

For matched origins, evaluate Brier and mean squared return error, where y is a binary target, p a probability, Y a log-return target, and Yhat a return forecast.



```math
\begin{aligned}
L^{\mathrm{Brier}}_{V,G}&=\frac1N\sum_i(p_{i,V,G}-y_i)^2\\
L^{\mathrm{MSE}}_{V,G}&=\frac1N\sum_i(\widehat Y_{i,V,G}-Y_i)^2\\
\Delta_{V,G}&=L_{V,G}-L_{0,0}\\
I_{V,G}&=L_{V,G}-L_{V,0}-L_{0,G}+L_{0,0}
\end{aligned}
```

Zero denotes no additional feature group. Negative delta favors the augmented variant over MTF alone. Negative interaction denotes an improvement beyond the sum of the separate reductions on the observed sample; positive interaction means less improvement (or more degradation) than additivity predicts. It is a descriptive contrast, not a causal treatment effect or statistical significance result. Overlapping hourly labels are dependent.

Worked example in arbitrary loss units: L00=10, LV0=9, L0G=8, LVG=6. Delta=6-10=-4 and interaction=6-9-8+10=-1. Both separate gains are present, and the combination improves by one further loss unit. The report stores four interaction contrasts for each loss and horizon.

## 3. Four synthetic controls

Use independent-return negative control, the Chapter XXX zero-mean GARCH control, the Chapter XXXI Markov directional control, and a new joint mean/variance control. Joint seed 20261129, 40 training days and 20 validation days around April 1, 2026 UTC. The state transition and mean values are the same as Chapter XXXI; the innovation process has the Chapter XXX GARCH parameters. Initial variance is unconditional and initial state is neutral, with no separate burn-in for this new control.



```math
\begin{aligned}
A&=\begin{bmatrix}.995&.0025&.0025\\.0025&.995&.0025\\.0025&.0025&.995\end{bmatrix}\\
\boldsymbol\mu&=(-0.0007,0,0.0007)\\
r_t&=\mu_{S_t}+e_t,\qquad e_t=\sqrt{v_t}z_t,\quad z_t\overset{\mathrm{iid}}{\sim}N(0,1)\\
v_{t+1}&=10^{-7}+0.12e_t^2+0.85v_t\\
v_0&=\frac{10^{-7}}{1-0.12-0.85}\\
C_t&=C_{t-1}\exp(r_t)
\end{aligned}
```

The simulator's latent state is scoring-only for ARI, never an input. The oracle variance is scored only for the earlier zero-mean GARCH control. No oracle second moment is provided for the joint control. The source block hashes identify the negative control; the other three controls are generated in memory. They are distinct simulated samples, not market data or independent estimates of actual trading performance.

## 4. Conditional variance versus raw second moment



```math
\begin{aligned}
E[r_{t+1}^2\mid\mathcal F_t]&=\operatorname{Var}(r_{t+1}\mid\mathcal F_t)+\left(E[r_{t+1}\mid\mathcal F_t]\right)^2\\
q_{t+1}&=r_{t+1}^2\\
L_{\mathrm{QLIKE}}&=\frac1N\sum_t\left(\ln\widehat q_{t+1\mid t}+\frac{q_{t+1}}{\widehat q_{t+1\mid t}}\right)\\
L_{\mathrm{MSE},q}&=\frac1N\sum_t(q_{t+1}-\widehat q_{t+1\mid t})^2
\end{aligned}
```

The reused Day 12 zero-mean GARCH recursion uses raw returns instead of mean-adjusted innovations. For nonzero conditional means it is a potentially misspecified second-moment proxy, not a correctly specified conditional variance forecast. Rolling RMS and the refit-window constant also estimate raw second moment after squaring. The report compares these using squared-return QLIKE/MSE/MAE. A correctly specified conditional mean and residual model is deferred. No one-hour variance forecast is supplied. Variance prediction and direction prediction remain distinct.

## 5. Expected counts and interpretation

Five-day GARCH warmup dominates four-day MTF warmup for complete aligned blocks; 13 final origins are excluded by strict 1h label availability. With all fits successful:



```math
\begin{aligned}
N_{\mathrm{eligible}}&=N_{\mathrm{candles}}-1440-13\\
N_{\mathrm{negative}}&=(68531,16115)\\
N_{\mathrm{variance}}&=(7187,4307)\\
N_{\mathrm{direction}}&=(10067,4307)\\
N_{\mathrm{joint}}&=(10067,4307)\\
N_{\mathrm{fits}}&=3\cdot3\cdot2\cdot2\cdot4=144
\end{aligned}
```

Pairs denote training and validation counts. Actual counts can be smaller if fits fail. Compare probability losses, accuracy, AUC, MAE, RMSE, sign accuracy and factorial contrasts within each control. Training prevalence, zero-return and training-mean forecasts remain benchmarks. State recovery ARI is meaningful only for controls with state labels. Do not choose a production variant based on four particular synthetic paths.

## 6. Run and verification

Existing requirements-xgboost.txt is sufficient. From repository root:

```bash
git pull origin main
python3 -m unittest tests.test_day14_factorial -v
python3 -m examples.day14_factorial data/history/synthetic_1y_plan --output data/factorial/synthetic_v1 --replay-delay-seconds 2
cat data/factorial/synthetic_v1/report.json
```

Five isolated tests rerun successfully in the assistant container: exact feature layouts, invalid input rejection, interaction arithmetic, deterministic joint generation, and matched small XGBoost fits (three trees, fixture encodings rather than the actual clustering fitter). NumPy 2.3.5, scikit-learn 1.8.0, XGBoost 3.1.3. This does not verify full upstream loader/MTF/GARCH/clustering integration or the full 144-fit experiment. Local Mac tests and full CLI pending. No additional downloads required for synthetic inputs. Output paths under gitignored data/: report.json and four prediction CSVs. New directory required; if present, read existing report or use another name. No saved deployable model.

## 7. Next gate

After reviewing the local report, implement purged walk-forward evaluation across multiple independently seeded paths, with each fold's transforms and training fitted only from its historical block. Predefine selection criteria and keep separate evaluation periods; no reuse of inspected synthetic validation as final evidence. Adaptive routing, calibration, risk/transaction-cost modeling, real-data rights, measured availability, XAU coverage and prospective paper evaluation remain pending. No live orders.

[Chapter XXXI](31_market_regimes.md) | [Chapter XXX](30_hybrid_garch_xgboost.md) | [Mathematical reference](mathematical_reference.md)
