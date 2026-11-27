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


## 8. User-reported factorial results — 27 November 2026

The supplied local report (synthetic factorial controls only) matches the fixed settings, nine configurations, source block checksums and expected eligible counts. All 144 scheduled XGBoost fits have corresponding reported metrics. Refits: negative 30, variance 5, direction 6, joint 6; total 47, zero failures. No holdout exported. This is report evidence, not independent verification of individual prediction rows, raw candles or Mac unit-test output.

| Control | Train | Validation | Refits |
|---|---:|---:|---:|
| Negative | 68531 | 16115 | 30 |
| Variance | 7187 | 4307 | 5 |
| Direction | 10067 | 4307 | 6 |
| Joint | 10067 | 4307 | 6 |

### Best observed scores, including benchmarks

| Control / horizon | Highest accuracy (fraction) | Lowest Brier | Lowest RMSE |
|---|---|---|---|
| Negative 5m | none_kmeans .507229 | prevalence .250003 | zero_return .001129074 |
| Negative 1h | none_gmm .521812 | none_gmm .249534 | none_gmm .003938199 |
| Variance 5m | prevalence .511261 | prevalence .249878 | training_mean .001784304 |
| Variance 1h | prevalence .498723 | prevalence .250833 | training_mean .005793091 |
| Direction 5m | none_none / rolling_none .621314 (tie) | none_gmm .221318 | garch_gmm .001949277 |
| Direction 1h | garch_gmm .716276 | none_gmm .154568 | garch_gmm .006851163 |
| Joint 5m | rolling_gmm .588112 | none_none / none_kmeans .237613 (tie) | none_none / none_kmeans .001726535 (tie) |
| Joint 1h | rolling_none .680288 | rolling_none .192140 | none_none .007076154 |

These retrospective minima are descriptive, not a validated selection rule. All regression RMSEs are unscaled log-return units. Variance-only prevalence and training mean win on both horizons for Brier and RMSE respectively. The directional control has best 1h AUC .806180 (garch_gmm); none_gmm has the best Brier/log loss. In the joint control, rolling_gmm has best 1h AUC .777633, but rolling_none has best accuracy/Brier/log loss. Different metrics need not select the same model. Negative-control deviations from chance are compatible with finite-sample variability; many variants and overlapping labels make naive independent-observation inference inappropriate.

### Full nine-variant results on the joint control

| Variant | 5m accuracy | 5m Brier | 5m RMSE | 1h accuracy | 1h Brier | 1h RMSE |
|---|---:|---:|---:|---:|---:|---:|
| none_none | .581844 | .237613 | .001726535 | .677502 | .192440 | .007076154 |
| none_kmeans | .581844 | .237613 | .001726535 | .677966 | .192761 | .007080412 |
| none_gmm | .582540 | .237724 | .001727233 | .679127 | .193351 | .007103856 |
| rolling_none | .585558 | .237696 | .001727333 | .680288 | .192140 | .007081345 |
| rolling_kmeans | .585558 | .237696 | .001727333 | .677502 | .192262 | .007085604 |
| rolling_gmm | .588112 | .237690 | .001727144 | .676109 | .192243 | .007113906 |
| garch_none | .579754 | .237640 | .001727198 | .677502 | .192440 | .007080150 |
| garch_kmeans | .579754 | .237640 | .001727198 | .677966 | .192761 | .007077675 |
| garch_gmm | .581379 | .237654 | .001727237 | .679127 | .193351 | .007101987 |
| Prevalence | .506617 | .250120 | — | .495240 | .252380 | — |
| Zero return | — | — | .001781245 | — | — | .008781343 |
| Training mean | — | — | .001784105 | — | — | .008865635 |

Joint 1h classifier metrics exactly match between each no-volatility variant and its GARCH-augmented counterpart in the report. Identical aggregate scores do not prove identical trees or individual predictions; inspect prediction rows before drawing stronger conclusions. No model promotion.

### Second-moment performance

| Control | Rolling QLIKE | GARCH QLIKE | Constant QLIKE | Rolling MSE | GARCH MSE | Constant MSE |
|---|---:|---:|---:|---:|---:|---:|
| Negative | -12.568774 | -12.571331 | -12.571573 | 3.347986e-12 | 3.339970e-12 | 3.339066e-12 |
| Variance | -11.664938 | -11.826310 | -11.602008 | 3.335467e-11 | 2.795035e-11 | 3.390648e-11 |
| Direction | -11.439396 | -11.563645 | -11.306662 | 4.719729e-11 | 4.500426e-11 | 4.957803e-11 |
| Joint | -11.649586 | -11.783411 | -11.654290 | 2.992671e-11 | 2.675538e-11 | 2.992718e-11 |

Variance-control oracle QLIKE -11.828890 and MSE 2.791022e-11. Constant wins on negative control. GARCH has lower second-moment losses than both practical benchmarks on all three structured controls, with joint MSE reduction about 10.60% vs constant. Nonzero-mean controls use raw squared returns, so these are second-moment results, not proof of correctly specified residual conditional variance. Reused negative and variance controls are not independent new experiments.

### Regime recovery

| Control | K-Means ARI | GMM ARI |
|---|---:|---:|
| Direction | .787153 | .826472 |
| Joint | .170666 | .167556 |

The joint generator uses the same innovation variance process across the three hidden mean regimes. Trend and volatility summaries can overlap strongly, and clusters can separate volatility levels rather than recover latent mean states. This is a plausible explanation for weak ARI, not a uniquely established cause. Do not equate cluster probabilities with direction probabilities.

### Interaction interpretation

| Control / horizon | GARCH+GMM Brier interaction | GARCH+GMM MSE interaction |
|---|---:|---:|
| Negative 5m | -1.497973e-6 | -1.284534e-11 |
| Negative 1h | -1.646166e-5 | 1.567694e-9 |
| Variance 5m | 5.469043e-5 | 1.755650e-9 |
| Variance 1h | .000216226 | -6.356093e-8 |
| Direction 5m | -3.244148e-5 | -1.499284e-9 |
| Direction 1h | -.000331788 | 8.271719e-8 |
| Joint 5m | -9.742773e-5 | -2.278908e-9 |
| Joint 1h | 0 | -8.312672e-8 |

A negative interaction can occur even when the combined model is worse than the baseline. Joint 1h Brier interaction is exactly zero for GARCH+GMM; Brier delta versus MTF is +.000911159. For joint rolling+GMM, Brier interaction -.000808265 but delta -.000196832, which is a smaller reduction than rolling alone (-.000299727). Interaction is not a model promotion criterion.

### Research decision

Retain MTF as the main candidate baseline; preserve all nine fixed candidates rather than choose a winner from these inspected simulations. Keep GARCH as a volatility/second-moment research candidate; no demonstrated universal directional benefit. Chapter XXXIII should implement purged chronological walk-forward on fresh independently seeded synthetic paths, with train-only scaling/clustering and supervised fits, label availability strictly before each training cutoff, disjoint evaluation origins, and causal GARCH updates. Report fold-level probability/return losses, matched baselines, variability, coverage and failure counts. Separate candidate selection from later evaluation; do not tune on evaluation blocks. Explicitly distinguish dependent folds and overlapping outcomes from independent simulation paths. No real-data permission, XAU coverage, measured availability, execution costs, final market holdout or prospective paper trading established.

## Follow-up

[Chapter XXXIII](33_purged_walkforward.md) implements fresh seeded purged walk-forward selection and separate assessment. Full local execution pending.
