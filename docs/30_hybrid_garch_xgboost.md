# Chapter XXX — Causal GARCH + XGBoost hybrid experiment

**27 November 2026.** M4 synthetic-only software experiment. No real market-data downloads or training, no holdout, no trade execution. The data permission gate in Chapter XXVII remains active.

## 1. Hypothesis and architecture

Does a conditional volatility estimate add useful information to an otherwise identical XGBoost model? Compare the original 12-feature 5m model and the 21-feature MTF model, each without volatility, with a one-day rolling RMS, and with a GARCH forecast. The additional feature is a one-step standard deviation, even for the one-hour direction/return target. No one-hour variance forecast is claimed. The rolling RMS is intentionally a different window from the existing rolling standard deviations. No model promotion on synthetic scores.

## 2. Zero-mean GARCH and causal parameter estimation

Chapter XXV used a two-stage AR(1) mean with GARCH residuals. This controlled experiment deliberately fixes the mean at zero because both synthetic controls have zero conditional mean. Therefore the innovation is the log return. This is not a jointly fitted AR-GARCH model; a causal AR mean feature is deferred.


```math
\begin{aligned}
r_t&=\ln(C_t/C_{t-1})=\varepsilon_t\\
r_t&=\sigma_t z_t,\qquad E[z_t\mid\mathcal F_{t-1}]=0,\quad E[z_t^2\mid\mathcal F_{t-1}]=1\\
\sigma_{t+1}^2&=\omega+\alpha r_t^2+\beta\sigma_t^2\\
\omega&>0,\qquad\alpha,\beta\ge0,\qquad\alpha+\beta<1
\end{aligned}
```

At a refit origin t, W=1440 past observed returns (five days of 5m bars) are used. Initial estimation at the first eligible origin; refit every 2880 observed returns (ten days). No full-sample fitting for past rows. Normalize by the historical mean square and minimize Gaussian QML using Day 7's constrained optimizer and three initializations:


```math
\begin{aligned}
s_t^2&=\frac1W\sum_{i=t-W+1}^{t}r_i^2\\
u_i&=r_i/s_t,\qquad v_i=\sigma_i^2/s_t^2\\
\mathcal L_t(\omega',\alpha,\beta)&=\frac1{2W}\sum_{i=t-W+1}^{t}\left[\ln v_i+\frac{u_i^2}{v_i}\right]\\
\widehat\omega_t&=s_t^2\widehat\omega'_t
\end{aligned}
```

The normalized initial variance is 1; replay the fitted recursion over that historical window to get the next-step state. Between refits freeze parameters and update state only after the next return is observed. Floors use 1e-8 times historical fit variance. If a fit fails, mark forecasts unavailable until the next scheduled attempt; record the event and exclude those origins for every variant. Reset state and warmup at gaps and at the start of each block. Validation may update GARCH using its own already-observed returns; XGBoost parameters remain frozen. No validation-driven hyperparameter tuning. Real computation/publication latency is unknown; the two-second delay is illustrative and does not include measured optimizer time.

### Worked recursion (invented normalized units)



```math
\begin{aligned}
\omega&=0.1,\quad\alpha=0.2,\quad\beta=0.7\\
r_t&=2,\quad \sigma_t^2=1\\
\widehat\sigma_{t+1\mid t}^2&=0.1+0.2(2^2)+0.7(1)=1.6\\
\widehat\sigma_{t+1\mid t}&=\sqrt{1.6}\approx1.265
\end{aligned}
```

Do not substitute the unknown future return into this forecast.

## 3. Comparison volatility features



```math
\begin{aligned}
\widehat\sigma_{t+1\mid t}^{\mathrm{rolling}}&=\sqrt{\frac1{288}\sum_{i=0}^{287}r_{t-i}^2}\\
\widehat\sigma_{t+1\mid t}^{\mathrm{GARCH}}&=\sqrt{\widehat\omega+\widehat\alpha r_t^2+\widehat\beta\widehat\sigma_t^2}\\
\widehat\sigma_{t+1\mid t}^{\mathrm{constant}}&=s_{t_f}
\end{aligned}
```

Here t_f is the most recent successful refit time. All features are in unscaled log-return standard-deviation units. RMS is appropriate for a zero-mean control, and differs from de-meaned sample standard deviation. The constant benchmark changes only at scheduled refits; it is not included in XGBoost inputs. We compare rolling, GARCH and refit-constant forecasts directly on next squared return:


```math
\begin{aligned}
q_{t+1}&=r_{t+1}^2\\
\mathrm{QLIKE}&=\frac1N\sum_t\left[\ln\widehat v_{t+1\mid t}+\frac{q_{t+1}}{\widehat v_{t+1\mid t}}\right]\\
\mathrm{MSE}_{v}&=\frac1N\sum_t(q_{t+1}-\widehat v_{t+1\mid t})^2,\qquad \widehat v=\widehat\sigma^2
\end{aligned}
```

The squared return proxy is noisy. A successful variance forecast need not predict the sign or conditional mean.

## 4. Matched hybrid feature vectors



```math
\begin{aligned}
X_t^{\mathrm{MTF}}&=\begin{bmatrix}X_t^{5m}\\X_t^{1h}\\X_t^{4h}\\X_t^{1D}\end{bmatrix}\\
X_t^{\mathrm{hybrid}}&=\begin{bmatrix}X_t^{\mathrm{MTF}}\\\widehat\sigma_{t+1\mid t}^{\mathrm{GARCH}}\end{bmatrix}\\
\widehat p_{t,h}&=\operatorname{sigmoid}(F_h^{\mathrm{class}}(X_t^{\mathrm{hybrid}}))\\
\widehat Y_{t,h}&=F_h^{\mathrm{reg}}(X_t^{\mathrm{hybrid}}),\quad h\in\{1,12\}
\end{aligned}
```

All six variants share 5m/MTF and variance eligibility. There are 23 distinct candidate input columns; one extra constant-variance column is used only for evaluation. The largest model uses 22 features.

| Variant | Model inputs | Count |
|---|---|---:|
| 5m_only | Base | 12 |
| 5m_rolling | Base + rolling RMS | 13 |
| 5m_garch | Base + GARCH | 13 |
| mtf_only | All timeframes | 21 |
| mtf_rolling | MTF + rolling RMS | 22 |
| mtf_garch | MTF + GARCH | 22 |

Same fixed XGBoost parameters as Chapter XXVIII; six variants x two horizons x two tasks = 24 fits per control. Compare accuracy, Brier, log loss, AUC, MAE, RMSE and sign accuracy; prevalence, zero return and training mean are no-skill baselines. Compare base and volatility additions on the same origins, not with previous larger-sample reports.

## 5. Negative and positive controls

Negative: the Chapter XXVII audited independent-innovation synthetic blocks (69,984 and 17,568 candles). Positive: a separately generated zero-mean GARCH process, seed 20261127, 1000 burn-in returns, 30 days train before April 1 2026 and 20 days validation from April 1. No holdout generated. Known simulator parameters:


```math
\begin{aligned}
z_t&\overset{\mathrm{iid}}{\sim}N(0,1),\quad r_t=\sqrt{v_t}z_t\\
v_{t+1}&=10^{-7}+0.12r_t^2+0.85v_t\\
v_\infty&=\frac{10^{-7}}{1-0.12-0.85}=3.3333\ldots\times10^{-6}
\end{aligned}
```

The positive control contains predictable conditional variance but deliberately no conditional direction or mean predictability. Its oracle variance is used only to score forecast losses, never as a model input. Even the oracle is not guaranteed to win a finite-sample squared-return loss comparison. The two controls are different simulated samples; compare models within each, not raw losses between them. No real-data permission granted by simulation.

The first eligible origin after five-day GARCH warmup has index 1440 (requiring 1441 candles), and the final 13 origins are reserved for strict 1h target availability on complete aligned blocks. Expected counts if every fit succeeds:


```math
\begin{aligned}
N_{\mathrm{eligible}}&=N_{\mathrm{candles}}-1440-13\\
N_{\mathrm{neg,train}}&=69984-1453=68531\\
N_{\mathrm{neg,val}}&=17568-1453=16115\\
N_{\mathrm{pos,train}}&=8640-1453=7187\\
N_{\mathrm{pos,val}}&=5760-1453=4307
\end{aligned}
```

Actual counts may be smaller if a refit fails. Daily warmup is shorter than the five-day GARCH warmup for these aligned blocks.

## 6. Run on Mac

```bash
git pull origin main
python3 -m unittest tests.test_day12_hybrid -v
python3 -m examples.day12_hybrid data/history/synthetic_1y_plan --output data/hybrid/synthetic_v1 --replay-delay-seconds 2
cat data/hybrid/synthetic_v1/report.json
```

Existing requirements-xgboost.txt includes SciPy, statsmodels and XGBoost via its nested requirements. No new dependencies. This experiment has more fits and periodic GARCH optimization; allow it to finish in the terminal. Outputs: report.json, negative_predictions.csv and positive_predictions.csv under gitignored data/. New output directory required. Tests are provided for recursion, future perturbation, fit failure, gap/reset and matched-model output. Direct container clone failed (DNS); new tests and CLI have not been independently executed. Run the unit tests before the experiment and report failures rather than assuming success.

## 7. Acceptance and next gate

Inspect counts, failed-fit logs, variance QLIKE/MSE on both controls, and matched directional/return metrics separately. A variance positive control tests variance learnability, not directional learnability. Do not tune or promote a model based on a single synthetic validation. A separate direction positive control can follow. Next candidate is training-only market-regime discovery (K-Means/GMM), with causal features and matched evaluation. Actual market-data licensing, untouched real holdout, measured feed latency, cost-aware prospective paper evaluation and XAU coverage remain open. No live orders.

[Chapter XXV](25_garch_volatility.md) | [Chapter XXIX](29_multitimeframe.md) | [Mathematical reference](mathematical_reference.md)


## 8. User-reported hybrid results — 27 November 2026

Local report supplied by the user. Expected counts and source block hashes match; negative control 68,531/16,115 train/validation pairs, positive control 7,187/4,307. Reported refits: 30 negative, five positive, zero failures. No holdout exported. This is evidence of a successful local CLI run, not independent inspection of raw data, unit-test output or individual predictions.

### Variance evaluation

| Control | Forecast | QLIKE | Variance MAE | Variance MSE |
|---|---|---:|---:|---:|
| Negative | Rolling RMS | -12.568774 | 1.242981e-6 | 3.347986e-12 |
| Negative | GARCH | -12.571331 | 1.236344e-6 | 3.339970e-12 |
| Negative | Refit constant | -12.571573 | 1.235977e-6 | 3.339066e-12 |
| Positive | Rolling RMS | -11.664938 | 3.400446e-6 | 3.335467e-11 |
| Positive | GARCH | -11.826310 | 3.058234e-6 | 2.795035e-11 |
| Positive | Refit constant | -11.602008 | 3.195871e-6 | 3.390648e-11 |
| Positive | Simulator oracle | -11.828890 | 3.089903e-6 | 2.791022e-11 |

Positive control: estimated GARCH reduces variance MSE by 17.5664% against refit constant and 16.2026% against rolling RMS. Its QLIKE is 0.224301 and 0.161372 lower, respectively. GARCH MSE is close to the oracle; GARCH happens to have lower variance MAE than the oracle on this finite sample. Squared-return proxy loss is noisy, and conditional variance is the squared-loss optimal forecast, not generally the absolute-loss optimal forecast. Negative control: constant wins all three variance losses; fitted GARCH alpha equals zero in 10 of 30 refits. Do not interpret nonzero fitted beta in these windows as evidence of true clustering.

### Parameter recovery (positive control)

| Parameter | Generating value | Arithmetic mean across five local refits |
|---|---:|---:|
| omega | 1e-7 | 9.241587e-8 |
| alpha | 0.12 | 0.116684 |
| beta | 0.85 | 0.852720 |
| alpha + beta | 0.97 | 0.969403 |

This descriptive mean is not an estimator with reported confidence bounds. Individual fits differ across windows. Successful optimizer status does not prove global optimality.

### Matched XGBoost direction/return evaluation

Accuracy and RMSE below summarize all variants; return units are unscaled log returns.

| Variant | Negative 5m accuracy | Negative 1h accuracy | Positive 5m accuracy | Positive 1h accuracy |
|---|---:|---:|---:|---:|
| Prevalence | 0.495625 | 0.484890 | 0.511261 | 0.498723 |
| 5m_only | 0.498852 | 0.491468 | 0.496866 | 0.501741 |
| 5m_rolling | 0.497983 | 0.490413 | 0.499420 | 0.506153 |
| 5m_garch | 0.497549 | 0.491095 | 0.498723 | 0.503599 |
| mtf_only | 0.506857 | 0.516910 | 0.497330 | 0.488507 |
| mtf_rolling | 0.502203 | 0.514179 | 0.498259 | 0.494312 |
| mtf_garch | 0.505430 | 0.514614 | 0.493151 | 0.494079 |

| Variant | Negative 5m RMSE | Negative 1h RMSE | Positive 5m RMSE | Positive 1h RMSE |
|---|---:|---:|---:|---:|
| Zero return | 0.001129074 | 0.003938280 | 0.001784405 | 0.005796846 |
| Training mean | 0.001129077 | 0.003938415 | 0.001784304 | 0.005793091 |
| 5m_only | 0.001129137 | 0.003944559 | 0.001792914 | 0.005787959 |
| 5m_rolling | 0.001129139 | 0.003948868 | 0.001791898 | 0.005818172 |
| 5m_garch | 0.001129220 | 0.003947371 | 0.001793712 | 0.005793181 |
| mtf_only | 0.001129154 | 0.003938832 | 0.001790994 | 0.005829231 |
| mtf_rolling | 0.001129108 | 0.003940072 | 0.001790912 | 0.005862798 |
| mtf_garch | 0.001129168 | 0.003939409 | 0.001791741 | 0.005827138 |

Positive control prevalence has best Brier and log loss at both horizons. Negative-control MTF-only achieves best 1h classification accuracy (0.516910), AUC (0.529810), Brier (0.249534) and log loss (0.692217); adding GARCH worsens each of these. Positive-control 5m-only has lowest 1h MAE/RMSE; neither GARCH addition beats it. Small differences do not establish reliable conditional direction or mean skill. Hourly outcomes overlap. No model promotion.

### Decision and mathematical interpretation

Under the known symmetric zero-mean simulator, the next return satisfies:

```math
\begin{aligned}
r_{t+1}&=\sqrt{v_{t+1}}z_{t+1},\quad z_{t+1}\sim N(0,1)\ \text{independent of }\mathcal F_t\\
E[r_{t+1}\mid\mathcal F_t]&=0\\
P(r_{t+1}>0\mid\mathcal F_t)&=\tfrac12\\
E[r_{t+1}^2\mid\mathcal F_t]&=v_{t+1}\\
\mathrm{MSE\ reduction}&=100\left(1-\frac{\mathrm{MSE}_{\mathrm{GARCH}}}{\mathrm{MSE}_{\mathrm{benchmark}}}\right)\%
\end{aligned}
```

Conditional variance can be predicted without conditional direction being predictable. Symmetry of future innovations also implies a symmetric aggregate return distribution in this zero-mean GARCH simulator. This is a simulator property, not an assertion about BTC/USD. Retain GARCH as a variance-research candidate and future risk/uncertainty input, not as an established directional feature. Next Chapter XXXI: training-only K-Means/GMM regime features, with train-only scaling and held-fixed fitted transforms on validation, matched feature ablations, and a separately generated known-regime/direction positive control before claiming learnability of directional structure. Do not assign economic meanings to arbitrary cluster IDs. Real-data authorization, XAU coverage, measured latency, cost-aware evaluation and untouched real holdout remain unresolved.

## Follow-up

[Chapter XXXI](31_market_regimes.md) implements training-only K-Means/GMM features and known-direction synthetic control, with GARCH held out for an isolated regime ablation. Local full CLI pending.
