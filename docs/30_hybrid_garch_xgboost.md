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
