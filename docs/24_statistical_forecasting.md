# Chapter XXIV — Statistical forecasting and baseline models

**26 November 2026.** Development replay only; not an untouched backtest. Five synthetic fixture tests passed in the assistant environment (NumPy 2.3.5, statsmodels 0.14.6). User's BTC dataset is not available to the assistant. No observed forecast accuracy or trade win rate yet.

## 1. Goal and notation

Compare four return-forecast candidates on the same previously collected 5m BTC/USD series. Let $C_t$ be the completed close, $z_t$ the log price, and $r_t$ the one-bar log return. One step means five minutes.

```math
\begin{aligned}
z_t&=\ln(C_t)\\
r_t&=z_t-z_{t-1}\\
Y_{t,h}&=\sum_{k=1}^{h}r_{t+k}=\ln\left(\frac{C_{t+h}}{C_t}\right)
\end{aligned}
```

The horizon is one or twelve bars (5m or 1h). Forecasts estimate conditional means; a positive conditional mean is not necessarily a probability above 50%.

## 2. M0 — Driftless log-price random walk / zero-return forecast


```math
\begin{aligned}
z_{t+1}&=z_t+\varepsilon_{t+1}\\
E(\varepsilon_{t+1}\mid\mathcal F_t)&=0\\
\widehat Y_{t,h}&=0
\end{aligned}
```

This is a no-skill return baseline. The log-price forecast equals the current log price; the corresponding price reference is the current close, not necessarily the conditional expected price after exponentiation. A separate training-majority UP/NOT_UP classifier is a directional benchmark. Zero-return forecasts map to NOT_UP; that does not imply an instruction to sell.

## 3. M1.1 — AR(1)


```math
\begin{aligned}
r_t&=c+\beta_1r_{t-1}+\varepsilon_t\\
\mu&=\frac{c}{1-\beta_1},\qquad |\beta_1|<1\\
r_t-\mu&=\beta_1(r_{t-1}-\mu)+\varepsilon_t\\
\widehat r_{t+1\mid t}&=\hat c+\hat\beta_1r_t
\end{aligned}
```

For the usual stationary AR(1) with white-noise innovations, the mean follows by taking expectation on both sides, yielding the stated relationship. An introductory OLS derivation uses the regression of current return on lagged return:


```math
\begin{aligned}
x_i&=r_{i-1},\quad y_i=r_i\\
\hat\beta_1&=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sum_i(x_i-\bar x)^2}\\
\hat c&=\bar y-\hat\beta_1\bar x
\end{aligned}
```

The implementation instead fits all ARMA candidates by statsmodels' state-space maximum likelihood for consistent comparison. In its ARIMA parameterization, const is the unconditional mean of the modeled return, not the AR intercept. Percent-unit const divided by 100 gives the mean log return. AR coefficients are dimensionless.

## 4. M1.2 — MA(1)


```math
\begin{aligned}
r_t&=\mu+\varepsilon_t+\theta_1\varepsilon_{t-1}\\
\widehat r_{t+1\mid t}&=\hat\mu+\hat\theta_1\hat\varepsilon_t
\end{aligned}
```

This uses the latest estimated innovation, not the average of the previous prices. At horizons beyond one bar the conditional mean reverts to the unconditional mean in the ideal MA(1) model.

## 5. M1.3 — ARMA(1,1)


```math
\begin{aligned}
r_t-\mu&=\beta_1(r_{t-1}-\mu)+\varepsilon_t+\theta_1\varepsilon_{t-1}\\
\widehat r_{t+1\mid t}&=\hat\mu+\hat\beta_1(r_t-\hat\mu)+\hat\theta_1\hat\varepsilon_t
\end{aligned}
```

The one-step forecast is obtained by the state-space filter rather than manually approximating the innovation. Fitted parameters are frozen at the training boundary. For subsequent horizons, future innovations have zero conditional mean:


```math
\begin{aligned}
m_1&=\widehat r_{t+1\mid t}\\
m_k&=\hat\mu+\hat\beta_1^{k-1}(m_1-\hat\mu)\\
\widehat Y_{t,h}&=\sum_{k=1}^h m_k
\end{aligned}
```

The same recursion applies to AR(1) and to MA(1) with AR coefficient zero. For an illustrative AR(1), not fitted market results:


```math
\begin{aligned}
\hat\mu&=0.0001,\quad \hat\beta_1=0.2,\quad r_t=0.001\\
\widehat r_{t+1\mid t}&=0.0001+0.2(0.001-0.0001)\\
&=0.00028\\
\widehat r_{t+2\mid t}&=0.0001+0.2(0.00028-0.0001)\\
&=0.000136
\end{aligned}
```

## 6. Evaluation

The entire seven-day series was already explored during EDA. Divide returns chronologically into 60% estimation, 20% validation and 20% development replay; neither later block is an untouched final holdout. Fit each candidate once on the first block. Filter sequentially with frozen parameters; a forecast for the next return uses only earlier observed returns. Do not smooth, refit, or use future observations to compute a forecast. For each horizon retain only origins whose whole outcome lies within its assigned evaluation block; drop boundary-crossing labels. Constant assumed publication delay is not verified historical availability. This implementation is an initial fixed-parameter experiment, not repeated walk-forward re-estimation. A future protocol will freeze split dates and purge overlapping outcome windows before model selection on substantially longer licensed history.

Compare forecast error:


```math
\begin{aligned}
\mathrm{MAE}&=\frac1N\sum_{i=1}^N|Y_i-\widehat Y_i|\\
\mathrm{RMSE}&=\sqrt{\frac1N\sum_{i=1}^N(Y_i-\widehat Y_i)^2}\\
\mathrm{Accuracy}&=\frac1N\sum_{i=1}^N\mathbf1\{\mathbf1(Y_i>0)=\mathbf1(\widehat Y_i>0)\}
\end{aligned}
```

Directional accuracy is not trading win rate. The sign of the mean forecast is only a simple direction rule. The independent training-majority benchmark uses training labels for each horizon. No probabilities, costs, trades or profitability claims are produced here.

Training diagnostics: AIC and BIC (compare only fitted models on the same training data), Ljung-Box at lag 12 with fitted AR/MA degrees of freedom removed, and ARCH-LM at lag 12 on training residuals after dropping 12 initialization observations. Read residual dependence and conditional variance as diagnostic clues. Optimizer nonconvergence is reported as a failed candidate, not silently accepted.

## 7. Run the experiment

From repo root after git pull:

```bash
python3 -m pip install -r requirements-eda.txt
python3 -m unittest tests.test_day06_statistical_baselines -v
python3 -m examples.day06_statistical_baselines data/coinbase/BTC-USD_300_1794528000_1795132800 --output data/baselines/btc_5m_smoke --replay-delay-seconds 2
```

Use your actual dataset directory and a fresh output directory. The script reads the audited close series directly; it does not require feature CSVs. It does not validate completion of the prior feature-generation step. The explicit two-second assumption matches the earlier smoke-test example and is not measured latency.

Outputs under gitignored data/baselines/: report.json (parameters, convergence failures, diagnostics, metrics, source audit) and predictions.csv (reference anchor, assumed decision, target close and actual/forecast return). For 2,015 uninterrupted returns, expect 1,209 training returns, 403 validation and 403 replay returns. Each successful model is evaluated on 403 one-step and 392 twelve-step forecasts in each later block.

| Experiment | MAE | RMSE | Direction accuracy |
|---|---|---|---|
| M0 zero return | Pending | Pending | Pending |
| M1.1 AR(1) | Pending | Pending | Pending |
| M1.2 MA(1) | Pending | Pending | Pending |
| M1.3 ARMA(1,1) | Pending | Pending | Pending |

Report results separately by horizon and evaluation block. An ARMA failure or lack of improvement is a valid result. If residual ARCH evidence motivates it, compare rolling volatility with GARCH next. Then compare Logistic Regression/Ridge and XGBoost using causal features, not the future labels as inputs.

## 8. Limitations and next gate

Five synthetic fixture tests passed locally in the assistant environment, including future-perturbation invariance for AR(1), horizon aggregation, metric arithmetic, split size and gap rejection. No real-data forecast has been executed by the assistant. Seven days are insufficient for evidence of stable forecasting skill; overlapping one-hour outcomes are dependent. The raw historical publication times remain unknown, and longer authorized history, prospective evaluation, XAU data, and Mac execution are pending. No order execution.

[Feature dataset](23_causal_feature_dataset.md) | [Research strategy](19_hybrid_selective_forecast_strategy.md)


## 9. User-reported first real-data baseline experiment — 26 November 2026

Evidence: full JSON pasted by the user after a local Mac run. Source CSV SHA256 reported as 589ca0a6fb76a9923673c3fe0cfd66fa231848ca913000e497856c6183d91a69, matching the earlier user-reported EDA hash. 2,016 contiguous 5m candles, 2,015 returns, 1,209 training, 403 validation and 403 development replay returns. No failed models reported. Raw files, predictions.csv and local unit-test output have not been independently inspected. Entire seven-day sample was already explored; this is not a final holdout.

### Estimated parameters and diagnostics

Return models were fitted on returns scaled by 100; const is the unconditional mean in percent units, sigma2 is in squared percent units, and AR/MA coefficients are dimensionless.

| Model | const | AR coefficient | MA coefficient | AIC | BIC | Ljung-Box p | ARCH-LM p |
|---|---:|---:|---:|---:|---:|---:|---:|
| AR(1) | -0.00782131 | 0.05354878 | — | -531.05068 | -515.75803 | 0.616444 | 7.38347e-9 |
| MA(1) | -0.00782182 | — | 0.05345724 | -531.03929 | -515.74664 | 0.615759 | 6.77205e-9 |
| ARMA(1,1) | -0.00782207 | 0.05204863 | 0.00141639 | -529.05077 | -508.66057 | 0.526544 | 7.54290e-9 |

AR(1) is a parsimonious working mean-model candidate based on training AIC/BIC; the AR(1)/MA(1) AIC difference is negligible (about 0.0114). No directional promotion. The AR(1) coefficient is small and not a demonstrated statistically significant or stable effect.

Using the fitted parameters and converting the mean to unscaled returns:

```math
\begin{aligned}
\hat\mu&=-0.00782131056/100\approx-0.0000782131\\
\hat\beta_1&=0.05354878\\
\widehat r_{t+1\mid t}&=\hat\mu+\hat\beta_1(r_t-\hat\mu)\\
\hat c&=(1-\hat\beta_1)\hat\mu
\end{aligned}
```

### Forecast comparison

MAE and RMSE are unscaled log returns. Direction accuracy is a percentage, with UP versus NOT_UP, not trading win rate.

| Block / horizon | Model | n | MAE | RMSE | Accuracy |
|---|---|---:|---:|---:|---:|
| Validation 5m | Zero | 403 | 0.0010491796 | 0.0015482865 | 53.5980% |
| Validation 5m | AR1 | 403 | 0.0010422528 | 0.0015413441 | 53.5980% |
| Validation 5m | MA1 | 403 | 0.0010420777 | 0.0015410894 | 53.8462% |
| Validation 5m | ARMA11 | 403 | 0.0010422469 | 0.0015413431 | 53.5980% |
| Replay 5m | Zero | 403 | 0.0013389417 | 0.0018486671 | 52.8536% |
| Replay 5m | AR1 | 403 | 0.0013371634 | 0.0018430246 | 51.1166% |
| Replay 5m | MA1 | 403 | 0.0013371355 | 0.0018429597 | 50.8685% |
| Replay 5m | ARMA11 | 403 | 0.0013371580 | 0.0018430247 | 51.1166% |
| Validation 60m | Zero | 392 | 0.0032973321 | 0.0045457632 | 56.1224% |
| Validation 60m | AR1 | 392 | 0.0032923507 | 0.0045521767 | 56.1224% |
| Validation 60m | MA1 | 392 | 0.0032917412 | 0.0045516668 | 56.1224% |
| Validation 60m | ARMA11 | 392 | 0.0032923358 | 0.0045521647 | 56.1224% |
| Replay 60m | Zero | 392 | 0.0045620751 | 0.0060508365 | 55.3571% |
| Replay 60m | AR1 | 392 | 0.0045675550 | 0.0060080957 | 55.3571% |
| Replay 60m | MA1 | 392 | 0.0045668354 | 0.0060072110 | 55.3571% |
| Replay 60m | ARMA11 | 392 | 0.0045675301 | 0.0060080586 | 55.3571% |

Training-majority direction accuracy equals zero-return direction accuracy in all four blocks. MA1 validation 5m improves by exactly one correct observation relative to majority, but loses to it in the later block. Numerical improvements are small and metric-dependent (60m MAE worsens in replay and RMSE worsens in validation). No reliable forecasting advantage established.

### Research decision

At nominal 5% significance, the training Ljung-Box tests at lag 12 do not reject residual autocorrelation nulls, whereas all ARCH-LM tests reject the no-ARCH null. This does not establish independence, exact model specification, or the reliability of asymptotic p-values under changing market regimes. Library warnings are future API notices, not fit failures.

Next experiment: compare constant training variance (V0), trailing sample variance (V0 rolling) and train-fitted GARCH(1,1) (V1) for one-step conditional variance. Keep mean specification and eligible origins consistent; initialize using training data, freeze parameters and update the variance state only after realized innovations become available. Measure QLIKE and squared-error loss on a common realized variance proxy; diagnose standardized residuals, convergence and persistence. Any 1h variance target needs an explicit aggregation and covariance assumption; do not multiply standard deviation by 12. No GARCH parameters or skill yet. GARCH is a volatility model, not an automatic direction signal. Longer authorized history and a fresh untouched chronological holdout remain required before promotion; gold access, feature export verification and original publication latency remain pending.


## Follow-up experiment

[Chapter XXV](25_garch_volatility.md) implements the proposed one-step variance comparison. Real GARCH results remain pending.
