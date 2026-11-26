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
