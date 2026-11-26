# Chapter XXIII — First causal feature and target dataset

26 November 2026. Development smoke test only. Based on the user's pasted full five-minute BTC/USD EDA report (not independently inspected raw data).

## Evidence and implications

Source CSV SHA256 reported: 589ca0a6fb76a9923673c3fe0cfd66fa231848ca913000e497856c6183d91a69. 2,016 consecutive candles, 2,015 returns; mean -0.0000707462811538, sample SD 0.00185007428132. Return quantiles at 1%, 50%, 99%: -0.0046000708, -0.0000888535, 0.0051181083.

Lag-1 return ACF/PACF are both 0.06401. Approximate white-noise pointwise reference bounds $\pm\frac{1.96}{\sqrt{2015}}\approx\pm0.0437$, not a robust or simultaneous significance test under heteroskedasticity. Most correlations are weak; lag 15 also crosses the approximate bounds and PACF lag 24 is marginally outside. Do not choose a high-order ARMA model from these observations. Compare zero-return, AR(1), and optionally MA(1) on new chronological evaluation data. No estimated AR coefficient or forecast skill claimed.

Squared-return ACF at lags 1–4: 0.14030, 0.17468, 0.11176, 0.13347. Several later lags are positive. This is evidence consistent with volatility clustering in this short sample, motivating rolling-volatility and GARCH experiments, not proof of GARCH superiority. Use Ljung-Box on squared fitted residuals and ARCH-LM later; compare out-of-sample conditional variance forecasts. EDA report covers a development sample already inspected in full, so it must not be used as the final untouched test set.

## First features

Close-derived features only. One time step represents a completed five-minute candle.

```math
\begin{aligned}z_t&=\ln(C_t)\\r_t&=z_t-z_{t-1}=\ln\left(\frac{C_t}{C_{t-1}}\right)\end{aligned}
```

### 1. Lagged returns (six features)

```math
\mathbf{x}_{t,\mathrm{lags}}=(r_t,r_{t-1},r_{t-2},r_{t-3},r_{t-6},r_{t-12})
```

Lag zero is the most recently completed return, never a future return.

### 2. Rolling mean (three features)

```math
\bar r_{t,w}=\frac{1}{w}\sum_{i=0}^{w-1}r_{t-i}=\frac{r_t+r_{t-1}+\cdots+r_{t-w+1}}{w}
```

Here $w\in\{6,12,24\}$ denotes 30 minutes, one hour, and two hours.

### 3. Rolling standard deviation (three features)

```math
s_{t,w}=\sqrt{\frac{\displaystyle\sum_{i=0}^{w-1}(r_{t-i}-\bar r_{t,w})^2}{w-1}}
```

This is a sample statistic of past returns, not a conditional volatility forecast.

### 4. Momentum (three features)

```math
\begin{aligned}M_{t,w}&=z_t-z_{t-w}\\&=\ln\left(\frac{C_t}{C_{t-w}}\right)\\&=\sum_{i=0}^{w-1}r_{t-i}=w\bar r_{t,w}\end{aligned}
```

Momentum is redundant with the corresponding rolling mean; retain for teaching and remove redundant columns in the initial linear model.

### 5. Historical targets (separate from the 15 numerical features)

```math
\begin{aligned}Y_t^{(5\mathrm{m})}&=\ln\left(\frac{C_{t+1}}{C_t}\right)\\Y_t^{(1\mathrm{h})}&=\ln\left(\frac{C_{t+12}}{C_t}\right)\end{aligned}
```



```math
y_t^{(h)}=\begin{cases}1,&Y_t^{(h)}>0\quad(\mathrm{UP})\\0,&Y_t^{(h)}\leq0\quad(\mathrm{NOT\ UP})\end{cases}
```

Future prices are used only for historical labels, never inputs. RSI, EMA, ATR and higher-timeframe features remain deferred to controlled ablations.

### 6. Future model experiments

AR(1) candidate for the return mean:

```math
\begin{aligned}r_t&=c+\beta_1r_{t-1}+\varepsilon_t\\\hat r_{t+1\mid t}&=\hat c+\hat\beta_1r_t\end{aligned}
```

GARCH(1,1) candidate for conditional variance:

```math
\sigma_t^2=\omega+\alpha_1\varepsilon_{t-1}^2+\beta_1\sigma_{t-1}^2
```

In AR(1), $c$ is the intercept, $\beta_1$ is the AR coefficient, and $\varepsilon_t$ is the innovation. In GARCH, $\omega$ is the variance constant, $\alpha_1$ weights the previous squared innovation, and $\beta_1$ weights the previous variance. The beta coefficients are distinct model parameters. No model has been fitted in this feature-generation step.

The audit uses the longest contiguous segment, preserves gaps, checks the source CSV against its manifest. The feature builder rejects gaps, uses 24-bar warmup, reserves 12 future bars, and writes features.csv and targets.csv separately. Match them only by anchor_close_utc. The original dataset contains unknown historical publication times. A required explicit replay-delay parameter models a constant assumed delay after each completed bar, not a measured feed delay. Never claim actual real-time causality or executable fills from this assumption; test sensitivity before deployment.

## Run

From repo root after git pull, with the exact earlier collector dataset:

    python3 -m unittest tests.test_day05_features -v
    python3 -m examples.day05_features data/coinbase/BTC-USD_300_1794528000_1795132800 --output data/features/btc_5m_smoke --replay-delay-seconds 2

The 2-second delay is illustrative and unverified. Expected paired rows, if the reported dataset is unchanged:

```math
N_{\mathrm{pairs}}=2016-24-12=1980
```

 Use the actual source directory, and a new output directory for each run. The generated data is gitignored. Five fixture tests supplied; execution pending. This step does not train any model.

## Next gate

Collect a substantially longer authorized BTC history (and XAU separately), freeze chronological validation/test boundaries before model selection, and purge training rows whose label availability overlaps a later split. Develop all preprocessing on training folds only. Keep this seven-day dataset for smoke tests. Use only manifest feature_columns as X, never targets or timestamp metadata. Report zero-return, training-majority, AR(1), then Logistic/Ridge before XGBoost. No live trading or win rate established.
