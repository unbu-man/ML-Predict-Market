# Chapter XXII — Exploratory data analysis and time-series diagnostics

26 November 2026. BTC/USD only in this step. Synthetic fixture tests passed in the assistant environment; the user's real CSV files, API permissions and local execution remain unverified.

## Learning goal

Audit the raw data before fitting models. A checksum confirms that a CSV matches its manifest, not that a provider's prices are true. Count missing bars, check UTC order and duplicates, verify OHLCV and compare the manifest with the actual timestamps. Analyze only the longest contiguous segment so we never calculate a five-minute return across a gap. This conservative exploratory choice discards shorter segments; it is not a final modeling policy. Require at least 80 candles as a diagnostic implementation guard, not as a scientific adequacy threshold.

## Mathematics

Let $C_t$ be the completed candle close, $r_t$ the realized log return, $n$ the number of returns, and $k$ the lag.

### Log price and realized return

```math
\begin{aligned}z_t&=\ln(C_t)\\r_t&=z_t-z_{t-1}\\&=\ln\left(\frac{C_t}{C_{t-1}}\right)\end{aligned}
```

This uses the prior close, not a future training label.

### Sample mean, variance and standard deviation

```math
\begin{aligned}\bar r&=\frac{1}{n}\sum_{t=1}^n r_t=\frac{r_1+r_2+\cdots+r_n}{n}\\s^2&=\frac{1}{n-1}\sum_{t=1}^n(r_t-\bar r)^2\\s&=\sqrt{\frac{1}{n-1}\sum_{t=1}^n(r_t-\bar r)^2}\end{aligned}
```

For the reported BTC sample:

```math
\begin{aligned}n&=2015\\\bar r&\approx-0.0000707463\\s&\approx0.0018500743\end{aligned}
```

### Rolling volatility (12 returns)

```math
\begin{aligned}\bar r_{t,12}&=\frac{1}{12}\sum_{i=0}^{11}r_{t-i}\\s_{t,12}&=\sqrt{\frac{1}{11}\sum_{i=0}^{11}(r_{t-i}-\bar r_{t,12})^2}\end{aligned}
```

This is one hour on the 5m series or twelve hours on the 1h series. It is not a volatility forecast.

### Autocovariance and ACF

```math
\begin{aligned}\hat\gamma_k&=\frac{1}{n}\sum_{t=k+1}^n(r_t-\bar r)(r_{t-k}-\bar r)\\\hat\gamma_0&=\frac{1}{n}\sum_{t=1}^n(r_t-\bar r)^2\\\hat\rho_k&=\frac{\hat\gamma_k}{\hat\gamma_0}\end{aligned}
```

The denominator n matches the biased autocovariance normalization used for the exploratory ACF. PACF measures partial linear association after controlling intermediate lags. The plots show descriptive estimates without confidence bands.

### ADF and KPSS

ADF (constant, AIC lag choice) tests the null of a unit root. KPSS tests the null of level stationarity. Read both together; failure to reject is not proof. Report lookup-table p-value bounds/warnings and acknowledge low power, changing regimes and short samples.

### Squared returns and volatility dependence

```math
q_t=r_t^2
```

Apply ACF to squared returns to explore time-varying volatility, not to prove ARCH/GARCH. Later test fitted residuals and compare out-of-sample volatility forecasts.

## Run from repository root

```bash
python3 -m pip install -r requirements-eda.txt
python3 -m unittest tests.test_day04_market_eda -v
ls data/coinbase/
```

Select the actual dataset directory printed by the Day 3 collector. For the exact seven-day example (13–20 November UTC), use:

```bash
python3 -m examples.day04_market_eda data/coinbase/BTC-USD_300_1794528000_1795132800 --output data/eda/btc_5m_smoke
python3 -m examples.day04_market_eda data/coinbase/BTC-USD_3600_1794528000_1795132800 --output data/eda/btc_1h_smoke
```

If your folder has a different name, substitute that name. If collection has not succeeded, return to Chapter XXI. Do not create fake real-market files. Output directories must not already exist; use another output path to preserve earlier runs.

Outputs: report.json, close.png, log_returns.png, rolling_volatility_12bars.png, acf_returns.png, pacf_returns.png, acf_squared_returns.png, return_histogram.png. Report JSON includes source checksum, gap count, segment size, sample statistics, ADF/KPSS results and warnings. Seven plots use observation indices or lag counts, not real-time axes.

## Interpretation and acceptance gate

Seven days are only a pipeline smoke test. Inspect data integrity and both timeframes; send the actual report.json and plots for review. For model development, obtain a longer licensed history and freeze a chronological final test interval BEFORE choosing model specifications or features. Run diagnostic/feature-selection work on training data only. Historical availability timestamps remain unverified; no prediction performance, gold results or live trades are established.

Next: causal feature engineering and target generation with explicit replay availability assumptions, contiguous-window checks, chronological purging and baseline comparisons.

See [Chapter XXI](21_first_historical_dataset.md), [Chapter XIX](19_hybrid_selective_forecast_strategy.md), and [Chapter XVIII](18_day01_forecast_contract_draft.md).


## First observed BTC/USD smoke-test results — 26 November 2026

Evidence: user's pasted terminal output from the local Mac EDA run for the 13–20 November 2026 UTC, five-minute dataset. The raw CSV, full manifest, full report.json, and plots have not been independently inspected.

- 2,016 received candles, zero missing intervals, 2,016 consecutive candles.
- 2,015 one-period returns implied by that uninterrupted segment.
- The EDA script reached its successful report-output message.

| Series | ADF statistic | ADF reported p | KPSS statistic | KPSS reported p |
|---|---:|---:|---:|---:|
| Log price | -2.48913889 | 0.11813936 | 5.25940402 | 0.01 (actual p < 0.01 per warning) |
| Log return | -42.07964180 | 0.0 (numerically very small) | 0.12294484 | 0.10 (actual p > 0.10 per warning) |

At alpha = 0.05, the constant-only ADF fails to reject a unit root for log price and rejects it for returns; KPSS rejects level stationarity for log price but does not reject it for returns. These are consistent with differencing log prices once as a starting model specification. Compare ARMA/AR on log returns with ARIMA(p,1,q) on log price; do not difference returns again automatically. These tests do not establish strict or conditional stationarity, future predictability, or GARCH benefit. The statsmodels future API warnings are not runtime errors; the KPSS warnings identify table bounds.

**Next gate:** inspect actual return ACF/PACF, squared-return ACF, rolling volatility and histogram from report.json and plots. Then prepare past-only features and +5m/+1h targets, after documenting availability assumptions and a purged evaluation protocol. Seven days is a smoke test; collect longer authorized history and reserve a chronological final holdout before model selection. Hourly EDA, XAU access, test-suite results, and publication latency are still unverified. No trading performance established.

## Full report follow-up

The user supplied the complete diagnostic JSON. See [Chapter XXIII](23_causal_feature_dataset.md) for actual return/squared-return ACF findings and the feature/target design. Plot review, raw data verification, and longer historical evaluation remain open.
