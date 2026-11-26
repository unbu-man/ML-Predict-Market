# Chapter XXII — Exploratory data analysis and time-series diagnostics

26 November 2026. BTC/USD only in this step. Synthetic fixture tests passed in the assistant environment; the user's real CSV files, API permissions and local execution remain unverified.

## Learning goal

Audit the raw data before fitting models. A checksum confirms that a CSV matches its manifest, not that a provider's prices are true. Count missing bars, check UTC order and duplicates, verify OHLCV and compare the manifest with the actual timestamps. Analyze only the longest contiguous segment so we never calculate a five-minute return across a gap. This conservative exploratory choice discards shorter segments; it is not a final modeling policy. Require at least 80 candles as a diagnostic implementation guard, not as a scientific adequacy threshold.

## Mathematics

Let C_t be the close of a completed candle.

- Log price: z_t = ln(C_t).
- One-bar realized log return: r_t = z_t - z_(t-1). This uses the prior close; it differs from the future-return training label in Chapter XVIII.
- Sample mean: sum(r_t)/n.
- Sample variance: sum((r_t - mean)^2)/(n-1).
- Rolling volatility: sample standard deviation of the most recent 12 returns (one hour on the 5m series; twelve hours on the 1h series). It is not a volatility forecast.
- ACF at lag k: sum((r_t - mean)(r_(t-k) - mean))/sum((r_t - mean)^2), using valid paired indices in the numerator. PACF measures partial linear association at a lag after controlling shorter lags. The plots show descriptive estimates without confidence bands.
- ADF (constant, AIC lag choice): null of a unit root. Small p-value is evidence against that null. KPSS (level stationarity): null of level stationarity. Read both tests together; failure to reject is not proof. Report lookup-table p-value bounds/warnings and acknowledge low power, changing regimes and short samples.
- ACF of squared returns: exploratory clue for time-varying volatility, not proof of ARCH/GARCH. Later test model residuals and compare out-of-sample volatility forecasts.

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
