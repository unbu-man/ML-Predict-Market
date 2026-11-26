# Chapter XXIII — First causal feature and target dataset

26 November 2026. Development smoke test only. Based on the user's pasted full five-minute BTC/USD EDA report (not independently inspected raw data).

## Evidence and implications

Source CSV SHA256 reported: 589ca0a6fb76a9923673c3fe0cfd66fa231848ca913000e497856c6183d91a69. 2,016 consecutive candles, 2,015 returns; mean -0.0000707462811538, sample SD 0.00185007428132. Return quantiles at 1%, 50%, 99%: -0.0046000708, -0.0000888535, 0.0051181083.

Lag-1 return ACF/PACF are both 0.06401. Approximate white-noise pointwise reference bounds +/- 1.96/sqrt(2015) = +/- 0.0437, not a robust or simultaneous significance test under heteroskedasticity. Most correlations are weak; lag 15 also crosses the approximate bounds and PACF lag 24 is marginally outside. Do not choose a high-order ARMA model from these observations. Compare zero-return, AR(1), and optionally MA(1) on new chronological evaluation data. No estimated AR coefficient or forecast skill claimed.

Squared-return ACF at lags 1–4: 0.14030, 0.17468, 0.11176, 0.13347. Several later lags are positive. This is evidence consistent with volatility clustering in this short sample, motivating rolling-volatility and GARCH experiments, not proof of GARCH superiority. Use Ljung-Box on squared fitted residuals and ARCH-LM later; compare out-of-sample conditional variance forecasts. EDA report covers a development sample already inspected in full, so it must not be used as the final untouched test set.

## First features

Close-derived only. Let z_t=ln(C_t), r_t=z_t-z_(t-1).

- Return lags 0,1,2,3,6,12 (lag 0 is the last completed one-bar return).
- Mean and sample standard deviation over trailing 6,12,24 returns (30m, 1h, 2h).
- Momentum = z_t-z_(t-w) over those windows. Note it equals the sum of those w returns, so it is redundant with w times rolling mean; retain for teaching and remove redundant columns in the initial linear-model feature set.
- Total 15 numerical features. RSI/EMA/ATR and higher timeframe features are deferred to controlled ablations.
- Targets are forward log returns for +5m/+1h and binary UP/NOT_UP. No labels are part of the feature allowlist.

The audit uses the longest contiguous segment, preserves gaps, checks the source CSV against its manifest. The feature builder rejects gaps, uses 24-bar warmup, reserves 12 future bars, and writes features.csv and targets.csv separately. Match them only by anchor_close_utc. The original dataset contains unknown historical publication times. A required explicit replay-delay parameter models a constant assumed delay after each completed bar, not a measured feed delay. Never claim actual real-time causality or executable fills from this assumption; test sensitivity before deployment.

## Run

From repo root after git pull, with the exact earlier collector dataset:

    python3 -m unittest tests.test_day05_features -v
    python3 -m examples.day05_features data/coinbase/BTC-USD_300_1794528000_1795132800 --output data/features/btc_5m_smoke --replay-delay-seconds 2

The 2-second delay is illustrative and unverified. Expected 1,980 paired rows = 2016 - 24 - 12, if the reported dataset is unchanged. Use the actual source directory, and a new output directory for each run. The generated data is gitignored. Five fixture tests supplied; execution pending. This step does not train any model.

## Next gate

Collect a substantially longer authorized BTC history (and XAU separately), freeze chronological validation/test boundaries before model selection, and purge training rows whose label availability overlaps a later split. Develop all preprocessing on training folds only. Keep this seven-day dataset for smoke tests. Use only manifest feature_columns as X, never targets or timestamp metadata. Report zero-return, training-majority, AR(1), then Logistic/Ridge before XGBoost. No live trading or win rate established.
