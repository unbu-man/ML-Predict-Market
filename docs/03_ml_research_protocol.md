# Chapter III — Machine Learning and Forecast Research Protocol

**Status:** Methodology, not a claim of results.

## Learning objectives
Design forecast targets, train on past-only features, compare honest baselines, prevent leakage and evaluate results after trading costs.

## The research question
For each of BTC/USD and XAU/USD independently, can features observed up to completed candle t predict next-candle price direction **out of sample** better than a simple baseline, and does any model-informed paper strategy survive realistic costs?

A possible regression target is the forward log return:

```math
r_{t+1}=\ln\left(\frac{P_{t+1}}{P_t}\right)
```

A direction label is defined by:

```math
y_t=\begin{cases}1,&r_{t+1}>0\\0,&r_{t+1}\leq0\end{cases}
```

 The future return is **a label, never a feature**. For signals, a neutral band may be more useful than predicting tiny movements that cannot cover spread and fees.

## Data provenance manifest (required)
Record: provider and venue, symbol, timestamp collection time (UTC), time window, interval, complete-candle policy, timezone conversion, price conventions, missing/duplicate bar counts, licensing and rate limits. Gold XAU/USD volume may be absent or an indicative provider-specific count, not centralized spot traded volume.

## First features
- Lagged log returns (1, 3, 6, 24 bars)
- Rolling past-only mean and standard deviation of returns
- EMA20 vs EMA50 and price-to-EMA distance
- RSI14 and ATR14 from completed candles
- Hour-of-day, day-of-week and session indicators, validated by market

Any normalization, imputation, threshold selection or feature selection must be **fitted on the training fold only**.

## Evaluation protocol
1. Freeze a historical sample and choose a final held-out date range.
2. Create sequential train / validation / test windows; avoid random shuffles.
3. Add a **gap/purge** where target horizons overlap boundaries.
4. Evaluate naive persistence, majority class and no-trade benchmarks.
5. Train Logistic Regression with a past-only preprocessing pipeline.
6. Compare XGBoost only after the baseline and feature set are established.
7. Tune hyperparameters on validation periods, then report one final test.
8. Run walk-forward re-training and report stability across regimes.
9. Save predictions with timestamp, target horizon, model version, fold and data fingerprint.

## Metrics
- **Classifier:** balanced accuracy, class-wise precision/recall, confusion matrix, ROC-AUC where appropriate, Brier score and calibration curves if probabilities are used.
- **Regression:** MAE/RMSE on returns, directional error, distribution and tail behavior.
- **Strategy:** net total return, maximum drawdown, Sharpe ratio with stated frequency assumptions, trade count, turnover, win/loss distribution and exposure.
- **Market realism:** spread, trading commissions, slippage, XAU/USD gold sessions and provider-specific rollover, if any, order size and data latency. A backtest is not evidence of future live performance.

A 60% accurate classifier can lose money if average losses and costs exceed the average wins. Avoid assuming predictive confidence equals probability of profit.

## Common mistakes
- Leakage from rolling functions that incorporate future observations
- Incorrect train/test split or fitting a scaler before the split
- Selecting a winning variant after repeatedly inspecting the final test
- Using mid-prices while ignoring executable bid/ask costs
- Optimizing on one short bull market
- Treating indicative gold quotes as executable centralized-exchange trades
- Reporting accuracy without a no-trade benchmark

## Practical exercise (Experiment EXP-001)
Collect BTC/USD hourly candles, validate data quality, compute past-only features and target, split chronologically, and train a majority-class baseline. Save a manifest, runnable notebook/script, confusion matrix and limitations. **Mark incomplete until executed.**

## Independent market experiments

- **EXP-BTC-001:** BTC/USD hourly direction and fee-aware backtest from real-USD spot exchange candles.
- **EXP-XAU-001:** XAU/USD hourly direction from a named gold quote provider. Do not label non-trading weekends or daily market breaks as normal missing candles. Document any broker CFD product features separately.
- Compare to naive and no-trade baselines **within** each asset first. Never infer cross-asset model quality from raw price scales or different executable quote assumptions.
