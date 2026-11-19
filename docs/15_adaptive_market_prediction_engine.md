# Chapter XV — Adaptive Market Prediction Engine (AMPE)

**Created:** 19 November 2026  
**Status:** Research architecture and hypotheses only. No model trained, no real-market accuracy verified and no execution.  
**Product sequence:** Prediction Engine → Auto Signal Bot → Custom Indicator → Paper Auto Bot.  
**Initial test markets:** actual USD-quoted BTC/USD (not BTC/USDT) and documented XAU/USD provider.  
**Timeframe ambition:** 1m, 5m, 1h, 4h, 1d. Initially issue forecasts after fully available 5m candles, predicting next 5m and next 1h.

## 1. Original research idea

Build a market-aware forecasting engine that estimates **three separate quantities**:

1. **Direction and expected return:** expected return and, if measured, calibrated probability of a positive next move.
2. **Magnitude and volatility:** how large and variable might that future move be? Avoid equating historical volatility with all sources of forecast uncertainty.
3. **Market regime:** do past-only features indicate a trending, ranging or high-volatility cluster, and does knowing this actually improve forward prediction?

A fourth component — a **reliability referee** — may allow the downstream signal system to **ABSTAIN / WAIT** when its forecast is unreliable, stale, too small relative to plausible trading costs, or unsupported by historical calibration. An abstention policy can reduce the number of alerts, so measure forecast quality **together with coverage**, not cherry-picked accuracy.

This combination is our **experimental architecture**, not a profitable strategy from a textbook.

## 2. Architecture

    Market provider and instrument registry
         │
    Historical OHLC + optional bid/ask + UTC sessions
         │
    Source audit: units, complete bars, available_at, missingness
         │
    Causal 5m features + confirmed 1h / optional 4h / 1d / 1m
         │
         ├── Direction / return head: zero-return, AR(1), Logistic, XGBoost
         ├── Volatility head: rolling variance → optional GARCH
         └── Regime head: optional train-only K-Means / GMM / TAR research
         │
    Versioned prediction record: expected return, horizon, uncertainty, model
         │
    Out-of-sample evaluation and reliability / abstention policy
         │
    Prediction journal → non-executing signals → indicator
         │
    Paper Auto Bot only in a later separately risk-gated phase

**Important:** For the 30-day MVP, there is NO requirement to combine all branches. The essential result is a well-defined, testable forecasting pipeline with simple baselines and honest out-of-sample reporting.

## 3. Mathematical contract

### 3.1 Training target and forecasting time

At time t, **after** a completed 5-minute candle has genuinely become available, define future horizon h:

\[
r_{t,h}=\ln\frac{C_{t+h}}{C_t}.
\]

The future close is an **outcome label** during backtesting, never an input to the forecast. Start with h = 5 minutes and h = 1 hour. For gold, write down what one hour means across session closures and do not silently advance through non-trading periods.

A simple directional label is:

\[
y_{t,h}=\mathbf 1(r_{t,h}>0).
\]

A NO_EDGE/NEUTRAL band may be added, but its value must be established on training/validation data without looking at final test results. Price-reference returns are NOT executable trade returns.

### 3.2 Forecast components

Conditional expected forward return:

\[
\widehat{\mu}_{t,h}\approx E[r_{t,h}\mid\mathcal F_t].
\]

Conditional variance, reflecting changing movement scale:

\[
\widehat{\sigma}_{t,h}^2\approx\operatorname{Var}(r_{t,h}\mid\mathcal F_t).
\]

Directional probability, if model is independently calibrated:

\[
\widehat{p}_{t,h}=P(r_{t,h}>0\mid\mathcal F_t).
\]

Here the information set \(\mathcal F_t\) includes only prices and features published by the decision timestamp. These three objects do not automatically create a safe BUY/SELL instruction.

### 3.3 Classical TSA baseline and volatility idea

- Zero expected return: benchmark for an efficient random-walk-like price.
- AR(1): past-only return benchmark, \(r_t=c+\phi r_{t-1}+\varepsilon_t\), with residual ACF diagnostics.
- ARMA/ARIMA: optional classical linear comparison if appropriate; not mandatory in a 30-day delivery.
- Rolling realized volatility: an initial scale estimator from past return windows.
- **Optional** GARCH(1,1): \(\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2\). This is about volatility, not itself direction.
- **Optional** TAR: threshold-dependent autoregressive dynamics; thresholds must be fitted to training data, and any threshold must be a lagged observable feature.
- **Optional** K-Means/GMM: historical state clustering; cluster IDs do not automatically mean bullish or bearish.

## 4. Multi-timeframe data rules

| Input | Idea | Research priority |
|---|---|---|
| 5m | Base forecasting clock, historical lagged returns | Essential |
| 1h | Confirmed broader context, no unfinished higher bar | Essential after 5m |
| 4h | Intermediate features | Ablation if enough data |
| 1d | Session-aware longer trend or risk context | Ablation if enough data |
| 1m | Optional micro-timing; much noisier and costlier | Later |

Record provider, exact symbol, quote type, price unit, session timezone, open/close UTC, **available_at**, is_complete, optional volume semantics, and missing/corrupt intervals. Aggregation cannot treat every market as 24/7 or unconditionally use UTC-midnight for gold daily bars. No cross-asset universal model performance assumption.

## 5. Model experiments to run IN ORDER

| Experiment | Model or added capability | Falsifiable question |
|---|---|---|
| M0 | Zero-return and majority/persistence forecasts | Can we beat no-skill baselines? |
| M1 | AR(1) return series | Is linear dependency useful? |
| M2 | Logistic Regression + lagged returns and volatility | Is direction forecast better and calibrated? |
| M3 | XGBoost on matched features | Do nonlinear interactions add value? |
| M4 | Add 1h then 4h, 1d and optionally 1m features | Which timeframe genuinely improves held-out results? |
| V1 | Rolling realized volatility | Can we predict movement scale? |
| V2 | Optional GARCH | Does conditional volatility help versus V1? |
| R1 | Optional train-only clusters | Do regimes improve forward predictions versus no-regime? |
| R2 | Optional TAR | Do explicit threshold dynamics beat AR(1)? |
| S1 | Forecast reliability / ABSTAIN | Does selective quality improve after accounting for reduced coverage? |

**Do not mark optional models as required for MVP.** M0/M1/M2 and one volatility baseline are the practical focus; M3 is a stretch experiment if acquisition and evaluation run smoothly.

## 6. Statistical and engineering acceptance rules

- Freeze the prediction timestamp, horizon, target and quote convention before fitting.
- Only completed and actually available 1h/4h/1d bars may be attached to 5m predictions; use backward as-of lookup.
- Chronological train, validation and untouched final test; purge/gap overlapping forward labels.
- Fit scalers, imputers, model parameters, calibrators and clusters only inside training folds or permitted validation fitting.
- Evaluate **per market and per horizon**, on matched periods and cost assumptions.
- For probabilities: log loss, Brier score, balanced accuracy, reliability/calibration and coverage.
- For return: MAE/RMSE versus the zero-return forecast.
- For volatility: variance/interval prediction quality with empirically checked coverage.
- Only downstream hypothetical trading tests may claim expectancy or drawdown, and must account for bid/ask, spread, slippage, fees and session limits.
- If a more complex model loses, preserve the negative result and favor the simpler benchmark.

## 7. Definition of a finished 30-day research MVP

A reproducible **non-executing** program that ingests lawful, documented source candles, validates timestamps and availability, builds point-in-time features, computes forward training targets correctly, fits at least two baselines, makes timestamped out-of-sample forecasts, reports uncertainty and evaluation, journals its predictions, and demonstrates a local forecast/signal message. For a blocked XAU feed, report source limitations honestly and never pretend a synthetic sample is validated live gold.

No guarantee of accuracy, profitability, universally transferable model, production availability, custody or actual trade execution.

## 8. Book inspiration vs engineering hypotheses

**Uploaded reference:** Jonathan D. Cryer and Kung-Sik Chan, *Time Series Analysis: With Applications in R*, 2nd ed. (Springer, 2008).

**From the book's theory:** Chapter 1 §1.2 (identification → fitting → diagnostics and parsimonious models), Chapters 2 and 4–8 (stationarity, AR/MA/ARIMA and diagnostic methods), Chapter 9 (forecasting and prediction limits), Chapter 12 (financial return dynamics and ARCH/GARCH), Chapter 15 (nonlinear threshold autoregression).

**Our proposed extensions:** simultaneous forecast output, reliability gate, five-timeframe point-in-time store, market adapters, selective WAIT, three products. These are engineering hypotheses and should be tested independently. We are NOT redistributing the textbook into the repository.

## 9. Next learning documents

- [30-day daily plan](13_30_day_math_signal_bot_plan.md)
- [Tomorrow's Day 01 checklist](16_day1_2026_11_20_kickoff.md)
- [Day 01 forecast-target template](17_forecast_target_contract_template.md)
- [Three-product long-term roadmap](14_prediction_to_three_products.md)
- [Editable Google Sheet tracker](https://docs.google.com/spreadsheets/d/1BlK-1Nl9fo4qgpTZ9tEFypvKVsfMJUyX1IYbEKrNqp8/edit)

**Progress on 19 November 2026:** documentation prepared; no experimental results yet.
