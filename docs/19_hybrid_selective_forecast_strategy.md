# Chapter XIX — Adaptive Hybrid ML + TSA: Selective Forecast Research

**Decision date:** 26 November 2026
**Status:** Founder agreed to adopt the research direction. Not a trained or validated model.
**Supersedes:** earlier ambiguity about whether XGBoost is the primary advanced candidate; preserves Chapter XVIII's draft target contract, Day 1 example, and 30-day core acceptance gates.
**Research scope:** BTC/USD and XAU/USD, independent models per asset and horizon (5m and 1h), with 1m/5m/1h/4h/1d input capability. No live orders.

## 1. Objective

Find whether an explainable nonlinear forecast with optional conditional volatility, market-state features and a selective decision policy improves unseen-data forecast quality and hypothetical **net** trade expectancy. Win rate is an evaluation metric, not the sole objective or a promised outcome. Prefer simpler models if they win. WAIT is a legitimate action, not a prediction class.

The first 30 days remain an engineering and research MVP. M0–M2 baseline comparisons are core; M3 XGBoost is the first advanced candidate when the data pipeline passes. M4–M6 are conditional experiments, not guarantees of completion before Day 30. Preserve the existing calendar and progress; do not backdate tasks or reset statuses.

## 2. Mathematical contract

Keep the Day 1 draft: reference-price log return at a confirmed 5m close t, with h=5m or 1h.

```math
r_{t,h}=\log(C_{t+h}/C_t),\quad y_{t,h}=\mathbf 1[r_{t,h}>0].
```

The binary target is UP versus NOT_UP (which includes exact zero); it is still a draft pending separate confirmation. No future price or label can enter the features. Separate forecast outputs: estimated conditional mean $\hat\mu_{t,h}$, calibrated probability $\hat p_{t,h}$ of positive return, and optional conditional scale $\hat\sigma_{t,h}$. Estimate at the same horizon and using the same information set; distinguish return variance from uncertainty about estimated parameters.

**XGBoost:** fit a classifier to y with binary log loss and a separate return regressor with a documented loss. Regularize shallow trees, select hyperparameters using past-only validation, compare identical features and samples. Avoid expensive unrestricted tuning; use CPU-friendly runs.

**GARCH(1,1):** $\sigma_{t+1}^2=\omega+\alpha\varepsilon_t^2+\beta\sigma_t^2$. Model conditional variance, not price direction. Check valid fitted parameters, convergence and residual diagnostics; generate forecast scale with information available at each historical decision. For multi-step horizons align the variance forecast with the return horizon; do not simply multiply one-step volatility by 12. Compare to trailing realized-volatility baseline. GARCH features for training the downstream model must also be causal, generated with rolling/expanding estimation or historical cross-fitted predictions, not full-sample fits.

**Market regime:** start with train-only standardized K-Means or GMM on lagged returns, volatility and trend features. Inspect cluster stability and sample counts; labels such as "trend" or "range" are interpretations, not verified states. HMM and TAR remain later comparisons. HMM state probabilities must be filtered (past/current only), not smoothed using future observations.

## 3. Ordered experiment ladder

| ID | Variant | Main test |
|---|---|---|
| M0 | Zero-return and training-majority / persistence direction | No-skill benchmark |
| M1 | AR(1) or justified ARIMA return benchmark | Linear temporal dependence |
| M2 | Logistic Regression and Ridge return benchmark | Simple supervised model |
| M3 | XGBoost classifier + return regressor | Nonlinear gain over M0–M2 |
| M3-TF | M3 with causal 1h context, then 4h/1d/1m independently | Incremental timeframe value |
| V0/V1 | Trailing volatility vs GARCH forecast | Volatility loss and empirical interval coverage |
| M4 | M3 + validated GARCH feature | Added value vs matched M3 |
| M5 | M4 + validated K-Means/GMM features | Added value vs M4; also compare M3 + regime-only |
| M6 | Best validated forecast model + selective decision policy | Quality vs coverage and hypothetical net expectancy |

M5 and M6 need not contain M4 if GARCH fails. Test one addition at a time. Preserve older Chapter XV experiment IDs as historical design; use this ladder in new model reports. Run per-market/per-horizon; use matched timestamps, identical data rules and cost assumptions. If the advanced candidate does not improve, keep the baseline or report no demonstrated edge.

## 4. Signal policy (research only)

Use a direction forecast with explicit zero-return treatment. An initial long candidate might require calibrated p_up >= a validation-selected upper threshold, estimated feasible net return > 0, acceptable uncertainty, and valid fresh market data. A down-side candidate requires appropriate evidence for strictly negative return and an instrument-specific feasible execution model; NOT_UP is not synonymous with a short trade. For BTC spot, a SELL can be an exit or bearish alert, not assumed short execution. Thresholds, coverage constraints, costs, expiry, entry/exit and exposure policy must be preregistered before final evaluation. No fixed 70%/80% threshold or win-rate expectation is adopted.

Otherwise output WAIT. Report all forecasts and rejected opportunities, not just accepted winners.

```math
\text{coverage}=\frac{N_{\rm accepted}}{N_{\rm eligible}},\quad
\text{selective accuracy}=\frac{N_{\rm correct,accepted}}{N_{\rm accepted}}.
```

Define completed simulated trades separately; win rate = positive net P&L trades / completed trades. Define treatment of break-even trades. For realized gross winners W and losses L and a per-trade average cost c, report empirical net expectancy (or compute mean net P&L directly); never subtract fees twice. Report turnover, sample size, payoff ratio, drawdown, cost stress, coverage, and uncertainty intervals. Directional accuracy does not equal trade win rate.

## 5. Testing and promotion rules

- Chronological, purged walk-forward folds and an untouched final holdout. Purge training outcomes overlapping next validation/test start, including 1h labels.
- Use validation folds for all selection, calibration, threshold tuning and feature selection; no tuning on final holdout. Apply preprocessing inside each fold. Include training-label availability checks as well as feature timestamps.
- Baseline comparisons: log loss and Brier score, balanced accuracy, calibration; MAE/RMSE for continuous return. Volatility baseline comparison and interval coverage. For overlapping labels, uncertainty estimates must respect dependence (e.g., suitable block bootstrap).
- On the same time periods, assess hypothetical trading with feasible quote-side fills after signal availability, spread, commission, slippage and market-specific constraints. Never treat a reference close as guaranteed fill. Report no-trade benchmark.
- Advance only when predefined, meaningful out-of-sample improvement and sufficient sample/coverage are observed across relevant periods, with limitations reported. Select minimum coverage/sample requirements and drawdown limits before testing; no invented numerical thresholds.
- After holdout, require separate prospective paper monitoring and drift/retraining policy, with no live execution in the MVP. Models can be withdrawn after degradation.

## 6. Implementation sequence and schedule

1. Finish Day 1 target review and local tests. Preserve existing work; do not reset the dates.
2. Source feasibility and causal candle ingestion (USD-quoted BTC, provider-specific XAU, lawful access), tests, features and forward targets.
3. Days 16–20 planned: chronological evaluation contract, no-skill/AR baseline, Logistic and Ridge, walk-forward scoring.
4. Days 21–25 planned: XGBoost first; calibration, timeframe ablation; rolling volatility/GARCH experiment and regime ablation as data and time permit.
5. Days 26–30 planned: selective WAIT policy, cost-aware paper-outcome simulation, non-executing demo and reproducibility audit. Defer unfinished add-ons rather than rushing.
6. Later: Auto Signal Bot, Custom Indicator and paper-first Auto Bot as separate product interfaces.

## 7. References and limitations

- [FreqAI documentation](https://www.freqtrade.io/en/stable/freqai/): inspiration for feature/label separation and retraining workflows, not evidence of a win rate.
- [Microsoft Qlib](https://github.com/microsoft/qlib): modular quantitative research architecture.
- Cryer & Chan, *Time Series Analysis: With Applications in R*, 2nd ed., chapters 9, 12, 15: forecasting, conditional variance and threshold models.
- [XGBoost documentation](https://xgboost.readthedocs.io/): implementation reference.

These are methodological references, not evidence that our hybrid is profitable. No trained model, real-market feed, live win rate, or paper-trading performance has been verified. The Day 1 draft and provider blockers remain in force.

[Daily Google Sheet](https://docs.google.com/spreadsheets/d/1BlK-1Nl9fo4qgpTZ9tEFypvKVsfMJUyX1IYbEKrNqp8/edit)
