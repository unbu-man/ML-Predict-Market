# Chapter XIII — 30-Day Mathematics + Signal Bot Engineering Plan

**Project:** ML-Predict-Market  
**Start:** 20 November 2026 | **Finish target:** 19 December 2026 | **Study time:** up to 5 hours/day (~150 hours)  
**Google Docs living plan:** https://docs.google.com/document/d/1tiXEHQBnZEJ26GNtQcuFg6RshQv2CVYUKn4nU3Jtwt8/edit?usp=drivesdk  
**Status:** Plan only, not yet implemented.  
**Primary deliverable:** Non-executing research Signal Bot for BTC/USD and XAU/USD, with validated completed-candle 5m/1h baseline, five-timeframe feature interface, mathematics, cost-aware backtesting, local alert journal and demo. **Auto Bot deferred**.

## Feasibility and definition of done

The 30-day goal is feasible as a **bounded engineering MVP**, not a robust profitable/production trading service. Five timeframes (1m,5m,1h,4h,1d) are feature targets, NOT five validated strategies. Two live market integrations depend on data-provider access and licensing; if XAU/USD is blocked, the adapter must remain honestly marked blocked while tests use clearly labeled fixtures.

### Required acceptance evidence

- [ ] True USD-quoted BTC/USD and documented XAU/USD quote sources with permission, units, sessions, prices and time semantics.
- [ ] Reproducible data ingestion, candle validator and data-quality manifests for 5m and 1h.
- [ ] Tested 1m,5m,1h,4h,1d feature schema/aggregation where reliable source data exists.
- [ ] Mathematics unit tests for log returns, EMA, RSI, ATR, volatility, cost-aware R, expectancy, drawdown.
- [ ] Deterministic, non-repainting 1h+5m candidate signal engine with WAIT/REJECTED/EXPIRED.
- [ ] 5m-only versus 1h+5m comparisons on chronological unseen data with nonzero costs.
- [ ] SQLite/local signal journal, one demonstrated non-executing alert path and reproducible demo.
- [ ] GitHub progress notes, test report, exact provider caveats and final limitations.

**Stretch only:** unsupervised regimes, XGBoost/nonlinear research and extra timeframe ablations. No live orders or performance promises.

## Architecture and tools

Provider adapters → instrument registry (symbol, units, product, sessions) → completed-candle store → validated multi-TF feature engine (1m,5m,1h,4h,1d) → versioned 1h/5m strategy → signal-quality/cost gate → research alerts + durable journal → historical and forward evaluation.

Stack: Python 3.12/3.13 (as supported locally), pandas, NumPy, matplotlib, httpx, pytest, SQLite. Add scikit-learn/XGBoost only after baseline passes.

## Mathematics (learn, implement, test)

- Log return: `r_t = ln(C_t / C_(t-1))` (no future bar).
- EMA: `EMA_t = alpha*C_t + (1-alpha)*EMA_(t-1)`, `alpha = 2/(n+1)`.
- True range: `TR_t = max(H-L, abs(H-prev_close), abs(L-prev_close))`; ATR = specified smoothing of TR.
- Rolling realized volatility: standard deviation of **past-only** returns; note session assumptions.
- Risk distance: `d = abs(entry-reference_stop)`; hypothetical long `R = (exit-entry)/d - costs_in_R`.
- Expectancy: `E[R] = win_rate*avg_win_R + (1-win_rate)*avg_loss_R - avg_cost_R`.
- Maximum drawdown: minimum of `equity/running_peak_equity - 1`.
- Unsupervised clustering: training-only standardized regimes; clusters ≠ future price labels.
- Supervised nonlinear: benchmark Logistic Regression then XGBoost on chronological splits.
- Scientific rule: **more bars or features do not guarantee more predictive value**.

## Day-by-day plan

### Week 1: Days 01–05: Scope, providers and mathematical foundation

*Research question:* What counts as a complete Candle? Why are BTC/USD and XAU/USD different data products?

#### Day 01 — 2026-11-20: Scope, architecture and definition of done

**Study / math:** Market independence vs asset-specific assumptions; identify data leakage.

**Implementation:** Write MVP contract, instrument registry draft, requirements, safety exclusions.

**Required evidence:** README decisions + 5 requirements/tests; explicitly no order methods.

#### Day 02 — 2026-11-21: Math I: market data representation

**Study / math:** OHLC, quote currency, units, log returns r_t=ln(C_t/C_(t−1)).

**Implementation:** Specify Candle schema with UTC open/close/available-at and bid/ask semantics.

**Required evidence:** Schema and valid/invalid candle examples.

#### Day 03 — 2026-11-22: BTC/USD source feasibility

**Study / math:** Exchange last price vs bid/ask; historical granularity and API limits.

**Implementation:** Verify genuine USD-quoted BTC spot provider, licensing, access and historical depth.

**Required evidence:** BTC provider manifest; do not substitute BTC/USDT.

#### Day 04 — 2026-11-23: XAU/USD source feasibility

**Study / math:** Gold troy ounce, spot indicative/CFD references, sessions/DST.

**Implementation:** Confirm permitted gold historical data provider with 5m/1h; document fallback if unavailable.

**Required evidence:** Gold provider manifest, session rules, risk/blocker note.

#### Day 05 — 2026-11-24: Python project foundation

**Study / math:** Functions, packaging, typing, pytest assertions and reproducibility.

**Implementation:** Set up pyproject, .venv, src package, configs, tests, CLI entry point.

**Required evidence:** Clean setup command + passing smoke test.

### Week 2: Days 06–10: Acquisition, five-timeframe schema and no-leakage engineering

*Research question:* How can we join 1m, 5m, 1h, 4h and 1d features without seeing future values?

#### Day 06 — 2026-11-25: BTC/USD collection pipeline

**Study / math:** API pagination, clock drift, rate limiting, OHLC data provenance.

**Implementation:** Download bounded historical USD Bitcoin 5m/1h candles; cache raw outside Git.

**Required evidence:** Re-runnable BTC ingestion with manifests, count and UTC times.

#### Day 07 — 2026-11-26: XAU/USD collection pipeline

**Study / math:** Quote price types, gaps from market closures vs outages.

**Implementation:** Download historical gold 5m/1h from verified source; preserve mid/bid/ask distinctions.

**Required evidence:** Gold ingestion or a documented blocked-data alternative; no fabricated feed.

#### Day 08 — 2026-11-27: Validation mathematics

**Study / math:** OHLC inequalities H>=max(O,C), L<=min(O,C); missingness and duplicates.

**Implementation:** Implement validators for chronology, duplicate keys, negative/inconsistent prices, incomplete bars.

**Required evidence:** Failing-fixture tests + two data quality summaries.

#### Day 09 — 2026-11-28: Five-timeframe data model

**Study / math:** Aggregated open-first/high-max/low-min/close-last; additive volume only.

**Implementation:** Represent 1m,5m,1h,4h,1d from native feeds or verified resampling; use session-aware boundaries.

**Required evidence:** Feature-store schema; demonstrate each interval from fixtures.

#### Day 10 — 2026-11-29: Causal alignment checkpoint

**Study / math:** As-of joins, publication latency, intervals not yet closed.

**Implementation:** Test at 11:05 use latest published 1h bar, never future 1h/4h/d; gold closed sessions.

**Required evidence:** Automated anti-lookahead and market-calendar tests.

### Week 3: Days 11–15: EDA and mathematics of indicators

*Research question:* Are price returns, trends, volatility and session effects accurately computed?

#### Day 11 — 2026-11-30: EDA Bitcoin

**Study / math:** Descriptive statistics, quantiles, skewness, fat tails, log-return histograms.

**Implementation:** Visualize BTC close, returns, missing intervals and intraday volatility.

**Required evidence:** BTC EDA report and chart generated from real/legally held historical data.

#### Day 12 — 2026-12-01: EDA Gold

**Study / math:** Gold daily-break distribution, session hours, spread and provider price conventions.

**Implementation:** Generate XAU price/returns/volatility plots; investigate gaps.

**Required evidence:** Gold EDA report (or clearly marked provisional provider limitations).

#### Day 13 — 2026-12-02: Math II: EMA and RSI

**Study / math:** EMA_t=alpha*C_t+(1-alpha)*EMA_(t−1), alpha=2/(n+1); RSI gains/losses.

**Implementation:** Implement EMA20/EMA50 and RSI14 with correct warm-up.

**Required evidence:** Unit tests against known small fixtures/reference values.

#### Day 14 — 2026-12-03: Math III: ATR and volatility

**Study / math:** TR_t=max(H-L, abs(H-C_prev), abs(L-C_prev)); ATR smoothing; sigma of past returns.

**Implementation:** Implement ATR14, rolling realized volatility and normalized ATR/price.

**Required evidence:** No future information; bounded, nonnegative metric tests.

#### Day 15 — 2026-12-04: Build multi-timeframe features

**Study / math:** Correlation and feature redundancy; 5m trigger with last confirmed 1h context.

**Implementation:** As-of join 5m,1h; make optional features from 1m,4h,1d; log feature availability.

**Required evidence:** Feature table + explicit null/not-ready reasons for each market.

### Week 4: Days 16–20: Deterministic strategy and signal quality

*Research question:* Can the same data produce the same explainable research signals every time?

#### Day 16 — 2026-12-05: Research strategy specification

**Study / math:** Trend/structure rules as hypotheses; signal vs forecast vs profit.

**Implementation:** Freeze versioned bullish/bearish/neutral 1h context and 5m pullback/trigger.

**Required evidence:** Readable strategy definition + fixture examples.

#### Day 17 — 2026-12-06: Deterministic signal engine

**Study / math:** State machines: WATCH, ARMED, TRIGGERED, CANCELLED, EXPIRED.

**Implementation:** Implement rule engine pure functions; do not connect Telegram yet.

**Required evidence:** Repeatable BUY/SELL/WAIT candidate log.

#### Day 18 — 2026-12-07: Hypothetical trade geometry

**Study / math:** Distance d=abs(entry−stop); R multiple; payoff ratio; stop/target ambiguity.

**Implementation:** Define entry at next available price, structural stop, expiry and cost assumptions.

**Required evidence:** Paper-outcome policy doc; prohibit ideal hindsight fills.

#### Day 19 — 2026-12-08: Risk-quality filters

**Study / math:** Spread/slippage as fraction of risk; stale data; session closure; exposure not execution.

**Implementation:** Implement rejection rules and clear reason codes for both assets.

**Required evidence:** Tests for session, stale data, bad risk, duplicate/rejected candidates.

#### Day 20 — 2026-12-09: Signal replay and tests

**Study / math:** Causality, deterministic replay, invariants and edge cases.

**Implementation:** Replay historical candles twice and compare all event hashes; test time boundaries.

**Required evidence:** Passing strategy/regression test suite.

### Week 5: Days 21–24: Backtests and statistics

*Research question:* Does the multi-timeframe strategy improve held-out results after estimated costs?

#### Day 21 — 2026-12-10: Math IV: expectancy and drawdown

**Study / math:** E[R]=win_rate*avg_win_R+(1-win_rate)*avg_loss_R−cost_R; max drawdown.

**Implementation:** Implement cost-aware hypothetical outcome journal; include bid/ask, commissions, slippage.

**Required evidence:** Signal-level net R and equity/drawdown computations.

#### Day 22 — 2026-12-11: Chronological backtest

**Study / math:** Time-ordered holdout, walk-forward splits, gap/purge overlapping outcomes.

**Implementation:** Run 5m-only vs 1h+5m baseline separately for BTC/gold; log every configuration.

**Required evidence:** Reproducible held-out backtest, no unsupported claims.

#### Day 23 — 2026-12-12: Math V: accuracy vs trading quality

**Study / math:** Confusion matrix, precision/recall, balanced accuracy, uncertainty, cost sensitivity.

**Implementation:** Report hit rate AND net expectancy, count, turnover, drawdown, regime shifts.

**Required evidence:** First per-market performance report and honest caveats.

#### Day 24 — 2026-12-13: Added-timeframe ablation

**Study / math:** Incremental value: same periods and costs; avoid data snooping.

**Implementation:** Compare 5m+1h against +4h, +1d, +1m when reliable data permits.

**Required evidence:** Ablation matrix; keep extra frames disabled if they add no robust value.

### Week 6: Days 25–27: Journal, notifications and forward observation

*Research question:* Can we deliver and track research alerts reliably without executing trades?

#### Day 25 — 2026-12-14: Signal journal infrastructure

**Study / math:** Event IDs, idempotency, SQLite, audit and provenance.

**Implementation:** Persist WATCH/TRIGGERED/REJECTED/EXPIRED alerts and theoretical outcomes.

**Required evidence:** Local storage, replay and duplicate-suppression tests.

#### Day 26 — 2026-12-15: Non-executing notifier

**Study / math:** Delivery semantics, latency, retries, secrets management.

**Implementation:** Build CLI output and optional Telegram alert, always marked RESEARCH ONLY.

**Required evidence:** Observed local message flow; no trading credentials or orders.

#### Day 27 — 2026-12-16: Prospective demo test

**Study / math:** Replay-vs-forward parity; watchdogs, stale-data shutoff.

**Implementation:** Run a bounded forward/demo observation session; investigate discrepancies.

**Required evidence:** Timestamped sample signals and alert latency/health report.

### Week 7: Days 28–30: Optional ML and final release gate

*Research question:* Do unsupervised regimes or nonlinear models add anything beyond the simple baseline?

#### Day 28 — 2026-12-17: Unsupervised optional experiment

**Study / math:** K-Means with train-only scaler; silhouette vs economic usefulness.

**Implementation:** Cluster historical return/volatility regimes; assign unseen points with fixed model.

**Required evidence:** Optional regime report; if core delayed, use day to fix reliability.

#### Day 29 — 2026-12-18: Nonlinear optional experiment

**Study / math:** Logistic Regression vs XGBoost, leakage avoidance, calibration.

**Implementation:** Train and compare simple vs nonlinear model on held-out periods; assess cost value.

**Required evidence:** Optional model card; no requirement for neural network profitability.

#### Day 30 — 2026-12-19: Final reproducibility and demo

**Study / math:** Sensitivity, uncertainty, what counts as complete vs profitable.

**Implementation:** Run fresh install/tests, two-market demonstration, publish reports, update GitHub and next-phase risks.

**Required evidence:** Complete researched Signal Bot MVP demo or transparently documented blocker.

## Comparison and release gates

A=5m only, B=5m+1h, C=B+4h, D=C+1d, E=D+1m, F=best predeclared baseline+training-only K-Means, G=nonlinear supervised model. Report each **per instrument**, with matched samples, walk-forward test, calibrated probabilities if any, realistic spreads/fees/slippage, sample size, alert frequency, false signals, net hypothetical expectancy and max drawdown. Stop added complexity if it does not improve honest held-out results.

**Day 05:** providers & basic tests documented. **Day 10:** no-lookahead/time alignment proven. **Day 20:** deterministic signal replay passes. **Day 24:** cost-aware held-out report exists. **Day 30:** full local signal demo/tests and honest limitations; not guaranteed profit or production service.

## Daily work allocation (~5h)

45m mathematics/theory → 2h implementation → 60m tests → 45m evaluation → 30m GitHub progress/commit.

## Provider and method sources

- [OANDA v20 granularity, candle timestamps and quote fields](https://developer.oanda.com/rest-live-v20/instrument-df/)
- [OANDA API account restrictions](https://developer.oanda.com/rest-live-v20/introduction/)
- [Coinbase Advanced Trade historical market data API](https://coinbase.github.io/coinbase-advanced-py/coinbase.rest.html)
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [TradingView repainting concept](https://www.tradingview.com/pine-script-docs/concepts/repainting/)

**Status on document creation:** tasks are *planned*. No data feeds, trained models, backtest results or signal delivery have been verified completed.

## Product roadmap clarification — 19 November 2026

The core is a **next-move prediction research model**, assessed against forecasting baselines on unseen chronological data. The Signal Bot is a downstream alert distribution component. After the 30-day MVP, add the custom Indicator and later a separately risk-gated **paper-only** Auto Bot; they are not required by Day 30. See [Chapter XIV](14_prediction_to_three_products.md).
