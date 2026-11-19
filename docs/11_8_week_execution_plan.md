# Chapter XI — Eight-Week Development & Learning Plan

**Created:** 19 November 2026  
**Product:** Non-executing BTC/USD and XAU/USD Signal Bot, with supporting indicator and performance reporting. **Auto Bot deferred.**
**Study capacity assumption:** Up to ~5 hours on a study day, adjustable around university. This is a planning guide, not a guaranteed delivery schedule. Aim for focused work 5 days per week; use remaining days to review or rest.

## Scope note: market-independent long-term architecture

The target feature library spans **1m, 5m, 1h, 4h, 1d** and more market types eventually, but the eight-week schedule remains **a focused BTC/USD and XAU/USD Signal Bot pilot, beginning with 5m+1h**. Reserve 1m, 4h, 1d, unsupervised regimes and nonlinear-model comparisons for incremental research after the initial baseline is stable. The separate [Chapter XII](12_multimarket_multitimeframe_unsupervised.md) contains the staged expansion, accuracy caveats and tests. This avoids expanding the MVP into an untestable collection of features.

## How to use each weekly chapter

For each week: **goal → concepts to study → exact engineering tasks → test cases → GitHub evidence → go/no-go**. Don't move forward solely because a calendar week ended. Record real results and failures in [Chapter VI](06_progress_and_decisions.md), with links to tests, code and reports.

### Week 1 — Environment, source choice and project skeleton

**Goal:** reproducible Python setup and provider-source decisions for BTC/USD and XAU/USD.

**Study:** Python modules, project packaging, Git branches, OHLC/bid/ask, real-USD BTC pairs versus stablecoin pairs, XAU/USD per-troy-ounce units, public price data licensing.

**Tasks:**
1. Clone repository and create \`.venv\`; initialize \`pyproject.toml\`, \`src/ml_predict_market/\` and \`tests/\`.
2. Install/pin test-verified \`pandas\`, \`numpy\`, \`matplotlib\`, \`pytest\` and HTTP client; do not prematurely install deep ML.
3. Identify a BTC/USD venue offering historical 5M candles, and a named XAU/USD provider with documented candle type and calendar.
4. Write provider manifests: endpoint, symbol, quote type, coverage, permission, interval, timestamp semantics, sessions, latency and costs.
5. Make \`pytest\` run with a first smoke test.

**Evidence:** README setup instructions, two source manifests, passing smoke test.  
**Go/no-go:** No valid provider licensing/data semantics? Resolve that before implementation.

### Week 2 — Collect, normalize and validate both market feeds

**Goal:** clean, timestamp-consistent completed 5M and 1H historical candles, separately per market.

**Study:** time zones/UTC, resampling, missingness, high/low validation, OHLC timestamp alignment, gold daily breaks and holidays.

**Tasks:**
1. Write provider adapters and a normalized Candle schema; store raw data outside Git.
2. Check sorting, duplicate open times, negative prices, inconsistent OHLC, incomplete bars, provider outages, market closures and session calendars.
3. Build H1 bars from *twelve complete contiguous 5M bars* only when valid for the provider calendar; compare with source-provided H1 bars.
4. Generate validation summaries per instrument and sample dates.
5. Test that BTC/USD rejects BTC/USDT and that gold volume can be \`null\`.

**Evidence:** \`src/.../data/\` adapters, validator tests and \`reports/data_quality_btc.md\`, \`reports/data_quality_xau.md\` with **actual** outcomes.
**Go/no-go:** No known-good backfill or unresolved publication timestamps means no research signals yet.

### Week 3 — EDA + non-repainting market features

**Goal:** study return behavior and implement correct 1H and 5M feature calculations.

**Study:** descriptive EDA, log returns, rolling volatility, EMA, RSI, ATR, swing highs/lows and warm-up; no future-candle information.

**Tasks:**
1. Produce BTC and gold exploratory notebooks or reproducible scripts with price/returns/volatility charts.
2. Compute EMA20, EMA50, RSI14, ATR14 from past **completed and available** bars.
3. At each 5M close, perform a backward-as-of lookup to the **last fully available 1H context**.
4. Test hour boundaries (e.g., 11:05 cannot see H1 11:00–12:00), provider publication delays and gold closures.
5. Generate diagnostic charts with explanations, not trading claims.

**Evidence:** EDA plots, feature module, no-leakage tests.
**Go/no-go:** Signals cannot use unfinished higher-timeframe bars.

### Week 4 — Signal logic and setup lifecycle

**Goal:** fully deterministic 1H/5M candidate signal engine.

**Study:** rule-based strategy specification, finite-state machines, market structure, invalidation, overfitting.

**Tasks:**
1. Implement a configurable 1H BULLISH/BEARISH/NEUTRAL state.
2. Implement 5M pullback detection and subsequent confirmation trigger.
3. Add explicit WAIT, WATCH, ARMED, TRIGGERED, REJECTED and EXPIRED events with reason codes.
4. Create a signal journal: UTC timestamps, signal ID, source, market, timeframe, rule version and explanation.
5. Add unit tests for duplicate alerts, stale source, missing bars, weekend gold closure, invalidation and replay determinism.

**Evidence:** versioned strategy, journal schema and test suite.
**Go/no-go:** For a fixed historical sample, identical signals must reappear on every replay.

### Week 5 — Backtesting and economic evaluation

**Goal:** measure whether the 1H+5M strategy beats simpler alternatives.

**Study:** forward labeling, chronological splits, transaction costs, expectancy, R multiples, drawdown, risk and bias.

**Tasks:**
1. Establish baselines: no signal / 5M-only / 1H-only / 1H+5M, using comparable periods and costs.
2. Define exact hypothetical entry quote, stop invalidation, exit policy, holding limit and expiry.
3. Simulate spread, fees and slippage; identify intrabar stop/target order ambiguity instead of assigning ideal fills.
4. Separate BTC/USD and XAU/USD reports; record all trials, even unsuccessful ones.
5. Evaluate on unseen chronological windows; stress test cost and rule sensitivity.

**Evidence:** reproducible backtest, config, full signal log, net expectancy and drawdown reports.
**Go/no-go:** If 1H+5M does not show credible incremental value, don't add more complexity; revise hypothesis.

### Week 6 — Non-executing Signal Bot MVP

**Goal:** operationally reliable research alerts, still with **no execution functions**.

**Study:** event queues, retries, idempotency keys, stale-data detection, rate limiting, Telegram bot/webhook security.

**Tasks:**
1. Publish local alert events with distinct WATCH / ENTRY_CANDIDATE / CANCELLED / EXPIRED statuses.
2. Add deduplication and durable storage; never send multiple alerts for the same finalized setup.
3. Include source, time, reason, hypothetical entry/stop reference, quote freshness and expiry.
4. Integrate an optional Telegram bot **after** local tests pass.
5. Log notification delivery time and compare with decision time. Never include broker order credentials.

**Evidence:** alert message fixtures, local delivery integration tests, delivery journal and latency report.
**Go/no-go:** Fail-safe on missing/stale data and closed market; no silent false entries.

### Week 7 — Forward paper observation and review

**Goal:** observe prospective signals and their hypothetical outcomes without live money.

**Study:** monitoring, false positives, calibration (only if probabilities exist), change control, market regimes.

**Tasks:**
1. Operate in DEMO/RESEARCH mode and record every alert and invalidation.
2. Reconcile future price outcomes under the previously frozen evaluation policy.
3. Report alert quality, error rate, signal frequency, spread/fee sensitivity, drawdown and latency.
4. Analyze BTC and gold independently; avoid learning from the same data used for final evaluation.
5. Do not promote an unproven approach to customers.

**Evidence:** dated forward paper-signal journal and reliability report.
**Go/no-go:** One week cannot prove long-run profitability; continue forward observation and require adequate sample size.

### Week 8 — ML research and portfolio-ready demo

**Goal:** compare simple ML features against the rule-based system without overstating performance.

**Study:** baseline classifiers, train-only preprocessing, walk-forward evaluation, feature importance, probability calibration and reproducible model cards.

**Tasks:**
1. Prepare time-ordered return and direction labels with no future leakage.
2. Benchmark no-skill/majority, Logistic Regression and XGBoost against the existing rule-only strategy; model per market.
3. Compare with and without MTF features on held-out windows. Do not assume ML improves results.
4. Write negative findings and limits; package a small chart/report demo of the **non-executing** Signal Bot.
5. Decide whether to continue signal-only research; Auto Bot stays deferred until a new explicit decision.

**Evidence:** a reproducible research report, model comparison, tests, demo and final README status.
**Go/no-go:** Document measured evidence, uncertainty and risks. No promise of returns.

## Suggested daily study block (~5 hours, flexible)

| Time | Activity |
|---|---|
| 45 minutes | Study one market, statistics or software concept |
| 2 hours | Implement one well-scoped feature |
| 1 hour | Write and run tests, explore failed cases |
| 45 minutes | Analyze outputs and record real results |
| 30 minutes | Commit documentation/progress and select next task |

## Next single task — Session 01 (start here)

**Read:** [Chapter IX data plan](09_btc_gold_data_plan.md) and [Chapter X Signal Bot spec](10_signal_bot_mvp.md).

1. Confirm initial 5M + 1H **BTC/USD** source is genuinely USD-quoted, with completed candles and acceptable access rights.
2. Confirm one **XAU/USD** source, its quote type, market hours and historical resolution.
3. Create a \`data_sources.md\` research note with provider, candle timestamp convention, interval and license.
4. Commit the note. **Do not** download data that you cannot lawfully use or redistribute.

**Session 01 is finished when:** two candidate providers have documented semantics and a reproducible plan for retrieving a bounded historical sample. No market data or model work is claimed completed at the time of writing.

## Project rules

- **Signal Bot first. Auto Bot deferred.**
- **Risk control first:** any candidate may be rejected for spread, invalid stop/reference, stale quote, closed gold session or missing candles.
- **No LLM/NLP dependency:** this project can use simple statistics and conventional ML; separately learned LLM skills can be revisited only if later research justifies it.
- **No model without baseline:** training accuracy is not evidence of out-of-sample trading value.
- **No future candles:** every feature and label must be aligned to what was knowable at the decision time.
- **No real trades or guaranteed profit:** demos and outcome reports are hypothetical; no order-placement methods belong in MVP.
