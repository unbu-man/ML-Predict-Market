# ML Predict Market — Multi-Market Signal Research Framework

**Founder / learner project · Started 18 November 2026 · Status: documentation only**

> **Long-term goal:** a market-independent prediction research engine powering three future products: automated non-executing signals, a custom market indicator and a separately risk-gated **paper-first** Auto Trading Bot. First research markets: BTC/USD and XAU/USD. Timeframes: 1m, 5m, 1h, 4h, 1d. No model or product is validated or live.

## Product priorities — 19 November 2026

| Product | What it does | Initial delivery | Trade execution? |
|---|---|---|---|
| Market Intelligence Indicator (MII) | Charts trend, momentum, volatility, BUY/SELL/WAIT research labels | Python charts, then optional TradingView Pine Script | No |
| Signal Bot | Sends timestamped, explainable and deduplicated market alerts | Dashboard, then Telegram | No |
| Auto Bot | Deferred idea, not in current MVP | Future only, after demonstrated signal quality and separate review | **Not implemented** |

**Active MVP workflow:** verified 5m/1h historical data → non-repainting signal rules → cost-aware backtest → signal journal and alerts. The architecture will expose all five timeframes, but add them incrementally after baseline evaluations. There is **no trade execution endpoint** in the MVP. The old Auto Bot concept remains documented for potential later review.

## Starting research defaults

- **BTC/USD:** Bitcoin priced in real USD. Study 1-hour completed crypto-spot candles from an identified USD-quoted venue (Coinbase `BTC-USD` is a candidate, not yet selected or downloaded).
- **XAU/USD:** Gold quoted in USD per troy ounce. Study independent historical gold quotes from a named provider; label whether they are indicative spot/OTC or broker CFD quotes. XAU/USD is **not** EUR/USD and is not an ordinary fiat-forex cross.
- **Market design:** An instrument registry + market-specific data adapters, calendars, units, costs and evaluation reports make the framework extensible to future assets. The indicator and Signal Bot share a versioned strategy interface; Auto Bot is deferred.
- **Timeframe scope:** 1m, 5m, 1h, 4h and 1d, with completed-candle UTC availability; 1m and higher contexts are introduced through controlled ablation experiments, not assumed useful.
- **Unsupervised research:** experiment with regime clustering (e.g., K-Means or Gaussian mixtures) on training-only data; cluster labels do not themselves predict direction.
- **Nonlinear research:** compare XGBoost against simpler supervised baselines; neural networks are optional experiments, not a necessary prerequisite.
- **Indicators:** EMA(20/50), RSI(14), ATR(14), with defined rules and no future-candle information.
- **Models:** Simple baselines first; then Logistic Regression and XGBoost. LSTM/Transformers are later research topics, not automatic improvements.
- **Validation:** Chronological holdout and walk-forward testing, transaction costs, slippage assumptions, and reproducibility.
- **Funding:** Virtual only. Do not use money needed for living expenses.

## Documentation (start here)

| Chapter | Purpose |
|---|---|
| [01 — Project vision](docs/01_vision_and_scope.md) | Define the products, users, boundaries and measurable success |
| [02 — System architecture](docs/02_system_architecture.md) | Understand data flow and the responsibilities of each module |
| [03 — ML research method](docs/03_ml_research_protocol.md) | Prepare data, labels, features, baselines and valid evaluations |
| [04 — Indicator and bots](docs/04_indicator_and_bots.md) | Specify rules, alerts, paper execution and acceptance tests |
| [05 — Project roadmap](docs/05_roadmap.md) | Follow implementation phases and graduation gates |
| [06 — Progress and decision log](docs/06_progress_and_decisions.md) | Track actual work, experiments, setbacks and next steps |
| [07 — First sprint](docs/07_first_sprint.md) | Complete a first seven-day research and engineering cycle |
| [08 — References and glossary](docs/08_references_and_glossary.md) | Essential concepts and authoritative reading links |
| [09 — BTC/USD and XAU/USD market plan](docs/09_btc_gold_data_plan.md) | Data, units, sessions, model and bot separation |
| [10 — 1H + 5M Signal Bot specification](docs/10_signal_bot_mvp.md) | Setup states, signals, risk checks and performance evaluation |
| [11 — Eight-week build and study plan](docs/11_8_week_execution_plan.md) | Weekly schedule, artifacts, acceptance tests and first-session instructions |
| [12 — Multi-market, five-timeframe and unsupervised learning](docs/12_multimarket_multitimeframe_unsupervised.md) | Generalizable architecture, 1m/5m/1h/4h/1d alignment, regime discovery, nonlinear ML, experiment plan |
| [13 — 30-day mathematics and Signal Bot build plan](docs/13_30_day_math_signal_bot_plan.md) | Day-by-day tasks, mathematical proofs/tests, clear MVP exit criteria and Google Docs plan |
| [14 — Prediction engine and three future products](docs/14_prediction_to_three_products.md) | Shared model, Auto Signal, Indicator, and a later paper Auto Bot |
| [15 — Adaptive Market Prediction Engine](docs/15_adaptive_market_prediction_engine.md) | Return, uncertainty, regime and selective WAIT research, with leakage-safe experiments |
| [16 — Day 01 kickoff](docs/16_day1_2026_11_20_kickoff.md) | Tasks and acceptance checks for 20 November 2026 |
| [17 — Forecast target contract template](docs/17_forecast_target_contract_template.md) | Fill before training any model |

## Research and build architecture

```text
BTC/USD exchange candles / XAU/USD gold quote provider
                 |
          Raw data storage
                 |
   Validation + session normalization
                 |
    Technical features + ML labels
                 |
        Versioned strategy engine
                 |
    Candidate BUY / SELL / WAIT
                 |
       Signal quality & risk gate
           /            \
     Chart/indicator   Signal journal
                           |
                   Notifications later
                           |
                  Outcome evaluation
```

## Proposed repository layout (not implemented yet)

```text
ML-predict-market/
├── README.md
├── CHANGELOG.md
├── .gitignore
├── docs/                      # Chapters, research and decisions
├── configs/                   # Venue, symbols, sessions, strategy versions
├── data/                      # Local only; never upload sensitive or large data
├── notebooks/                 # EDA and exploratory learning
├── src/ml_predict_market/
│   ├── data/                   # OHLCV ingestion and validation
│   ├── features/               # Lagged returns, EMA, RSI, ATR
│   ├── models/                 # Baselines and ML training
│   ├── indicators/             # Visual market overlays
│   ├── strategies/             # Versioned candidate signals
│   ├── alerts/                 # Signal delivery and dedupe
│   ├── risk/                   # Signal quality and cost gates
│   └── evaluation/             # Historical replay and signal outcomes
├── tests/
└── reports/                   # Experiment results, charts, metadata
```

## How to study and document as a Data Science student

1. Read **goal → theory → formula → exercise → test → outcome** in the relevant chapter.
2. Implement one small experiment. Save the exact data source, time period, features, train/test split and random seed.
3. Compare the result with a **simple baseline** and a **no-trade** benchmark, after estimated fees/spreads.
4. Record real results (including failures) in the [progress log](docs/06_progress_and_decisions.md). Do not mark work complete merely because it is planned.
5. Use a pull request with tests and a concise change log for each meaningful milestone.

### Current completion status (19 November 2026)

- [x] Vision drafted
- [x] Requirements for three tools drafted
- [x] Initial scope and evaluation rules documented
- [x] Update two research instruments to BTC/USD and XAU/USD
- [ ] Verify separate BTC/USD and XAU/USD data providers
- [ ] Market data downloaded and verified
- [ ] Initial EDA notebook executed
- [ ] Rules tested without look-ahead bias
- [ ] Baseline ML model trained
- [ ] Walk-forward backtesting completed
- [ ] Signal Bot implemented
- [x] Auto Bot deferred from active MVP
- [x] Document long-term market-agnostic architecture and five-timeframe research scope
- [ ] Confirm provider support for all five timeframes on both test markets
- [ ] Demonstrate incremental value of added timeframes and unsupervised features on unseen data
- [ ] 1H+5M Signal Bot implemented and independently tested
- [ ] Historical and forward paper-signal outcome reports

**Disclaimer:** This is educational/experimental research, not a licensed brokerage, financial advice, or a guarantee of returns. Actual trading and regulated services require separate legal, operational and risk review.

## Product direction clarification — 19 November 2026

The research **Prediction Engine** estimates next-horizon direction or return first. Its outputs may later power (1) a non-executing Auto Signal Bot, (2) an explanatory chart Indicator, and (3) a separately risk-gated **paper-only** Auto Bot. These are future modules, not three completed 30-day deliverables. See [Chapter XIV](docs/14_prediction_to_three_products.md) and the future-product tab of the [Google Sheet tracker](https://docs.google.com/spreadsheets/d/1BlK-1Nl9fo4qgpTZ9tEFypvKVsfMJUyX1IYbEKrNqp8/edit).

## Prepared for 20 November 2026

The **Adaptive Market Prediction Engine (AMPE)** is the current mathematical research hypothesis: predict forward return/direction, model volatility/uncertainty separately, optionally study market regimes and validate a selective WAIT policy. This is not yet a trained model or proven trading strategy. See [Chapter XV](docs/15_adaptive_market_prediction_engine.md) and [Day 01 checklist](docs/16_day1_2026_11_20_kickoff.md). The [Google Sheet](https://docs.google.com/spreadsheets/d/1BlK-1Nl9fo4qgpTZ9tEFypvKVsfMJUyX1IYbEKrNqp8/edit) contains separate AMPE and Day 01 tracking tabs while preserving the original 30-day tracker.

## Day 01 implementation underway — 20 November 2026

A [draft forecasting target contract](docs/18_day01_forecast_contract_draft.md), [data provider research](docs/data_source_research.md), and [pure-Python example](examples/day01_targets.py) with [11 synthetic fixture tests](tests/test_day01_targets.py) have been added. To run from repo root:

```bash
python3 -m unittest discover -s tests -p 'test_day01*.py' -v
python3 -m examples.day01_targets
```

**Important:** Tests passed in an isolated assistant environment, not on the user's Mac; real data source access and user signoff remain pending. No trained predictive models exist yet.

## Current research strategy — 26 November 2026

[Chapter XIX — Adaptive Hybrid ML + TSA](docs/19_hybrid_selective_forecast_strategy.md) is the current experiment specification. XGBoost is the first advanced forecasting candidate after no-skill, classical TSA and simple supervised baselines. GARCH, causal regime features and selective prediction are individually tested additions, not presumed improvements. Preserve Day 1's draft and existing tracker dates; no live trade execution or proven win rate. The earlier 30-day plan remains a bounded engineering MVP.

## Data foundation — 26 November 2026

[Chapter XX](docs/20_day02_data_foundation.md) provides a read-only Coinbase connectivity probe and six fixture tests. Actual provider data access, licensing, Mac execution and test results remain unverified. Run the probe only after checking permitted use; no credentials or orders.

## Python 3.12 test-discovery troubleshooting

Run from the repository root (the directory containing `README.md`, `examples/` and `tests/`):

```bash
python3 -m unittest discover -s ./tests -t . -v
```

`tests/__init__.py` makes the local test directory a regular Python package. The explicit start and top-level paths also avoid ambiguous resolution of another installed package called `tests`. If discovery still fails, check `pwd` and `ls tests` before changing the Python environment. The presence of a discovery fix does not establish that all tests pass on your machine.

## First historical data collection

[Chapter XXI](docs/21_first_historical_dataset.md) provides the seven-day BTC/USD collector and instructions. Six new fixture tests are supplied; real feed access, licensing and local execution remain pending. No model is trained.

## Market EDA and TSA

[Chapter XXII](docs/22_market_eda_tsa.md) documents the data audit and exploratory diagnostics. Four new fixture tests passed in the assistant environment; real dataset execution remains pending. Install requirements-eda.txt and run the chapter commands from the repository root.

## First local market-data evidence — 26 November 2026

The user reported a successful five-minute BTC/USD EDA run: 2,016 candles, zero missing intervals. ADF/KPSS support starting with log returns rather than log price in this seven-day sample. [Results and limitations](docs/22_market_eda_tsa.md#first-observed-btcusd-smoke-test-results--26-november-2026). This is a pipeline smoke test, not a validated prediction model.

## Feature and target engineering

[Chapter XXIII](docs/23_causal_feature_dataset.md) records the user's full EDA findings and introduces a close-only development dataset with separate features and labels, an explicit assumed replay delay, and five fixture tests pending execution. Longer history and purged chronological evaluation are required before model claims.
