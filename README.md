# ML Predict Market — Multi-Market Signal Research Framework

**Founder / learner project · Started 18 November 2026 · Status: documentation only**

> **Long-term goal:** a configurable, non-executing Signal Bot research framework that can be tested on multiple market types. **First validation markets:** BTC/USD and XAU/USD. **Supported research timeframe targets:** 1m, 5m, 1h, 4h, 1d. The Auto Bot remains **deferred**. No strategy is yet validated or implemented.

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
