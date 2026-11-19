# ML Predict Market — Bitcoin & Gold Research Lab

**Founder / learner project · Started 18 November 2026 · Status: documentation only**

> **Current MVP: build and validate a non-executing Signal Bot first** for BTC/USD and XAU/USD. The indicator supports research/visualization; the Auto Bot is **deferred**. No strategy has yet been validated and no live trading is implemented.

## Product priorities — 19 November 2026

| Product | What it does | Initial delivery | Trade execution? |
|---|---|---|---|
| Market Intelligence Indicator (MII) | Charts trend, momentum, volatility, BUY/SELL/WAIT research labels | Python charts, then optional TradingView Pine Script | No |
| Signal Bot | Sends timestamped, explainable and deduplicated market alerts | Dashboard, then Telegram | No |
| Auto Bot | Deferred idea, not in current MVP | Future only, after demonstrated signal quality and separate review | **Not implemented** |

**Active workflow:** historical 5M/1H data → validated, non-repainting signal rules → cost-aware backtest → signal journal and alerts. There is **no trade execution endpoint** in the MVP. The old Auto Bot concept remains documented for potential later review.

## Starting research defaults

- **BTC/USD:** Bitcoin priced in real USD. Study 1-hour completed crypto-spot candles from an identified USD-quoted venue (Coinbase `BTC-USD` is a candidate, not yet selected or downloaded).
- **XAU/USD:** Gold quoted in USD per troy ounce. Study independent historical gold quotes from a named provider; label whether they are indicative spot/OTC or broker CFD quotes. XAU/USD is **not** EUR/USD and is not an ordinary fiat-forex cross.
- **Market design:** Separate data adapters, market calendars, costs and prediction evaluations. The indicator and Signal Bot share a versioned strategy interface; Auto Bot is deferred.
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
- [ ] 1H+5M Signal Bot implemented and independently tested
- [ ] Historical and forward paper-signal outcome reports

**Disclaimer:** This is educational/experimental research, not a licensed brokerage, financial advice, or a guarantee of returns. Actual trading and regulated services require separate legal, operational and risk review.
