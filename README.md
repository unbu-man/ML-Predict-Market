# ML Predict Market — Forex & Crypto Research Lab

**Founder / learner project · Started 18 November 2026 · Status: documentation only**

> Build an explainable **market indicator**, a **signal bot**, and an **automated paper-trading bot**, using a shared Data Science and quantitative research pipeline. **No strategy has yet been validated; no live trading is implemented.**

## Our three products

| Product | What it does | Initial delivery | Trade execution? |
|---|---|---|---|
| Market Intelligence Indicator (MII) | Charts trend, momentum, volatility, BUY/SELL/WAIT research labels | Python charts, then optional TradingView Pine Script | No |
| Signal Bot | Sends timestamped, explainable and deduplicated market alerts | Dashboard, then Telegram | No |
| Auto Bot | Applies position limits and executes simulated orders | Paper account, ledger, kill switch | **Paper only** |

The bots are **separate customer experiences** built on a shared, reproducible **Strategy Engine**. A BUY label does not automatically authorize an order.

## Starting research defaults

- **Crypto:** BTC/USDT, 1-hour candles, spot data from one recorded venue; add ETH/USDT later.
- **Forex:** EUR/USD from a specified FX quote provider; add USD/JPY later. FX is not a centralized exchange and has session, timezone and spread differences.
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

## Research and build architecture

```text
Crypto exchange candles / FX quote provider
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
       /           |           \
  Indicator     Signal bot    Risk engine
   charts          alerts          |
                              Auto paper bot
                                   |
                          Simulated fills, ledger
                                   |
                         Backtests and monitoring
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
│   ├── risk/                   # Paper-only risk controls
│   └── simulation/             # Paper fill and ledger engine
├── tests/
└── reports/                   # Experiment results, charts, metadata
```

## How to study and document as a Data Science student

1. Read **goal → theory → formula → exercise → test → outcome** in the relevant chapter.
2. Implement one small experiment. Save the exact data source, time period, features, train/test split and random seed.
3. Compare the result with a **simple baseline** and a **no-trade** benchmark, after estimated fees/spreads.
4. Record real results (including failures) in the [progress log](docs/06_progress_and_decisions.md). Do not mark work complete merely because it is planned.
5. Use a pull request with tests and a concise change log for each meaningful milestone.

### Current completion status (18 November 2026)

- [x] Vision drafted
- [x] Requirements for three tools drafted
- [x] Initial scope and evaluation rules documented
- [ ] Market data downloaded and verified
- [ ] Initial EDA notebook executed
- [ ] Rules tested without look-ahead bias
- [ ] Baseline ML model trained
- [ ] Walk-forward backtesting completed
- [ ] Signal Bot implemented
- [ ] Auto **paper** Bot implemented

**Disclaimer:** This is educational/experimental research, not a licensed brokerage, financial advice, or a guarantee of returns. Actual trading and regulated services require separate legal, operational and risk review.
