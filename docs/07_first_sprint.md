# Chapter VII — First Seven-Day Sprint

**Suggested first sprint:** the next seven working sessions after project documentation.  
**Goal:** reproducible BTC/USDT data inspection and a basic indicator hypothesis; not a trading bot yet.

## Day 1 — Select and understand market data
- [ ] Choose a specific crypto exchange/venue and BTC/USDT symbol.
- [ ] Review OHLCV, bid/ask, completed hourly candles, UTC and data usage terms.
- [ ] Record source, coverage, sampling, rate limits and known limitations.

**Artifact:** docs/data_provider_choice.md (planned).

## Day 2 — Set up Python
- [ ] Create isolated Python virtual environment.
- [ ] Start with numpy, pandas, matplotlib and pytest; pin verified versions when chosen.
- [ ] Create src, tests, notebooks and local data directories.
- [ ] Confirm test runner executes.

**Example macOS/Linux:**

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install numpy pandas matplotlib pytest
    python -m pytest

Do not commit .venv or credentials.

## Day 3 — Download a bounded sample
- [ ] Use provider rules to download **completed** 1-hour historical candles.
- [ ] Preserve UTC time, source and retrieval date.
- [ ] Save raw data locally; commit only a small licensed test fixture if allowed.

**Artifact:** collection script + dataset manifest.

## Day 4 — Audit market data
- [ ] Check sorted unique timestamps, missing hours, duplicates, OHLC consistency.
- [ ] Note exchange-specific missing data; avoid blindly filling trading gaps.
- [ ] Add validator tests including negative and inconsistent prices.

**Artifact:** reports/data_quality_001.md (planned).

## Day 5 — Exploratory Data Analysis
- [ ] Plot close, hourly log returns and rolling historical volatility.
- [ ] Explore missingness, tails and time-of-day distributions.
- [ ] Explain what is **descriptive**, not predictive.

**Artifact:** notebooks/01_market_eda.ipynb (planned).

## Day 6 — Indicator hypothesis
- [ ] Compute EMA20/50, RSI14 and ATR14 on completed candles.
- [ ] Explain warm-up NaNs and align signals to decision time.
- [ ] Produce descriptive BUY/SELL/WAIT rules and test non-repainting.

**Artifact:** charts + tests (planned).

## Day 7 — Reproducibility and progress review
- [ ] Re-run scripts from a clean environment.
- [ ] Confirm timestamps, source and data permissions.
- [ ] Write results, errors and next sprint tasks in Chapter VI.

**Artifact:** one documented weekly report with links to code/tests.

## Definition of done
Source is documented; data issues are quantified; EDA and indicators are reproducible; no future information appears in signal features. No claims of model accuracy or profitability are made.

**Week 2 preview:** baseline time-series classification, walk-forward evaluation and the first fee-aware historical strategy replay.
