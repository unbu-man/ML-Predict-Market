# Chapter I — Vision and Project Scope

**Status:** Proposed design, not implemented. **Started:** 18 November 2026.

## What you will learn
Distinguish a forecast, an indicator, a signal and an execution policy. Translate an idea into engineering requirements with measurable acceptance criteria.

## Goal
Research an explainable BTC/USD and XAU/USD market analysis toolkit as a Data Science student. The project eventually contains three products, **powered by one shared data and strategy engine**.

| Product | Role | Initial implementation |
|---|---|---|
| Market Intelligence Indicator (MII) | Explain trend, momentum, volatility and signal conditions on charts | Python visualization, optional Pine Script later |
| Signal Bot | Send timestamped BUY/SELL/WAIT *research alerts* with reasons and caveats | Local event log, dashboard/Telegram later |
| Auto Bot | Apply strict position/risk checks then execute simulated orders | Paper-trading only, independent risk engine and kill switch |

A forecast estimates a future outcome. An indicator summarizes market observations. A signal is a *candidate decision*. An execution policy adds constraints that may reject the signal entirely.

## Scope
- **Bitcoin research:** BTC/USD spot candles, 1-hour interval, from one identified exchange with true USD quote currency.
- **Gold research:** XAU/USD (USD per troy ounce of gold), 1-hour quotes from one documented source. Identify indicative spot, OTC, or CFD-linked quotes accurately; no real execution.
- **First hypotheses:** EMA20/EMA50 + RSI14 + ATR14; simple ML baselines then Logistic Regression/XGBoost.
- **First technology:** Python, pandas, NumPy, matplotlib, pytest; FastAPI and a dashboard only after the research core passes tests.

These are starting assumptions, not validated trading parameters. BTC/USD spot markets are often available 24/7, subject to venue outages. XAU/USD historical quotes have gold-provider-specific weekly sessions, breaks, holidays and sometimes no meaningful volume field.

## Out of scope for first version
No real-money orders, customer deposits, custody, leverage, CFDs, perpetual futures, personal advice, copy-trading or profit claims. Production brokerage activities need separate jurisdiction-by-jurisdiction authorization and security review.

## What success means
A research milestone is complete only when **source, sample window, methods, tests, out-of-sample results and limitations** are reproducible. A paper-bot milestone requires simulated fills, a reconciled virtual ledger, no duplicate fills and non-bypassable risk controls.

## Exercise
Write a one-paragraph user problem statement: who needs this indicator, which decision it informs, what can go wrong, and what evidence would stop you shipping it? Put your answer in Chapter VI.

## Review questions
1. Why can 60% direction accuracy still lose money after costs?
2. What separates the signal bot from the auto bot?
3. Which differences between BTC/USD and gold XAU/USD are represented in the data model?
