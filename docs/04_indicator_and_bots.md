> **Scope update — 19 November 2026:** The active deliverable is the non-executing Signal Bot. The indicator is a research visualization. The Auto Bot below is preserved as **deferred historical design**, not current build scope. See [Chapters X](10_signal_bot_mvp.md) and [XI](11_8_week_execution_plan.md).

# Chapter IV — Indicator, Signal Bot and Auto Bot

**Status:** Product specs; not implemented.

## A. Market Intelligence Indicator (MII v0.1)
Purpose: visually explain market state rather than promise profitable entries.

### Candidate components
- EMA(20) and EMA(50): short- and longer-horizon trend
- RSI(14): momentum characterization, not a deterministic reversal trigger
- ATR(14): historical range, used for risk-range visualization
- Candle timestamp, timezone, data source and completeness status

### Starting rules (unvalidated hypotheses)
- **BUY candidate:** completed-candle close > EMA20, EMA20 > EMA50, RSI between 50 and 70.
- **SELL candidate:** completed-candle close < EMA20, EMA20 < EMA50, RSI between 30 and 50.
- **WAIT:** otherwise, or if data is stale/insufficient.
- Warm-up periods produce no signals. A strategy version is stored with each label.
- ATR may define a *hypothetical* stop distance; never present a stop as a loss guarantee.

### Acceptance tests
Repeatable results from identical data; chart and backend match; signals reference only finalized candles; alert values never change after a candle closes (non-repainting); chart explains WHY BUY/SELL/WAIT occurred.

## B. Signal Bot v0.1
Purpose: publish research alerts, not execute trades.

**Input:** a versioned SignalEvent from the strategy engine.  
**Output example:**

    [PAPER/RESEARCH] BTC/USD · 1h
    Candle closed: 2026-11-19 12:00 UTC
    Candidate: BUY
    Reasons: EMA20>EMA50, Close>EMA20, RSI=54
    Source: exchange X; strategy: mii-0.1
    Not a guarantee. Not an executable order.

Publish through a local log before attempting Telegram/web delivery. Apply deduplication by signal_id, expiration checks, rate limiting, retry policy and opt-out. Record delivery attempts and failures. No broker credentials or ability to place orders.

## C. Auto Bot v0.1 — Paper Trading Only
Purpose: demonstrate automated decision-to-simulated-trade flow under independent safety constraints.

**Pre-order checks:** active paper account; current quote; open market/session; permitted instrument; nonzero balance; quantity/tick precision; max order notional; max portfolio exposure; max daily loss; retry/idempotency policy; kill switch.

A BUY/SELL label may be rejected. Logs must show the rejection reason. The simulator uses explicit fees, spread and slippage parameters and records an immutable paper fill and virtual balance/position update.

**Never include live-order methods or API keys in the first implementation.** Live market integration would be a new project phase with compliance and security gates.

## D. Shared signal schema
Minimum fields: signal_id, strategy_version, symbol, venue/source, candle_close_utc, timeframe, action, reasons, feature_values, generated_at_utc, model_version_or_null.

## E. ML integration later
A calibrated direction forecast can be a separate input to a decision policy after honest out-of-sample testing. Do not relabel an arbitrary indicator agreement score as 'confidence 90%' or a probability of profit.

## Exercise
Write three unit-test cases:
1. Indicator returns WAIT for insufficient warm-up data.
2. Signal Bot sends the same signal at most once.
3. Paper Auto Bot rejects a BUY if the kill switch is active.

## Both markets, one system

One indicator, signal bot and paper-only auto bot support BTC/USD and XAU/USD using the same signal schema. Each instrument has independent signals, stop distances, position sizes, quote provenance and model evaluations. XAU/USD signals must respect provider-defined gold market closures; BTC/USD may continue evaluating during weekends. Never map BTC/USDT stablecoin bars into BTC/USD without explicit conversion and research justification.
