# Chapter X — Signal Bot MVP (1H + 5M) for BTC/USD and XAU/USD

**Date:** 19 November 2026  
**Status:** Hypothesis/design only — unimplemented and not empirically validated.  
**Product:** Research alerts ONLY. No automatic orders, live execution, custody or client funds. LLM/NLP is a separate education topic and not required.

## Goals

1. Detect potential market setups on BTC/USD and XAU/USD with explainable, testable rules.
2. Use the **last completed 1-hour candle** for context, and a **completed 5-minute candle** for trigger.
3. Filter or cancel signals on data freshness, session state, implausible risk, estimated costs and duplicates.
4. Compare against simpler baselines in honest chronological tests, keeping all alerts/outcomes.
5. Publish clear non-executing alerts only after replay and quality tests pass.

## Version 0.1 — hypothesis, not a proven strategy

**1H context**
- BULLISH if last confirmed 1H close > its EMA20 > EMA50.
- BEARISH if last confirmed 1H close < EMA20 < EMA50.
- NEUTRAL otherwise, including insufficient warm-up or stale data.

**5M entry research**
- For BULLISH 1H, watch for a completed 5M pullback to/through EMA20.
- Only a **later, completed** 5M close above EMA20 and the highest high of the **preceding three completed 5M candles** is a BUY candidate.
- Mirror for bearish SELL candidates. For spot BTC, SELL should not be mislabeled as an authorized short.
- All thresholds (lookback, timeout, stop distance and cost limits) are research parameters to preregister and compare on validation data, not tuned on the final test.

**Quality gate**
- Current source and confirmed candles, no stale 1H context, valid session, non-duplicate signal, a plausible structural invalidation/stop distance, costs small enough to justify recording a candidate.
- If any critical requirement fails, emit WAIT or CANCELLED with a reason. The engine never automatically places a trade.

**Signal stages**

\`\`\`text
NO_SETUP -> H1_BIAS -> WATCH_PULLBACK -> ARMED -> 5M_TRIGGERED
     -> RISK_AND_COST_CHECK -> ALERT / REJECTED
     -> MONITORING -> EXPIRED / INVALIDATED / HYPOTHETICAL_OUTCOME
\`\`\`

Each state transition receives a UTC timestamp, source, symbol and strategy version.

## Critical anti-leakage rule

At 11:05 UTC, a 5M signal can see the H1 bar **ending at 11:00**, once that bar has actually been published, but never the final H1 bar ending at 12:00. All joins are **backward-as-of on available_at**, not naive equal timestamps. Test the boundary, publication latency, candle incompleteness and replay behavior.

## Per-instrument reality

| | BTC/USD | XAU/USD |
|---|---|---|
| Feed | Actual USD-quoted BTC spot exchange | Documented gold provider, with indicative/OTC/CFD quote type recorded |
| Sessions | Normally 24/7 except outages | Provider-specific weekday sessions, daily breaks and holidays |
| Price unit | USD per BTC | USD per troy ounce |
| Volume | Venue traded BTC when supplied | May be missing/tick-only |
| Costs | Exchange bid/ask spread and fees | Provider spread and any product-specific financing assumptions |
| Backtests | Independent data provenance | Independent source/quote/calendar verification |

**Do not** silently substitute BTC/USDT for BTC/USD, or treat indicative gold quotes as executable physical-gold prices.

## Signal output

Example fields (not live values):
- signal_id, source/provider, instrument, quote_type, mode=RESEARCH_ONLY
- 1H bar close UTC and available_at, 5M trigger bar close UTC and available_at
- state, direction BUY_CANDIDATE / SELL_CANDIDATE / WAIT / CANCELLED
- reason codes, rule version, optional experimental entry reference and stop/target reference
- planned expiry, data freshness, estimated costs, notification sent time and final research outcome

## Performance evaluation: must compare with baselines

The central **experimental question** is whether 1H+5M improves outcomes versus a 5M trigger alone. Report per market:
- Count of all opportunities, candidates, rejects, alerts and expirations; coverage and false-signal definition.
- Win/loss rate under a *documented* simulated trigger/exit policy.
- **Average hypothetical net R per signal after spread, slippage and fees**, not win rate alone.
- Maximum hypothetical drawdown, signal frequency, payoff ratio, losing streaks.
- Alert latency, dedupe rate and data-staleness incidents.
- Sensitivity across chronological periods, market regimes, reasonable cost scenarios and timeframe variations.

Use walk-forward chronological validation with a held-out final period. Account for target overlap and ambiguous stop/target sequencing inside a 5M candle. Report failures and uncertainty. Never label model scores as winning probabilities without calibration.

## Engineering acceptance tests

- [ ] 1H incomplete candle is never used for 5M signal.
- [ ] Same historical input and version produce identical signals.
- [ ] Signals only occur on complete, available candles.
- [ ] Weekend BTC trading and closed gold sessions behave independently.
- [ ] Spread/fees/slippage are not zero by default in replay.
- [ ] Alert duplicate IDs are suppressed and stale setups expire.
- [ ] Rejected setups and no-signal periods are recorded.
- [ ] There is no order-placement endpoint in the Signal Bot.
- [ ] Separate BTC/USD and XAU/USD held-out evaluations are produced.

## Sources and technical reading
- [TradingView — Repainting](https://www.tradingview.com/pine-script-docs/concepts/repainting/) for future/unfinished higher-timeframe signals.
- [TradingView — Other data and timeframes](https://www.tradingview.com/pine-script-docs/faq/other-data-and-timeframes/) for HTF reference behavior.
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) for forward validation (may require purge when labels overlap).
- [Coinbase Advanced Trade](https://coinbase.github.io/coinbase-advanced-py/coinbase.rest.html) as a **candidate** BTC-USD source; not yet verified.
- [OANDA API](https://developer.oanda.com/rest-live-v20/instrument-ep/) as an example of gold historical market-data APIs; gold access and product type vary.

**Completion status:** Design drafted; all actual tests, data feeds, signals, alerts, and backtest performance **not started**.

## Scope extension — five-timeframe architecture (not five-timeframe strategy yet)

The project now targets 1m / 5m / 1h / 4h / 1d feature availability for supported markets. This chapter is still the **baseline 1h+5m v0.1** experiment. Additional timeframes must be tested as independent research additions, not enabled automatically. Longer-timeframe data must be finalized and available before any shorter-timeframe decision. Cross-provider resampling and session boundaries must be explicit. A train-only regime cluster might later be a feature, not a guaranteed signal or substitute for forward evaluation. See [Chapter XII](12_multimarket_multitimeframe_unsupervised.md).
