# Chapter XVIII — Day 1 Working Forecast Contract v0.1

**Date:** 20 November 2026
**State:** Working draft, not founder-approved or trained. Initial design choices have been specified for review.
**Related:** [AMPE strategy](15_adaptive_market_prediction_engine.md), [Day 1 checklist](16_day1_2026_11_20_kickoff.md), [source research](data_source_research.md), [Google Sheet](https://docs.google.com/spreadsheets/d/1BlK-1Nl9fo4qgpTZ9tEFypvKVsfMJUyX1IYbEKrNqp8/edit).

## Objective

Build a causal next-market-move forecasting engine. At each completed and **published** 5-minute candle on BTC/USD or XAU/USD, predict a **future reference-price log return** over (a) the next 5 minutes or (b) the next hour. Forecasting is separate from candidate signals, chart indicators, and any later paper trading. No live orders or profit claims.

## Prediction and label math

Let t be the timestamp at the **end** of a 5-minute source candle. A prediction is not issued before that candle is available. For h in {5m, 1h}, define:

```math
\begin{aligned}r_{t,h}&=\ln\left(\frac{C_{t+h}}{C_t}\right)\\y_{t,h}&=\begin{cases}1,&r_{t,h}>0\\0,&r_{t,h}\leq0\end{cases}\end{aligned}
```

Here $C_t$ is the anchor close, $C_{t+h}$ is the future close, and $h$ is the forecast horizon.

A zero return is **NOT_UP**, not necessarily a decline. A distinct ternary UP/DOWN/NEUTRAL scheme can be researched later and must be explicitly versioned. We expect to compare a no-skill return forecast, majority direction classifier, AR(1), and Logistic Regression before nonlinear methods.

The future close $C_{t+h}$ is only known after time passes and may be used to form a **retrospective training label**. It MUST NOT enter features available to the model at time t. The target is a **price reference change**, not realizable net trade return.

### Time horizon and gold closures

For h = 5m, demand exactly the **next complete consecutive 5-minute candle**. For h = 1h, demand **12 consecutive complete 5-minute bars** with timestamps spanning exactly one continuous hour, in the same provider-defined market session. Drop retrospective labels if the window crosses a missing bar, a session break, a weekend or a holiday. Do not silently treat the 12th gold trading candle after a closure as a one-hour elapsed-time return. An alternative *next trading-hour* target needs a separate name and experimental contract.

## Input publication rule

At 11:05 UTC, a 5-minute candle ending at 11:05 and the last completed 1-hour context ending at 11:00 may enter the feature set **only after each is published**. The 11:00–12:00 one-hour candle cannot provide final values at 11:05. Use fields `open_time_utc`, `close_time_utc`, `available_at_utc`; do not assume timestamps from all vendors have identical meanings.

## Minimum normalized candle contract

- `provider_id`, `provider_symbol`, `instrument`, `asset_class`, `product_type`
- `timeframe`, `open_time_utc`, `close_time_utc`, `available_at_utc`, `is_complete`
- `open`, `high`, `low`, `close`, optional `bid/ask`, nullable `volume` and volume meaning
- `market_calendar_id`, `session_id`, quote unit, source/licensing notes

Data validation: strictly positive finite prices, $H_t\geq\max(O_t,C_t)$, $L_t\leq\min(O_t,C_t)$, no duplicated source+instrument+timeframe+start keys, no future/incomplete candles, gap handling per source and session.

## Mathematics example (synthetic only)

Current price $100,000. Future (only observed later) $100,100 after 5m.

```math
\begin{aligned}r&=\ln\left(\frac{100100}{100000}\right)\\&=\ln(1.001)\\&\approx0.000999500333\\100r&\approx0.0999500333\\y&=1\quad(\mathrm{UP})\end{aligned}
```

The log return expressed as a percentage is approximately +0.0999500333%.

A forecast for this movement cannot use the future $100,100 value.

## Learning and implementation evidence

- [x] Candidate prediction times, horizons, labels and no-gap rules **drafted**.
- [x] Pure Python example and **11 local synthetic-fixture tests run and passed** in assistant environment: `examples/day01_targets.py`, `tests/test_day01_targets.py`. This is **not** a prediction performance result.
- [ ] User confirms the target and the UP/NOT_UP label choice.
- [ ] Historical providers accessed with legal data rights; real price data quality checked.
- [ ] User runs the tests on their Mac, checks output and records actual result.

Run from the repository root:

    python3 -m unittest discover -s tests -p 'test_day01*.py' -v
    python3 -m examples.day01_targets

**Status:** Day 1 **In progress**; target definition remains a *proposal* until reviewed. Day 2 will generalize the 5-minute-only illustrative Candle into a proper typed multi-timeframe production schema.

## 26 November continuation

Proceed provisionally with both numeric return and UP/NOT_UP labels for engineering, pending explicit review and Mac tests. See [Chapter XX](20_day02_data_foundation.md). Provider access and licensing remain unresolved; Day 1 is not marked complete.
