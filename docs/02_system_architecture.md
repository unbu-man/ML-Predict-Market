# Chapter II — System Architecture and Safety

**Status:** Design only.

## Learning objectives
Practice modular software architecture, typed event contracts, deterministic replay, observability and safety boundaries.

## Data and decision flow

    MarketDataProvider -> RawStore -> DataValidator
                                         |
                                     FeatureEngine
                                         |
                             RuleStrategy / ML Research
                                         |
                                CandidateSignal
                            /          |          \
                      Indicator    AlertPublisher  PaperRiskEngine
                     (charts)       (no orders)        |
                                                SimulatedOrderRouter
                                                        |
                                                PaperFill -> VirtualLedger
                                                        |
                                                   EvaluationStore

The **auto-bot risk engine is independent** of indicator scores, ML probabilities and message delivery. No module in v0.1 uses credentials capable of real order placement.

## Contracts to design

### Candle
Fields: symbol, venue/provider, asset_class, timeframe, open_time_utc, close_time_utc, open, high, low, close, volume_or_null, bid_or_ask_if_available, complete.

Rules: nonnegative, internally consistent OHLC values; UTC timestamps; one row per provider/symbol/interval; explicit completeness and instrument precision. Missing data is reported, not silently fabricated.

### SignalEvent
Fields: signal_id, strategy_version, symbol, timeframe, source, candle_close_utc, created_at_utc, action (BUY/SELL/WAIT), feature_snapshot, explanation, model_version_or_null.

Signals are derived **only from completed candles available before decision time**. A score based on indicators is not a calibrated probability.

### PaperOrderProposal
Fields: account_id, signal_id, symbol, side, proposed_qty, max_notional, strategy_version, expiry, idempotency_key.

RiskEngine may return: APPROVED, REJECTED (with reason), or EXPIRED. The simulator records the configured bid/ask spread, fees, slippage and fill assumptions.

## Boundaries and tests
- DataValidator: reject negative prices, invalid high/low and duplicate timestamps.
- FeatureEngine: test warm-up periods and look-ahead leakage.
- StrategyEngine: deterministic results for a fixed data set and version.
- SignalBot: duplicate alerts are not sent twice; stale feeds do not create new alerts.
- PaperRiskEngine: kill switch, position exposure and daily loss limits always win over a BUY signal.
- PaperLedger: balances, fills and positions reconcile; retries cannot double-fill.
- MarketCalendar: FX weekends/holidays and crypto venue interruptions are handled separately.

## Proposed repository directories
- src/ml_predict_market/data — providers, snapshots and checks
- src/ml_predict_market/features — technical and statistical features
- src/ml_predict_market/models — pipelines, training and inference
- src/ml_predict_market/strategies — versioned rules
- src/ml_predict_market/indicators — chart outputs
- src/ml_predict_market/alerts — signal formatting/delivery
- src/ml_predict_market/risk — risk gates
- src/ml_predict_market/simulation — orders and ledger
- tests/ — unit, property, integration, time-leakage and replay tests

## Exercise
Draw a sequence diagram for one completed hourly candle that triggers a candidate BUY but gets rejected by the paper risk engine. Record the rejection reason.
