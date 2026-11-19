# Chapter XIV — Prediction Engine to Three Product Modules

**Created:** 19 November 2026  
**Status:** Design proposal only; not implemented or tested.  
**Google Sheet:** https://docs.google.com/spreadsheets/d/1BlK-1Nl9fo4qgpTZ9tEFypvKVsfMJUyX1IYbEKrNqp8/edit

## Mission and sequence

Create one market-independent **Prediction Research Engine** first. Train and validate next-move direction or future-return models using market-specific point-in-time features. The first validation markets are BTC/USD (genuine USD spot) and XAU/USD (documented gold quote type). Five desired feature timeframes are 1m, 5m, 1h, 4h and 1d. Product development follows **Prediction Engine → Auto Signal Bot → Custom Market Indicator → Paper Auto Bot**. The signal bot and indicator may be built in parallel once the prediction schema stabilizes.

## Four explicit boundaries

**0. Prediction research engine.** Inputs: validated source candles and features whose *available_at* precedes prediction time; labels: exactly defined forward log return over 5m and 1h horizons. Outputs: model_id/version, market/provider, UTC as-of cutoff, prediction horizon, estimated return and, only if truly calibrated, probability and uncertainty; label time when later outcome becomes observable. Compare against persistence, no-skill, Logistic Regression, then optional XGBoost. Train-only unsupervised clustering is a separate feature ablation.

**1. Auto Signal Bot (no orders).** Consumes confirmed model forecasts, estimated fees/spread/slippage, market session and stale-feed status. Produces WAIT / BUY_CANDIDATE / SELL_CANDIDATE, expiry, reasoning from actual measured features, alert journal and optional Telegram notifications. Even if model has no demonstrated edge, observational research alerts may remain useful if prominently labeled as experimental.

**2. Custom Indicator (visual only).** Renders confirmed trend/regime/forecast overlays and alert states on charts using the same versioned backend events. Test no look-ahead, no repainting, matching bar timestamps and visual parity with Python result. Optional TradingView Pine Script is a display/proxy adapter, not necessarily able to run the trained Python model internally.

**3. Auto Bot (paper first; deferred).** Independent risk/order-intent processor, with maximum hypothetical position sizing/exposure, fees and slippage, stops, drawdown gate, kill switch, simulation fill ledger, idempotency and failure reconciliation. A signal MUST NOT directly become an execution order. No live-money trading in this research MVP.

## Prediction event contract (draft)

`prediction_id, instrument_id, provider_id, source_price_type, timeframes_used, feature_cutoff_utc, available_at_utc, model_version, horizon, target_definition, prediction_value, calibrated_probability_nullable, uncertainty_metadata_nullable, expiry_utc, stale_data, market_open, outcome_nullable`.

A probability is not trustworthy just because an ML API returned a number: calibration must be evaluated on out-of-sample periods. A model working for BTC/USD does not imply it works for gold or other markets.

## Independent acceptance gates

| Stage | Evidence required before adding complexity |
|---|---|
| Prediction Core | Correct point-in-time data, reproducible baseline models, chronological held-out report, leakage tests, model card |
| Auto Signal | No trading credentials/order calls, stale signal rejection, durable deduplicated alerts, tested expiry and forward replay |
| Indicator | Matches backend predictions on the exact historical bar and uses no unpublished higher-timeframe data |
| Paper Auto Bot | Paper-only positions, strict independent risk limits, audit and reconciliation tests, kill switch and separately reviewed scope |
| Future new markets | Data/licensing/session/cost adapters + independently measured out-of-sample metrics |
| Any future live trading | Explicit authorization, applicable regulation review, security/operational controls; not covered by this plan |

**30-day scope:** complete a bounded research-grade prediction engine with historical evaluation and a non-executing demonstration, not all three full products and not a guarantee of profitable forecasting. Keep the daily Google Sheet tracker in the first tab; the second tab **After 30 Days — Products** is for future status and personal progress notes.
