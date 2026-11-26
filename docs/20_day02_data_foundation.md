# Chapter XX — Data foundation and provider access probe

**26 November 2026. Status:** implementation draft; no verified historical dataset yet.

## Target decision
Proceed with both reference log-return regression and UP/NOT_UP classification as a provisional engineering default. This keeps the existing Day 1 code and Chapter XIX strategy compatible. The user's agreement to continue is not an explicit approval of every target convention. Retain exact-zero treatment, uninterrupted 5m/1h horizons, and pending local review.

## Step 1 — Historical BTC/USD probe
Read the [Coinbase candle endpoint](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles). A bounded retrieval attempt through the assistant web environment did not return a payload. This does not prove Coinbase is inaccessible from the user's Mac.

After confirming permitted research use, run from repo root:

    python3 -m examples.day02_coinbase_probe --start 2026-11-20T00:00:00Z --end 2026-11-20T01:00:00Z
    python3 -m unittest discover -s tests -v

The probe requests 12 five-minute bars for a one-hour window. It sorts, validates OHLC, rejects conflicting duplicates, filters out-of-window bars, and reports missing timestamps. It does not save prices or fabricate absent candles. The six fixture tests are provided for local execution; their results are not yet verified. No API key required for this public endpoint.

## Step 2 — XAU/USD
Check eligible OANDA account access, permitted market-data use and XAU_USD instrument availability. Do not deposit funds or share credentials merely to obtain research data. See https://developer.oanda.com/rest-live-v20/instrument-df/. If access is blocked, retain a separate unverified gold adapter and work on BTC first.

## Step 3 — Next engineering milestone
Only after a successful probe, plan paginated raw retrieval, immutable manifests, session-aware gap auditing, OHLC normalization, publication timestamp policy and causal feature alignment. Historical API responses do not reveal actual original publication latency; do not pretend the retrieval time or candle end establishes exact historical availability. Document conservative replay assumptions and test sensitivity before forecasting.

## Learning
A one-hour elapsed target on a 5m grid spans twelve consecutive intervals. Missing timestamps are not zero returns. OHLC validity requires high >= max(open, close) and low <= min(open, close). This is data quality, not model accuracy. No training or order placement yet.

[Strategy](19_hybrid_selective_forecast_strategy.md) | [Day 1 draft](18_day01_forecast_contract_draft.md)
