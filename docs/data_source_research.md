# Day 1 — BTC/USD and XAU/USD Data Source Research

**Date:** 20 November 2026
**Status:** Official developer docs inspected. **No user account, API credentials, historical data ingestion, full license check or actual source access verified yet.**

## BTC/USD — Candidate: Coinbase Exchange `BTC-USD`

- Official REST endpoint: `GET https://api.exchange.coinbase.com/products/BTC-USD/candles`.
- Coinbase Exchange documentation describes `granularity` values `60, 300, 900, 3600, 21600, 86400` seconds (1m, 5m, 15m, 1h, 6h, 1d). **No native 4h** in this endpoint; use carefully validated 1h resampling if suitable.
- Maximum response: **300 candles per request**; backfilling requires bounded windows and request throttling. Coinbase warns that historical periods can have no records if no trades occur. This is a data gap, not automatically a zero price change.
- Candle bucket `time` is the **start time**, so normalize the 5m bar end as `start + 5m`; document observed source delay before assigning `available_at`.
- Historical OHLC based on trades is not the same thing as a buyable/sellable bid/ask price. Store fee/spread assumptions separately.
- Documentation: https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles
- **Unknowns before coding adapter:** product status from API, actual connectivity, backfill history, rate limits, data-use rights, candle publication latency and quality report.

## XAU/USD — Candidate: OANDA v20 `XAU_USD` (if entitled)

- OANDA v20 instrument documentation defines native M1, M5, H1, H4 and D granularities, and a `complete` candle flag.
- Its historical candle endpoint supports mid (M), bid (B) and ask (A) quote components. We must choose one consistent label price and describe limitations in trade-cost simulation.
- An eligible v20 account and access token are required for authenticated API use. The instrument list depends on regulatory division/account, and access is **not** guaranteed for a Cambodia-based account. Treat `XAU_USD` as **candidate symbol, not an account-verified instrument**.
- Gold may represent a provider's CFD/OTC or indicative quote, not a universally executable spot bullion market. Verify actual instrument type, USD per troy ounce conventions, daily breaks, weekend hours, holidays, DST and daily candle alignment.
- API docs: https://developer.oanda.com/rest-live-v20/instrument-df/
- Account availability: https://developer.oanda.com/rest-live-v20/account-ep/
- Access restrictions: https://developer.oanda.com/rest-live-v20/introduction/
- Endpoint: https://developer.oanda.com/rest-live-v20/pricing-ep/
- **Unknowns:** regional eligibility, accessible `XAU_USD` in account instrument list, licensing, actual history and pricing components.

## Day 1 next steps

| Check | Coinbase `BTC-USD` | OANDA `XAU_USD` |
|---|---|---|
| Documented granularities for 5m+1h | Yes | Yes |
| Five-frame support | 4h needs derivation | M1/M5/H1/H4/D in docs; account support untested |
| Correct exact pair usable on account | Not live-verified | Not verified |
| History / quality tested | Not yet | Not yet |
| License / distribution rights assessed | Pending | Pending |
| Provider sessions/cost model | Pending | Pending |
| Status | **Candidate** | **Candidate** |

If gold access is unavailable, keep its adapter a **blocked integration** with clearly tagged synthetic test fixtures; do not claim historical XAU/USD model results. Never commit credentials or client data. We can validate the mathematical engine separately from provider access.

**No data ingestion or asset-specific signal accuracy is claimed here.**
