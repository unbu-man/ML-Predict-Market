# Chapter IX — BTC/USD and XAU/USD Market Data Plan

**Confirmed instrument focus:** BTC/USD (Bitcoin/USD) and XAU/USD (Gold/USD).  
**State:** Planning, no market data verified or downloaded.

## Goal
Research and build one reusable indicator, one signal bot and one paper-only auto bot for both markets with separate market adapters, risk controls and empirical validation.

## Instrument matrix

| Requirement | BTC/USD | XAU/USD |
|---|---|---|
| Economic meaning | One Bitcoin priced in actual US dollars | One troy ounce of gold quoted in US dollars |
| Product research | Exchange-traded BTC spot | Provider-defined gold spot indicative/OTC or gold CFD quote **clearly labeled** |
| Candidate source | Coinbase Advanced Trade, product ID BTC-USD | Historical gold source not chosen; OANDA is only one candidate |
| Timeframe | 1-hour completed candles | 1-hour completed candles |
| Session | Normally 24/7, with venue outages possible | Provider-set gold weekday sessions, daily breaks and holidays |
| Volume | Venue-specific BTC traded volume if provided | Often missing or provider/tick-specific (not comparable) |
| Costs | Exchange commissions, bid/ask spread and slippage | Broker/provider spread; financing only if relevant to modeled CFD |
| Evaluation | Separate train/test windows and paper cost models | Separate train/test windows and paper cost models |

Coinbase API reference: https://coinbase.github.io/coinbase-advanced-py/coinbase.rest.html  
Gold sessions reference (one provider): https://www.oanda.com/bvi-en/cfds/hours-of-operation/

## Data registry (planning example; no API has been tested)

```yaml
instruments:
  BTC/USD:
    asset_class: crypto_spot
    quote_asset: USD
    candle_interval: 1h
    price_unit: USD_per_BTC
    candidate_provider: coinbase_advanced
    candidate_provider_symbol: BTC-USD
    calendar: provider_exchange_24x7
    status: unverified
  XAU/USD:
    asset_class: gold_quote_research
    quote_asset: USD
    candle_interval: 1h
    price_unit: USD_per_troy_ounce
    candidate_provider: unselected
    candidate_provider_symbol: null
    calendar: provider_gold_schedule
    status: unverified
```

## Data manifest needed for **each** instrument

- Provider, venue/entity, symbol and exact product type (exchange spot, indicative spot, OTC, CFD reference).
- Price field type: last-traded, bid/ask, mid or indicative quote.
- UTC timestamps and candle close semantics; provider time zone, DST, holidays and session breaks.
- Units and precision; whether volume exists, and what it represents.
- Collection method, coverage, row count, duplicate/stale/gap checks, and permissible redistribution.
- Market-specific backtest spreads, fees, slippage, and financing conventions if applicable.

## Data validation tests
- Reject BTC/USDT masquerading as BTC/USD.
- Reject gold samples labeled USD per BTC, and BTC samples labeled USD per troy ounce.
- Weekend may have BTC/USD candles but **no** valid XAU/USD session candles.
- Do not fill session closures as though gold traded continuously.
- Missing gold volume remains nullable; no invented volume.
- Models never use incomplete or future candles.
- Gold CFD reference data never implies executable physical-gold spot prices.

## Milestones
- [ ] Confirm a BTC/USD historical venue/product and licensing terms.
- [ ] Confirm an XAU/USD historical source, quote type, historical coverage and licensing.
- [ ] Download and audit each market independently (real results logged in Chapter VI).
- [ ] Plot indicators for each instrument with UTC completed-candle logic.
- [ ] Train/evaluate separate baseline classifiers with chronological splits.
- [ ] Evaluate separate costs, drawdowns and no-trade baselines.
- [ ] Integrate both instruments into the non-executing Signal Bot.
- [ ] Integrate both into the **paper-only** Auto Bot with separate risk settings.

**No actual market feed, strategy, prediction or trading bot has been implemented yet.**
