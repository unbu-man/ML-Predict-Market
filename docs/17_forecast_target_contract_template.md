# Chapter XVII — Day 01 Forecast Target Contract (Template)

**Version:** 0.1 draft, to complete on 20 November 2026  
**Status:** Not yet signed off; blank fields and unchecked decisions are intentional.

## Prediction clock

- [ ] Decision occurs after a genuinely complete and published 5m candle.
- [ ] Time zone and timestamp semantics: UTC; source close_time, available_at are distinct.
- [ ] Actual BTC/USD source is USD spot and its exact symbol is recorded.
- [ ] XAU/USD provider, quote product type, pricing unit and trading calendar are recorded.
- [ ] Future 5m label definition, including missing bars and market closures.
- [ ] Future 1h label definition, including gold daily breaks, holidays and DST.
- [ ] UP/DOWN target definition or fixed neutral threshold.
- [ ] Features allowed at time t and forbidden future data.
- [ ] Evaluation split, target-overlap gap, baseline scores and leakage tests.
- [ ] No order-execution endpoints or credentials in this research component.

## Outcome variable, not a known prediction

\[
r_{t,h}=\ln(C_{t+h}/C_t).
\]

At a hypothetical current reference close 100000 and a future next 5m close 100100, the training outcome is ~0.0009995. Do not treat this outcome as information available at time t.

## Provider research table

| Market | Source and exact symbol | Product / units | 5m and 1h history | Calendar, breaks, DST | Licensing/fees | Verified? |
|---|---|---|---|---|---|---|
| BTC/USD | TBD | actual USD-quoted spot BTC | TBD | venue-specific | TBD | No |
| XAU/USD | TBD | USD/troy ounce; product class TBD | TBD | gold sessions TBD | TBD | No |

## First model baseline specification

- Zero-return regression benchmark; train-only majority/persistence classifier.
- AR(1) on past returns as a classical TSA comparison.
- Logistic Regression classification with lagged, past-only features.
- MAE/RMSE and log loss/Brier/balanced accuracy measured on chronological future data.
- Forecast volatility with rolling past variance first. GARCH/TAR/clusters remain optional experiments.

**Approval rule:** The project cannot begin fitting models until the clock, source, future label convention and train/test availability are defensibly documented.
