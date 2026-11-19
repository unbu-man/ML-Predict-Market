# Chapter XII — Multi-Market, Five-Timeframe & Unsupervised-Learning Research

**Date:** 19 November 2026  
**Status:** Architecture and experimental plan only. No models, feeds, signals, or paper performance have been validated.  
**Product priority:** Non-executing Signal Bot. **Auto Bot deferred.**  
**Initial validation instruments:** BTC/USD and XAU/USD. Long-term design goal: extensible to additional legally accessible, documented price markets.

## 1. Main research thesis — hypotheses, not assumptions

The founder wants one platform able to examine five intervals: **1m, 5m, 1h, 4h, 1d**. We will research whether combining these **adds incremental predictive or decision value** relative to simple signals. More granular history gives more rows but not five independent data sources. Bars across timeframes are highly correlated, and 1m data can have substantial noise, latency, spread and storage costs.

**Primary research question:** Do additional completed-candle timeframes, unsupervised market-regime features or nonlinear price models improve **held-out, cost-aware** Signal Bot outcomes on BTC/USD and XAU/USD?

Secondary question: Can new markets be added via a provider adapter and instrument configuration without rewriting strategy logic, while recognizing that models and parameters may require market-specific retraining?

## 2. Proposed role of every timeframe

| Interval | Primary research use | Notes |
|---|---|---|
| 1d | Broad regime, slow directional context and multi-day volatility | Anchoring is **provider session-based**, not necessarily UTC midnight |
| 4h | Intermediate trend, swings, support/resistance regions | Four-hour boundaries and closures depend on provider/calendar |
| 1h | Setup context, structure and medium-term risk | Baseline higher timeframe |
| 5m | Setup trigger, confirmation and invalidation | Baseline lower timeframe |
| 1m | Optional micro-timing and quote/spread diagnostics | Lowest priority until data/replay quality justifies it |

**Versioned research ladder:** A=5m-only; B=1h+5m; C=4h+1h+5m; D=1d+4h+1h+5m; E=all five including 1m. Then test regime and nonlinear features on the **best validated, predeclared** baseline. Do not tune on final held-out data.

## 3. Multi-market adapter architecture

    Provider adapter -> Instrument registry -> Data audit -> Market calendar
                                                           |
                                                Completed-candle store
                                                           |
                                                Multi-timeframe feature store
                                                           |
                                          Rules / optional regime model / ML
                                                           |
                                              Signal policy + quality gate
                                                           |
                                       Signal journal / visualization / alerts
                                                           |
                                                Forward outcome evaluation

Each market has a separate, configured provider, trading calendar, pricing/quantity units, volume meaning, quote type, costs and data-licensing record. The shared engine **does not assume all symbols trade 24/7, in USD, with a centralized volume, or at a given pip/lot size**.

### Suggested instrument contract (design, not live code)

- instrument_id, provider_id, provider_symbol, asset_class, product_type
- base_asset, quote_asset, price_unit, quantity_unit, tick_size, quantity_step
- timezone and market_calendar_id, closure/holiday schedule, session anchor
- candle_timeframe, candle_open_utc, candle_close_utc, **available_at_utc**, is_complete
- open, high, low, close, volume_nullable, volume_semantics, bid/ask if available
- expected_spread, fees, slippage method, financing if applicable, licensing_notes

**Examples:** BTC/USD means USD per BTC from a recorded USD spot venue, **not** BTC/USDT. XAU/USD is USD per troy ounce of gold from a named provider; an indicative OTC quote is not guaranteed to represent an executable price or physical metal trade.

### Future-market extension gate

A new market is **not automatically supported** by giving it a symbol. It passes only when its source is lawful and documented, bars validate, sessions and costs are modeled, signals can be replayed with proper timestamps, and it has its **own** out-of-sample evaluation. Start with separate per-market strategies and models; cross-market transfer is a distinct experiment.

## 4. Resampling and time alignment

### Source of truth
Where reliable, consider provider historical **1m** candles as one possible base store, then aggregate up to 5m, 1h and 4h. This is **not mandatory**: 1m data might be unavailable, license-restricted, short-lived or too costly for a full backfill; native higher-timeframe bars can be used if they are consistent and verified.

For each valid, complete aggregation interval:

- open = first available interval open
- high = maximum of highs
- low = minimum of lows
- close = last completed interval close
- volume = sum **only** if the provider's volume is meaningful and additive
- mark incomplete if required base bars are missing; never manufacture complete candles
- publish derived bar only after last constituent bar is complete **and available**
- define 1d and 4h boundaries with the instrument/provider **session calendar**, not a universal 24-hour clock

BTC/USD exchange feeds may be 24/7 (interrupted by outages). Gold sources may have weekend closures, daily breaks, holidays and DST behavior. Do **not** aggregate across these gaps or fill closed markets as normal trading.

### An example available-at test

At 11:05 UTC, the engine may use a 1h candle closing at 11:00 **only once published**. It may not see final OHLC/indicators from the 11:00–12:00 candle. Similarly, the 4h and 1d indicators must come from their last genuinely available, completed bars.

Use a backward-as-of join keyed on **available_at_utc** and instrument/source. All feature pipelines must respect causal availability, not just row timestamp. Account for published-data delays and timing ambiguity at simultaneous candle closes.

**Acceptance:** No future higher-timeframe value influences any lower-timeframe signal, in historical or forward replay. TradingView's confirmed HTF technique and Python as-of alignment both need tests.

## 5. Unsupervised learning: regime discovery (research experiment)

**What it is:** clustering past-only observations into groups with similar return/volatility/trend characteristics. Clusters may later be interpreted as trending, ranging or stressed periods *after* inspecting their training-set summaries. They are **not automatically profit classes or future forecasts**.

### Candidate features (avoid raw cross-asset price levels)
- rolling past-only log return and volatility, with a clear window
- normalized EMA20–EMA50 spread / ATR and past-only trend slope
- ATR divided by price, range expansion, relative realized volatility
- drawdown/distance from rolling high using past-only observations
- time/session features, if appropriate for market

### Progressive models
1. K-Means: fast and interpretable starter clusters; labels are arbitrary IDs, not bullish/bearish truth.
2. Gaussian Mixture: optional soft probabilities of mixture components; **not** calibrated chance of winning a trade.
3. HDBSCAN/DBSCAN: exploratory nonlinear-density clusters and outliers. Confirm how cluster membership is assigned **out of sample** before incorporating them into a live signal path; not all clustering APIs support ordinary predict on unseen rows.
4. Temporal state models (such as HMM) only later if a causal, online filtering method is implemented; do not use future-aware smoothing of held-out sequences.

### Causal training and use
- Fit transformations, feature selection, imputation and clustering **only on each training fold**.
- Never refit to include final-test or future observations before reporting held-out performance.
- Fix a reproducible out-of-sample label-assignment policy; e.g., KMeans.predict on the trained scaler's transform of the new row.
- Inspect regime stability by time and asset, not cluster separation alone.
- Test the regime feature **against the same signal strategy without regime features**.

**Possible result:** "Regime 2 correlates with wider spreads and higher return variability" is an observation; "Regime 2 guarantees a SELL signal" is not justified.

## 6. Nonlinear machine learning (separate from unsupervised learning)

- Start with a *naive* baseline and a train-only-fitted Logistic Regression.
- Compare nonlinear XGBoost or another tree ensemble for interaction effects across timeframes.
- Only try LSTM/Transformers if there is enough data and evidence they beat the simpler models.
- Predict an explicitly defined forward **return / direction / volatility horizon**, rather than assuming a clustering result is a price forecast.
- Train and evaluate BTC/USD and XAU/USD **independently** at first; normalize by returns/ATR, not one universal price scale.
- Keep models independent of trade execution. An ML score is just one input to the non-executing Signal Bot policy.

### Look-ahead and overfitting protections
- Chronological train/validation/final-test windows; walk-forward evaluation.
- Purge/gap where labels overlap decisions and avoid fit-on-all-data preprocessing.
- Document exact data publication and bar-close times for each feature.
- Guard against repeated hyperparameter search and cherry-picking the best backtest.
- Re-evaluate costs, multiple market regimes and gold trading breaks.

## 7. Model comparison and performance criteria

| Experiment | Inputs | Why |
|---|---|---|
| A | 5m only | Signal baseline |
| B | 1h + 5m | Test existing core hypothesis |
| C | 4h + 1h + 5m | Test intermediate structure benefit |
| D | 1d + 4h + 1h + 5m | Test slow-regime benefit |
| E | 1m + 5m + 1h + 4h + 1d | Test if 1m adds enough value to justify noise/cost |
| F | Best predeclared baseline + train-only unsupervised regime | Test incremental regime value |
| G | Best baseline + supervised nonlinear model | Test incremental predictive value |

For each market, report: signal count and coverage; balanced accuracy / calibration if defined; false signals; gross and **net hypothetical expectancy**; volatility and drawdown; fee/spread/slippage sensitivity; latency; stability across chronological out-of-sample folds. Include no-signal/no-trade and feasible market benchmarks. A cluster quality score does not replace economic performance measurement.

Test comparisons on **matched times and data availability** to avoid one strategy benefiting from more favorable windows. Record negative results.

## 8. Staged implementation plan

**Current eight-week signal pilot (unchanged core):**
- Foundation and data providers → 5m/1h BTC/USD and XAU/USD quality checks → EDA and deterministic signal rules → cost-aware replay → non-executing signals and outcome tracking → simple supervised baseline.

**Post-baseline research gates (scope extension, no fixed promised finish):**
1. Implement instrument registry + timeframe adapter API and test a third hypothetical instrument contract with synthetic candles (no unsupported real-market claims).
2. Add native/resampled 4h features; test A versus B versus C.
3. Add session-aware 1d features; test D versus C.
4. Add optional 1m, only if provider quality, storage, alert latency and costs are acceptable; test E versus D.
5. Run K-Means/GMM train-only regime experiments F; evaluate OOS stability and incremental signal outcomes.
6. Run XGBoost versus simple supervised baselines G; only introduce deep learning if justified.
7. Select simpler, robust variants based on held-out evidence; preserve alternative results.

### Engineering tests

- [ ] All five timeframes have explicit interval, session and availability definitions.
- [ ] No incomplete 1h/4h/1d candle appears in a 1m/5m decision.
- [ ] Resampled OHLC matches native source on valid complete bars.
- [ ] Gold daily breaks, holidays and DST do not create fabricated bars.
- [ ] BTC/USD real-USD and XAU/USD per-troy-ounce units remain correct.
- [ ] Regime scaler and clustering are fit only on training data.
- [ ] Unsupervised cluster labels are assigned reproducibly to unseen observations.
- [ ] Nonlinear models beat clearly stated baselines *only if actual measured evidence supports this*.
- [ ] Additional timeframes are evaluated with a saved cost-aware ablation table.
- [ ] New market adapters fail closed when source/licensing/session metadata is missing.
- [ ] There is no real trading order endpoint in this project.

## 9. Sources for the engineering methodology

- [pandas — time series and OHLC resampling](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)
- [TradingView — higher/lower timeframe data](https://www.tradingview.com/pine-script-docs/faq/other-data-and-timeframes/)
- [TradingView — repainting and confirmed higher timeframes](https://www.tradingview.com/pine-script-docs/concepts/repainting/)
- [scikit-learn — time series validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [scikit-learn — clustering methods and limitations](https://scikit-learn.org/stable/modules/clustering.html)
- [XGBoost documentation](https://xgboost.readthedocs.io/)

**Decision status:** market-independent architecture and five-frame feature research are goals; the Signal Bot remains the first product, with BTC/USD and XAU/USD as the only initial real market tests. There are no actual performance results yet.
