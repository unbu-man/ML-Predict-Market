# Chapter VI — Progress, Decisions and Experiment Journal

**Last documentary review:** 19 November 2026  
**Current status:** Requirements and learning chapters prepared; no executable feature verified.

## Project scorecard

| Workstream | Status | Next verifiable artifact |
|---|---|---|
| Vision, three product definitions | Documented | User and scope review |
| Architecture and risk boundaries | Documented | Typed event contracts |
| BTC/USD data provider | Not started | USD-quoted exchange source manifest |
| Candle ingestion and validation | Not started | Script + quality report |
| EDA | Not started | Notebook + figures |
| Indicator prototype | Not started | Deterministic chart + tests |
| Forecasting baselines | Not started | Walk-forward fold report |
| Signal bot | Not started | Message log + dedupe tests |
| Cost-aware backtest | Not started | Net P&L report |
| Auto bot | **Deferred — out of current MVP** | Revisit only after explicit product decision |
| Five-timeframe framework | Documented architecture, not implemented | Verify 1m/5m/1h/4h/1d provenance and session boundaries |
| Unsupervised regimes | Proposed research, not implemented | Define features, training-only preprocessing and OOS assignment method |
| XAU/USD gold quotes | Not started | Gold quote source/type and session report |

## Working checklist
- [x] Select scope: BTC/USD crypto and XAU/USD gold (no stocks)
- [x] Specify indicator, non-executing signal bot, paper-only auto bot
- [x] Draft initial architecture and research methodology
- [ ] Choose BTC/USD and XAU/USD data providers, verify licenses and document both 1-hour sample windows
- [ ] Create Python project scaffold and tests
- [ ] Download and validate completed candles
- [ ] Run exploratory analysis and write results
- [ ] Compute indicator features and test no future-data use
- [ ] Run majority/persistence baseline, then Logistic Regression
- [ ] Compare with XGBoost using walk-forward testing
- [ ] Evaluate estimated spread, fees and slippage
- [ ] Deliver research alerts
- [x] Auto Bot deferred; no order APIs in current project
- [x] Record market-agnostic design and 1m/5m/1h/4h/1d target
- [ ] Document data volume, provider rates, licensing, session/DST and OHLC aggregation per market
- [ ] Benchmark 1h+5m, then selectively add 4h, 1d and 1m
- [ ] Test unsupervised regimes with train-only fitting and genuine OOS assignment
- [ ] Test whether nonlinear supervised models beat simpler baselines after costs
- [ ] Evaluate 1H+5M setups against 5M-only and 1H-only baselines
- [ ] Record signal outcome, net hypothetical expectancy, drawdown, alert latency and rejection rate per market

## Decisions (working defaults, not finalized performance claims)

| ID | Date | Decision | Why | State |
|---|---|---|---|---|
| ML-001 | 2026-11-18 | One common strategy engine for three products | Prevent inconsistent signals | Proposed |
| ML-002 | 2026-11-18 | Signal Bot cannot execute | Separate analysis from execution | Scope rule |
| ML-003 | 2026-11-18 | Auto Bot paper only | Avoid real-money exposure in MVP | Scope rule |
| ML-004 | 2026-11-18 | Original BTC/USDT and EUR/USD plan | Historical initial proposal | Superseded by ML-007 |
| ML-005 | 2026-11-18 | Benchmarks before neural networks | Detect whether complexity helps | Proposed |
| ML-006 | 2026-11-18 | Preserve negative/null results | Research integrity | Mandatory |
| ML-007 | 2026-11-19 | Two markets: BTC/USD and XAU/USD (correcting XAU/USA) | Founder decision | Confirmed scope |
| ML-008 | 2026-11-19 | Keep shared engine but separate market adapters, costs, sessions and experimental evaluations | Different instrument semantics | Design policy |
| ML-009 | 2026-11-19 | Prioritize Signal Bot, keep indicator as supporting visualization, defer Auto Bot | Founder decision | Confirmed active scope |
| ML-010 | 2026-11-19 | Research 1H completed-candle context and 5M entry confirmation | Founder idea; not proven profitable | Hypothesis |
| ML-011 | 2026-11-19 | Use an eight-week build/study sequence with documented test gates | Practical development proposal | Proposed schedule |
| ML-012 | 2026-11-19 | Target all supported market classes in long-term architecture; begin empirical validation with BTC/USD and XAU/USD | Founder wants more reusable signal platform | Confirmed long-term goal, not completed capability |
| ML-013 | 2026-11-19 | Support 1m, 5m, 1h, 4h, 1d but add each to strategy only after ablation | More context may help but has costs/leakage risks | Research specification |
| ML-014 | 2026-11-19 | Research unsupervised market regimes and nonlinear predictive models separately | Compare hypothesis to simpler baselines | Proposed experiments |
| ML-015 | 2026-11-19 | Set a 30-day, 5h/day target for a bounded research-grade Signal Bot MVP including mathematical testing | User requested an achievable full build | Confirmed schedule goal; no implementation completed |

## Experiments

| ID | Research question | Status | Results |
|---|---|---|---|
| EXP-BTC-001 | BTC/USD baseline versus naive next-hour prediction | Not run | None |
| EXP-XAU-001 | XAU/USD baseline versus naive next-hour prediction | Not run | None |
| EXP-002 | Does XGBoost outperform a well-tuned Logistic Regression baseline? | Not run | None |
| EXP-003 | Do signals exceed realistic costs in out-of-sample simulation? | Not run | None |
| EXP-MTF-001 | Does the 1H+5M signal setup outperform a simpler 5M-only rule after costs, separately on both markets? | Not run | None |
| EXP-MTF-002 | Do incremental 4h, 1d and 1m features improve out-of-sample signal outcomes? | Not run | None |
| EXP-REGIME-001 | Does a train-fitted unsupervised regime feature improve out-of-sample results? | Not run | None |
| EXP-NONLINEAR-001 | Does XGBoost with MTF features beat simple supervised models net of costs? | Not run | None |
| EXP-TRANSFER-001 | Can the same interface validate a third market without hardcoded symbols, calendars or units? | Not run | None |

## Copyable session template

### YYYY-MM-DD — Session title
**Objective:**  
**Theory learned:**  
**Code/commit:**  
**Dataset and period:**  
**Tests and outputs:**  
**Observed results:**  
**Problems and fixes:**  
**Decision:**  
**Next action:**  

Do not mark “done” before runnable proof exists. A forecast chart without a held-out comparison is not evidence of predictive ability.

## 2026-11-19 — New product decision

**ML-016:** Build a predictive market engine first. After the 30-day research MVP, reuse its timestamped, horizon-specific forecasts in three product surfaces: automated non-executing signals, chart indicator and future **paper-first** automated execution with independent risk gates. **Status:** planned; none implemented. Follow-on work is in [Chapter XIV](14_prediction_to_three_products.md) and the Google Sheet future-product tab.

## 2026-11-19 — AMPE research design decision

**ML-017:** Record an **unvalidated** adaptive forecasting hypothesis: return/direction model + separate volatility model + optional train-only market regime + calibrated forecast reliability and ABSTAIN behavior. Inspiration: Cryer and Chan's time-series model diagnostics, forecasting, financial volatility and threshold models (Chapters 1, 9, 12, 15). Compare each component against simpler baselines on chronological unseen data. See [Chapter XV](15_adaptive_market_prediction_engine.md) and [Day 01](16_day1_2026_11_20_kickoff.md). **No execution or performance result yet.**

## 2026-11-20 — Day 01 progress (ML-018)

**Working decision proposed:** forecast exactly +5m and +1h reference log returns at verified 5m bar closes, initially with UP vs NOT_UP binary label and no cross-session/gap target windows. [Draft contract](18_day01_forecast_contract_draft.md). Pending founder review.

**Data source review:** Coinbase Exchange `BTC-USD` and OANDA v20 `XAU_USD` are *candidate adapters* based on official documentation. No actual historical feed, licensing/access rights, source latency or gold instrument eligibility verified. [Source notes](data_source_research.md).

**Math tests:** 11 pure-Python, synthetic fixture unit tests passed in an isolated environment (label arithmetic, bad prices/OHLC, missing interval, market-session change, and publication-time cutoff); code committed. Neither model skill nor live platform correctness established.

**Status:** Day 1 In progress until target approved, provider questions investigated, local tests confirmed, and progress journal completed.

## 2026-11-26 — ML-019: Adopt hybrid selective forecast research

Founder agreed to the XGBoost + TSA + optional volatility/regime + selective WAIT direction. See [Chapter XIX](19_hybrid_selective_forecast_strategy.md) for ordered M0–M6 experiments, mathematical contracts and holdout/paper promotion gates. Win rate is secondary to net expectancy, risk, coverage and forecast quality. No numerical win-rate or probability threshold is approved. No source access or trained model is established by this decision. Day 1 target review and local setup remain open.

## 2026-11-26 — Data foundation implementation draft

Prepared a bounded BTC-USD connectivity probe and six fixture tests, pending execution. Web retrieval did not return a valid candle payload. XAU access remains unverified. Chapter XX records the provisional dual regression/classification target and next gates. Day 1 remains in progress; no fabricated market data or model results.

## 2026-11-26 — Historical dataset starter

Added bounded Coinbase BTC-USD collection and export with gap report, hash and unknown historical publication time, plus six fixture tests. Actual test execution and real data access remain pending. See [Chapter XXI](21_first_historical_dataset.md).

## 2026-11-26 — Market EDA/TSA starter

Added CSV/manifest audit, contiguous-segment log-return diagnostics, ADF/KPSS, ACF/PACF, rolling volatility and squared-return ACF, seven plots, and JSON report. Four synthetic-fixture tests passed in assistant environment. Real BTC data and gold provider access remain unverified. See [Chapter XXII](22_market_eda_tsa.md).

## 2026-11-26 — First user-reported BTC EDA success

User's local terminal report shows 2,016 five-minute BTC/USD candles, zero gaps, and successful EDA output. ADF/KPSS are consistent with nonstationary log price and level-stationary log returns in this sample. [Recorded interpretation](22_market_eda_tsa.md#first-observed-btcusd-smoke-test-results--26-november-2026). Evidence is pasted output, not independent CSV/manifest verification. ACF/PACF, squared-return ACF, longer history and hourly/XAU checks remain open. No forecast or trade performance measured.

## 2026-11-26 — ML-023: EDA-informed feature starter

Full user-pasted report shows weak return autocorrelation (lag 1 = 0.064) and stronger squared-return dependence (lag 2 = 0.175). Retain AR(1) baseline, investigate rolling volatility and later GARCH; no validated model selection. Added separate close-only feature/label generator with five tests pending execution. See [Chapter XXIII](23_causal_feature_dataset.md). Seven-day sample is development only; original publication latency and raw data not independently verified.
