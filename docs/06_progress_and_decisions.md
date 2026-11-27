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

## 2026-11-26 — ML-024: Statistical baseline implementation

Added zero-return, AR(1), MA(1) and ARMA(1,1) 5m/1h forecasts with a separate training-majority direction benchmark, train-only estimation, causal filtering, within-block targets and residual diagnostics. Five synthetic tests passed in the assistant environment. The already-inspected seven-day series is development only. Actual forecast scores, longer history, feature-generation verification, gold access and provider latency remain pending. See [Chapter XXIV](24_statistical_forecasting.md).

## 2026-11-26 — ML-024 local result and volatility research decision

User reported all three statistical candidates fit on the BTC development sample. Numerical improvements over zero return were small and metric-dependent; no consistent directional improvement over training majority. AR(1) retained as parsimonious mean candidate (AIC -531.05068, BIC -515.75803); MA(1) nearly tied by AIC. All training ARCH-LM p-values below 1e-8, motivating a GARCH vs constant/rolling variance experiment, not a directional promotion. [Full comparison](24_statistical_forecasting.md#9-user-reported-first-real-data-baseline-experiment--26-november-2026). Raw files and test-suite output not independently verified; no untouched holdout or trade performance.

## 2026-11-26 — ML-025: Conditional variance implementation

Added fixed AR(1) mean, train-only two-stage Gaussian QML GARCH(1,1), constant/rolling24 benchmarks and 5m causal variance replay, with six synthetic tests passed. The seven-day sample remains development only. Actual scores, Mac CLI execution, longer history, feature export and XAU access are pending. See [Chapter XXV](25_garch_volatility.md).

## 2026-11-26 — ML-025 local volatility results

User reported GARCH11 had lower QLIKE and variance MSE than both constant and rolling24 in validation and development replay (403 observations each); rolling24 won validation MAE. Persistence 0.94661, standardized ARCH-LM p=0.85976, training diagnostics only. Retain GARCH for later causal feature ablation, not as a proven direction predictor. Next M2 Logistic/Ridge baseline with verified feature export, chronological purge and train-only scaling; longer history and untouched holdout still required. [Results](25_garch_volatility.md#8-user-reported-btcusd-garch-results--26-november-2026).

## 2026-11-26 — ML-026: supervised baseline implementation

Added four fixed Logistic/Ridge models, 12 allowlisted inputs after removing redundant momentum, chronological strict availability purge, train-only scaling and no-skill probability/return benchmarks. Five synthetic tests passed in the assistant container. Real feature export, scores, matched TSA comparison, longer history and XAU coverage remain pending. See [Chapter XXVI](26_supervised_baselines.md).

## 2026-11-26 — ML-026 local supervised results

User supplied feature manifest and M2 report: 1,980 pairs; expected purged splits; all four supervised models completed. Logistic underperformed prevalence on accuracy, Brier, log loss and AUC in every block. Ridge underperformed training mean in every block. Retain as negative-result benchmarks; no direction-model promotion. Prioritize longer authorized history and frozen holdout before serious selection; fixed XGBoost smoke test may proceed as implementation work. [Results](26_supervised_baselines.md#9-user-reported-m2-results--26-november-2026).

## 2026-11-27 — ML-027: Data permission gate and historical infrastructure

Reviewed Coinbase market data terms (section 3, August 2026 update); applicable ML and dissemination permissions not established. Pause real Coinbase-data ML and review public derived documentation. Candidate Bitstamp/Kaggle provenance and rights need review. Added local 1m-to-5m importer with fixed UTC train/validation/reserved-holdout dates, gap preservation, source and output hashes, no holdout export. Five synthetic tests and full synthetic CLI run passed. No market data fetched or published. Existing visibility unchanged. See [Chapter XXVII](27_historical_data_expansion.md).

## 2026-11-27 — ML-027 local synthetic pipeline result

User-reported manifest matches plan: 69,984 training and 17,568 validation 5m synthetic candles, no missing bars, no holdout exported. Recorded checksums in [Chapter XXVII](27_historical_data_expansion.md#6-user-reported-synthetic-pipeline-result--27-november-2026). Software milestone only; raw files and local tests not independently verified. Next provider-neutral gap-aware features and synthetic XGBoost comparison. Real data permissions and coverage remain unresolved.

## 2026-11-27 — ML-028: Gap-aware XGBoost software experiment

Added synthetic-only manifest/checksum audit, segment-reset feature windows, strict boundary cutoff shared by horizons, and fixed shallow XGBoost classifier/regressor alongside matched M2 and no-skill benchmarks. Five synthetic fixture tests passed in the assistant container. Full local history run pending, no real data or holdout used. See [Chapter XXVIII](28_xgboost.md).

## 2026-11-27 — ML-028 local XGBoost result

User-reported full synthetic run completed: 69,947 training and 17,531 validation pairs, expected hashes/settings, holdout unexported. Both learned classifiers lose to prevalence on probability losses; training mean beats learned regressors on MAE/RMSE. Expected negative-control behavior, no market skill demonstrated. Next: causal 5m/1h/4h/1D alignment and matched MTF ablation, synthetic only. 1m inputs deferred because not present in exported blocks. [Results](28_xgboost.md#9-user-reported-synthetic-xgboost-results--27-november-2026).

## 2026-11-27 — ML-029: Causal multi-timeframe ablation

Added nine higher-frame features to the 12 base features; complete UTC buckets, segment and block resets, strict target cutoff and matched 21-feature eligibility for all variants. Fixed XGBoost settings; 16 independent fits at 5m/1h. Five isolated tests passed with copied upstream helpers; full manifest integration, Day 10 regression and local CLI pending. No holdout access or real market data used. See [Chapter XXIX](29_multitimeframe.md).

## 2026-11-27 — ML-029 local multi-timeframe result

User-reported full synthetic MTF CLI completed with expected 68,820/16,404 train/validation pairs, 21 features, fixed settings and matching hashes. Full MTF 1h classification accuracy 51.3655%, AUC 0.527637; 18-feature variant has lowest 1h MAE/RMSE, zero-return wins 5m regression. Independent synthetic returns, overlapping hourly labels and multiple comparisons preclude interpreting gains as durable skill. No feature promotion. Next causal GARCH + XGBoost matched ablation with a known-variance positive control. [Results](29_multitimeframe.md#9-user-reported-synthetic-mtf-results--27-november-2026).

## 2026-11-27 — Mathematics documentation convention

User requested that every formula taught be preserved in GitHub. Added [mathematical reference](mathematical_reference.md) for general equations through Chapter XXIX and added the proposed hybrid feature equations to the Chapter XXIX preview. Future chapters must contain rendered formulas, definitions, assumptions, derivations and implementation links, updating the reference as needed. Existing experiment-specific numeric results remain in their chapters subject to the data rights gate. No model or experiment code changed.

## 2026-11-27 — ML-030: Causal hybrid feature implementation

Added zero-mean GARCH using 1,440 observed returns per fit, refit every 2,880, 288-return rolling RMS control, recursive state updates and gap resets. Matched six XGBoost feature groups per horizon/task, known-GARCH positive and independent-innovation negative controls. Five tests supplied, not executed; direct container clone failed due to DNS. Local reports pending. No real-data use or holdout. [Chapter XXX](30_hybrid_garch_xgboost.md).

## 2026-11-27 — ML-030 local hybrid results

User-reported complete synthetic experiment: counts 68,531/16,115 and 7,187/4,307, 35 refits with no failures. Positive-control GARCH reduces variance MSE by 17.57% vs refit constant, 16.20% vs rolling RMS; negative-control constant wins. Direction/return ablations do not demonstrate a reliable GARCH benefit. Retain volatility candidate, no market promotion. Local tests/raw files not independently inspected. Next K-Means/GMM and known-regime/direction positive control. [Results](30_hybrid_garch_xgboost.md#8-user-reported-hybrid-results--27-november-2026).

## 2026-11-27 — ML-031: Regime discovery and directional control

Added training-only scaling and K-Means/GMM on four causal MTF features, 21/24/24-feature matched XGBoost ablation, independent-return negative and persistent Markov-regime directional positive control, scoring-only ARI. Five new isolated unit tests passed; complete CLI and Mac results pending. GARCH excluded for isolated regime effect; joint ablation deferred. No real data, holdout or model promotion. [Chapter XXXI](31_market_regimes.md).
