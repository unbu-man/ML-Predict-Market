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
