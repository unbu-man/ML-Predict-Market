# Chapter VI — Progress, Decisions and Experiment Journal

**Last documentary review:** 19 November 2026  
**Current status:** Requirements and learning chapters prepared; no executable feature verified.

## Project scorecard

| Workstream | Status | Next verifiable artifact |
|---|---|---|
| Vision, three product definitions | Documented | User and scope review |
| Architecture and risk boundaries | Documented | Typed event contracts |
| BTC/USDT data provider | Not started | Source manifest |
| Candle ingestion and validation | Not started | Script + quality report |
| EDA | Not started | Notebook + figures |
| Indicator prototype | Not started | Deterministic chart + tests |
| Forecasting baselines | Not started | Walk-forward fold report |
| Signal bot | Not started | Message log + dedupe tests |
| Cost-aware backtest | Not started | Net P&L report |
| Auto bot | Not started | Paper engine + kill-switch tests |
| Forex support | Not started | FX feed and session report |

## Working checklist
- [x] Select project scope: Crypto + Forex, with no stocks in first version
- [x] Specify indicator, non-executing signal bot, paper-only auto bot
- [x] Draft initial architecture and research methodology
- [ ] Choose provider, confirm licensing, and document BTC/USDT 1-hour sample window
- [ ] Create Python project scaffold and tests
- [ ] Download and validate completed candles
- [ ] Run exploratory analysis and write results
- [ ] Compute indicator features and test no future-data use
- [ ] Run majority/persistence baseline, then Logistic Regression
- [ ] Compare with XGBoost using walk-forward testing
- [ ] Evaluate estimated spread, fees and slippage
- [ ] Deliver research alerts
- [ ] Implement independently risk-gated paper orders

## Decisions (working defaults, not finalized performance claims)

| ID | Date | Decision | Why | State |
|---|---|---|---|---|
| ML-001 | 2026-11-18 | One common strategy engine for three products | Prevent inconsistent signals | Proposed |
| ML-002 | 2026-11-18 | Signal Bot cannot execute | Separate analysis from execution | Scope rule |
| ML-003 | 2026-11-18 | Auto Bot paper only | Avoid real-money exposure in MVP | Scope rule |
| ML-004 | 2026-11-18 | Start BTC/USDT, hourly, then EUR/USD | Minimize scope before FX complexity | Working default |
| ML-005 | 2026-11-18 | Benchmarks before neural networks | Detect whether complexity helps | Proposed |
| ML-006 | 2026-11-18 | Preserve negative/null results | Research integrity | Mandatory |

## Experiments

| ID | Research question | Status | Results |
|---|---|---|---|
| EXP-001 | Does a baseline beat naive next-hour prediction? | Not run | None |
| EXP-002 | Does XGBoost outperform a well-tuned Logistic Regression baseline? | Not run | None |
| EXP-003 | Do signals exceed realistic costs in out-of-sample simulation? | Not run | None |

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
