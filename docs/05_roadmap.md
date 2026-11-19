# Chapter V — Development Roadmap and Milestone Gates

**Planning date:** 18 November 2026. No timetable is a guarantee.

| Phase | Learning and engineering focus | Required acceptance evidence |
|---|---|---|
| 0. Foundation | Python environment, Git practices, requirements, risk boundaries | Documented setup + test runner |
| 1. Historical data | Crypto candle ingestion, manifests, gaps/duplicates/precision | Data quality report + UTC sample |
| 2. EDA | Return distributions, volatility regimes, sessions | Reproducible notebook + charts |
| 3. Custom indicator | EMA/RSI/ATR + deterministic signal rules | Plot, warm-up tests, no look-ahead |
| 4. Signal bot | Signal event contract, log, dedupe, expiration | Alert event replay + tests |
| 5. ML baselines | Majority, persistence, Logistic Regression | Walk-forward and held-out metrics |
| 6. Advanced ML | XGBoost, optional LSTM if justified | Comparison, leakage audit, model cards |
| 7. Backtesting | Cost-aware strategy replay, trade log, P&L | Net return, drawdown, sensitivity analysis |
| 8. Paper auto bot | Risk gates, order simulation, ledger, kill switch | Idempotency and reconciliation tests |
| 9. FX extension | EUR/USD feed and provider-specific session/spread rules | Independent FX data and replay tests |
| 10. Demo product | Dashboard, explanation, monitoring, security review | End-to-end paper-only demonstration |

## Suggested sequence
Start small. Do not build a large AI system before proving that your data pipeline is trustworthy. Prefer a single historical BTC/USDT hourly sample and reproducible rules. Only then move from alerts to paper orders.

## Implementation policy
- Each milestone gets a goal, hypothesis, code, tests, measured results and a changelog entry.
- Proposed and completed are different statuses.
- Failing hypotheses stay in the repository; include why they failed.
- Keep the engine instrument-agnostic but provider-, market-calendar- and cost-aware.
- Store no secrets or large/licensed raw market data in Git.

## Definition of done for the first public demo
A visitor can inspect a completed-candle chart, see an explainable signal, view an audit trail of simulated actions and reproduce the experiment. The demo labels every trade and performance result as **simulated**. No live-trading endpoint exists.

## Future business caveat
A research application is not a license to provide regulated brokerage, custody, discretionary trading or personalized signals. Analyze each country/product permission separately before any real-money or customer-facing expansion.
