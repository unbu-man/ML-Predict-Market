# Chapter V — Development Roadmap and Milestone Gates

**Planning date:** 18 November 2026. No timetable is a guarantee.

| Phase | Learning and engineering focus | Required acceptance evidence |
|---|---|---|
| 0. Foundation | Python environment, Git practices, requirements, risk boundaries | Documented setup + test runner |
| 1. Historical data | BTC/USD and XAU/USD ingestion, independent source manifests and market calendars | Data quality report + UTC sample |
| 2. EDA | Return distributions, volatility regimes, sessions | Reproducible notebook + charts |
| 3. MTF signal engine | Versioned 1H context + 5M trigger state machine | Deterministic tests and HTF look-ahead prevention |
| 4. Research indicator | Plot 1H/5M rules, trigger and invalidation | Chart/backend parity and no-repainting tests |
| 5. Cost-aware backtesting | Compare 5M-only vs 1H+5M and no-signal baselines | Chronological results, costs, drawdown, ambiguity policy |
| 6. Signal alert journal | States, dedupe, expiry, stale-data and session filtering | Local event log + replay tests |
| 7. Forward demo signals | Non-executing dashboard or Telegram alert delivery | Latency, alert lifecycle and prospective outcome log |
| 8. ML improvement research | Logistic Regression/XGBoost vs rule baseline; unsupervised regime clustering as a separate experiment | Walk-forward and ablation results, train-only clustering |
| 9. Gold market validation | XAU/USD source classification, daily breaks and quote costs | Separate gold replay and data-quality evidence |
| 10. Demo review | Monitoring, explanation and reproducibility | End-to-end **signal-only** demo, no order methods |

## Suggested sequence
Start small. Do not build a large AI system before proving that your data pipeline is trustworthy. Prefer a historical BTC/USD hourly sample, then independently validate XAU/USD gold quotes and reproducible rules. Only then evaluate optional ML improvements. **Auto trading is deferred** unless the founder deliberately reopens the scope.

## Implementation policy
- Each milestone gets a goal, hypothesis, code, tests, measured results and a changelog entry.
- Proposed and completed are different statuses.
- Failing hypotheses stay in the repository; include why they failed.
- Keep the engine instrument-agnostic but provider-, market-calendar- and cost-aware.
- Store no secrets or large/licensed raw market data in Git.

## Definition of done for the first public demo
A visitor can inspect a completed-candle chart, see an explainable signal, view an audit trail of research alerts, cancelled setups and historical/forward paper outcomes. Every performance result is labeled **simulated** and no trade endpoint exists.

## Future business caveat
A research application is not a license to provide regulated brokerage, custody, discretionary trading or personalized signals. Analyze each country/product permission separately before any real-money or customer-facing expansion.

## Detailed schedule
See [Chapter XI — Eight-week build and study plan](11_8_week_execution_plan.md). Each week ends with a GitHub commit, tests, a progress note and an explicit go/no-go check. The timeline is illustrative and can move with coursework or data-provider constraints.

## Long-term extension — five timeframes and more markets

The full research target is 1m / 5m / 1h / 4h / 1d with market adapters for future crypto, metals, FX and other lawful, available price feeds. Initial implementation **still begins with 5m + 1h**, on BTC/USD and XAU/USD. Test additions in this order: 4h context, 1d context, 1m execution/timing; then train-only unsupervised regimes; then nonlinear ML. Include each added layer only when it produces measurable, stable improvement after costs in held-out testing. Do not equate more rows, more features or more clusters with higher forecast accuracy. See [Chapter XII](12_multimarket_multitimeframe_unsupervised.md).
