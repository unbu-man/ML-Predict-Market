# Chapter XXXIV — Adaptive model selection with delayed feedback

**27 November 2026.** Synthetic-only reliability experiment. No real market data, market holdout access or live orders. The data permission gate remains active.

## 1. Delayed feedback and model universe

The objective is to select a candidate architecture using recent out-of-sample forecast outcomes rather than the current hidden state or future returns. Reuse Chapter XXXIII's expanding-window model fits (30/40/50 training days and three 10-day evaluation folds per path). The nine factorial architectures are always fitted on matched observations. Classification also permits the training-prevalence probability; regression also permits zero return and training mean (ten and eleven candidates respectively). The policy can fall back to simple forecasts. It does not submit orders.

For an anchor a_i, common assumed delay delta and forecast horizon h in five-minute bars, define:


```math
\begin{aligned}
d_i&=a_i+\delta\\
b_{i,h}&=a_i+300h+\delta\\
\mathcal M_{i,h}&=\{j<i:b_{j,h}<d_i\}
\end{aligned}
```

Here d is the decision time and b is the assumed label availability. Strict inequality is deliberate: an outcome available at the same instant is not yet used. The common assumed delay cancels when comparing these two timestamps. With two-second delay and decision at 11:00:02, the one-hour forecast made at 10:00:02 is not yet eligible; the one made at 09:55:02 is eligible. This conservative ordering is not measured latency. No labels from purged/nonpredicted origins enter the selector.

## 2. Rolling historical prediction loss

For candidate c and already-generated forecast p for UP or Yhat for return, squared loss is:


```math
\begin{aligned}
\ell_{j,c}^{\mathrm{class}}&=(p_{j,c}-y_j)^2\\
\ell_{j,c}^{\mathrm{reg}}&=(\widehat Y_{j,c}-Y_j)^2\\
\mathcal W_{i,h}&=\operatorname{last}_{288}(\mathcal M_{i,h})\\
\widehat L_{i,h,c}&=\frac1{|\mathcal W_{i,h}|}\sum_{j\in\mathcal W_{i,h}}\ell_{j,c}
\end{aligned}
```

Brier and MSE match the locked-selection objectives of Chapter XXXIII. Last 288 means matured eligible prediction origins, not necessarily 24 calendar hours because coverage gaps and purged boundaries can occur. Each candidate is scored on the same historical origins, even when it was not selected (shadow forecasting). Classification and regression are routed separately for each horizon. No probability calibration yet.

## 3. Causal switching rule

Initialize each path/horizon/task with its Chapter XXXIII locked architecture, derived solely from the four selection paths of that chapter. Require 48 matured origins before switching. Review every 12 eligible prediction origins (including index zero; no review at other indices). Let c_prev be the previous selected candidate:


```math
\begin{aligned}
c_i&=\begin{cases}
\arg\min_c\widehat L_{i,h,c},& i\bmod12=0,\ |\mathcal W_{i,h}|\ge48\\
c_{i-1},&\text{otherwise}
\end{cases}\\
\widehat p_i^{\mathrm{adaptive}}&=p_{i,c_i}\\
\widehat Y_i^{\mathrm{adaptive}}&=\widehat Y_{i,c_i}
\end{aligned}
```

Use the appropriate task-specific candidate in each output. Keep the current candidate if tied for minimum, otherwise fixed ordering (benchmarks first, then ascending feature count and lexical name). Parameters are fixed engineering choices, not optimized on the new paths. The policy is a simple hard switch, not a learned gating network or an online weight update. Rolling memory persists across expanding folds on the same path; reset across paths and tasks. The previously scored forecasts retain the model versions that actually generated them. No retrospective refit of those predictions.

Worked example: suppose two classifiers have three matured probability forecasts (.7,.6,.8) and (.4,.5,.6), respectively, with observed labels (1,0,1). Their Brier losses are:


```math
\begin{aligned}
L_A&=(.3^2+.6^2+.2^2)/3=0.163333\ldots\\
L_B&=(.6^2+.5^2+.4^2)/3=0.256666\ldots
\end{aligned}
```

A has smaller historical loss. In this toy example three rows illustrate arithmetic only; the real policy requires 48 matured rows before a review can change the selected model.

## 4. Performance assessment and switching audit

Evaluate adaptive, locked, MTF-only and simple benchmarks on the same new path origins. Record Brier, log loss, AUC and accuracy, and MAE, RMSE and sign accuracy. Report the number of origins allocated to each candidate and the number of switches:


```math
\begin{aligned}
N_c&=\sum_i\mathbf1\{c_i=c\}\\
S&=\sum_{i=2}^{N}\mathbf1\{c_i\ne c_{i-1}\}\\
\Delta L&=L_{\mathrm{adaptive}}-L_{\mathrm{locked}}
\end{aligned}
```

Negative delta indicates lower matched loss, not greater trading profitability. Switching a model is not switching a trading position. All observed scores are path-level pooled losses; report per control, not an aggregate across different return scales. Hourly outcomes overlap and folds reuse historical data, so naive independent-trial tests are not justified.

## 5. Fresh paths and fixed parameters

Use the same four simulated processes as Chapter XXXIII with fresh seeds 34001, 34101, 34201 and 34301, one path per control. Start at the same illustrative epoch, generate 60 simulated days, no market data. Same GARCH fitting and label purging, with zero-mean GARCH a raw second-moment proxy in nonzero-mean controls. The selector never receives simulator identity, latent state or true generator parameters as inputs. Four paths, twelve folds, nine architectures, two horizons and two tasks imply:


```math
N_{\mathrm{fits}}=4(3)(9)(2)(2)=432
```

Expected 24 GARCH refits and 8,604 matched evaluation origins per horizon/path if no failed refits. The full run is an offline replay, not a streaming live engine: future rows are present in memory but the switching rule only calculates historical losses after label maturity. A finite-input validation scan checks all supplied values, not a prediction feature. One new path per process is limited evidence; the chosen adaptive rule may underperform, and no tuning on these assessment paths is allowed. After inspection they are no longer fresh.

## 6. Run and verification

First complete Chapter XXXIII and keep its full report.json. Its local results have not yet been supplied to the assistant. The new CLI checks the report's scope, variant list, original selection seeds/fold counts, parameters and delay, and recomputes the four locked choices from selection folds only. The prior report hash is recorded; this is a consistency check, not independent proof of provenance. Do not pass summary.json, and do not change the selection based on Chapter XXXIII assessment results.

```bash
git pull origin main
python3 -m unittest tests.test_day16_adaptive -v
python3 -m examples.day16_adaptive --walkforward-report data/walkforward/synthetic_v1/report.json --output data/adaptive/synthetic_v1 --replay-delay-seconds 2
cat data/adaptive/synthetic_v1/report.json
```

If Chapter XXXIII has not completed, run its documented command first. If an output directory exists, read the earlier report or use a fresh output name. No new libraries. Outputs are staged and published to a new directory: report.json includes locked choices, metrics, selection counts, switches, fold coverage and refit events; predictions.csv includes selected candidate IDs, matured window sizes, adaptive/locked/MTF forecasts and actual outcomes. All under gitignored data/.

Five isolated tests actually executed in the assistant container with NumPy 2.3.5 and passed: strict maturity, switches and ties, future-outcome perturbation, replay on synthetic candidate predictions, rejection of incorrect selection seeds. The complete upstream GARCH/MTF/XGBoost integration, actual Chapter XXXIII report import and full 432-fit CLI have not been executed here. Run local tests first and report any errors.

## 7. Next gate

Compare adaptive and locked forecasts on all four controls, including no-skill benchmarks and selection stability. No production model promotion from synthetic results. Next probability calibration on a separately designed historical calibration segment and fresh evaluation paths, followed by cost-aware selective WAIT decisions. Real-data authorization, XAU coverage, measured latency, real holdout and prospective paper trading remain unresolved.

[Chapter XXXIII](33_purged_walkforward.md) | [Mathematical reference](mathematical_reference.md)
