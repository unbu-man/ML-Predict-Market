# Chapter XXXIII — Purged walk-forward validation and locked selection

**27 November 2026.** Synthetic-only reliability research. No real market dataset or reserved market holdout accessed; data-rights gate remains active.

## 1. Information boundary and fold design

Let a denote a five-minute candle's completed close, h the number of future five-minute bars (1 or 12), and delta the assumed data delay. Label availability and the common strict training cutoff are:



```math
\begin{aligned}
Y_{a,h}&=\ln(C_{a+300h}/C_a)\\
T_{\mathrm{label}}(a,h)&=a+300h+\delta\\
\mathcal T_k&=\{a:a+3600+\delta<B_k\}\\
\mathcal E_k&=\{a:B_k\le a,\quad a+3600+\delta<B_k+10D\}
\end{aligned}
```

D=86400 seconds; B_k is the start of fold k's evaluation period. Use the stricter one-hour rule for both tasks to preserve matched origins. Each fold refits scaler, K-Means/GMM and supervised models on eligible training rows only. The next fold may train on prior evaluation observations once their labels are available; no assessment results enter the architecture selection. No additional embargo beyond the common label-availability purge is claimed.

| Fold | Expanding training history | Evaluation origin interval |
|---|---|---|
| 1 | First 30 days | Day 30 to day 40 |
| 2 | First 40 days | Day 40 to day 50 |
| 3 | First 50 days | Day 50 to day 60 |

Day boundaries are elapsed offsets from the start (day zero); right endpoints exclusive. Causal MTF and GARCH features are built once on a continuous simulated path, retaining past context across boundaries. Each row uses only completed historical returns and historical-prefix GARCH refits, not the full path fit; clustering remains a batch training-only transform for each fold. The strict common label cutoff applies again at each fold end. Simulated publication latency is not measured. A failed GARCH refit may exclude matched rows for all candidates. Insufficient matched rows stop the run rather than silently dropping a fold.

Worked boundary example: with delta=2 seconds and B=12:00:00, a=11:00:00 has a one-hour label at 12:00:02 and is excluded from training. At a=10:55:00 the label is available at 11:55:02 and is eligible if its features exist.

## 2. Fresh synthetic paths

Use four processes from Chapter XXXII: independent Gaussian 5m returns (sd=.001125), zero-mean GARCH innovations, persistent Markov directional regimes, and joint Markov means with GARCH innovations. These are newly generated 60-day paths beginning at an illustrative 2024-01-01 UTC epoch; no existing history directory or real data is read. State starts neutral; variance starts at unconditional GARCH variance, without a separate burn-in. Transition stay probability .995, alternatives .0025 each; means (-.0007,0,.0007), directional regime sigmas (.001,.0025,.001); GARCH omega=1e-7, alpha=.12, beta=.85. In nonzero-mean controls the reused zero-mean GARCH feature remains a potentially misspecified raw second-moment proxy.

One selection seed per control (33001+100c, c=0,1,2,3 in the fixed control order) and two different assessment seeds per control (33002+100c, 33003+100c). Selection is completed before assessment begins. Generator choices are engineered controls, not estimated market dynamics. The assessment paths are fresh relative to earlier chapters but their seeds are now public; after inspection they are no longer fresh assessment evidence.

## 3. Coverage and computational size

For complete paths without failed volatility fits, the first eligible origin is close index 1440 (1441 candles), and each ten-day evaluation interval has 2880 five-minute slots. The last 12 origins are removed by the common strict label cutoff.



```math
\begin{aligned}
N_{\mathrm{train},k}&=288(30+10(k-1))-1440-13\\
N_{\mathrm{eval},k}&=10(288)-12=2868\\
\mathrm{Coverage}_k&=N_{\mathrm{eligible,eval},k}/2868\\
N_{\mathrm{paths}}&=4(1+2)=12\\
N_{\mathrm{folds}}&=12(3)=36\\
N_{\mathrm{fits}}&=36(9)(2)(2)=1296
\end{aligned}
```

Expected training counts 7187, 10067, 12947; evaluation counts 2868 per fold. Six scheduled GARCH refits per 60-day path, or 72 attempts total. Coverage is relative to eligible calendar-grid origins after label-boundary purging, not trade coverage. Report actual counts and minimum coverage. No parallel path fits; reuse the Chapter XXVIII two-worker XGBoost settings. The default experiment is substantially heavier than Chapter XXXII. Run it once, not repeatedly on the same seeds to tune the model.

## 4. Loss aggregation and selection

Choose separately for each forecast horizon and task among the nine factorial candidates, using Brier for classification and MSE for regression. Let L_{p,k,c} be a candidate's loss in fold k on selection path p; L_{p,k,0} is the matched MTF-only loss. Use a relative loss so that different simulator scales do not dominate the regression comparison. The base loss must be positive.



```math
\begin{aligned}
L^{\mathrm{class}}&=\frac1N\sum_i(p_i-y_i)^2\\
L^{\mathrm{reg}}&=\frac1N\sum_i(\widehat Y_i-Y_i)^2=\mathrm{RMSE}^2\\
R_{p,k,c}&=\frac{L_{p,k,c}}{L_{p,k,0}}\\
S_c&=\frac1P\sum_{p=1}^{P}\frac1K\sum_{k=1}^{K}R_{p,k,c}\\
c^*&=\arg\min_c S_c
\end{aligned}
```

Here P=4 selection paths, K=3 folds. Equal weight per fold and path; an exact tie prefers fewer features then lexical name. Four locked choices, one per horizon/task, are global across controls; no routing using simulator identity or oracle state. A score below one means lower mean relative loss than MTF-only, not necessarily a lower error than a no-skill benchmark. Example ratios .98, 1.02 and .97 give path mean .99, or 1% lower relative loss on average. Prevalence, zero-return and training-mean benchmarks are evaluated but not included in the nine-way architecture selection.

## 5. Locked assessment

On the two fresh assessment paths per control, report mean absolute Brier or MSE across folds and paths for the locked candidate, MTF and the simple benchmarks. All candidates' detailed metrics are also preserved for research auditing, but inspecting them must not change the locked choices. Only the architecture is locked; learned parameters are re-estimated in each expanding historical fold.



```math
\begin{aligned}
\bar L_{p,c}&=\frac1K\sum_{k=1}^{K}L_{p,k,c}\\
\bar L_c&=\frac1{P_a}\sum_{p=1}^{P_a}\bar L_{p,c}\\
s_L&=\sqrt{\frac{\sum_{p,k}(L_{p,k,c}-\bar L_c)^2}{P_a K-1}}
\end{aligned}
```

P_a=2 per control, K=3. MSE is in squared unscaled log-return units; its square root is the root of mean fold MSE, not mean fold RMSE. Fold SD is descriptive only: expanding training sets and overlapping hourly targets induce dependence; six folds do not equal six independent trials. Two seeds per control are a small assessment, not sufficient for reliable statistical significance or tail-risk claims. Do not promote a model unless future authorized real-data evidence supports it.

## 6. Implementation, verification and run

Use the existing dependencies and run from the repo root:

```bash
git pull origin main
python3 -m unittest tests.test_day15_walkforward -v
python3 -m examples.day15_walkforward --output data/walkforward/synthetic_v1 --replay-delay-seconds 2
cat data/walkforward/synthetic_v1/summary.json
```

Four isolated tests actually ran and passed in the assistant container (deterministic fresh simulation, purge and disjoint evaluation windows, selection/assessment isolation, small real K-Means/GMM and nine-way XGBoost model fit on fixtures with three trees). NumPy 2.3.5, scikit-learn 1.8.0, XGBoost 3.1.3. The copied Day 13 and Day 14 Python blobs match their previously committed hashes. Direct GitHub clone failed due to DNS, so full upstream GARCH/MTF integration, CSV writing through the CLI and the full 1296-fit run are not verified here. Run local unit tests first. Output is staged then published to a new directory; no overwrite. It contains summary.json (compact locked selection, assessment and coverage/refit audit), report.json (fold-level metrics and fit events), and predictions.csv (all candidates, phase/control/seed/fold identifiers). No existing data/history input is needed. Progress prints one line per path. A failed run should not be treated as a completed assessment.

## 7. Next gate

Review the locked choices against simple benchmarks on each fresh assessment process and check failure/coverage audit. Do not rerank using the assessment table. Next investigate causal adaptive selection versus fixed locked candidates and a separate calibration protocol with fresh paths, followed by selective WAIT logic and cost-aware paper-only evaluation. Real market permissions, XAU data, measured publication/computation latency and untouched real-market evaluation are still unresolved. No live orders, no evidence of market profitability.

[Chapter XXXII](32_combined_hybrid.md) | [Mathematical reference](mathematical_reference.md)
