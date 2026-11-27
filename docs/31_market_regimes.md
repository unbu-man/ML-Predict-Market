# Chapter XXXI — Unsupervised market-regime research: K-Means and GMM

**27 November 2026.** M5 synthetic-only experiment. No real data or holdout access; data-rights gate remains active. 

## 1. Regime inputs and standardization

Use four causal features already present in the 21-column MTF matrix: 24-bar mean and sample standard deviation of 5m returns, and mean and sample standard deviation of the three latest completed 1h returns. Only these four are standardized for clustering. StandardScaler is fit once on training observations, then frozen for validation; a constant feature uses scale one.


```math
\begin{aligned}
\mu_j&=\frac1{N_{\mathrm{train}}}\sum_{i\in\mathrm{train}}x_{ij}\\
s_j&=\sqrt{\frac1{N_{\mathrm{train}}}\sum_{i\in\mathrm{train}}(x_{ij}-\mu_j)^2}\\
z_{ij}&=(x_{ij}-\mu_j)/s_j
\end{aligned}
```

This is batch training of an unsupervised transformation for future predictions, not an online estimate available at every earlier training origin. No validation data or future-return labels are used to fit clustering.

## 2. K-Means: hard assignment

Partition standardized vectors into K=3 clusters by minimizing within-cluster squared Euclidean distance:


```math
\begin{aligned}
J&=\sum_{i=1}^N\sum_{k=1}^K a_{ik}\|z_i-c_k\|_2^2\\
a_{ik}&=\mathbf1\{k=\arg\min_j\|z_i-c_j\|_2^2\}\\
\frac{\partial J}{\partial c_k}&=2\sum_i a_{ik}(c_k-z_i)=0\\
c_k&=\frac{\sum_i a_{ik}z_i}{\sum_i a_{ik}}
\end{aligned}
```

The assignment and centroid update alternate until convergence (or the implementation limit). The centroid formula follows from differentiating at fixed assignments for a nonempty cluster. In Python use KMeans with three clusters, 10 initializations and seed 42. The training fit is not guaranteed to find a global minimum. Encode the predicted label as a three-element one-hot vector, not an ordinal number.

Worked example in two standardized dimensions: centers c0=(-1,0), c1=(0,1), c2=(1,0), point z=(0.8,0.1). Squared distances are 3.25, 1.45 and 0.05, respectively. The encoded cluster is (0,0,1). These numbers are illustrative, not market centers.

## 3. Gaussian Mixture Model: soft assignment

Assume a mixture density of K multivariate Gaussian components. In d dimensions, each has weight pi_k, mean m_k, and positive-definite covariance Sigma_k.


```math
\begin{aligned}
p(z)&=\sum_{k=1}^K\pi_k\,\mathcal N(z\mid m_k,\Sigma_k),\quad \sum_k\pi_k=1,\quad \pi_k\ge0\\
\mathcal N(z\mid m,\Sigma)&=\frac{\exp[-\frac12(z-m)^\top\Sigma^{-1}(z-m)]}{(2\pi)^{d/2}|\Sigma|^{1/2}}\\
\ell&=\sum_i\ln\left[\sum_k\pi_k\mathcal N(z_i\mid m_k,\Sigma_k)\right]
\end{aligned}
```

Use diagonal covariance, regularization 1e-6 on the standardized variance diagonal, three initializations, max 300 iterations, seed 42. Diagonal covariance is an engineering simplification, not a claim of independent features in markets.

Expectation-Maximization (EM) uses Bayes' rule in the E step:


```math
\gamma_{ik}=P(S_i=k\mid z_i)=
\frac{\pi_k\mathcal N(z_i\mid m_k,\Sigma_k)}
{\sum_j\pi_j\mathcal N(z_i\mid m_j,\Sigma_j)}
```

For the M step, maximize the expected complete-data log likelihood with responsibilities fixed. Using N_k as the effective membership, the weight, mean and diagonal variance updates (with regularization epsilon) are:


```math
\begin{aligned}
N_k&=\sum_i\gamma_{ik}\\
\pi_k&=N_k/N\\
m_k&=\frac1{N_k}\sum_i\gamma_{ik}z_i\\
(\Sigma_k)_{jj}&=\frac1{N_k}\sum_i\gamma_{ik}(z_{ij}-m_{kj})^2+\epsilon
\end{aligned}
```

Repeat E/M steps until convergence. The implementation rejects nonconverged fits. Example unnormalized component scores 0.02, 0.03, 0.05 sum to 0.10, giving responsibilities (0.2,0.3,0.5). This is a soft distribution over clusters, not a directional probability. GMM assigns each row independently under its fitted mixture; unlike a hidden Markov model it has no transition model.

## 4. Synthetic directional positive control

Keep the existing independent-return history as negative control. Generate an independent 40-day training period ending April 1, 2026 UTC and 20-day validation period starting that date, seed 20261128. Three latent states have persistence 0.995 per 5m step, with equal transition probability 0.0025 to either other state.


```math
\begin{aligned}
A&=\begin{bmatrix}.995&.0025&.0025\\.0025&.995&.0025\\.0025&.0025&.995\end{bmatrix}\\
A_{ij}&=P(S_{t+1}=j\mid S_t=i)\\
r_t&=\mu_{S_t}+\sigma_{S_t}\xi_t,\qquad \xi_t\overset{\mathrm{iid}}{\sim}N(0,1)\\
\boldsymbol\mu&=(-0.0007,0,0.0007)\\
\boldsymbol\sigma&=(0.001,0.0025,0.001)\\
C_t&=C_{t-1}\exp(r_t)
\end{aligned}
```

For the simulator only, conditional on the known current state i:


```math
\begin{aligned}
E[r_{t+1}\mid S_t=i]&=\sum_j A_{ij}\mu_j\\
P(r_{t+1}>0\mid S_t=i)&=\sum_j A_{ij}\Phi(\mu_j/\sigma_j)
\end{aligned}
```

Phi is the standard normal CDF. This mathematical structure deliberately creates a persistent directional signal in two states. No state ID, generator parameters or future observations enter any fitted feature or prediction model. Current-state truth is used only after clustering for validation scoring. No reserved June–July holdout is generated. The simulator is deliberately simple and not a model of actual BTC returns.

## 5. Matched regime ablation



```math
\begin{aligned}
X_t^{(0)}&=X_t^{\mathrm{MTF}}\\
X_t^{(K)}&=[X_t^{\mathrm{MTF}},a_{t1},a_{t2},a_{t3}]\\
X_t^{(G)}&=[X_t^{\mathrm{MTF}},\gamma_{t1},\gamma_{t2},\gamma_{t3}]
\end{aligned}
```

Compare 21, 24 and 24 features, respectively. Each receives identical 5m and 1h targets, origins and fixed XGBoost settings (100 shallow trees from Chapter XXVIII). Refit all variants on training only; no validation hyperparameter tuning. Prevalence, zero return and training mean are retained. GARCH is intentionally not included in this chapter, isolating the regime effect before a joint ablation. The four clustering inputs are already in the MTF feature set, so cluster encodings are transformations, not genuinely new raw information.

For state recovery, use the adjusted Rand index (ARI) on validation only for the positive control. It compares two partitions independent of numeric label permutations. Let n_ij be the contingency counts, a_i and b_j their row and column sums, and N the total.


```math
\begin{aligned}
U&=\sum_{ij}\binom{n_{ij}}2,\quad V=\sum_i\binom{a_i}2,\quad W=\sum_j\binom{b_j}2\\
E&=VW/\binom N2\\
\mathrm{ARI}&=\frac{U-E}{\tfrac12(V+W)-E}
\end{aligned}
```

Use the scikit-learn convention for degenerate partitions. ARI=1 for identical partitions up to relabeling, around zero for random agreement, and possibly negative. It is not forecasting accuracy. GMM hard assignments for ARI use argmax of responsibilities; XGBoost receives the soft responsibilities. Cluster IDs may permute between algorithms and controls; interpret fitted centroids rather than automatically assigning bull/bear labels. The negative control has no true state labels.

Expected matched counts for complete aligned blocks using the existing strict cutoff and daily warmup:


```math
\begin{aligned}
N_{\mathrm{neg,train}}&=69984-1164=68820\\
N_{\mathrm{neg,val}}&=17568-1164=16404\\
N_{\mathrm{pos,train}}&=40(288)-1164=10356\\
N_{\mathrm{pos,val}}&=20(288)-1164=4596
\end{aligned}
```

No market skill follows from successful positive-control learnability. A model may learn direction without accurately recovering latent states, or recover states without improving direction relative to an existing baseline.

## 6. Implementation and tests

Use existing requirements-xgboost.txt. The new module's five unit tests passed in the assistant container with NumPy 2.3.5, scikit-learn 1.8.0 and XGBoost 3.1.3 (encoding sums, validation-perturbation invariance, nonfinite rejection, deterministic simulation and matched model outputs; the small model test uses ten trees). Direct git access from the container failed due to DNS, so these tests exercised the new module in isolation; full repository loader/MTF/CLI integration and full-size experiment remain pending. No claim that the Mac tests passed.

From repo root:
```bash
git pull origin main
python3 -m unittest tests.test_day13_regimes -v
python3 -m examples.day13_regimes data/history/synthetic_1y_plan --output data/regimes/synthetic_v1 --replay-delay-seconds 2
cat data/regimes/synthetic_v1/report.json
```

Existing output directory is protected. Read the report or choose a fresh directory. Outputs under gitignored data/: report.json and two prediction CSVs. Report includes training/validation cluster counts, standardized GMM covariance, centers in original units, component weights, mean responsibilities, positive-control validation ARI, metrics and hashes. Inspect convergence, sample counts, cluster occupancy and positive/negative control behavior. No clustering artifacts are deployed.

## 7. Next gate

Assess direction learnability on the known-regime control versus independent-return control and matched XGBoost baseline. Then consider a GARCH x regime factorial ablation and a separate purged walk-forward research protocol. Do not select a production feature group based on this simulation. HMM, TAR, adaptive routing, calibration and selective risk rules are later experiments, not implemented here. Authorized real-data coverage, XAU/USD source, measured latency, costs and prospective paper trading remain unresolved.

[Chapter XXX](30_hybrid_garch_xgboost.md) | [Chapter XXIX](29_multitimeframe.md) | [Mathematical reference](mathematical_reference.md)


## 8. User-reported local results — 27 November 2026

The user supplied the full local report. Counts match the expected common origins: negative control 68,820 train / 16,404 validation; positive control 10,356 / 4,596. The source block hashes and fixed settings match the earlier experiments. No holdout exported. This establishes a user-reported successful CLI run, not independent verification of raw prediction rows or Mac unit-test output.

### Classification metrics

| Control | Horizon | Variant | Accuracy | Brier | Log loss | ROC-AUC |
|---|---|---|---:|---:|---:|---:|
| Negative | 5m | Prevalence | .495001 | .250003 | .693153 | .500000 |
| Negative | 5m | MTF | .498537 | .250126 | .693401 | .500125 |
| Negative | 5m | K-Means | .503109 | .250124 | .693397 | .501116 |
| Negative | 5m | GMM | .494392 | .250136 | .693420 | .498051 |
| Negative | 1h | Prevalence | .484760 | .250060 | .693266 | .500000 |
| Negative | 1h | MTF | .513655 | .249618 | .692383 | .527637 |
| Negative | 1h | K-Means | .513655 | .249618 | .692383 | .527637 |
| Negative | 1h | GMM | .516277 | .249629 | .692407 | .526341 |
| Positive | 5m | Prevalence | .439948 | .252754 | .698658 | .500000 |
| Positive | 5m | MTF | .624891 | .220523 | .631542 | .674442 |
| Positive | 5m | K-Means | .624238 | .220524 | .631500 | .676106 |
| Positive | 5m | GMM | .628372 | .220221 | .630811 | .675938 |
| Positive | 1h | Prevalence | .373586 | .260813 | .714816 | .500000 |
| Positive | 1h | MTF | .722150 | .152015 | .452221 | .820644 |
| Positive | 1h | K-Means | .715187 | .151248 | .449931 | .823794 |
| Positive | 1h | GMM | .723890 | .150244 | .445302 | .825397 |

All three learned variants recognize some deliberately injected directional structure. GMM adds only 0.3481 and 0.1741 percentage points in accuracy over MTF at 5m and 1h, respectively. K-Means has the best positive-control 5m AUC, GMM the best positive-control 1h AUC and probability losses at both horizons. Training prevalence is a weak fixed-direction baseline under the observed shift in class balance. Retrospectively, the validation majority-class fractions on the positive control are .560052 and .626414, respectively; these are descriptive oracle fractions, not deployable baseline forecasts. The independent-return negative control is still near chance, with modest finite-sample deviations. Do not promote a variant based on this single simulation.

### Regression metrics

| Control | Horizon | Variant | MAE | RMSE | Sign accuracy |
|---|---|---|---:|---:|---:|
| Negative | 5m | Zero return | .000899706 | .001129797 | .504999 |
| Negative | 5m | Training mean | .000899708 | .001129800 | .495001 |
| Negative | 5m | MTF | .000900021 | .001129944 | .499329 |
| Negative | 5m | K-Means | .000900021 | .001129944 | .499329 |
| Negative | 5m | GMM | .000900113 | .001129991 | .495977 |
| Negative | 1h | Zero return | .003151525 | .003947176 | .515240 |
| Negative | 1h | Training mean | .003151590 | .003947291 | .484760 |
| Negative | 1h | MTF | .003149267 | .003947232 | .513655 |
| Negative | 1h | K-Means | .003149267 | .003947232 | .513655 |
| Negative | 1h | GMM | .003149687 | .003947612 | .516398 |
| Positive | 5m | Zero return | .001489944 | .001967679 | .439948 |
| Positive | 5m | Training mean | .001496239 | .001972658 | .439948 |
| Positive | 5m | MTF | .001407157 | .001913682 | .626414 |
| Positive | 5m | K-Means | .001406653 | .001913718 | .628372 |
| Positive | 5m | GMM | .001406611 | .001913199 | .624674 |
| Positive | 1h | Zero return | .007575368 | .008678124 | .373586 |
| Positive | 1h | Training mean | .007735168 | .008844117 | .373586 |
| Positive | 1h | MTF | .005090199 | .006764983 | .721715 |
| Positive | 1h | K-Means | .005079111 | .006749584 | .728460 |
| Positive | 1h | GMM | .005060560 | .006734089 | .734552 |

Unscaled log-return units. Positive GMM achieves the lowest MAE/RMSE at both horizons. At 1h it lowers RMSE by about 22.40% relative to zero return, but only 0.46% relative to MTF. Negative K-Means regression scores exactly match MTF at both horizons; equal aggregate metrics alone cannot establish that the fitted trees or individual predictions are identical. Negative 1h GMM has higher accuracy but worse probability losses than MTF. Different metrics measure different objectives.

### State recovery and fitted cluster interpretation

Validation ARI is .799345 for K-Means and .837534 for GMM on the directional simulator. ARI is a chance-adjusted agreement metric, not a percentage of correctly classified observations. No latent state recovery score is defined for the independent-return control.

Positive-control centers in original log-return units (column order: mean_return_24, std_return_24, 1h_mean3, 1h_std3):

| Algorithm | ID | Mean 5m return | 5m std | Mean 1h return | 1h std | Descriptive interpretation |
|---|---:|---:|---:|---:|---:|---|
| K-Means | 0 | .00004543 | .00241245 | .00061433 | .00818340 | Higher-volatility, near-neutral |
| K-Means | 1 | -.00068625 | .00104835 | -.00813394 | .00336687 | Negative trend |
| K-Means | 2 | .00067755 | .00100397 | .00797751 | .00326456 | Positive trend |
| GMM | 0 | -.00070998 | .00098657 | -.00847742 | .00313585 | Negative trend |
| GMM | 1 | .00070130 | .00096384 | .00842446 | .00305095 | Positive trend |
| GMM | 2 | .00000930 | .00220938 | .00009368 | .00754180 | Higher-volatility, near-neutral |

These are feature centroids, not maximum-likelihood estimates of the underlying Markov generator's emission parameters. Numeric cluster IDs differ between algorithms. K-Means training counts [2905,4154,3297] and validation counts [2286,677,1633] imply a shift in occupancy. GMM mean responsibilities shift from [.354765,.287732,.357503] in training to [.121423,.325972,.552605] in validation. Soft mean responsibilities are not hard class counts. K-Means also partitions the independent-return sample, showing why cluster existence does not establish genuine economic regimes.

### Mathematical interpretation and next gate

```math
\begin{aligned}
\Delta\mathrm{Accuracy}_{\mathrm{pp}}&=100(A_{\mathrm{augmented}}-A_{\mathrm{base}})\\
\mathrm{RMSE\ reduction}_{\%}&=100\left(1-\frac{\mathrm{RMSE}_{\mathrm{augmented}}}{\mathrm{RMSE}_{\mathrm{base}}}\right)\\
\Delta\mathrm{ARI}&=\mathrm{ARI}_{\mathrm{GMM}}-\mathrm{ARI}_{\mathrm{KMeans}}
\end{aligned}
```

With positive-control 1h accuracy .7238903394 versus .7221496954, the incremental GMM gain is .1740644 percentage points; RMSE .0067340891 versus .0067649828 gives .4567% relative reduction. ARI gain is .0381883, not 3.82 percentage points of classification accuracy.

Next Chapter XXXII: implement matched factorial ablation across variance features (none, rolling, causal GARCH) and regime encodings (none, K-Means, GMM), while fixing the MTF base and model settings. Consider all three controls (independent returns, zero-mean GARCH, persistent directional states) and an explicitly designed joint mean/variance control; any nonzero-mean GARCH feature needs an appropriate causal residual definition or clear labeling as a misspecified zero-mean proxy. Separate variance prediction from direction/return tasks. Compare paired losses within each control and add purged walk-forward evaluation across independently seeded simulations before adaptive selection. Preserve no-holdout and no-real-data gates; no trading performance, model promotion, or real-data rights established.

## Follow-up

[Chapter XXXII](32_combined_hybrid.md) implements the nine matched factorial variants and joint control. Full local experiment pending.
