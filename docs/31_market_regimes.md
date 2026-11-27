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
