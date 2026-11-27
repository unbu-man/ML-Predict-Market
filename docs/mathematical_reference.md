# Mathematical reference — Adaptive Market Prediction Engine (AMPE)

**27 November 2026.** Living study handbook for the research project. Each equation should be taught with its assumptions, variable definitions, derivation where appropriate, a worked example, implementation and evaluation. General equations below are not claims of market predictability. Existing chapter-specific fitted parameters and results belong in their original experiment records; this handbook does not duplicate restricted market-data-derived numerical results. Synthetic-only development remains the current gate.

## Navigation

| Topic | Full lesson |
|---|---|
| Forecast targets and information clock | [III](03_ml_research_protocol.md), [XV](15_adaptive_market_prediction_engine.md), [XVII](17_forecast_target_contract_template.md), [XVIII](18_day01_forecast_contract_draft.md) |
| Descriptive statistics, ACF and stationarity | [XXII](22_market_eda_tsa.md) |
| Lagged returns and rolling features | [XXIII](23_causal_feature_dataset.md) |
| Random walk, AR, MA and ARMA | [XXIV](24_statistical_forecasting.md) |
| ARCH/GARCH, variance forecasts and diagnostics | [XXV](25_garch_volatility.md) |
| Logistic and Ridge regression | [XXVI](26_supervised_baselines.md) |
| Candle aggregation and dataset counts | [XXVII](27_historical_data_expansion.md) |
| XGBoost derivatives and tree weights | [XXVIII](28_xgboost.md) |
| Multi-timeframe alignment and ablations | [XXIX](29_multitimeframe.md) |
| Hybrid GARCH + XGBoost (proposed) | [XXIX preview](29_multitimeframe.md#10-preview--hybrid-garch--xgboost) |
| Indicator and risk mathematics | [XIII](13_30_day_math_signal_bot_plan.md), [XIX](19_hybrid_selective_forecast_strategy.md) |

## 1. Return, target and probability

Let C be a completed closing price, t the current base-bar index, h the number of future bars, and F_t the information available at the decision. The base bar is five minutes; h=1 or 12 corresponds to five minutes or one hour.



```math
\begin{aligned}
z_t&=\ln C_t\\
r_t&=z_t-z_{t-1}=\ln(C_t/C_{t-1})\\
R_t^{\mathrm{simple}}&=C_t/C_{t-1}-1\\
Y_{t,h}&=\sum_{k=1}^{h}r_{t+k}=\ln(C_{t+h}/C_t)\\
y_{t,h}&=\mathbf1\{Y_{t,h}>0\}
\end{aligned}
```



```math
\begin{aligned}
\widehat\mu_{t,h}&\approx E[Y_{t,h}\mid\mathcal F_t]\\
\widehat\sigma_{t,h}^2&\approx\operatorname{Var}(Y_{t,h}\mid\mathcal F_t)\\
\widehat p_{t,h}&\approx P(Y_{t,h}>0\mid\mathcal F_t)
\end{aligned}
```

A zero return is NOT_UP. Positive expected return is not equivalent to probability of UP greater than one half. A class is not an execution decision. Log-return percent units mean 100 times the log return, not exactly the simple percentage return.

## 2. Descriptive statistics and time-series diagnostics

For n observed returns, sample variance uses denominator n-1. Population-style autocovariance estimates here use denominator n.



```math
\begin{aligned}
\bar r&=\frac1n\sum_{t=1}^{n}r_t\\
s^2&=\frac1{n-1}\sum_{t=1}^{n}(r_t-\bar r)^2,\qquad s=\sqrt{s^2}\\
\widehat\gamma_k&=\frac1n\sum_{t=k+1}^{n}(r_t-\bar r)(r_{t-k}-\bar r)\\
\widehat\rho_k&=\widehat\gamma_k/\widehat\gamma_0
\end{aligned}
```



```math
\operatorname{Var}(r_t)=\gamma_0,\qquad q_t=r_t^2,\qquad
\text{approximate pointwise white-noise ACF bounds}=\pm1.96/\sqrt n
```

The ACF reference bounds are approximate, not simultaneous or universally valid under heteroskedasticity. PACF measures linear association at lag k controlling intermediate lags.

The following diagnostic formulas are general teaching definitions; refer to the relevant library specification for exact test configuration and p-value calculation.



```math
\begin{aligned}
Q_{\mathrm{LB}}&=n(n+2)\sum_{k=1}^{m}\frac{\widehat\rho_k^2}{n-k}\\
\varepsilon_t^2&=a_0+\sum_{j=1}^{m}a_j\varepsilon_{t-j}^2+u_t\\
LM_{\mathrm{ARCH}}&\approx N R^2
\end{aligned}
```

Ljung-Box tests residual serial correlation; ARCH-LM uses an auxiliary regression of squared residuals. N is the auxiliary sample size (software may apply degrees-of-freedom corrections). ADF tests a unit-root null; KPSS tests a stationarity null. Failing to reject is not proof of a null. Use nominal p-values cautiously for dependent or changing data.

## 3. Causal feature engineering



```math
\begin{aligned}
\mathbf x_{t,\mathrm{lags}}&=(r_t,r_{t-1},r_{t-2},r_{t-3},r_{t-6},r_{t-12})\\
\bar r_{t,w}&=\frac1w\sum_{i=0}^{w-1}r_{t-i}\\
s_{t,w}&=\sqrt{\frac1{w-1}\sum_{i=0}^{w-1}(r_{t-i}-\bar r_{t,w})^2}\\
M_{t,w}&=z_t-z_{t-w}=w\bar r_{t,w}
\end{aligned}
```

Windows w=6,12,24. Momentum is redundant with rolling mean for this definition; omit it for the existing linear baselines. With a contiguous segment of N candles, warmup w_max and reserved horizon h_max, the initial paired count is N-w_max-h_max before further boundary purges. Never bridge missing candles.

## 4. Random walk and classical TSA



```math
\begin{aligned}
z_{t+1}&=z_t+\varepsilon_{t+1}\\
E(\varepsilon_{t+1}\mid\mathcal F_t)&=0\\
\widehat Y_{t,h}^{\mathrm{zero}}&=0
\end{aligned}
```



```math
\begin{aligned}
\mathrm{AR}(1):\quad r_t&=c+\phi r_{t-1}+\varepsilon_t\\
\mu&=c/(1-\phi),\quad |\phi|<1\\
r_t-\mu&=\phi(r_{t-1}-\mu)+\varepsilon_t\\
\widehat r_{t+1\mid t}&=\hat\mu+\hat\phi(r_t-\hat\mu)\\
\mathrm{MA}(1):\quad r_t&=\mu+\varepsilon_t+\theta\varepsilon_{t-1}\\
\widehat r_{t+1\mid t}&=\hat\mu+\hat\theta\hat\varepsilon_t\\
\mathrm{ARMA}(1,1):\quad r_t-\mu&=\phi(r_{t-1}-\mu)+\varepsilon_t+\theta\varepsilon_{t-1}\\
\widehat r_{t+1\mid t}&=\hat\mu+\hat\phi(r_t-\hat\mu)+\hat\theta\hat\varepsilon_t
\end{aligned}
```



```math
\begin{aligned}
\hat\phi_{\mathrm{OLS}}&=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sum_i(x_i-\bar x)^2},\quad x_i=r_{i-1},\ y_i=r_i\\
\hat c_{\mathrm{OLS}}&=\bar y-\hat\phi\bar x\\
m_k&=\hat\mu+\hat\phi^{k-1}(m_1-\hat\mu)\\
\widehat Y_{t,h}&=\sum_{k=1}^{h}m_k\\
\mathrm{AIC}&=2K-2\log\widehat L\\
\mathrm{BIC}&=K\log n-2\log\widehat L
\end{aligned}
```

Here K is the parameter count and L is likelihood. OLS is a teaching estimator; Chapter XXIV implements likelihood fitting. AR coefficient phi is also called beta_1 in earlier chapters. For MA use phi=0 for the forecast recursion after the first step. Compare AIC/BIC only on comparable likelihoods/data; neither proves forecasting performance.

## 5. ARCH and GARCH conditional variance



```math
\begin{aligned}
\varepsilon_t&=\sigma_t z_t^{*}\\
E(z_t^{*}\mid\mathcal F_{t-1})&=0,\quad E[(z_t^{*})^2\mid\mathcal F_{t-1}]=1\\
\mathrm{ARCH}(1):\quad \sigma_t^2&=\omega+\alpha_1\varepsilon_{t-1}^2\\
\mathrm{GARCH}(1,1):\quad \sigma_t^2&=\omega+\alpha_1\varepsilon_{t-1}^2+\beta_1\sigma_{t-1}^2\\
\widehat\sigma_{t+1\mid t}^2&=\omega+\alpha_1\varepsilon_t^2+\beta_1\sigma_t^2\\
\omega&>0,\quad\alpha_1,\beta_1\geq0,\quad\alpha_1+\beta_1<1\\
v_\infty&=\frac{\omega}{1-\alpha_1-\beta_1}
\end{aligned}
```

The starred z is a standardized innovation, not log price. Alpha measures reaction to squared innovations, beta measures variance persistence; their sum governs expected variance decay under the model. The final constraint gives finite unconditional innovation variance under the stated assumptions.



```math
\begin{aligned}
s_0^2&=\frac1{N_{\mathrm{train}}}\sum_{\mathrm{train}}\varepsilon_t^2\\
u_t&=\varepsilon_t/s_0,\quad v_t=\sigma_t^2/s_0^2\\
\mathcal L_{\mathrm{QML}}&=\frac1{2N_{\mathrm{train}}}\sum_{\mathrm{train}}\left(\ln v_t+u_t^2/v_t\right)\\
\widehat v_{t+1\mid t}^{\mathrm{constant}}&=s_0^2\\
\bar\varepsilon_{t,w}&=\frac1w\sum_{i=0}^{w-1}\varepsilon_{t-i}\\
\widehat v_{t+1\mid t}^{\mathrm{rolling}}&=\frac1{w-1}\sum_{i=0}^{w-1}(\varepsilon_{t-i}-\bar\varepsilon_{t,w})^2
\end{aligned}
```

Example (invented normalized units): omega=0.1, alpha=0.2, beta=0.7, current squared innovation=4 and variance=1.



```math
\widehat\sigma_{t+1\mid t}^2=0.1+0.2(4)+0.7(1)=1.6,\qquad
\widehat\sigma_{t+1\mid t}=\sqrt{1.6}\approx1.265
```

This is a one-step innovation volatility, not a direction probability. Never multiply standard deviation by twelve to obtain one-hour volatility. Under a zero-mean martingale-difference return specification, aggregate conditional variance is the sum of expected one-step variances; AR mean dynamics require covariance/impulse-response treatment.

## 6. Logistic and Ridge Regression



```math
\begin{aligned}
\mu_j&=\frac1{N_{\mathrm{train}}}\sum_{\mathrm{train}}x_{ij}\\
s_j&=\sqrt{\frac1{N_{\mathrm{train}}}\sum_{\mathrm{train}}(x_{ij}-\mu_j)^2}\\
z_{ij}&=(x_{ij}-\mu_j)/s_j\\
\eta_i&=\beta_0+\sum_{j=1}^{p}\beta_jz_{ij}\\
p_i&=\frac1{1+\exp(-\eta_i)}\\
\log\frac{p_i}{1-p_i}&=\eta_i
\end{aligned}
```



```math
\min_{\beta_0,\boldsymbol\beta}\left[-\sum_i\{y_i\ln p_i+(1-y_i)\ln(1-p_i)\}
+\frac1{2C}\sum_{j=1}^p\beta_j^2\right]
```



```math
\begin{aligned}
\widehat Y_i&=\beta_0+\sum_{j=1}^{p}\beta_jz_{ij}\\
J_{\mathrm{ridge}}&=\sum_i(Y_i-\widehat Y_i)^2+\lambda\sum_{j=1}^p\beta_j^2\\
(X^\top X+\lambda I)\widehat{\boldsymbol\beta}&=X^\top\mathbf Y\\
\widehat{\boldsymbol\beta}&=(X^\top X+\lambda I)^{-1}X^\top\mathbf Y
\end{aligned}
```

The Ridge closed form assumes centered data and an unpenalized intercept. Fit scaling on training rows only; a constant feature uses unit scaling. For the existing fixed models C=1 and lambda=1. Logistic score 0.4 gives probability about 0.5987; that is an illustrative calculation, not a market signal.

## 7. XGBoost mathematics



```math
\begin{aligned}
\widehat y_i^{(K)}&=\widehat y_i^{(0)}+\eta\sum_{k=1}^{K}f_k(x_i)\\
p_i&=\frac1{1+e^{-\widehat y_i^{(K)}}}\\
\mathcal L&=\sum_i\ell(y_i,\widehat y_i)+\sum_k\Omega(f_k)\\
\Omega(f)&=\gamma T+\frac{\lambda}{2}\sum_{j=1}^T w_j^2\\
g_i&=\frac{\partial\ell(y_i,\widehat y_i)}{\partial\widehat y_i}\\
h_i&=\frac{\partial^2\ell(y_i,\widehat y_i)}{\partial\widehat y_i^2}\\
\widetilde{\mathcal L}^{(k)}&=\sum_i[g_i f_k(x_i)+\tfrac12 h_i f_k(x_i)^2]+\Omega(f_k)
\end{aligned}
```



```math
\begin{aligned}
\ell_{\mathrm{reg}}&=\tfrac12(y-\widehat y)^2,\quad g=\widehat y-y,\quad h=1\\
\ell_{\mathrm{class}}&=-y\ln p-(1-y)\ln(1-p),\quad g=p-y,\quad h=p(1-p)\\
G_j&=\sum_{i\in I_j}g_i,\quad H_j=\sum_{i\in I_j}h_i\\
J_j(w)&=G_jw+\tfrac12(H_j+\lambda)w^2\\
\frac{\partial J_j}{\partial w}&=G_j+(H_j+\lambda)w=0\\
w_j^*&=-G_j/(H_j+\lambda)
\end{aligned}
```



```math
\mathrm{Gain}=\tfrac12\left[
\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}
-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma
```

Here K is the tree count, eta learning rate, T leaves, w leaf score, gamma split penalty, lambda L2 penalty; G and H are gradient/Hessian sums. Classification derivatives use raw logit. The second-order approximation ignores terms constant with respect to the new tree; shrinkage is applied after fitting. Child weight is a Hessian sum, not always a row count. See Chapter XXVIII for the worked leaf-weight example.

## 8. Timeframe aggregation and as-of alignment

With D seconds per larger timeframe, t an aligned 5m open timestamp and delta an assumed delay:



```math
\begin{aligned}
m&=D/300,\qquad b(t)=D\left\lfloor t/D\right\rfloor\\
T_{\mathrm{close}}&=b(t)+D\\
O^{(D)}&=O_1,\quad H^{(D)}=\max_i H_i,\quad L^{(D)}=\min_i L_i\\
C^{(D)}&=C_m,\quad V^{(D)}=\sum_{i=1}^{m}V_i\\
t_{\mathrm{decision}}&=a+\delta\\
t_{\mathrm{available},k}^{(D)}&=T_{\mathrm{close},k}^{(D)}+\delta\\
k^*(a,D)&=\max\{k:t_{\mathrm{available},k}^{(D)}\le t_{\mathrm{decision}}\}
\end{aligned}
```



```math
\begin{aligned}
r_k^{(D)}&=\ln(C_k^{(D)}/C_{k-1}^{(D)})\\
\bar r_{k,3}^{(D)}&=\tfrac13\sum_{i=0}^2r_{k-i}^{(D)}\\
s_{k,3}^{(D)}&=\sqrt{\tfrac12\sum_{i=0}^2(r_{k-i}^{(D)}-\bar r_{k,3}^{(D)})^2}\\
a+3600+\delta&<T_{\mathrm{end\ exclusive}}
\end{aligned}
```

D=3600,14400,86400 gives m=12,48,288. Only complete buckets within a continuous segment are used; four higher-frame closes are needed for three returns. The last inequality is the common strict label-availability cutoff for the 1h horizon. The existing MTF experiment resets at block starts and gaps. In real data use measured publication/computation times, not the constant delay assumption.



```math
\begin{aligned}
N_{\mathrm{expected}}&=365\cdot24\cdot60/5=105120\\
N_{\mathrm{features}}&=12+3+3+3=21\\
\Delta L_D&=L_D-L_{\mathrm{5m\ only}}
\end{aligned}
```

Compare losses on matched anchors; negative differences favor the added feature group on that evaluation block.

## 9. Hybrid GARCH + XGBoost (proposed, not implemented here)



```math
\begin{aligned}
\widehat\sigma^2_{t+1\mid t}&=\widehat\omega_t+\widehat\alpha_{1,t}\varepsilon_t^2+
\widehat\beta_{1,t}\widehat\sigma_t^2\\
\widehat\sigma_{t+1\mid t}&=\sqrt{\widehat\sigma^2_{t+1\mid t}}\\
X_t^{\mathrm{hybrid}}&=\begin{bmatrix}
X_t^{5m}\\X_t^{1h}\\X_t^{4h}\\X_t^{1D}\\\widehat\sigma_{t+1\mid t}
\end{bmatrix}\\
\widehat p_{t,h}&=\operatorname{sigmoid}\left(F_h^{\mathrm{class}}(X_t^{\mathrm{hybrid}})\right)\\
\widehat Y_{t,h}&=F_h^{\mathrm{reg}}(X_t^{\mathrm{hybrid}})
\end{aligned}
```

The last element is a single one-step volatility feature for either forecast horizon; it is not a one-hour volatility forecast. GARCH parameters must be estimated using only information available at the historical fit date, then updated causally. Do not use full-training fitted parameters for earlier origins. Compare base, base plus rolling volatility, and base plus GARCH on identical eligible observations. The proposed 21+1 feature vector is a candidate, not a promoted model.

## 10. Evaluation and trading-risk formulas



```math
\begin{aligned}
\mathrm{Accuracy}&=\frac1N\sum_i\mathbf1\{y_i=\widehat y_i\}\\
\mathrm{MAE}&=\frac1N\sum_i|Y_i-\widehat Y_i|\\
\mathrm{RMSE}&=\sqrt{\frac1N\sum_i(Y_i-\widehat Y_i)^2}\\
\mathrm{Brier}&=\frac1N\sum_i(p_i-y_i)^2\\
\mathrm{LogLoss}&=-\frac1N\sum_i[y_i\ln p_i+(1-y_i)\ln(1-p_i)]\\
\mathrm{AUC}&=P(s^+>s^-)+\tfrac12P(s^+=s^-)\\
q_{t+1}&=\varepsilon_{t+1}^2\\
\mathrm{QLIKE}&=\frac1N\sum_t[\ln\widehat v_{t+1\mid t}+q_{t+1}/\widehat v_{t+1\mid t}]\\
\mathrm{MSE}_{v}&=\frac1N\sum_t(q_{t+1}-\widehat v_{t+1\mid t})^2
\end{aligned}
```

AUC compares randomly drawn positive and negative example scores, including half credit for ties. QLIKE here may be negative; use identical variance units and origins. Squared innovation is a noisy realized-variance proxy. Accuracy is not trading win rate.

Candidate indicator formulas (future experiments, not validated rules):



```math
\begin{aligned}
\alpha_{\mathrm{EMA}}&=2/(n+1)\\
\mathrm{EMA}_t&=\alpha_{\mathrm{EMA}} C_t+(1-\alpha_{\mathrm{EMA}})\mathrm{EMA}_{t-1}\\
\mathrm{TR}_t&=\max(H_t-L_t,|H_t-C_{t-1}|,|L_t-C_{t-1}|)\\
\mathrm{ATR}_t&=((n-1)\mathrm{ATR}_{t-1}+\mathrm{TR}_t)/n\\
U_t&=\max(C_t-C_{t-1},0),\quad D_t=\max(C_{t-1}-C_t,0)\\
RS_t&=\overline U_t/\overline D_t,\quad \mathrm{RSI}_t=100-\frac{100}{1+RS_t}
\end{aligned}
```

ATR uses Wilder smoothing after an initial n-bar average; RSI uses Wilder-smoothed gains and losses. Define zero-loss/flat handling explicitly before implementation.



```math
\begin{aligned}
d&=|P_{\mathrm{entry}}-P_{\mathrm{stop}}|>0\\
R_{\mathrm{net,long}}&=(P_{\mathrm{exit}}-P_{\mathrm{entry}})/d-c_R\\
E[R_{\mathrm{net}}]&=pW+(1-p)L-c\\
\mathrm{MDD}_{\mathrm{signed}}&=\min_t\left(\frac{E_t}{\max_{u\le t}E_u}-1\right)\\
\mathrm{Coverage}&=N_{\mathrm{accepted}}/N_{\mathrm{eligible}}\\
\mathrm{SelectiveAccuracy}&=N_{\mathrm{correct,accepted}}/N_{\mathrm{accepted}}
\end{aligned}
```

W and L are mean gross winning and losing outcomes (L nonpositive), p is trade win fraction, c is average cost in the same units; do not subtract costs twice. E_t is positive equity. Report denominators and undefined zero-denominator metrics explicitly. These are future research evaluation formulas; no live orders or trading returns established.

## 11. Documentation rule for every subsequent chapter

- Place every substantive mathematical formula taught in chat into the corresponding GitHub learning chapter, using fenced math blocks for displays and single-dollar inline math.
- Define variables, indices, units, assumptions and information availability; show derivations and illustrative calculations when relevant.
- Add new general formulas to this reference or link the exact chapter section if it already contains them; keep experimental numeric substitutions with their original experiment (subject to data rights).
- Explain the Python implementation and tests; distinguish actual run evidence from illustrative examples and planned work.
- Do not treat undocumented formulas or synthetic results as a validated market strategy.

This reference covers the main general equations through Chapter XXIX and the Chapter XXX hybrid preview. It is not a verbatim transcript of all previous chats. Future formulas and corrections should be added alongside each implementation.

## 12. Chapter XXX — Implemented causal volatility features and controls

See [Chapter XXX](30_hybrid_garch_xgboost.md) for the complete derivation, worked example and implementation. The controls fix conditional mean at zero, so returns are innovations. Fit using a rolling historical prefix of W=1440 returns and refit every 2880. No fitted parameters from future observations enter earlier rows.


```math
\begin{aligned}
s_t^2&=\frac1W\sum_{i=t-W+1}^{t}r_i^2\\
\widehat\sigma_{t+1\mid t}^{\mathrm{rolling}}&=\sqrt{\frac1{288}\sum_{i=0}^{287}r_{t-i}^2}\\
\widehat\sigma_{t+1\mid t}^{\mathrm{GARCH}}&=\sqrt{\widehat\omega+\widehat\alpha r_t^2+\widehat\beta\widehat\sigma_t^2}\\
r_t&=\sqrt{v_t}z_t,\quad z_t\overset{\mathrm{iid}}{\sim}N(0,1)\\
v_{t+1}&=10^{-7}+0.12r_t^2+0.85v_t\\
v_\infty&=10^{-7}/(1-0.12-0.85)
\end{aligned}
```

The last three lines define a synthetic positive control with predictable variance, not predictable sign. The oracle variance is scoring-only. The independent synthetic control remains the negative control. For complete aligned blocks with successful refits, common eligible count is N minus 1440 warmup indices and 13 final reserved origins. See Chapter XXX for limitations and the matched comparison.

### Chapter XXX result interpretation — variance versus direction


```math
\begin{aligned}
E[r_{t+1}\mid\mathcal F_t]&=0\\
P(r_{t+1}>0\mid\mathcal F_t)&=\tfrac12\\
E[r_{t+1}^2\mid\mathcal F_t]&=v_{t+1}\\
\mathrm{MSE\ reduction}&=100\left(1-\frac{\mathrm{MSE}_{\mathrm{model}}}{\mathrm{MSE}_{\mathrm{benchmark}}}\right)\%
\end{aligned}
```

The first three equalities apply to the symmetric zero-mean GARCH simulator with positive conditional variance, not generally to markets. The MSE reduction formula requires a positive benchmark MSE and matched units/origins. See [Chapter XXX results](30_hybrid_garch_xgboost.md#8-user-reported-hybrid-results--27-november-2026) for numeric examples.

## 13. Chapter XXXI — K-Means, GMM and synthetic directional regimes

[Complete mathematical lesson and derivations](31_market_regimes.md).

### Regime inputs and standardization

Use four causal features already present in the 21-column MTF matrix: 24-bar mean and sample standard deviation of 5m returns, and mean and sample standard deviation of the three latest completed 1h returns. Only these four are standardized for clustering. StandardScaler is fit once on training observations, then frozen for validation; a constant feature uses scale one.


```math
\begin{aligned}
\mu_j&=\frac1{N_{\mathrm{train}}}\sum_{i\in\mathrm{train}}x_{ij}\\
s_j&=\sqrt{\frac1{N_{\mathrm{train}}}\sum_{i\in\mathrm{train}}(x_{ij}-\mu_j)^2}\\
z_{ij}&=(x_{ij}-\mu_j)/s_j
\end{aligned}
```

This is batch training of an unsupervised transformation for future predictions, not an online estimate available at every earlier training origin. No validation data or future-return labels are used to fit clustering.

### K-Means: hard assignment

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

### Gaussian Mixture Model: soft assignment

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

### Synthetic directional positive control

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

### Matched regime ablation



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

### Chapter XXXI results — incremental model comparison


```math
\begin{aligned}
\Delta\mathrm{Accuracy}_{\mathrm{pp}}&=100(A_{\mathrm{augmented}}-A_{\mathrm{base}})\\
\mathrm{RMSE\ reduction}_{\%}&=100\left(1-\frac{\mathrm{RMSE}_{\mathrm{augmented}}}{\mathrm{RMSE}_{\mathrm{base}}}\right)\\
\Delta\mathrm{ARI}&=\mathrm{ARI}_{\mathrm{GMM}}-\mathrm{ARI}_{\mathrm{KMeans}}
\end{aligned}
```

Accuracy inputs are fractions; the result is in percentage points. RMSE reduction is relative to a positive matched baseline error. ARI difference is on the chance-adjusted partition agreement scale. See [Chapter XXXI results](31_market_regimes.md#8-user-reported-local-results--27-november-2026).
