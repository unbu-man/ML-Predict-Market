# Chapter XXV — GARCH and conditional volatility forecasting

26 November 2026. One-step (5m) variance experiment; development replay only. Six synthetic fixture tests passed in an isolated assistant container. No BTC GARCH fit or skill measurement yet.

## 1. Why GARCH?

Chapter XXIV's user-reported AR(1) training residuals had Ljung-Box p = 0.61644 and ARCH-LM p = 7.38347e-9. Weak return predictability and significant squared-residual dependence motivate a conditional variance experiment. These are not guarantees of future skill.

## 2. Mean and variance equations



```math
\begin{aligned}
r_t&=\mu+\phi_1(r_{t-1}-\mu)+\varepsilon_t\\
\varepsilon_t&=\sigma_t z_t\\
E(z_t\mid\mathcal F_{t-1})&=0,\qquad E(z_t^2\mid\mathcal F_{t-1})=1\\
\sigma_t^2&=\omega+\alpha_1\varepsilon_{t-1}^2+\beta_1\sigma_{t-1}^2
\end{aligned}
```

Here $\mu$ is the long-run mean return, $\phi_1$ the AR coefficient, $\varepsilon_t$ the innovation, and $z_t$ the standardized innovation. The variance constant is $\omega$; $\alpha_1$ weights the previous squared innovation, and $\beta_1$ the previous conditional variance. Volatility is the standard deviation $\sigma_t$, not the variance.

### Parameter constraints and unconditional variance



```math
\omega>0,\quad\alpha_1\geq0,\quad\beta_1\geq0,\quad\alpha_1+\beta_1<1
```

Assuming covariance stationarity and finite second moments, take expectations of the GARCH equation:


```math
\begin{aligned}
v&=\omega+\alpha_1v+\beta_1v\\
v(1-\alpha_1-\beta_1)&=\omega\\
v&=\frac{\omega}{1-\alpha_1-\beta_1}
\end{aligned}
```

Persistence close to one indicates slowly decaying variance shocks; it does not indicate directional predictability.

### Worked example (synthetic normalized units)



```math
\begin{aligned}
\omega&=0.1,\quad\alpha_1=0.2,\quad\beta_1=0.7\\
\varepsilon_t&=2,\quad\sigma_t^2=1\\
\sigma_{t+1}^2&=0.1+0.2(2^2)+0.7(1)\\
&=1.6\\
\sigma_{t+1}&=\sqrt{1.6}\approx1.265
\end{aligned}
```

These are not fitted BTC parameters.

## 3. Estimation

Fit the AR(1) mean by maximum likelihood on the first 60% of returns, as in Chapter XXIV. Freeze its parameters and compute innovations from the observed return minus its prior conditional mean. Discard 12 initial residuals. Normalize innovations by the root mean square of training innovations only. Estimate a GARCH(1,1) variance model in a second stage by minimizing the Gaussian quasi-negative log likelihood (ignoring the constant):


```math
\begin{aligned}
s_0^2&=\frac{1}{N_{\mathrm{train}}}\sum_{t\in\mathrm{train}}\varepsilon_t^2\\
u_t&=\frac{\varepsilon_t}{s_0}\\
v_t&=\frac{\sigma_t^2}{s_0^2}\\
\mathcal L&=\frac{1}{2N_{\mathrm{train}}}\sum_{t\in\mathrm{train}}\left(\ln v_t+\frac{u_t^2}{v_t}\right)
\end{aligned}
```

This is two-stage QML, not jointly estimated AR-GARCH. SciPy L-BFGS-B uses three starting points; normalized omega is bounded [1e-6,10], persistence is capped at 0.999, and the ARCH share is bounded [0,1]. Report failure if no optimizer converges. A numerical floor of 1e-8 times training mean squared innovation protects positive forecasts. Initial normalized variance is one, estimated using training data; training residual diagnostics are in-sample, not historical out-of-sample features. Convert forecast variances and omega back to squared log-return units. No Gaussian tail-probability or direction-probability claim.

## 4. Compare three models at identical origins

V0.1 constant variance (training innovation mean square):


```math
\widehat v_{t+1\mid t}^{(0)}=\frac{1}{N_{\mathrm{train}}}\sum_{i\in\mathrm{train}}\varepsilon_i^2
```

V0.2 trailing 24-innovation sample variance (two hours):


```math
\begin{aligned}
\bar\varepsilon_{t,24}&=\frac1{24}\sum_{i=0}^{23}\varepsilon_{t-i}\\
\widehat v_{t+1\mid t}^{(\mathrm{roll})}&=\frac1{23}\sum_{i=0}^{23}(\varepsilon_{t-i}-\bar\varepsilon_{t,24})^2
\end{aligned}
```

V1 GARCH(1,1):


```math
\widehat v_{t+1\mid t}^{(G)}=\widehat\omega+\widehat\alpha_1\varepsilon_t^2+\widehat\beta_1\widehat v_{t\mid t-1}^{(G)}
```

Freeze all fitted parameters. Update rolling and GARCH state only after each innovation has become observable under the explicit constant replay-delay assumption. The current unknown innovation cannot enter its own variance forecast. Score validation and development replay separately, using the same mean and origins.

## 5. Loss functions and diagnostics

Use the next squared innovation as a noisy proxy for realized conditional variance:


```math
q_{t+1}=\varepsilon_{t+1}^2
```



```math
\begin{aligned}
\mathrm{QLIKE}&=\frac1N\sum_t\left[\ln(\widehat v_{t+1\mid t})+\frac{q_{t+1}}{\widehat v_{t+1\mid t}}\right]\\
\mathrm{MSE}_{v}&=\frac1N\sum_t(q_{t+1}-\widehat v_{t+1\mid t})^2
\end{aligned}
```

Lower losses are better; QLIKE in this form can be negative and comparisons must use the same units. Also report variance MAE and mean predicted vs mean realized proxy. Standardize innovations by predicted standard deviation and inspect Ljung-Box on standardized and squared standardized training residuals and ARCH-LM at lag 12. These are exploratory, nominal p-values; they do not establish model adequacy or forecast superiority. Compare both blocks rather than select a winner based on one score.

## 6. Run on your Mac

```bash
git pull origin main
python3 -m pip install -r requirements-volatility.txt
python3 -m unittest tests.test_day07_garch -v
python3 -m examples.day07_garch data/coinbase/BTC-USD_300_1794528000_1795132800 --output data/volatility/btc_5m_smoke --replay-delay-seconds 2
```

The output directory must be new. With the unchanged 2,016-candle dataset expect 1,197 training innovations (1,209 returns less 12 initialization residuals), 403 validation forecasts and 403 development replay forecasts per model. Output report.json and variance_forecasts.csv under gitignored data/. The two-second replay delay is not measured.

## 7. Acceptance gate

Six isolated synthetic tests passed: variance recursion, future-innovation invariance with frozen parameters, parameter constraints, QLIKE arithmetic, synthetic end-to-end experiment, and gap rejection. Real BTC scores, CLI integration on the user's Mac, and longer history remain pending. No 1h variance forecast in this first experiment. Do not simply multiply standard deviation by twelve. Compare performance, inspect boundary parameters and diagnostics; GARCH need not win. Before adding GARCH forecasts as supervised features, generate training-fold forecasts causally with historical fits; never use this full training-sample fitted path as if it were available in real time. Collect longer authorized history and reserve a fresh untouched chronological holdout. No trading win rate or live execution.

[Chapter XXIV](24_statistical_forecasting.md) | [Chapter XIX](19_hybrid_selective_forecast_strategy.md)


## 8. User-reported BTC/USD GARCH results — 26 November 2026

Evidence: local report JSON pasted by user; 2,016 contiguous candles, source SHA256 589ca0a6fb76a9923673c3fe0cfd66fa231848ca913000e497856c6183d91a69, matching earlier reports. 1,197 training innovations and 403 forecasts in each later block. Raw CSV, individual variance forecasts and local test output have not been independently inspected. This entire week was previously explored.

### Estimated equation

```math
\begin{aligned}
\widehat{\sigma}_t^2={}&2.1458335504\times10^{-7}\\
&+0.1154662146\varepsilon_{t-1}^2\\
&+0.8311436839\widehat{\sigma}_{t-1}^2\\
\widehat\alpha+\widehat\beta&=0.9466098986<1\\
\widehat v_\infty&=4.0191599047\times10^{-6}
\end{aligned}
```

The fitted long-run innovation standard deviation is about 0.2005% per five-minute interval, conditional on the model assumptions, not a price range or confidence guarantee. The initial variance is 3.7579181499e-6. The AR mean is -7.8213105637e-5 and AR coefficient 0.053548778. These are estimates for one development sample.

### Variance forecast comparison

QLIKE in the implementation's unscaled log-return units can be negative. All scores are lower-is-better. Variance MAE is in squared log-return units; variance MSE is in fourth-power log-return units.

| Period | Model | QLIKE | Variance MAE | Variance MSE | Mean forecast variance |
|---|---|---:|---:|---:|---:|
| Validation | Constant | -11.8594491880 | 3.8165805607e-6 | 6.8314750701e-11 | 3.7579181499e-6 |
| Validation | Rolling24 | -12.1737352962 | 2.7215313458e-6 | 6.4820800152e-11 | 2.4098394592e-6 |
| Validation | GARCH11 | -12.2152956738 | 2.8677479007e-6 | 5.7798334533e-11 | 2.8922866244e-6 |
| Development replay | Constant | -11.5877567280 | 4.0349530424e-6 | 5.6928949129e-11 | 3.7579181499e-6 |
| Development replay | Rolling24 | -11.5160760846 | 3.7896690110e-6 | 5.8546131890e-11 | 3.3605230610e-6 |
| Development replay | GARCH11 | -11.6466564095 | 3.7296056018e-6 | 5.6394683741e-11 | 3.5667573056e-6 |

Mean squared innovation proxy: validation 2.3757417600e-6, replay 3.3967397868e-6. GARCH has the lowest QLIKE and MSE in both blocks, but rolling24 wins validation MAE. GARCH lowers MSE relative to constant by about 15.4% in validation and 0.94% in replay. QLIKE differences versus constant are -0.3558465 and -0.0588997; do not compute percentage changes on negative log losses. No uncertainty intervals or paired loss significance tests yet. The improvement in the later block is modest.

### Training diagnostics

| Diagnostic (lag 12) | p-value |
|---|---:|
| Standardized innovation Ljung-Box | 0.6218275652 |
| Squared standardized innovation Ljung-Box | 0.8666328582 |
| Standardized innovation ARCH-LM | 0.8597582976 |

Compared with Chapter XXIV's AR(1) ARCH-LM p = 7.38347e-9, standardized residuals no longer reject the no-ARCH null at nominal 5%. These are in-sample diagnostic p-values, not proof of independence, homoskedasticity, exact specification, or future skill.

### Decision and next gate

Retain GARCH as a promising V1 research candidate, not a promoted production feature. The next supervised experiment is M2: Logistic Regression (UP/NOT_UP) and Ridge Regression (forward log return), separately for 5m and 1h. Verify Chapter XXIII feature/target files and hashes first, use the manifest allowlist, drop redundant momentum columns for the linear baselines, fit scalers on training only, purge any labels whose assumed availability overlaps the next block, and compare with zero-return and training-majority benchmarks. Do not add full-sample fitted GARCH to training inputs: generate causal volatility features in a later historical-fold ablation (M4), comparing otherwise identical models on matched origins. Keep the existing week as a smoke test and obtain longer authorized data with a fresh chronological holdout before making model-selection or trading-performance claims. No one-hour variance forecast, actual publication latency, independent raw-data verification, XAU access, trading win rate or live orders established.
