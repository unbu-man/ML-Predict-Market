"""One-step conditional variance comparison on a development sample only."""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

FLOOR = 1e-8  # variance floor relative to training innovation mean square


def variance_path(e, omega, alpha, beta, initial=1.0):
    """v[t] is fixed before observing e[t]. Inputs are normalized innovations."""
    v = np.empty(len(e))
    prev = initial
    for t in range(len(e)):
        v[t] = max(prev, FLOOR)
        prev = omega + alpha * e[t]**2 + beta * v[t]
    return v


def fit_garch(e):
    def objective(x):
        omega, persistence, share = x
        alpha, beta = persistence*share, persistence*(1-share)
        v = variance_path(e, omega, alpha, beta)
        return float(.5*np.mean(np.log(v)+e**2/v))
    candidates = [minimize(objective, x, method="L-BFGS-B",
                           bounds=[(1e-6,10), (0,.999), (0,1)])
                  for x in ((.1,.9,.1),(.3,.7,.3),(.8,.2,.5))]
    good = [r for r in candidates if r.success and np.isfinite(r.fun)]
    if not good:
        raise ValueError("No converged GARCH fit")
    best = min(good, key=lambda r:r.fun)
    omega,p,share = map(float,best.x)
    return omega,p*share,p*(1-share)


def losses(e, v):
    q = e**2
    return {"n": len(e), "qlike": float(np.mean(np.log(v)+q/v)),
            "variance_mae": float(np.mean(np.abs(q-v))),
            "variance_mse": float(np.mean((q-v)**2)),
            "mean_predicted_variance": float(np.mean(v)),
            "mean_squared_innovation": float(np.mean(q))}


def experiment(segment, delay):
    if not 0 <= delay < 300 or len(segment) < 200:
        raise ValueError("Need >=200 candles and assumed delay in [0,299]")
    if any(t % 300 or not np.isfinite(c) or c<=0 or (i and t!=segment[i-1][0]+300)
           for i,(t,c) in enumerate(segment)):
        raise ValueError("Invalid or noncontiguous 5m candles")
    r = np.diff(np.log([c for _,c in segment]))
    cut,end = int(.6*len(r)),int(.8*len(r))
    fit = ARIMA(r[:cut]*100, order=(1,0,0), trend="c").fit()
    if not fit.mle_retvals.get("converged",False):
        raise ValueError("AR(1) estimation failed")
    params = dict(zip(fit.param_names, map(float,fit.params)))
    mu,phi = params["const"]/100,params["ar.L1"]
    mean = np.r_[mu,mu+phi*(r[:-1]-mu)]
    # Drop initial 12 residuals. Train-only normalization and variance initialization.
    e = (r-mean)[12:]
    cut,end = cut-12,end-12
    scale2 = float(np.mean(e[:cut]**2))
    if not np.isfinite(scale2) or scale2<=0:
        raise ValueError("No positive training innovation variance")
    u = e/np.sqrt(scale2)
    omega,alpha,beta = fit_garch(u[:cut])
    normalized = variance_path(u,omega,alpha,beta)
    paths = {"constant": np.full(len(e),scale2),
             "rolling24": np.array([np.var(e[max(0,t-24):t],ddof=1) if t>=24 else scale2
                                      for t in range(len(e))]),
             "garch11": normalized*scale2}
    paths = {k:np.maximum(v,FLOOR*scale2) for k,v in paths.items()}
    report = {"scope":"5m development replay only", "assumed_delay_seconds":delay,
              "historical_publication_times_verified":False,
              "train_innovations":cut, "ar1_mean_log_return":mu,"ar1_phi":phi,
              "garch":{"omega_log_return_squared":omega*scale2,"alpha":alpha,"beta":beta,
                       "persistence":alpha+beta,"long_run_variance":omega*scale2/(1-alpha-beta),
                       "initialization_variance":scale2},"metrics":{}}
    standardized = e[:cut]/np.sqrt(paths["garch11"][:cut])
    report["train_diagnostics"] = {"standardized_ljung_box_p_lag12":float(acorr_ljungbox(standardized,lags=[12],return_df=True)["lb_pvalue"].iloc[0]),
        "squared_standardized_ljung_box_p_lag12":float(acorr_ljungbox(standardized**2,lags=[12],return_df=True)["lb_pvalue"].iloc[0]),
        "standardized_arch_lm_p_lag12":float(het_arch(standardized,nlags=12)[1])}
    rows=[]
    for split,lo,hi in (("validation",cut,end),("development_replay",end,len(e))):
        report["metrics"][split]={k:losses(e[lo:hi],v[lo:hi]) for k,v in paths.items()}
        for t in range(lo,hi):
            anchor=segment[t+12][0]+300
            rows.append({"split":split,"anchor_close_utc":datetime.fromtimestamp(anchor,timezone.utc).isoformat(),
                         "decision_epoch_assumed":anchor+delay,"target_close_epoch":anchor+300,
                         "realized_squared_innovation":float(e[t]**2),
                         **{k:float(v[t]) for k,v in paths.items()}})
    return report,rows


def main():
    from examples.day04_market_eda import audit
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--replay-delay-seconds",type=int,required=True)
    args=p.parse_args()
    segment,source=audit(args.dataset)
    if source["interval_seconds"]!=300:
        p.error("Use 5m candles")
    report,rows=experiment(segment,args.replay_delay_seconds)
    report["source_audit"]=source
    payload=json.dumps(report,indent=2,allow_nan=False)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/"variance_forecasts.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/"report.json").write_text(payload,encoding="utf-8")
    print(payload)
    print("Report:",args.output)


if __name__=="__main__":
    main()
