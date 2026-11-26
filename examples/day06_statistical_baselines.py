"""Fixed-parameter chronological return forecasts; development replay only."""
import argparse
import csv
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

ORDERS = {"AR1": (1, 0, 0), "MA1": (0, 0, 1), "ARMA11": (1, 0, 1)}


def horizons(first, mu, phi, h):
    """Sum conditional return means, not future observed returns."""
    return sum(mu + phi**k * (first-mu) for k in range(h))


def fit_forecasts(returns, cut, order):
    # Fit in percent units to improve numerical conditioning.
    scaled = np.asarray(returns, dtype=float)*100
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter("always")
        fit = ARIMA(scaled[:cut], order=order, trend="c").fit()
        if not fit.mle_retvals.get("converged", False):
            raise ValueError("Optimizer did not converge")
        # Filtering with frozen training parameters is causal. No smoothing or refit.
        replay = ARIMA(scaled, order=order, trend="c").filter(fit.params)
        params = dict(zip(fit.param_names, map(float, fit.params)))
        residual = np.asarray(fit.resid)[12:]
        lb = acorr_ljungbox(residual, lags=[12], model_df=order[0]+order[2], return_df=True)
        lm = het_arch(residual, nlags=12, ddof=order[0]+order[2])
    report = {"parameters_percent_units": params, "aic": float(fit.aic), "bic": float(fit.bic),
              "train_ljung_box_p": float(lb["lb_pvalue"].iloc[0]), "train_arch_lm_p": float(lm[1]),
              "warnings": [str(w.message) for w in notes]}
    return np.asarray(replay.fittedvalues)/100, params.get("const", 0)/100, params.get("ar.L1", 0), report


def score(actual, predicted):
    a, p = np.asarray(actual), np.asarray(predicted)
    return {"n": len(a), "mae": float(np.mean(np.abs(a-p))),
            "rmse": float(np.sqrt(np.mean((a-p)**2))),
            "direction_accuracy": float(np.mean((a>0)==(p>0)))}


def experiment(segment, delay):
    if not 0 <= delay < 300 or len(segment) < 150:
        raise ValueError("Need >=150 contiguous 5m candles and assumed delay in [0,299]")
    if any(t % 300 or not np.isfinite(c) or c <= 0 or (i and t != segment[i-1][0]+300)
           for i,(t,c) in enumerate(segment)):
        raise ValueError("Invalid or noncontiguous candles")
    r = np.diff(np.log([c for _,c in segment]))
    n = len(r)
    cut, end = int(.6*n), int(.8*n)
    report = {"scope": "development replay only; previously inspected history", "return_count": n,
              "train_returns": cut, "validation_returns": end-cut, "replay_returns": n-end,
              "assumed_delay_seconds": delay, "historical_publication_times_verified": False,
              "models": {}, "metrics": {}, "failed_models": {}}
    models = {"zero_return": (np.zeros(n),0.,0.)}
    for name, order in ORDERS.items():
        try:
            first, mu, phi, info = fit_forecasts(r,cut,order)
            if not np.all(np.isfinite(first)):
                raise ValueError("Nonfinite forecasts")
            models[name] = (first,mu,phi)
            report["models"][name] = info
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            report["failed_models"][name] = str(exc)
    rows = []
    for h in (1,12):
        # Majority labels are learned only on outcomes entirely inside training.
        majority = int(np.mean([np.sum(r[j:j+h])>0 for j in range(cut-h+1)]) > .5)
        for split, lo, hi in (("validation",cut,end),("development_replay",end,n)):
            indices = range(lo, hi-h+1)
            actual = [float(np.sum(r[j:j+h])) for j in indices]
            key = f"{split}_{h*5}m"
            report["metrics"][key] = {"train_majority_direction_accuracy": float(np.mean((np.array(actual)>0)==majority)), "models": {}}
            for name,(first,mu,phi) in models.items():
                predicted = [float(horizons(first[j],mu,phi,h)) for j in indices]
                report["metrics"][key]["models"][name] = score(actual,predicted)
                for j,a,p in zip(indices,actual,predicted):
                    # r[j] is unknown at the close of candle j; outcomes start after that close.
                    anchor = segment[j][0]+300
                    rows.append({"split":split,"model":name,"horizon_minutes":5*h,
                                 "anchor_close_utc":datetime.fromtimestamp(anchor,timezone.utc).isoformat(),
                                 "decision_epoch_assumed":anchor+delay,"target_close_epoch":anchor+300*h,
                                 "actual_log_return":a,"forecast_log_return":p})
    return report, rows


def main():
    from examples.day04_market_eda import audit
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replay-delay-seconds", type=int, required=True)
    args = p.parse_args()
    segment, source = audit(args.dataset)
    if source["interval_seconds"] != 300:
        p.error("Use five-minute data")
    report, rows = experiment(segment, args.replay_delay_seconds)
    report["source_audit"] = source
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/"predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    print("Report:", args.output)


if __name__ == "__main__":
    main()
