"""Causal zero-mean GARCH features and matched synthetic XGBoost ablations."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from examples.day07_garch import FLOOR, fit_garch, variance_path, losses
from examples.day10_xgboost import HORIZONS, PARAMS, load, classify, regress
from examples.day11_multitimeframe import FEATURES, build_mtf
from examples.day09_history import VALIDATION

FIT_WINDOW = 1440  # five days of 5m returns
REFIT_EVERY = 2880  # ten days
ROLL_WINDOW = 288  # one day
COLUMNS = FEATURES + ("rolling_rms_288", "garch_sigma", "constant_rms")
VARIANTS = {
    "5m_only": tuple(range(12)),
    "5m_rolling": tuple(range(12)) + (21,),
    "5m_garch": tuple(range(12)) + (22,),
    "mtf_only": tuple(range(21)),
    "mtf_rolling": tuple(range(21)) + (21,),
    "mtf_garch": tuple(range(21)) + (22,),
}
TRUE_PARAMS = {"omega": 1e-7, "alpha": .12, "beta": .85}
SEED = 20261127


def forecast_paths(r, fit_window=FIT_WINDOW, refit_every=REFIT_EVERY,
                   roll_window=ROLL_WINDOW, fit=fit_garch):
    """Row i forecasts r[i] using exactly r[:i]; last row forecasts beyond input.

    Returns columns rolling RMS, GARCH sigma, refit-window RMS. No centered
    mean estimation: the two synthetic controls have zero conditional mean.
    """
    r = np.asarray(r, dtype=float)
    if (r.ndim != 1 or not np.isfinite(r).all()
            or not 2 <= roll_window <= fit_window or refit_every < 1):
        raise ValueError("Invalid variance forecast inputs")
    out = np.full((len(r)+1, 3), np.nan)
    events = []
    active = None
    for i in range(fit_window, len(r)+1):
        if (i-fit_window) % refit_every == 0:
            active = None
            event = {"observed_returns": i, "history_start": i-fit_window}
            hist = r[i-fit_window:i]
            scale2 = float(np.mean(hist**2))
            try:
                if not np.isfinite(scale2) or scale2 <= 0:
                    raise ValueError("No positive variance in historical window")
                omega, alpha, beta = map(float, fit(hist/np.sqrt(scale2)))
                if not (np.isfinite([omega,alpha,beta]).all() and omega>0
                        and alpha>=0 and beta>=0 and alpha+beta<1):
                    raise ValueError("Invalid fitted GARCH parameters")
                v = float(variance_path(np.r_[hist/np.sqrt(scale2),0.],
                                       omega,alpha,beta)[-1]*scale2)
                w = omega*scale2
                if not np.isfinite(v) or v<=0:
                    raise ValueError("Invalid variance state")
                active = (w,alpha,beta,scale2,v)
                event.update(status="ok",omega=w,alpha=alpha,beta=beta)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                event.update(status="failed",error=str(exc))
            events.append(event)
        elif active is not None:
            w,alpha,beta,scale2,v = active
            v = max(w+alpha*r[i-1]**2+beta*v,FLOOR*scale2)
            active = (w,alpha,beta,scale2,v)
        if active is not None:
            _,_,_,scale2,v = active
            rolling = max(float(np.mean(r[i-roll_window:i]**2)),FLOOR*scale2)
            out[i] = np.sqrt([rolling,v,scale2])
    return out,events


def build_hybrid(segments, stop, delay, events, fit=fit_garch):
    """All variants share full-MTF and GARCH eligibility; no bridging gaps."""
    aa,xx=[],[]
    yy={h:[] for h in HORIZONS}
    for segment in segments:
        if len(segment)<FIT_WINDOW+14 or segment[FIT_WINDOW][0]+300+3600+delay>=stop:
            continue
        a,x,y=build_mtf([segment],stop,delay)
        r=np.diff(np.log([c for _,c in segment]))
        path,notes=forecast_paths(r,fit=fit)
        for note in notes:
            note["fit_anchor_epoch"]=int(segment[0][0]+300+300*note["observed_returns"])
            events.append(note)
        i=(a-(segment[0][0]+300))//300
        full=np.column_stack((x,path[i]))
        keep=np.isfinite(full).all(axis=1)
        if keep.any():
            aa.append(a[keep]);xx.append(full[keep])
            for h in HORIZONS: yy[h].append(y[h][keep])
    if not aa:
        raise ValueError("No matched origins after GARCH warmup/failures")
    return np.concatenate(aa),np.vstack(xx),{h:np.concatenate(v) for h,v in yy.items()}


def positive_control():
    """Independent seed, 30d train + 20d validation, known GARCH innovations.

    Oracle next variance is returned solely for scoring, never for fitting or
    feature generation. No June/July holdout is generated or read.
    """
    rng=np.random.default_rng(SEED)
    w,a,b=(TRUE_PARAMS[k] for k in ("omega","alpha","beta"))
    v=w/(1-a-b)
    for _ in range(1000):
        e=np.sqrt(v)*rng.normal()
        v=w+a*e**2+b*v
    start=VALIDATION-30*86400
    segments={"train":[],"validation":[]}; oracle={}
    log_c=np.log(100.)
    for t in range(start,VALIDATION+20*86400,300):
        e=np.sqrt(v)*rng.normal()
        log_c+=e
        v=w+a*e**2+b*v
        segments["train" if t<VALIDATION else "validation"].append((t,float(np.exp(log_c))))
        oracle[t+300]=float(v)
    return segments,oracle


def experiment(blocks, oracle=None):
    at,xt,yt=blocks["train"];av,xv,yv=blocks["validation"]
    if len(at)<100 or len(av)<20 or at[-1]+3600>=av[0]:
        raise ValueError("Insufficient or overlapping matched blocks")
    report={"horizons":{},"variance_5m":{}}
    actual=yv["5m"]
    for name,j in (("rolling_rms_288",21),("garch11",22),("refit_constant",23)):
        report["variance_5m"][name]=losses(actual,xv[:,j]**2)
    if oracle is not None:
        report["variance_5m"]["simulator_oracle"]=losses(actual,np.array([oracle[int(a)] for a in av]))
    rows=[]
    for h in HORIZONS:
        y,actual=yt[h],yv[h]
        up,observed=(y>0).astype(int),(actual>0).astype(int)
        if len(np.unique(up))!=2:
            raise ValueError("Both training classes required")
        probs={"prevalence":np.full(len(av),up.mean())}
        preds={"zero_return":np.zeros(len(av)),"training_mean":np.full(len(av),y.mean())}
        for name,cols in VARIANTS.items():
            cl=XGBClassifier(objective="binary:logistic",eval_metric="logloss",**PARAMS)
            rg=XGBRegressor(objective="reg:squarederror",**PARAMS)
            cl.fit(xt[:,cols],up);rg.fit(xt[:,cols],y*100)
            probs[name]=cl.predict_proba(xv[:,cols])[:,1].astype(float)
            preds[name]=rg.predict(xv[:,cols]).astype(float)/100
        report["horizons"][h]={"classification":{k:classify(observed,p) for k,p in probs.items()},
                               "regression":{k:regress(actual,p) for k,p in preds.items()}}
        for j,a in enumerate(av):
            rows.append({"horizon":h,"anchor_epoch":int(a),"actual_log_return":float(actual[j]),
                         "actual_up":int(observed[j]),"rolling_sigma":float(xv[j,21]),
                         "garch_sigma":float(xv[j,22]),"refit_constant_sigma":float(xv[j,23]),
                         **{k+"_p_up":float(p[j]) for k,p in probs.items()},
                         **{k+"_return":float(p[j]) for k,p in preds.items()}})
    report["counts"]={k:len(v[0]) for k,v in blocks.items()}
    return report,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--replay-delay-seconds",type=int,required=True)
    args=p.parse_args()
    if args.output.exists():
        p.error("Output exists; read the report or choose a new directory")
    delay=args.replay_delay_seconds
    if not 0<=delay<300: p.error("Invalid assumed delay")
    events=[]
    m,negative=load(args.dataset,delay,builder=lambda s,b,d:build_hybrid(s,b,d,events))
    neg,nrows=experiment(negative)
    segments,oracle=positive_control(); positive_events=[]
    positive={k:build_hybrid([s],s[-1][0]+300,delay,positive_events) for k,s in segments.items()}
    pos,prows=experiment(positive,oracle)
    for report,notes in ((neg,events),(pos,positive_events)):
        report["garch_refits"]={"attempted":len(notes),"failed":sum(e["status"]=="failed" for e in notes),
                               "events":notes}
    report={"scope":"synthetic controls only; no market-performance inference",
            "holdout_exported":False,"historical_publication_times_verified":False,
            "assumed_delay_seconds":delay,"garch_mean":"zero",
            "fit_window_returns":FIT_WINDOW,"refit_every_returns":REFIT_EVERY,"rolling_window_returns":ROLL_WINDOW,
            "variants":{k:[COLUMNS[i] for i in v] for k,v in VARIANTS.items()},"xgboost_params":PARAMS,
            "source_block_hashes":{k:v["csv_sha256"] for k,v in m["blocks"].items()},
            "positive_control":{"seed":SEED,"true_parameters":TRUE_PARAMS,"train_days":30,"validation_days":20},
            "negative":neg,"positive":pos}
    payload=json.dumps(report,indent=2,allow_nan=False)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,rows in (("negative_predictions.csv",nrows),("positive_predictions.csv",prows)):
        with (args.output/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/"report.json").write_text(payload,encoding="utf-8")
    print(payload)


if __name__=="__main__":
    main()
