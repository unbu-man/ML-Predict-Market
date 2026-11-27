"""Synthetic-only, gap-aware matched M2/M3 experiment. No holdout or trades."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

LAGS = (0, 1, 2, 3, 6, 12)
WINDOWS = (6, 12, 24)
INPUTS = tuple([f"return_lag_{k}" for k in LAGS] +
               [f"{s}_{w}" for w in WINDOWS for s in ("mean_return", "std_return")])
HORIZONS = {"5m": 1, "1h": 12}
PARAMS = dict(n_estimators=100, max_depth=2, learning_rate=.05, min_child_weight=20,
              reg_lambda=5., reg_alpha=0., subsample=1., colsample_bytree=1.,
              tree_method="hist", random_state=42, n_jobs=2)
HEADER = ["open_time_utc", "close_time_utc", "available_at_utc", "open", "high", "low", "close", "volume"]


def epoch(s):
    t = datetime.fromisoformat(s)
    if t.utcoffset() != timedelta(0) or t.microsecond:
        raise ValueError("Expected whole-second UTC")
    return int(t.timestamp())


def read_block(folder, name, meta):
    a, b = epoch(meta["start_utc"]), epoch(meta["end_exclusive_utc"])
    if a >= b or a % 300 or b % 300 or meta["expected"] != (b-a)//300:
        raise ValueError("Invalid block bounds")
    path = folder/(name+".csv")
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != meta["csv_sha256"]:
            raise ValueError("Block checksum mismatch")
    segments, count = [], 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER:
            raise ValueError("Unexpected candle columns")
        for row in reader:
            t = epoch(row[HEADER[0]])
            op, hi, lo, cl, vol = (float(row[k]) for k in HEADER[3:])
            if (None in row or not a <= t < b or t % 300 or epoch(row[HEADER[1]]) != t+300
                    or row[HEADER[2]] != "" or not np.isfinite([op,hi,lo,cl,vol]).all()
                    or min(op,hi,lo,cl)<=0 or hi<max(op,cl) or lo>min(op,cl) or vol<0
                    or (segments and t<=segments[-1][-1][0])):
                raise ValueError("Invalid candle, timestamp or ordering")
            if not segments or t != segments[-1][-1][0]+300:
                segments.append([])
            segments[-1].append((t,cl)); count+=1
    if (not segments or count!=meta["received"] or meta["missing"]!=meta["expected"]-count
            or max(map(len,segments))!=meta["longest_contiguous"]):
        raise ValueError("Block manifest mismatch")
    return segments


def build(segments, stop, delay):
    """Common origins for both horizons; reset at every gap and block start."""
    if type(delay) is not int or not 0<=delay<300:
        raise ValueError("Invalid assumed delay")
    xs, anchors = [], []
    ys = {h:[] for h in HORIZONS}
    for segment in segments:
        if len(segment)<37:
            continue
        t = np.asarray([s[0] for s in segment], dtype=np.int64)
        z = np.log([s[1] for s in segment]); r = np.diff(z)
        if np.any(np.diff(t)!=300) or not np.isfinite(z).all():
            raise ValueError("Invalid contiguous segment")
        i = np.arange(24,len(t)-12)
        # Strict label availability cutoff at the block boundary for both horizons.
        i = i[t[i]+300+12*300+delay<stop]
        if not len(i):
            continue
        columns = [r[i-1-k] for k in LAGS]
        for w in WINDOWS:
            window = np.lib.stride_tricks.sliding_window_view(r,w)[i-w]
            columns.extend([window.mean(axis=1),window.std(axis=1,ddof=1)])
        xs.append(np.column_stack(columns)); anchors.append(t[i]+300)
        for h,k in HORIZONS.items():
            ys[h].append(z[i+k]-z[i])
    if not xs:
        raise ValueError("No eligible complete windows")
    return np.concatenate(anchors),np.vstack(xs),{h:np.concatenate(v) for h,v in ys.items()}


def load(folder, delay):
    m = json.loads((folder/"manifest.json").read_text())
    if (m["provider"]!="synthetic" or m["scope"]!="synthetic pipeline test only"
            or m["symbol"]!="BTC-USD" or m["interval_seconds"]!=300 or m["holdout_exported"] is not False
            or set(m["blocks"])!={"train","validation"}):
        raise ValueError("Synthetic development export only")
    bounds = ("2025-08-01T00:00:00+00:00", "2026-04-01T00:00:00+00:00",
              "2026-06-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00")
    a,b = m["blocks"]["train"],m["blocks"]["validation"]
    if (a["start_utc"],a["end_exclusive_utc"],b["end_exclusive_utc"],m["holdout_end_exclusive_utc"])!=bounds or b["start_utc"]!=bounds[1] or m["holdout_start_utc"]!=bounds[2]:
        raise ValueError("Frozen dates changed")
    return m, {name:build(read_block(folder,name,meta),epoch(meta["end_exclusive_utc"]),delay)
               for name,meta in m["blocks"].items()}


def classify(y,p):
    return dict(accuracy=float(accuracy_score(y,p>.5)), brier=float(brier_score_loss(y,p)),
                log_loss=float(log_loss(y,np.clip(p,1e-15,1-1e-15),labels=[0,1])),
                roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None)


def regress(y,p):
    return dict(mae=float(np.mean(np.abs(y-p))), rmse=float(np.sqrt(np.mean((y-p)**2))),
                direction_accuracy=float(np.mean((y>0)==(p>0))))


def experiment(blocks):
    at,xt,yt = blocks["train"]
    av,xv,yv = blocks["validation"]
    if len(at)<100 or len(av)<20 or at[-1]+3600>=av[0]:
        raise ValueError("Insufficient or overlapping blocks")
    report, rows = {}, []
    for h in HORIZONS:
        y, actual = yt[h],yv[h]; up=(y>0).astype(int); observed=(actual>0).astype(int)
        if len(np.unique(up))!=2:
            raise ValueError("Both training classes required")
        classifiers = {"logistic":make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000)),
                       "xgb_classifier":XGBClassifier(objective="binary:logistic",eval_metric="logloss",**PARAMS)}
        regressors = {"ridge":make_pipeline(StandardScaler(),Ridge(alpha=1)),
                      "xgb_regressor":XGBRegressor(objective="reg:squarederror",**PARAMS)}
        probs = {"prevalence":np.full(len(av),up.mean())}
        preds = {"zero_return":np.zeros(len(av)),"training_mean":np.full(len(av),y.mean())}
        for name,model in classifiers.items():
            model.fit(xt,up); probs[name]=model.predict_proba(xv)[:,1].astype(float)
        for name,model in regressors.items():
            model.fit(xt,y*100); preds[name]=model.predict(xv).astype(float)/100
        report[h] = {"training_up_rate":float(up.mean()),"classification":{k:classify(observed,p) for k,p in probs.items()},
                     "regression":{k:regress(actual,p) for k,p in preds.items()}}
        for j,a in enumerate(av):
            rows.append({"horizon":h,"anchor_epoch":int(a),"actual_log_return":float(actual[j]),
                         "actual_up":int(observed[j]),**{k+"_p_up":float(p[j]) for k,p in probs.items()},
                         **{k+"_return":float(p[j]) for k,p in preds.items()}})
    return report,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--replay-delay-seconds",type=int,required=True)
    args=p.parse_args()
    if args.output.exists():
        p.error("Output exists; choose a new directory or read the earlier report")
    manifest,blocks=load(args.dataset,args.replay_delay_seconds)
    metrics,rows=experiment(blocks)
    report={"scope":"synthetic software test only; not market performance","holdout_exported":False,
            "assumed_delay_seconds":args.replay_delay_seconds,"inputs":list(INPUTS),"xgboost_params":PARAMS,
            "source_block_hashes":{k:v["csv_sha256"] for k,v in manifest["blocks"].items()},
            "counts":{k:len(v[0]) for k,v in blocks.items()},"horizons":metrics}
    payload=json.dumps(report,indent=2,allow_nan=False)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/"report.json").write_text(payload,encoding="utf-8")
    print(payload)


if __name__=="__main__":
    main()
