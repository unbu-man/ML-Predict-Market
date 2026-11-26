"""M2 Logistic/Ridge development replay; no live orders."""
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

KEY = "anchor_close_utc"
FEATURES = tuple([f"return_lag_{k}" for k in (0,1,2,3,6,12)] +
                 [f"{kind}_{w}" for w in (6,12,24) for kind in ("mean_return","std_return","momentum")])
INPUTS = tuple(k for k in FEATURES if not k.startswith("momentum_"))
HORIZONS = {"5m": 1, "1h": 12}


def epoch(s):
    t = datetime.fromisoformat(s)
    if t.utcoffset() != timedelta(0) or t.microsecond:
        raise ValueError("Expected whole-second UTC timestamp")
    return int(t.timestamp())


def read_verified(folder, name, manifest):
    path = folder/name
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["file_sha256"][name]:
        raise ValueError(f"Checksum mismatch: {name}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("Missing or duplicate CSV columns")
        return set(columns), list(reader)


def load(folder):
    m = json.loads((folder/"manifest.json").read_text())
    delay = m["replay_delay_seconds_assumed"]
    n = m["source_candles"]
    if (m["feature_columns"] != list(FEATURES) or m["join_key"] != KEY or
            m["warmup_bars"] != 24 or m["future_bars_reserved"] != 12 or
            type(delay) is not int or not 0 <= delay < 300 or
            type(n) is not int or n < 250):
        raise ValueError("Unsupported feature manifest")
    fc, f = read_verified(folder,"features.csv",m)
    tc, t = read_verified(folder,"targets.csv",m)
    expected_targets = {KEY} | {f"{prefix}_{h}{suffix}" for h in HORIZONS for prefix,suffix in
                                (("future_return",""),("up",""),("target_close","_utc"),("label_available","_assumed_utc"))}
    if fc != set(FEATURES) | {KEY,"decision_time_assumed_utc"} or tc != expected_targets:
        raise ValueError("Unexpected feature or label columns")
    if len(f) != len(t) or len(f) != m["rows"] or len(f) != n-36:
        raise ValueError("Feature/target row count mismatch")
    anchor = np.array([epoch(row[KEY]) for row in f],dtype=np.int64)
    if np.any(anchor%300) or np.any(np.diff(anchor)!=300):
        raise ValueError("Anchors must be unique, ordered and contiguous")
    x = np.array([[float(row[k]) for k in INPUTS] for row in f])
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite feature")
    labels = {h:[] for h in HORIZONS}
    for a,fr,tr in zip(anchor,f,t):
        if epoch(tr[KEY])!=a or epoch(fr["decision_time_assumed_utc"])!=a+delay:
            raise ValueError("Feature/target join or decision time mismatch")
        for h,k in HORIZONS.items():
            y=float(tr[f"future_return_{h}"])
            if (not np.isfinite(y) or int(tr[f"up_{h}"])!=int(y>0) or
                epoch(tr[f"target_close_{h}_utc"])!=a+300*k or
                epoch(tr[f"label_available_{h}_assumed_utc"])!=a+300*k+delay):
                raise ValueError("Invalid label or availability")
            labels[h].append(y)
    return m,anchor,x,{h:np.array(y) for h,y in labels.items()}


def splits(source_candles,h):
    """Use Chapter XXIV return-index boundaries, strict availability purge."""
    n=source_candles-1
    j=np.arange(24,source_candles-12)
    a,b=int(.6*n),int(.8*n)
    return {"train":np.flatnonzero(j+h<a),
            "validation":np.flatnonzero((j>=a)&(j+h<b)),
            "development_replay":np.flatnonzero((j>=b)&(j+h<=n))}


def classify(y,p):
    return {"accuracy":float(accuracy_score(y,p>.5)),"brier":float(brier_score_loss(y,p)),
            "log_loss":float(log_loss(y,np.clip(p,1e-15,1-1e-15),labels=[0,1])),
            "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}


def regress(y,p):
    return {"mae":float(np.mean(np.abs(y-p))),"rmse":float(np.sqrt(np.mean((y-p)**2))),
            "direction_accuracy":float(np.mean((y>0)==(p>0)))}


def experiment(m,anchor,x,labels):
    report={"scope":"previously inspected development sample only","source_sha256":m["source_sha256"],
            "assumed_delay_seconds":m["replay_delay_seconds_assumed"],"historical_publication_times_verified":False,
            "inputs":list(INPUTS),"horizons":{}}
    rows=[]
    for h,k in HORIZONS.items():
        idx=splits(m["source_candles"],k)
        train=idx["train"]; y=labels[h]; up=(y>0).astype(int)
        if min(map(len,idx.values()))<20 or len(np.unique(up[train]))!=2:
            raise ValueError("Insufficient split size or training class variation")
        logit=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000))
        ridge=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
        logit.fit(x[train],up[train]);ridge.fit(x[train],y[train]*100)
        if np.max(logit[-1].n_iter_)>=2000:
            raise ValueError("Logistic optimizer reached iteration limit")
        prevalence=float(up[train].mean());mean=float(y[train].mean())
        result={"counts":{s:len(i) for s,i in idx.items()},"purged_rows":len(x)-sum(map(len,idx.values())),
                "train_up_rate":prevalence,"train_mean_log_return":mean,"metrics":{},
                "logistic_coefficients_standardized":dict(zip(INPUTS,map(float,logit[-1].coef_[0]))),
                "ridge_coefficients_standardized_percent_units":dict(zip(INPUTS,map(float,ridge[-1].coef_)))}
        for split in ("validation","development_replay"):
            i=idx[split]; p=logit.predict_proba(x[i])[:,1]; r=ridge.predict(x[i])/100
            result["metrics"][split]={"logistic":classify(up[i],p),
                  "training_prevalence":classify(up[i],np.full(len(i),prevalence)),
                  "ridge":regress(y[i],r),"zero_return":regress(y[i],np.zeros(len(i))),
                  "training_mean":regress(y[i],np.full(len(i),mean))}
            for j,prob,pred in zip(i,p,r):
                rows.append({"horizon":h,"split":split,"anchor_epoch":int(anchor[j]),
                             "actual_log_return":float(y[j]),"actual_up":int(up[j]),
                             "logistic_p_up":float(prob),"ridge_log_return":float(pred)})
        report["horizons"][h]=result
    return report,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",type=Path);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(); report,rows=experiment(*load(args.dataset))
    payload=json.dumps(report,indent=2,allow_nan=False)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/"report.json").write_text(payload,encoding="utf-8")
    print(payload)
    print("Report:",args.output)


if __name__=="__main__":
    main()
