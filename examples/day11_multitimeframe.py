"""Synthetic-only causal multi-timeframe XGBoost ablation."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from examples.day10_xgboost import INPUTS, HORIZONS, PARAMS, build, load, classify, regress

FRAMES = (("1h", 3600), ("4h", 14400), ("1d", 86400))
FEATURES = INPUTS + tuple(f"{name}_{feature}" for name, _ in FRAMES
                          for feature in ("return", "mean3", "std3"))
VARIANTS = {"5m_only": 12, "plus_1h": 15, "plus_4h": 18, "plus_1d": 21}


def higher_features(segment, period):
    """Complete aligned buckets in one continuous 5m segment; close-only features."""
    if period not in dict(FRAMES).values():
        raise ValueError("Unsupported timeframe")
    t = np.asarray([s[0] for s in segment], dtype=np.int64)
    c = np.asarray([s[1] for s in segment], dtype=float)
    if not len(t) or np.any(t % 300) or np.any(np.diff(t) != 300) or not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError("Invalid continuous segment")
    width = period // 300
    first = (-int(t[0]) % period) // 300
    count = max(0, (len(t)-first)//width)
    if count < 4:
        return np.empty(0,dtype=np.int64), np.empty((0,3))
    ends = first + width*np.arange(1,count+1)-1
    r = np.diff(np.log(c[ends]))
    w = np.lib.stride_tricks.sliding_window_view(r,3)
    return t[ends[3:]]+300, np.column_stack((r[2:], w.mean(axis=1), w.std(axis=1,ddof=1)))


def align(anchors, closes, values, delay):
    """No higher candle may be used before its assumed availability."""
    j = np.searchsorted(closes+delay, anchors+delay, side="right")-1
    out = np.full((len(anchors),3),np.nan)
    valid = j >= 0
    out[valid] = values[j[valid]]
    return out


def build_mtf(segments, stop, delay):
    if type(delay) is not int or not 0 <= delay < 300:
        raise ValueError("Invalid assumed delay")
    aa, xx = [], []
    yy = {h:[] for h in HORIZONS}
    for segment in segments:
        if len(segment)<37 or segment[24][0]+300+3600+delay>=stop:
            continue
        anchors, base, targets = build([segment],stop,delay)
        columns = [base]
        for _, period in FRAMES:
            closes, values = higher_features(segment,period)
            columns.append(align(anchors,closes,values,delay))
        x = np.column_stack(columns)
        keep = np.isfinite(x).all(axis=1)
        if keep.any():
            aa.append(anchors[keep]); xx.append(x[keep])
            for h in HORIZONS:
                yy[h].append(targets[h][keep])
    if not aa:
        raise ValueError("No origins with complete 5m and higher-timeframe windows")
    return np.concatenate(aa), np.vstack(xx), {h:np.concatenate(v) for h,v in yy.items()}


def experiment(blocks):
    at,xt,yt = blocks["train"]
    av,xv,yv = blocks["validation"]
    if len(at)<100 or len(av)<20 or at[-1]+3600>=av[0]:
        raise ValueError("Insufficient or overlapping matched blocks")
    report, rows = {}, []
    for h in HORIZONS:
        y,actual = yt[h],yv[h]
        up,observed = (y>0).astype(int),(actual>0).astype(int)
        if len(np.unique(up))!=2:
            raise ValueError("Both training classes required")
        probs={"prevalence":np.full(len(av),up.mean())}
        preds={"zero_return":np.zeros(len(av)),"training_mean":np.full(len(av),y.mean())}
        for name,n in VARIANTS.items():
            cl=XGBClassifier(objective="binary:logistic",eval_metric="logloss",**PARAMS)
            rg=XGBRegressor(objective="reg:squarederror",**PARAMS)
            cl.fit(xt[:,:n],up); rg.fit(xt[:,:n],y*100)
            probs[name]=cl.predict_proba(xv[:,:n])[:,1].astype(float)
            preds[name]=rg.predict(xv[:,:n]).astype(float)/100
        report[h]={"training_up_rate":float(up.mean()),
                   "classification":{k:classify(observed,p) for k,p in probs.items()},
                   "regression":{k:regress(actual,p) for k,p in preds.items()}}
        for j,a in enumerate(av):
            rows.append({"horizon":h,"anchor_epoch":int(a),"actual_log_return":float(actual[j]),
                         "actual_up":int(observed[j]),**{k+"_p_up":float(p[j]) for k,p in probs.items()},
                         **{k+"_return":float(p[j]) for k,p in preds.items()}})
    return report,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",type=Path); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--replay-delay-seconds",type=int,required=True)
    args=p.parse_args()
    if args.output.exists():
        p.error("Output exists; read earlier report or choose a new directory")
    m,blocks=load(args.dataset,args.replay_delay_seconds,builder=build_mtf)
    metrics,rows=experiment(blocks)
    report={"scope":"synthetic software test only; not market performance","holdout_exported":False,
            "historical_publication_times_verified":False,"assumed_delay_seconds":args.replay_delay_seconds,
            "features":list(FEATURES),"variants":{k:list(FEATURES[:n]) for k,n in VARIANTS.items()},
            "xgboost_params":PARAMS,"source_block_hashes":{k:v["csv_sha256"] for k,v in m["blocks"].items()},
            "counts":{k:len(v[0]) for k,v in blocks.items()},"horizons":metrics}
    payload=json.dumps(report,indent=2,allow_nan=False)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.output/"report.json").write_text(payload,encoding="utf-8")
    print(payload)


if __name__=="__main__":
    main()
