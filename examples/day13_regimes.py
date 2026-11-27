"""Synthetic-only K-Means/GMM regime ablation with a known-direction control."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, accuracy_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

REGIME_COLS = (10, 11, 13, 14)
REGIME_NAMES = ("mean_return_24", "std_return_24", "1h_mean3", "1h_std3")
K = 3
SEED = 20261128
MEANS = np.array([-.0007, 0., .0007])
SIGMAS = np.array([.001, .0025, .001])
TRANSITION = np.full((K,K), .0025)
np.fill_diagonal(TRANSITION, .995)
BOUNDARY = int(datetime(2026,4,1,tzinfo=timezone.utc).timestamp())
HORIZONS = ("5m", "1h")


def fit_regimes(train, validation):
    """Batch training-only transforms. No labels or validation fit."""
    if any(x.ndim!=2 or x.shape[1]!=21 or not np.isfinite(x).all() for x in (train,validation)):
        raise ValueError("Expected finite 21-column MTF matrices")
    scaler=StandardScaler().fit(train[:,REGIME_COLS])
    z=scaler.transform(train[:,REGIME_COLS]); q=scaler.transform(validation[:,REGIME_COLS])
    km=KMeans(n_clusters=K,n_init=10,random_state=42).fit(z)
    gm=GaussianMixture(n_components=K,covariance_type="diag",reg_covar=1e-6,
                       n_init=3,max_iter=300,random_state=42).fit(z)
    if not gm.converged_ or len(np.unique(km.labels_))!=K:
        raise ValueError("Degenerate or nonconverged clustering")
    def encode(x):
        labels=km.predict(x); p=gm.predict_proba(x)
        return np.eye(K)[labels],p
    kt,gt=encode(z); kv,gv=encode(q)
    report={"scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist(),
            "kmeans_centers_original_units":scaler.inverse_transform(km.cluster_centers_).tolist(),
            "kmeans_train_counts":kt.sum(axis=0).astype(int).tolist(),
            "kmeans_validation_counts":kv.sum(axis=0).astype(int).tolist(),
            "gmm_means_original_units":scaler.inverse_transform(gm.means_).tolist(),
            "gmm_covariance_diagonal_scaled":gm.covariances_.tolist(),"gmm_weights":gm.weights_.tolist(),
            "gmm_train_mean_responsibility":gt.mean(axis=0).tolist(),
            "gmm_validation_mean_responsibility":gv.mean(axis=0).tolist()}
    return (kt,gt),(kv,gv),report


def positive_control(train_days=40, validation_days=20):
    """Markov regimes with known conditional direction; state is scoring-only."""
    rng=np.random.default_rng(SEED); state=1; logc=np.log(100.)
    segments={"train":[],"validation":[]}; truth={}
    for t in range(BOUNDARY-train_days*86400,BOUNDARY+validation_days*86400,300):
        state=int(rng.choice(K,p=TRANSITION[state]))
        logc+=MEANS[state]+SIGMAS[state]*rng.normal()
        segments["train" if t<BOUNDARY else "validation"].append((t,float(np.exp(logc))))
        truth[t+300]=state
    return segments,truth


def classify(y,p):
    return dict(accuracy=float(accuracy_score(y,p>.5)),brier=float(brier_score_loss(y,p)),
                log_loss=float(log_loss(y,np.clip(p,1e-15,1-1e-15),labels=[0,1])),
                roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None)


def regress(y,p):
    return dict(mae=float(np.mean(np.abs(y-p))),rmse=float(np.sqrt(np.mean((y-p)**2))),
                direction_accuracy=float(np.mean((y>0)==(p>0))))


def experiment(blocks, params, truth=None):
    at,xt,yt=blocks["train"]; av,xv,yv=blocks["validation"]
    if len(at)<100 or len(av)<20 or at[-1]+3600>=av[0]:
        raise ValueError("Insufficient or overlapping matched blocks")
    (kt,gt),(kv,gv),diagnostics=fit_regimes(xt,xv)
    if truth is not None:
        actual=np.array([truth[int(a)] for a in av])
        diagnostics["validation_state_ari"]={"kmeans":float(adjusted_rand_score(actual,kv.argmax(axis=1))),
                                             "gmm":float(adjusted_rand_score(actual,gv.argmax(axis=1)))}
    variants={"mtf_only":(xt,xv),"mtf_kmeans":(np.column_stack((xt,kt)),np.column_stack((xv,kv))),
              "mtf_gmm":(np.column_stack((xt,gt)),np.column_stack((xv,gv)))}
    report={"counts":{"train":len(at),"validation":len(av)},"clustering":diagnostics,"horizons":{}}
    rows=[]
    for h in HORIZONS:
        y,actual=yt[h],yv[h]; up=(y>0).astype(int); observed=(actual>0).astype(int)
        if len(np.unique(up))!=2: raise ValueError("Both training classes required")
        probs={"prevalence":np.full(len(av),up.mean())}
        preds={"zero_return":np.zeros(len(av)),"training_mean":np.full(len(av),y.mean())}
        for name,(a,b) in variants.items():
            cl=XGBClassifier(objective="binary:logistic",eval_metric="logloss",**params)
            rg=XGBRegressor(objective="reg:squarederror",**params)
            cl.fit(a,up); rg.fit(a,y*100)
            probs[name]=cl.predict_proba(b)[:,1].astype(float)
            preds[name]=rg.predict(b).astype(float)/100
        report["horizons"][h]={"classification":{k:classify(observed,p) for k,p in probs.items()},
                              "regression":{k:regress(actual,p) for k,p in preds.items()}}
        for j,a in enumerate(av):
            rows.append({"horizon":h,"anchor_epoch":int(a),"actual_log_return":float(actual[j]),
                         "actual_up":int(observed[j]),"kmeans_id":int(kv[j].argmax()),
                         **{f"gmm_p{k}":float(gv[j,k]) for k in range(K)},
                         **{k+"_p_up":float(p[j]) for k,p in probs.items()},
                         **{k+"_return":float(p[j]) for k,p in preds.items()}})
    return report,rows


def main():
    from examples.day10_xgboost import load, PARAMS
    from examples.day11_multitimeframe import build_mtf, FEATURES
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--replay-delay-seconds",type=int,required=True)
    args=p.parse_args()
    if args.output.exists(): p.error("Output exists; read report or choose a new directory")
    if tuple(FEATURES[i] for i in REGIME_COLS)!=REGIME_NAMES: raise ValueError("Feature order changed")
    delay=args.replay_delay_seconds
    m,negative=load(args.dataset,delay,builder=build_mtf)
    segments,truth=positive_control()
    positive={k:build_mtf([s],s[-1][0]+300,delay) for k,s in segments.items()}
    neg,nrows=experiment(negative,PARAMS);pos,prows=experiment(positive,PARAMS,truth)
    report={"scope":"synthetic controls only; no market-performance inference","holdout_exported":False,
            "historical_publication_times_verified":False,"assumed_delay_seconds":delay,
            "regime_features":list(REGIME_NAMES),"n_clusters":K,"gmm_covariance_type":"diag",
            "base_features":list(FEATURES),"variants":{"mtf_only":21,"mtf_kmeans":24,"mtf_gmm":24},
            "xgboost_params":PARAMS,"source_block_hashes":{k:v["csv_sha256"] for k,v in m["blocks"].items()},
            "positive_control":{"seed":SEED,"train_days":40,"validation_days":20,"means":MEANS.tolist(),
                                "sigmas":SIGMAS.tolist(),"transition":TRANSITION.tolist()},
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
