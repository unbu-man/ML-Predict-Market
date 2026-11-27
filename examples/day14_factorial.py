"""Synthetic-only matched 3x3 volatility/regime factorial experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

VOL = ('none', 'rolling', 'garch')
REG = ('none', 'kmeans', 'gmm')
VARIANTS = {f'{v}_{g}': 21 + int(v != 'none') + 3*int(g != 'none')
            for v in VOL for g in REG}
JOINT_SEED = 20261129
JOINT_OMEGA = 1e-7
JOINT_ALPHA = .12
JOINT_BETA = .85
JOINT_MEANS = np.array([-.0007, 0., .0007])
JOINT_TRANSITION = np.full((3, 3), .0025)
np.fill_diagonal(JOINT_TRANSITION, .995)


def joint_control(boundary, train_days=40, validation_days=20):
    rng = np.random.default_rng(JOINT_SEED)
    v = JOINT_OMEGA/(1-JOINT_ALPHA-JOINT_BETA)
    state, z = 1, np.log(100.)
    segments, truth = {'train': [], 'validation': []}, {}
    for t in range(boundary-train_days*86400, boundary+validation_days*86400, 300):
        state = int(rng.choice(3, p=JOINT_TRANSITION[state]))
        e = np.sqrt(v)*rng.normal()
        z += JOINT_MEANS[state]+e
        v = JOINT_OMEGA+JOINT_ALPHA*e**2+JOINT_BETA*v
        segments['train' if t<boundary else 'validation'].append((t, float(np.exp(z))))
        truth[t+300] = state
    return segments, truth


def design(x, encodings):
    if x.ndim != 2 or x.shape[1] != 24 or not np.isfinite(x).all() or np.any(x[:,21:]<=0):
        raise ValueError('Expected 24 finite hybrid columns with positive RMS values')
    if any(a.shape != (len(x),3) or not np.isfinite(a).all() or np.any(a<0) or
           not np.allclose(a.sum(axis=1),1) for a in encodings):
        raise ValueError('Invalid regime encodings')
    out = {}
    for v in VOL:
        for g in REG:
            cols = [x[:,:21]]
            if v != 'none': cols.append(x[:,21:22] if v=='rolling' else x[:,22:23])
            if g != 'none': cols.append(encodings[0 if g=='kmeans' else 1])
            out[f'{v}_{g}'] = np.column_stack(cols)
    return out


def contrasts(losses):
    base = losses['none_none']
    return {'delta_vs_base': {k: float(v-base) for k,v in losses.items()},
            'interaction': {f'{v}_{g}': float(losses[f'{v}_{g}']-losses[f'{v}_none']-
                                              losses[f'none_{g}']+base)
                            for v in VOL[1:] for g in REG[1:]}}


def classification(y,p):
    return {'accuracy':float(np.mean((p>.5)==y)), 'brier':float(brier_score_loss(y,p)),
            'log_loss':float(log_loss(y,np.clip(p,1e-15,1-1e-15),labels=[0,1])),
            'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}


def regression(y,p):
    return {'mae':float(np.mean(np.abs(y-p))), 'rmse':float(np.sqrt(np.mean((y-p)**2))),
            'direction_accuracy':float(np.mean((y>0)==(p>0)))}


def second_moment(y,v):
    q=y**2
    return {'n':len(y),'qlike':float(np.mean(np.log(v)+q/v)),
            'mse':float(np.mean((q-v)**2)), 'mae':float(np.mean(np.abs(q-v)))}


def experiment(blocks, params, fitter, truth=None, oracle=None):
    at,xt,yt=blocks['train']; av,xv,yv=blocks['validation']
    if len(at)<100 or len(av)<20 or at[-1]+3600>=av[0]:
        raise ValueError('Insufficient or overlapping matched blocks')
    et,ev,diag=fitter(xt[:,:21],xv[:,:21])
    train,val=design(xt,et),design(xv,ev)
    if truth is not None:
        states=np.array([truth[int(a)] for a in av])
        diag['validation_state_ari']={k:float(adjusted_rand_score(states,e.argmax(axis=1)))
                                     for k,e in zip(REG[1:],ev)}
    report={'counts':{'train':len(at),'validation':len(av)}, 'clustering':diag,
            'second_moment_5m':{k:second_moment(yv['5m'],xv[:,j]**2)
                               for k,j in [('rolling',21),('garch',22),('refit_constant',23)]},
            'horizons':{}}
    if oracle is not None:
        report['second_moment_5m']['simulator_oracle']=second_moment(yv['5m'],np.array([oracle[int(a)] for a in av]))
    rows=[]
    for h in ('5m','1h'):
        y,actual=yt[h],yv[h]; up,observed=y>0,actual>0
        if len(np.unique(up))!=2: raise ValueError('Both training classes required')
        probs={'prevalence':np.full(len(av),up.mean())}
        preds={'zero_return':np.zeros(len(av)),'training_mean':np.full(len(av),y.mean())}
        for k in VARIANTS:
            cl=XGBClassifier(objective='binary:logistic',eval_metric='logloss',**params)
            rg=XGBRegressor(objective='reg:squarederror',**params)
            cl.fit(train[k],up.astype(int)); rg.fit(train[k],y*100)
            probs[k]=cl.predict_proba(val[k])[:,1].astype(float)
            preds[k]=rg.predict(val[k]).astype(float)/100
        report['horizons'][h]={'classification':{k:classification(observed,p) for k,p in probs.items()},
                               'regression':{k:regression(actual,p) for k,p in preds.items()},
                               'contrasts':{'brier':contrasts({k:float(np.mean((probs[k]-observed)**2)) for k in VARIANTS}),
                                            'mse':contrasts({k:float(np.mean((preds[k]-actual)**2)) for k in VARIANTS})}}
        for j,a in enumerate(av):
            rows.append({'horizon':h,'anchor_epoch':int(a),'actual_log_return':float(actual[j]),
                         'actual_up':int(observed[j]), 'rolling_rms':float(xv[j,21]),'garch_rms_proxy':float(xv[j,22]),
                         'kmeans_id':int(ev[0][j].argmax()),
                         **{f'gmm_p{k}':float(ev[1][j,k]) for k in range(3)},
                         **{k+'_p_up':float(p[j]) for k,p in probs.items()},
                         **{k+'_return':float(p[j]) for k,p in preds.items()}})
    return report,rows


def main():
    from examples.day09_history import VALIDATION
    from examples.day10_xgboost import load, PARAMS
    from examples.day12_hybrid import build_hybrid, positive_control as garch_control, COLUMNS, FIT_WINDOW, REFIT_EVERY, ROLL_WINDOW
    from examples.day13_regimes import fit_regimes, positive_control as direction_control, REGIME_COLS, REGIME_NAMES
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay-delay-seconds',type=int,required=True)
    args=p.parse_args()
    if args.output.exists(): p.error('Output exists; read report or choose a new directory')
    if tuple(COLUMNS[i] for i in REGIME_COLS)!=REGIME_NAMES or tuple(COLUMNS[21:])!=('rolling_rms_288','garch_sigma','constant_rms'):
        raise ValueError('Feature order changed')
    delay=args.replay_delay_seconds
    events=[]
    m,negative=load(args.dataset,delay,builder=lambda s,b,d:build_hybrid(s,b,d,events))
    controls={'negative':(negative,events,None,None)}
    for name,generated in [('variance',garch_control()),('direction',direction_control()),('joint',joint_control(VALIDATION))]:
        segments,truth_or_oracle=generated; notes=[]
        blocks={k:build_hybrid([s],s[-1][0]+300,delay,notes) for k,s in segments.items()}
        controls[name]=(blocks,notes,truth_or_oracle if name!='variance' else None,truth_or_oracle if name=='variance' else None)
    report={'scope':'synthetic factorial controls only; no market-performance inference','holdout_exported':False,
            'historical_publication_times_verified':False,'assumed_delay_seconds':delay,'variants':VARIANTS,
            'xgboost_params':PARAMS,'source_block_hashes':{k:v['csv_sha256'] for k,v in m['blocks'].items()},
            'fit_window_returns':FIT_WINDOW,'refit_every_returns':REFIT_EVERY,'rolling_window_returns':ROLL_WINDOW,
            'garch_mean':'zero; raw second-moment proxy when conditional mean is nonzero',
            'joint_control':{'seed':JOINT_SEED,'train_days':40,'validation_days':20,'means':JOINT_MEANS.tolist(),
                             'transition':JOINT_TRANSITION.tolist(),'omega':JOINT_OMEGA,'alpha':JOINT_ALPHA,'beta':JOINT_BETA},
            'controls':{}}
    all_rows={}
    for name,(blocks,notes,truth,oracle) in controls.items():
        result,rows=experiment(blocks,PARAMS,fit_regimes,truth,oracle)
        result['refits']={'attempted':len(notes),'failed':sum(e['status']=='failed' for e in notes),'events':notes}
        report['controls'][name]=result; all_rows[name]=rows
    payload=json.dumps(report,indent=2,allow_nan=False)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,rows in all_rows.items():
        with (args.output/(name+'_predictions.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/'report.json').write_text(payload,encoding='utf-8')
    print(payload)


if __name__=='__main__':
    main()
