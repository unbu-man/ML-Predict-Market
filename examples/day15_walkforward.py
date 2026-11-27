"""Fresh synthetic paths: purged walk-forward selection and assessment."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from examples.day14_factorial import VARIANTS

DAY = 86400
START = 1704067200  # 2024-01-01 UTC; simulated timestamps only
CONTROLS = ('independent', 'variance', 'direction', 'joint')
SEEDS = {'selection': (33001,), 'assessment': (33002, 33003)}
HORIZONS = ('5m', '1h')
TASKS = {'classification': 'brier', 'regression': 'mse'}


def simulate(control, seed, days=60):
    if control not in CONTROLS or days < 1:
        raise ValueError('Invalid synthetic control')
    rng = np.random.default_rng(seed)
    means = np.array([-.0007, 0., .0007])
    sigmas = np.array([.001, .0025, .001])
    state, v, z = 1, 1e-7/.03, np.log(100.)
    result = []
    for t in range(START, START+days*DAY, 300):
        if control in ('direction', 'joint'):
            p = np.full(3, .0025); p[state] = .995
            state = int(rng.choice(3, p=p))
        e = rng.normal()* (np.sqrt(v) if control in ('variance','joint') else
                           sigmas[state] if control == 'direction' else .001125)
        r = e + (means[state] if control in ('direction','joint') else 0.)
        if control in ('variance','joint'):
            v = 1e-7 + .12*e**2 + .85*v
        z += r
        result.append((t, float(np.exp(z))))
    return result


def split_fold(block, cutoff, end, delay):
    a,x,y = block
    if (type(delay) is not int or not 0<=delay<300 or cutoff>=end or
            np.any(np.diff(a)<=0) or len(x)!=len(a) or any(len(v)!=len(a) for v in y.values())):
        raise ValueError('Invalid fold inputs')
    available = a+3600+delay
    train = available < cutoff
    val = (a>=cutoff)&(available<end)
    def subset(mask):
        return a[mask],x[mask],{h:v[mask] for h,v in y.items()}
    if train.sum()<100 or val.sum()<20:
        raise ValueError('Insufficient matched origins; stop rather than skip fold')
    return {'train':subset(train),'validation':subset(val)}


def loss(report, horizon, task, candidate):
    m = report['horizons'][horizon][task][candidate]
    return m['brier'] if task=='classification' else m['rmse']**2


def path_score(folds, h, task, candidate, relative=False):
    values=[]
    for f in folds:
        v=loss(f,h,task,candidate)
        if relative:
            base=loss(f,h,task,'none_none')
            if base<=0: raise ValueError('Nonpositive base loss')
            v/=base
        values.append(v)
    return float(np.mean(values))


def select(paths):
    if not paths or any(p['phase']!='selection' for p in paths):
        raise ValueError('Selection paths only')
    result={}
    for h in HORIZONS:
        result[h]={}
        for task in TASKS:
            scores={k:float(np.mean([path_score(p['folds'],h,task,k,True) for p in paths])) for k in VARIANTS}
            chosen=min(scores,key=lambda k:(scores[k],VARIANTS[k],k))
            result[h][task]={'candidate':chosen,'mean_relative_loss':scores}
    return result


def summarize(paths, locked):
    out={}
    for control in CONTROLS:
        group=[p for p in paths if p['control']==control]
        if not group: raise ValueError('Missing control')
        out[control]={}
        for h in HORIZONS:
            out[control][h]={}
            for task in TASKS:
                chosen=locked[h][task]['candidate']
                benchmarks=('prevalence',) if task=='classification' else ('zero_return','training_mean')
                keys=list(dict.fromkeys((chosen,'none_none',*benchmarks)))
                metrics={}
                for k in keys:
                    per_path=[path_score(p['folds'],h,task,k) for p in group]
                    all_folds=[loss(f,h,task,k) for p in group for f in p['folds']]
                    metrics[k]={'mean_loss':float(np.mean(per_path)), 'path_mean_losses':per_path,
                                'fold_loss_sd':float(np.std(all_folds,ddof=1)) if len(all_folds)>1 else None}
                out[control][h][task]={'locked_candidate':chosen,'loss':TASKS[task], 'metrics':metrics}
    return out


def run_path(control, seed, phase, delay, builder, experiment, params, fitter, writer):
    segment=simulate(control,seed); notes=[]
    block=builder([segment],START+60*DAY,delay,notes)
    result={'control':control,'seed':seed,'phase':phase,'folds':[],
            'refits':{'attempted':len(notes),'failed':sum(n['status']=='failed' for n in notes),'events':notes}}
    for k in range(3):
        cutoff=START+(30+10*k)*DAY; end=cutoff+10*DAY
        split=split_fold(block,cutoff,end,delay)
        report,rows=experiment(split,params,fitter)
        report.update(fold=k+1,cutoff_epoch=cutoff,end_exclusive_epoch=end,
                      first_eval_anchor=int(split['validation'][0][0]),last_eval_anchor=int(split['validation'][0][-1]),
                      expected_eval_origins=2868,coverage=len(split['validation'][0])/2868)
        result['folds'].append(report)
        for row in rows: writer.writerow(dict(phase=phase,control=control,seed=seed,fold=k+1,**row))
    return result


def main():
    from examples.day10_xgboost import PARAMS
    from examples.day12_hybrid import build_hybrid, COLUMNS
    from examples.day13_regimes import fit_regimes, REGIME_COLS, REGIME_NAMES
    from examples.day14_factorial import experiment
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay-delay-seconds',type=int,required=True)
    args=p.parse_args(); delay=args.replay_delay_seconds
    if args.output.exists() or not 0<=delay<300: p.error('Choose new output and valid delay')
    if tuple(COLUMNS[i] for i in REGIME_COLS)!=REGIME_NAMES or tuple(COLUMNS[21:])!=('rolling_rms_288','garch_sigma','constant_rms'):
        raise ValueError('Feature order changed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'scope':'fresh synthetic walk-forward research only; no market performance',
            'market_holdout_accessed':False,'historical_publication_times_verified':False,
            'assumed_delay_seconds':delay,'seeds_by_phase':SEEDS,'variants':VARIANTS,'params':PARAMS,
            'garch_mean':'zero; raw second-moment proxy for nonzero means',
            'initial_train_days':30,'evaluation_days':10,'folds_per_path':3,'paths':[]}
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        stage=Path(tmp)/'export';stage.mkdir()
        with (stage/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
            fields=['phase','control','seed','fold','horizon','anchor_epoch','actual_log_return','actual_up',
                    'rolling_rms','garch_rms_proxy','kmeans_id',*[f'gmm_p{k}' for k in range(3)],
                    *[k+'_p_up' for k in ('prevalence',*VARIANTS)],
                    *[k+'_return' for k in ('zero_return','training_mean',*VARIANTS)]]
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for phase,seeds in SEEDS.items():
                for ci,control in enumerate(CONTROLS):
                    for seed in seeds:
                        actual_seed=seed+100*ci
                        print(f'{phase}: {control}, seed {actual_seed}',flush=True)
                        report['paths'].append(run_path(control,actual_seed,phase,delay,build_hybrid,experiment,PARAMS,fit_regimes,writer))
                if phase=='selection':
                    report['locked_selection']=select(report['paths'])
            assessment=[p for p in report['paths'] if p['phase']=='assessment']
            report['assessment_summary']=summarize(assessment,report['locked_selection'])
            report['audit']={'paths':len(report['paths']),
                             'folds':sum(len(p['folds']) for p in report['paths']),
                             'refits_attempted':sum(p['refits']['attempted'] for p in report['paths']),
                             'refits_failed':sum(p['refits']['failed'] for p in report['paths']),
                             'min_coverage':min(f['coverage'] for p in report['paths'] for f in p['folds'])}
        (stage/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        (stage/'summary.json').write_text(json.dumps({k:report[k] for k in ('scope','audit','locked_selection','assessment_summary')},indent=2,allow_nan=False),encoding='utf-8')
        if args.output.exists(): raise FileExistsError(str(args.output))
        stage.rename(args.output)
    print(f'Completed: {args.output / "summary.json"}')


if __name__=='__main__':
    main()
