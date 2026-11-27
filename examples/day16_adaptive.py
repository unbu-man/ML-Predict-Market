"""Delayed-feedback adaptive selection on fresh synthetic walk-forward paths."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

WINDOW = 288
MIN_HISTORY = 48
REVIEW_EVERY = 12
SEED = 34001


def route(anchors, predictions, actual, horizon, delay, initial, window=WINDOW,
          minimum=MIN_HISTORY, review_every=REVIEW_EVERY):
    """Use only matured historical errors. Returns selected column indices."""
    a = np.asarray(anchors, dtype=np.int64)
    p, y = np.asarray(predictions, float), np.asarray(actual, float)
    if (a.ndim != 1 or p.ndim != 2 or y.shape != a.shape or len(p) != len(a)
            or not len(a) or np.any(np.diff(a) <= 0) or horizon not in (1, 12)
            or type(delay) is not int or not 0 <= delay < 300
            or not np.isfinite(p).all() or not np.isfinite(y).all()
            or not 0 <= initial < p.shape[1] or not 1 <= minimum <= window
            or review_every < 1):
        raise ValueError('Invalid adaptive inputs')
    chosen, history = np.empty(len(a), int), np.empty(len(a), int)
    current, matured = initial, 0
    errors = []
    for i, t in enumerate(a):
        while matured < i and a[matured]+300*horizon+delay < t+delay:
            errors.append((p[matured]-y[matured])**2)
            matured += 1
        history[i] = min(matured, window)
        if i % review_every == 0 and history[i] >= minimum:
            scores = np.mean(errors[-window:], axis=0)
            # Keep the current candidate on an exact tie; otherwise fixed column order.
            if scores[current] > np.min(scores):
                current = int(np.argmin(scores))
        chosen[i] = current
    return chosen, history


def locked_choices(report, select, controls, variants):
    if report.get('scope') != 'fresh synthetic walk-forward research only; no market performance' or report.get('variants') != variants:
        raise ValueError('Expected Chapter XXXIII full report')
    paths = [p for p in report['paths'] if p['phase']=='selection']
    if sorted((p['control'],p['seed']) for p in paths) != sorted((c,33001+100*i) for i,c in enumerate(controls)) or any(len(p['folds'])!=3 for p in paths):
        raise ValueError('Unexpected selection paths')
    locked = select(paths)
    if locked != report['locked_selection']:
        raise ValueError('Locked selection mismatch')
    return locked


class Collector:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(row)


def replay(rows, locked, variants, delay, classification, regression):
    results, output = {}, []
    for h, bars in [('5m',1),('1h',12)]:
        batch = [r for r in rows if r['horizon']==h]
        a = np.array([r['anchor_epoch'] for r in batch])
        result = {}
        exports = [dict(phase=r['phase'],control=r['control'],seed=r['seed'],fold=r['fold'],
                        horizon=h,anchor_epoch=int(r['anchor_epoch']),actual_log_return=r['actual_log_return'],
                        actual_up=r['actual_up']) for r in batch]
        for task, suffix, actual_key, benchmarks, score in [
            ('classification','_p_up','actual_up',('prevalence',),classification),
            ('regression','_return','actual_log_return',('zero_return','training_mean'),regression)]:
            keys = list(benchmarks)+sorted(variants,key=lambda k:(variants[k],k))
            fixed = locked[h][task]['candidate']
            p = np.array([[r[k+suffix] for k in keys] for r in batch]); y = np.array([r[actual_key] for r in batch])
            if task=='classification' and (np.any((p<0)|(p>1)) or not np.isin(y,[0,1]).all()):
                raise ValueError('Invalid probability or class')
            chosen, history = route(a,p,y,bars,delay,keys.index(fixed))
            adaptive = p[np.arange(len(a)),chosen]
            metrics = {'adaptive':score(y,adaptive), 'locked':score(y,p[:,keys.index(fixed)]),
                       'mtf_only':score(y,p[:,keys.index('none_none')])}
            metrics.update({k:score(y,p[:,keys.index(k)]) for k in benchmarks})
            result[task] = {'locked_candidate':fixed,'metrics':metrics,
                            'selection_counts':{k:int(np.sum(chosen==j)) for j,k in enumerate(keys)},
                            'switches':int(np.sum(np.diff(chosen)!=0)), 'n':len(a)}
            for i,r in enumerate(exports):
                r.update({task+'_selected':keys[chosen[i]],task+'_history':int(history[i]),
                          task+'_adaptive':float(adaptive[i]),task+'_locked':float(p[i,keys.index(fixed)]),
                          task+'_mtf':float(p[i,keys.index('none_none')])})
        results[h]=result
        output.extend(exports)
    return results,output


def main():
    from examples.day10_xgboost import PARAMS
    from examples.day12_hybrid import build_hybrid
    from examples.day13_regimes import fit_regimes
    from examples.day14_factorial import VARIANTS, experiment, classification, regression
    from examples.day15_walkforward import CONTROLS, run_path, select
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--walkforward-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay-delay-seconds',type=int,required=True)
    args=p.parse_args(); delay=args.replay_delay_seconds
    if args.output.exists() or not 0<=delay<300: p.error('Choose new output and valid delay')
    raw=args.walkforward_report.read_bytes(); prior=json.loads(raw)
    locked=locked_choices(prior,select,CONTROLS,VARIANTS)
    if prior['params']!=PARAMS or prior['assumed_delay_seconds']!=delay:
        raise ValueError('Prior settings differ')
    report={'scope':'fresh synthetic adaptive replay only; no market performance',
            'market_holdout_accessed':False,'historical_publication_times_verified':False,
            'prior_report_sha256':hashlib.sha256(raw).hexdigest(), 'locked_selection':locked,
            'assumed_delay_seconds':delay,'window_matured_origins':WINDOW,'minimum_history':MIN_HISTORY,
            'review_every_eligible_origins':REVIEW_EVERY,'params':PARAMS,'controls':{}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        stage=Path(tmp)/'export';stage.mkdir()
        with (stage/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
            writer=None
            for i,c in enumerate(CONTROLS):
                seed=SEED+100*i; collector=Collector()
                print(f'adaptive: {c}, seed {seed}',flush=True)
                path=run_path(c,seed,'adaptive',delay,build_hybrid,experiment,PARAMS,fit_regimes,collector)
                metrics,rows=replay(collector.rows,locked,VARIANTS,delay,classification,regression)
                report['controls'][c]={'seed':seed,'refits':path['refits'],
                    'fold_counts':[v['counts'] for v in path['folds']],
                    'min_coverage':min(v['coverage'] for v in path['folds']),'horizons':metrics}
                if writer is None:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
                writer.writerows(rows)
        (stage/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        if args.output.exists(): raise FileExistsError(str(args.output))
        stage.rename(args.output)
    print(f'Completed: {args.output / "report.json"}')


if __name__=='__main__':
    main()
