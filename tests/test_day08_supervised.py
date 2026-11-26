import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from examples.day08_supervised import FEATURES, INPUTS, HORIZONS, KEY, load, splits, experiment


def iso(t):
    return datetime.fromtimestamp(t,timezone.utc).isoformat()


def fixture(folder,n=300):
    rng=np.random.default_rng(42); f=[]; t=[]
    for j in range(24,n-12):
        a=j*300
        f.append({KEY:iso(a),"decision_time_assumed_utc":iso(a+2),
                  **dict(zip(FEATURES,rng.normal(size=len(FEATURES))))})
        row={KEY:iso(a)}
        for h,k in HORIZONS.items():
            y=float(rng.normal(0,.002));row.update({f"future_return_{h}":y,f"up_{h}":int(y>0),
                  f"target_close_{h}_utc":iso(a+300*k),f"label_available_{h}_assumed_utc":iso(a+300*k+2)})
        t.append(row)
    hashes={}
    for name,rows in (("features.csv",f),("targets.csv",t)):
        p=folder/name
        with p.open("w",newline="") as out:
            w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (folder/"manifest.json").write_text(json.dumps({"feature_columns":list(FEATURES),"join_key":KEY,
          "warmup_bars":24,"future_bars_reserved":12,"source_candles":n,"rows":n-36,
          "replay_delay_seconds_assumed":2,"file_sha256":hashes,"source_sha256":"synthetic"}))


class SupervisedTests(unittest.TestCase):
    def test_split_purge(self):
        for h in (1,12):
            d=splits(2016,h);j=np.arange(24,2004)
            self.assertLess(j[d['train'][-1]]+h,j[d['validation'][0]])
            self.assertLess(j[d['validation'][-1]]+h,j[d['development_replay'][0]])
    def test_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixture(p);m,a,x,y=load(p)
            self.assertEqual(x.shape,(264,12));self.assertEqual(len(y['1h']),264)
            self.assertFalse(any('momentum' in k for k in INPUTS))
    def test_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixture(p)
            with (p/'targets.csv').open('a') as f:f.write('tampered')
            with self.assertRaisesRegex(ValueError,'Checksum'):load(p)
    def test_future_perturbation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixture(p);m,a,x,y=load(p)
            _,r=experiment(m,a,x,y)
            j=splits(m['source_candles'],1)['development_replay'];x[j]+=100;y['5m'][j]*=-1
            _,q=experiment(m,a,x,y)
            self.assertEqual([z for z in r if z['split']=='validation'],[z for z in q if z['split']=='validation'])
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixture(p);report,rows=experiment(*load(p))
            json.dumps(report,allow_nan=False)
            self.assertEqual(set(report['horizons']),{'5m','1h'})
            self.assertTrue(all(0<=r['logistic_p_up']<=1 for r in rows))
