import csv
import hashlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from examples.day10_xgboost import build, experiment, read_block, HEADER


def segment(start,n,seed=42):
    r=np.random.default_rng(seed).normal(0,.002,n-1)
    return list(zip(range(start,start+300*n,300),100*np.exp(np.r_[0,np.cumsum(r)])))


def iso(t):
    return datetime.fromtimestamp(t,timezone.utc).isoformat()


class XGBoostTests(unittest.TestCase):
    def test_features_and_boundary(self):
        a,x,y=build([segment(0,100)],30000,2)
        self.assertEqual(x.shape,(63,12))
        self.assertTrue(np.all(a+3600+2<30000))
        self.assertAlmostEqual(x[0,0],np.log(segment(0,100)[24][1]/segment(0,100)[23][1]))
    def test_gap(self):
        a,x,y=build([segment(0,50),segment(18000,50)],33000,2)
        self.assertEqual(len(a),27)
        self.assertFalse(np.any((a>50*300)&(a<18000+25*300)))
    def test_future_perturbation(self):
        s=segment(0,100);a,x,y=build([s],30000,2)
        s[40]=(s[40][0],s[40][1]*2)
        _,q,_=build([s],30000,2)
        np.testing.assert_allclose(x[:16],q[:16])
    def test_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f=p/'train.csv'
            with f.open('w',newline='') as out:
                w=csv.writer(out);w.writerow(HEADER);w.writerow([iso(0),iso(300),'',10,11,9,10,1])
            m=dict(start_utc=iso(0),end_exclusive_utc=iso(300),expected=1,received=1,missing=0,
                   longest_contiguous=1,csv_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            self.assertEqual(read_block(p,'train',m),[[(0,10.)]])
            f.write_text('tampered')
            with self.assertRaisesRegex(ValueError,'checksum'):read_block(p,'train',m)
    def test_matched_models(self):
        blocks={'train':build([segment(0,250)],75000,2),
                'validation':build([segment(75000,100,43)],105000,2)}
        report,rows=experiment(blocks)
        self.assertEqual(len(rows),126)
        self.assertIn('xgb_classifier',report['5m']['classification'])
        self.assertIn('xgb_regressor',report['1h']['regression'])
        self.assertTrue(all(0<=r['xgb_classifier_p_up']<=1 for r in rows))
