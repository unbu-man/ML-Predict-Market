import unittest
import numpy as np
from examples.day06_statistical_baselines import horizons, score, fit_forecasts, experiment

class BaselineTests(unittest.TestCase):
    def test_horizon_sum(self):
        self.assertAlmostEqual(horizons(.03,.01,.5,3),.065)
    def test_metrics(self):
        s=score([.01,-.01],[0,0])
        self.assertAlmostEqual(s['mae'],.01)
        self.assertAlmostEqual(s['direction_accuracy'],.5)
    def test_causal_fixed_fit(self):
        r=np.random.default_rng(8).normal(0,.002,250)
        p,*_=fit_forecasts(r,150,(1,0,0))
        r[190:]+=0.05
        q,*_=fit_forecasts(r,150,(1,0,0))
        np.testing.assert_allclose(p[:191],q[:191],atol=1e-10)
    def test_replay_split(self):
        r=np.random.default_rng(9).normal(0,.002,300)
        segment=list(zip(range(0,301*300,300),100*np.exp(np.r_[0,np.cumsum(r)])))
        report,rows=experiment(segment,2)
        self.assertEqual(report['metrics']['validation_60m']['models']['zero_return']['n'],49)
        self.assertEqual(report['metrics']['development_replay_5m']['models']['zero_return']['n'],60)
        self.assertTrue(all(row['decision_epoch_assumed']<row['target_close_epoch'] for row in rows))
    def test_gap_rejected(self):
        s=[(300*i,100.) for i in range(160) if i!=10]
        with self.assertRaises(ValueError):
            experiment(s,2)
