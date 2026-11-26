import unittest
import numpy as np
from examples.day07_garch import variance_path, fit_garch, losses, experiment

class VolatilityTests(unittest.TestCase):
    def test_recursion(self):
        np.testing.assert_allclose(variance_path(np.array([2.,1.,0.]),.1,.2,.7),[1.,1.6,1.42])
    def test_future_invariance(self):
        e=np.array([.1,.2,1.,.4]);p=variance_path(e,.1,.2,.7)
        e[2:]=10;np.testing.assert_allclose(p[:3],variance_path(e,.1,.2,.7)[:3])
    def test_fit_constraints(self):
        w,a,b=fit_garch(np.random.default_rng(13).normal(size=300))
        self.assertTrue(w>0 and a>=0 and b>=0 and a+b<1)
    def test_loss(self):
        self.assertAlmostEqual(losses(np.array([1.,1.]),np.ones(2))["qlike"],1)
    def test_full_experiment(self):
        r=np.random.default_rng(42).normal(0,.002,300)
        s=list(zip(range(0,301*300,300),100*np.exp(np.r_[0,np.cumsum(r)])))
        report,rows=experiment(s,2)
        self.assertEqual(len(rows),120)
        self.assertEqual(report["metrics"]["validation"]["garch11"]["n"],60)
        self.assertTrue(all(row["garch11"]>0 and row["decision_epoch_assumed"]<row["target_close_epoch"] for row in rows))
    def test_gap(self):
        with self.assertRaises(ValueError):
            experiment([(i*300,100.) for i in range(210) if i!=10],2)
