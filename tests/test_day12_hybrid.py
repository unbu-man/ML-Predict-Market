import unittest
import numpy as np
from examples.day12_hybrid import forecast_paths, positive_control, build_hybrid, experiment, FIT_WINDOW, VARIANTS


def fixed_fit(_):
    return .1,.2,.7


class HybridTests(unittest.TestCase):
    def test_recursion_and_warmup(self):
        r=np.array([1.,-1.,2.,-2.,3.])
        out,events=forecast_paths(r,fit_window=2,refit_every=10,roll_window=2,fit=fixed_fit)
        self.assertTrue(np.isnan(out[:2]).all())
        # For history [1,-1], RMS and both conditional variances equal one.
        np.testing.assert_allclose(out[2],[1.,1.,1.])
        self.assertAlmostEqual(out[3,1]**2,1.6)
        self.assertEqual(len(events),1)
    def test_future_invariance_at_refit(self):
        r=np.random.default_rng(9).normal(size=30)
        out,_=forecast_paths(r,fit_window=5,refit_every=4,roll_window=3,fit=fixed_fit)
        r[13:]+=20
        q,_=forecast_paths(r,fit_window=5,refit_every=4,roll_window=3,fit=fixed_fit)
        np.testing.assert_allclose(out[:14],q[:14],equal_nan=True)
    def test_failed_refit(self):
        def failure(_):
            raise ValueError("forced failure")
        out,events=forecast_paths(np.ones(12),fit_window=4,refit_every=4,roll_window=2,fit=failure)
        self.assertTrue(np.isnan(out).all())
        self.assertTrue(all(x["status"]=="failed" for x in events))
    def test_positive_control_and_gap_reset(self):
        s,oracle=positive_control()
        self.assertEqual(len(s["train"]),30*288)
        self.assertEqual(len(s["validation"]),20*288)
        self.assertTrue(all(v>0 for v in oracle.values()))
        first=s["train"][:6*288];second=s["train"][7*288:13*288]
        a,x,y=build_hybrid([first,second],second[-1][0]+300,2,[],fit=fixed_fit)
        self.assertEqual(x.shape[1],24)
        self.assertFalse(np.any((a>first[-1][0]+300)&(a<second[FIT_WINDOW][0]+300)))
        self.assertTrue(np.all(a+3600+2<second[-1][0]+300))
    def test_matched_model_outputs(self):
        rng=np.random.default_rng(3)
        def block(start,n):
            return np.arange(start,start+300*n,300),rng.normal(size=(n,24)),{h:rng.normal(0,.001,n) for h in ("5m","1h")}
        # Positive RMS columns required by the variance scoring.
        train=block(0,200);val=block(100000,60)
        for b in (train,val): b[1][:,21:]=.001
        report,rows=experiment({"train":train,"validation":val})
        self.assertEqual(len(rows),120)
        self.assertEqual(set(report["horizons"]["5m"]["classification"]),{"prevalence"}|set(VARIANTS))
        self.assertTrue(all(0<=r["mtf_garch_p_up"]<=1 for r in rows))
