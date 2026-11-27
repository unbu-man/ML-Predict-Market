import unittest
import numpy as np
from examples.day13_regimes import fit_regimes, positive_control, experiment, TRANSITION


def fixture():
    rng=np.random.default_rng(4)
    x=rng.normal(size=(300,21)); q=rng.normal(size=(90,21))
    return x,q


class RegimeTests(unittest.TestCase):
    def test_encodings(self):
        x,q=fixture(); (kt,gt),(kv,gv),_=fit_regimes(x,q)
        for a in (kt,gt,kv,gv):
            self.assertEqual(a.shape[1],3)
            np.testing.assert_allclose(a.sum(axis=1),1)
            self.assertTrue(np.all((a>=0)&(a<=1)))
        self.assertTrue(np.all((kv==0)|(kv==1)))
    def test_validation_cannot_change_fit(self):
        x,q=fixture(); a,_,m=fit_regimes(x,q); b,_,n=fit_regimes(x,q+100)
        for i in range(2): np.testing.assert_allclose(a[i],b[i])
        for k in ("scaler_mean","scaler_scale","kmeans_centers_original_units","gmm_means_original_units","gmm_weights"):
            np.testing.assert_allclose(m[k],n[k])
    def test_bad_features(self):
        x,q=fixture(); x[0,0]=np.nan
        with self.assertRaises(ValueError): fit_regimes(x,q)
    def test_control(self):
        a,t=positive_control(2,2); b,u=positive_control(2,2)
        self.assertEqual((a,t),(b,u)); self.assertEqual(len(a["train"]),576)
        self.assertEqual(len(a["validation"]),576)
        self.assertTrue(all(t[s+300] in (0,1,2) for block in a.values() for s,_ in block))
        np.testing.assert_allclose(TRANSITION.sum(axis=1),1)
    def test_matched_models(self):
        x,q=fixture(); rng=np.random.default_rng(5)
        at=np.arange(300)*300;av=100000+np.arange(90)*300
        blocks={"train":(at,x,{h:rng.normal(0,.001,len(x)) for h in ("5m","1h")}),
                "validation":(av,q,{h:rng.normal(0,.001,len(q)) for h in ("5m","1h")})}
        report,rows=experiment(blocks,dict(n_estimators=10,max_depth=2,n_jobs=2,random_state=42))
        self.assertEqual(len(rows),180)
        self.assertEqual(set(report["horizons"]["5m"]["classification"]),{"prevalence","mtf_only","mtf_kmeans","mtf_gmm"})
        self.assertTrue(all(0<=r["mtf_gmm_p_up"]<=1 for r in rows))
