import unittest
import numpy as np
from examples.day11_multitimeframe import higher_features, align, build_mtf, experiment


def segment(start,n,seed=42):
    r=np.random.default_rng(seed).normal(0,.002,n-1)
    return list(zip(range(start,start+300*n,300),100*np.exp(np.r_[0,np.cumsum(r)])))


class MultiTimeframeTests(unittest.TestCase):
    def test_complete_buckets(self):
        s=[(300*i,float(i+1)) for i in range(1,60)]
        t,x=higher_features(s,3600)
        np.testing.assert_array_equal(t,[18000])
        r=np.diff(np.log([24,36,48,60]))
        np.testing.assert_allclose(x[0],[r[-1],r.mean(),r.std(ddof=1)])
    def test_asof(self):
        x=align(np.array([14300,14400,15000]),np.array([14400]),np.array([[1.,2.,3.]]),2)
        self.assertTrue(np.isnan(x[0]).all()); np.testing.assert_array_equal(x[1:],[[1,2,3],[1,2,3]])
    def test_gap_reset(self):
        s1=segment(0,5*288); s2=segment(6*86400,5*288)
        a,x,y=build_mtf([s1,s2],11*86400,2)
        self.assertFalse(np.any((a>5*86400)&(a<10*86400)))
        self.assertEqual(x.shape[1],21)
    def test_causality_and_cutoff(self):
        s=segment(0,6*288); a,x,y=build_mtf([s],6*86400,2)
        self.assertEqual(len(a),6*288-1164)
        self.assertTrue(np.all(a+3600+2<6*86400))
        i=5*288+1; s[i]=(s[i][0],s[i][1]*2)
        b,q,_=build_mtf([s],6*86400,2)
        np.testing.assert_array_equal(a,b)
        keep=a<=s[i][0]; np.testing.assert_allclose(x[keep],q[keep])
    def test_matched_models(self):
        blocks={"train":build_mtf([segment(0,6*288)],6*86400,2),
                "validation":build_mtf([segment(6*86400,5*288,43)],11*86400,2)}
        report,rows=experiment(blocks)
        self.assertEqual(len(rows),2*len(blocks["validation"][0]))
        self.assertIn("plus_1d",report["5m"]["classification"])
        self.assertIn("5m_only",report["1h"]["regression"])
        self.assertTrue(all(0<=r["plus_1d_p_up"]<=1 for r in rows))
