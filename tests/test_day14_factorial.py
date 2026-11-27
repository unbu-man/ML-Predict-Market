import unittest
import numpy as np
from examples.day14_factorial import design, contrasts, joint_control, experiment, VARIANTS


def fixture(n, seed):
    x=np.random.default_rng(seed).normal(size=(n,24)); x[:,21:]=.001
    return x


def enc(x):
    k=np.eye(3)[np.arange(len(x))%3]; g=np.full((len(x),3),1/3)
    return k,g


class FactorialTests(unittest.TestCase):
    def test_design(self):
        x=fixture(20,1); e=enc(x); d=design(x,e)
        self.assertEqual({k:v.shape[1] for k,v in d.items()},VARIANTS)
        np.testing.assert_array_equal(d['garch_gmm'][:,21],x[:,22])
        np.testing.assert_array_equal(d['garch_gmm'][:,22:],e[1])
        np.testing.assert_array_equal(d['rolling_kmeans'][:,21],x[:,21])
    def test_invalid(self):
        x=fixture(20,1); x[0,22]=0
        with self.assertRaises(ValueError): design(x,enc(x))
    def test_interaction(self):
        loss={k:10. for k in VARIANTS}; loss.update(garch_none=9.,none_gmm=8.,garch_gmm=6.)
        c=contrasts(loss)
        self.assertEqual(c['interaction']['garch_gmm'],-1.)
        self.assertEqual(c['delta_vs_base']['garch_gmm'],-4.)
    def test_joint(self):
        a,t=joint_control(0,2,2); b,u=joint_control(0,2,2)
        self.assertEqual((a,t),(b,u)); self.assertEqual(len(a['train']),576)
        self.assertTrue(all(c>0 and np.isfinite(c) for s in a.values() for _,c in s))
        self.assertTrue(all(t[i+300] in (0,1,2) for s in a.values() for i,_ in s))
    def test_matched_models(self):
        rng=np.random.default_rng(3)
        def block(start,n):
            return np.arange(start,start+300*n,300),fixture(n,start+1),{h:rng.normal(0,.001,n) for h in ('5m','1h')}
        blocks={'train':block(0,200),'validation':block(100000,60)}
        result,rows=experiment(blocks,dict(n_estimators=3,max_depth=2,n_jobs=2,random_state=42),
                               lambda x,q:(enc(x),enc(q),{}))
        self.assertEqual(len(rows),120)
        self.assertEqual(set(result['horizons']['5m']['classification']),set(VARIANTS)|{'prevalence'})
        self.assertTrue(all(0<=r['garch_gmm_p_up']<=1 for r in rows))
        self.assertEqual(len(result['horizons']['1h']['contrasts']['mse']['interaction']),4)
