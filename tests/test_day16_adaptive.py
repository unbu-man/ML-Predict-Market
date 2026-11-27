import unittest
import numpy as np
from examples.day16_adaptive import route, replay, locked_choices


class AdaptiveTests(unittest.TestCase):
    def test_strict_maturity(self):
        a=np.arange(30)*300; p=np.zeros((30,2)); y=np.zeros(30)
        _,n=route(a,p,y,12,2,0,minimum=1,review_every=1)
        self.assertEqual(n[12],0);self.assertEqual(n[13],1)

    def test_switch_and_tie(self):
        a=np.arange(30)*300; p=np.tile([0.,1.],(30,1));y=np.ones(30)
        c,n=route(a,p,y,1,2,0,window=5,minimum=3,review_every=1)
        self.assertTrue(np.all(c[:4]==0));self.assertTrue(np.all(c[4:]==1))
        c,_=route(a,np.zeros_like(p),y,1,2,1,minimum=1,review_every=1)
        self.assertTrue(np.all(c==1))

    def test_future_labels_cannot_change_earlier_choices(self):
        a=np.arange(80)*300;p=np.tile([0.,1.],(80,1)); y=np.ones(80)
        c,_=route(a,p,y,12,2,0,window=5,minimum=1,review_every=1)
        y[30:]=0
        d,_=route(a,p,y,12,2,0,window=5,minimum=1,review_every=1)
        np.testing.assert_array_equal(c[:43],d[:43]);self.assertNotEqual(c[-1],d[-1])

    def test_replay(self):
        rows=[]; variants={'none_none':21,'none_gmm':24}
        for h in ('5m','1h'):
            for i in range(80):
                rows.append(dict(phase='adaptive',control='joint',seed=42,fold=1,horizon=h,anchor_epoch=i*300,
                    actual_log_return=.001,actual_up=1,prevalence_p_up=.8,none_none_p_up=.1,none_gmm_p_up=.4,
                    zero_return_return=0.,training_mean_return=.001,none_none_return=-.001,none_gmm_return=0.))
        locked={h:{t:{'candidate':'none_none'} for t in ('classification','regression')} for h in ('5m','1h')}
        def score(y,p):return {'loss':float(np.mean((y-p)**2))}
        r,out=replay(rows,locked,variants,2,score,score)
        self.assertEqual(len(out),160)
        self.assertGreater(r['5m']['classification']['selection_counts']['prevalence'],0)
        self.assertGreater(r['5m']['regression']['selection_counts']['training_mean'],0)

    def test_lock_rejects_wrong_seed(self):
        r={'scope':'fresh synthetic walk-forward research only; no market performance','variants':{},
           'paths':[{'phase':'selection','control':'joint','seed':7,'folds':[{}, {}, {}]}]}
        with self.assertRaises(ValueError):locked_choices(r,lambda p:{},('joint',),{})


if __name__=='__main__':unittest.main()
