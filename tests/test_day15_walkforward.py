import unittest
import numpy as np
from examples.day15_walkforward import simulate, split_fold, select, summarize, VARIANTS, CONTROLS, HORIZONS
from examples.day14_factorial import experiment
from examples.day13_regimes import fit_regimes


def fake_path(phase,control='joint',winner='none_gmm'):
    cls={k:{'brier':.2 if k==winner else .25} for k in VARIANTS}
    cls['prevalence']={'brier':.25}
    reg={k:{'rmse':.001 if k==winner else .002} for k in VARIANTS}
    reg.update(zero_return={'rmse':.003},training_mean={'rmse':.003})
    return {'phase':phase,'control':control,'folds':[{'horizons':{h:{'classification':cls,'regression':reg} for h in HORIZONS}}]*3}


class WalkForwardTests(unittest.TestCase):
    def test_fresh_paths(self):
        for c in CONTROLS:
            a=simulate(c,42,2)
            self.assertEqual(a,simulate(c,42,2))
            self.assertNotEqual(a,simulate(c,43,2))
            self.assertEqual(len(a),576)
            self.assertTrue(all(p>0 and np.isfinite(p) for _,p in a))
    def test_purge_and_disjoint(self):
        a=np.arange(1200)*300
        block=(a,np.zeros((len(a),24)),{h:np.zeros(len(a)) for h in HORIZONS})
        first=split_fold(block,120000,180000,2)
        second=split_fold(block,180000,240000,2)
        self.assertTrue(np.all(first['train'][0]+3602<120000))
        self.assertTrue(np.all(first['validation'][0]+3602<180000))
        self.assertGreater(second['validation'][0][0],first['validation'][0][-1])
        self.assertEqual(len(first['validation'][0]),188)
        self.assertGreater(len(second['train'][0]),len(first['train'][0]))
    def test_selection_isolation(self):
        p=fake_path('selection'); locked=select([p])
        self.assertEqual(locked['1h']['classification']['candidate'],'none_gmm')
        with self.assertRaises(ValueError): select([p,fake_path('assessment')])
        a=[fake_path('assessment',c,'garch_gmm') for c in CONTROLS]
        summary=summarize(a,locked)
        self.assertEqual(summary['joint']['1h']['regression']['locked_candidate'],'none_gmm')
        self.assertAlmostEqual(summary['joint']['1h']['regression']['metrics']['none_gmm']['mean_loss'],.002**2)
    def test_actual_small_factorial(self):
        rng=np.random.default_rng(21); a=np.arange(700)*300
        x=rng.normal(size=(len(a),24)); x[:,21:]=.001
        block=(a,x,{h:rng.normal(0,.001,len(a)) for h in HORIZONS})
        fold=split_fold(block,120000,180000,2)
        r,rows=experiment(fold,dict(n_estimators=3,max_depth=2,n_jobs=2,random_state=42),fit_regimes)
        self.assertEqual(len(rows),376)
        self.assertIn('garch_gmm',r['horizons']['1h']['classification'])
        self.assertTrue(all(0<=v['garch_gmm_p_up']<=1 for v in rows))


if __name__=='__main__': unittest.main()
