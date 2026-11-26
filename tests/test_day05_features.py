import math
import unittest
from examples.day05_features import build, FEATURE_COLUMNS, META_COLUMNS


def fixture(n=60):
    return [(300*i, 100 + i + (i%3)*0.1) for i in range(n)]


class FeatureTests(unittest.TestCase):
    def test_shape_and_target(self):
        s = fixture()
        x,y = build(s, 2)
        self.assertEqual(len(x), 24)
        self.assertEqual(len(y), 24)
        self.assertEqual(set(x[0]), set(FEATURE_COLUMNS+META_COLUMNS))
        self.assertAlmostEqual(y[0]["future_return_5m"], math.log(s[25][1]/s[24][1]))
        self.assertAlmostEqual(y[0]["future_return_1h"], math.log(s[36][1]/s[24][1]))
        self.assertEqual(y[0]["up_5m"], 1)
        self.assertEqual(x[0]["anchor_close_utc"], y[0]["anchor_close_utc"])

    def test_future_perturbation(self):
        s = fixture()
        x,y = build(s, 2)
        t,c = s[25]
        s[25] = (t,c*1.1)
        xx,yy = build(s, 2)
        self.assertEqual(x[0],xx[0])
        self.assertNotEqual(y[0]["future_return_5m"],yy[0]["future_return_5m"])

    def test_gap_rejected(self):
        with self.assertRaisesRegex(ValueError,"missing"):
            build(fixture()[:25]+fixture()[26:],2)

    def test_invalid_delay(self):
        for d in (-1,300,2.5):
            with self.assertRaises(ValueError):
                build(fixture(),d)

    def test_flat_targets_not_up(self):
        x,y = build([(300*i,100.0) for i in range(37)],2)
        self.assertEqual(len(x),1)
        self.assertEqual(y[0]["up_5m"],0)
        self.assertEqual(y[0]["up_1h"],0)
