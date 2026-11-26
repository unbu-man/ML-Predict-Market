import unittest
from examples.day02_coinbase_probe import normalize, parse_utc

class TestProbe(unittest.TestCase):
    def test_sort_and_filter(self):
        a=[0,99,103,100,102,10]
        b=[300,101,105,102,104,9]
        self.assertEqual(normalize([b,a,a,[600,100,105,103,104,1]],0,600),[tuple(a),tuple(b)])
    def test_gap_preserved(self):
        self.assertEqual(len(normalize([[300,99,103,100,102,10]],0,900)),1)
    def test_invalid_envelope(self):
        with self.assertRaises(ValueError):
            normalize([[0,99,101,100,102,10]],0,300)
    def test_conflicting_duplicate(self):
        with self.assertRaises(ValueError):
            normalize([[0,99,103,100,102,10],[0,99,103,100,101,10]],0,300)
    def test_invalid_number(self):
        with self.assertRaises(ValueError):
            normalize([[0,99,103,100,float("nan"),10]],0,300)
    def test_naive_time_rejected(self):
        with self.assertRaises(ValueError):
            parse_utc("2026-11-20T00:00:00")
