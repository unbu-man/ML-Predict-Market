import csv
import tempfile
import unittest
from pathlib import Path
from examples.day09_history import aggregate, export, read_minutes, synthetic, HEADER


class HistoryTests(unittest.TestCase):
    def test_ohlcv_aggregation(self):
        rows=[(60*i,10+i,12+i,9+i,11+i,2) for i in range(5)]
        self.assertEqual(list(aggregate(rows,0,300)),[(0,10,16,9,15,10)])

    def test_missing_minute(self):
        rows=list(synthetic(0,600)); del rows[2]
        self.assertEqual([r[0] for r in aggregate(rows,0,600)],[300])

    def test_duplicate_rejected(self):
        rows=list(synthetic(0,600));rows.insert(2,rows[1])
        with self.assertRaises(ValueError):list(aggregate(rows,0,600))

    def test_holdout_prices_not_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(HEADER);w.writerow([0,10,12,9,11,2]);w.writerow([60,'UNREAD','','','',''])
            self.assertEqual(len(list(read_minutes(p,0,60))),1)

    def test_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'result';bounds=(0,600,1200,1800)
            r=export(synthetic(0,1200),p,'synthetic','test',boundaries=bounds)
            self.assertEqual(r['blocks']['train']['received'],2)
            self.assertEqual(r['blocks']['validation']['missing'],0)
            self.assertFalse(r['holdout_exported'])
            self.assertEqual({x.name for x in p.iterdir()},{'train.csv','validation.csv','manifest.json'})
            with self.assertRaises(FileExistsError):export(synthetic(0,1200),p,'synthetic','test',boundaries=bounds)
