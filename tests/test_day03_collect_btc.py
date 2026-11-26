import csv
import json
import tempfile
import unittest
from pathlib import Path
from examples.day03_collect_btc import windows, collect, save

def fixture(left, right, interval):
    return [[t, 99, 102, 100, 101, 2] for t in reversed(range(left, right, interval))]

class CollectorTests(unittest.TestCase):
    def test_windows(self):
        self.assertEqual(list(windows(0, 251*300, 300)), [(0,75000),(75000,75300)])
    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            list(windows(1, 600, 300))
    def test_multiple_pages(self):
        rows, report = collect(0, 251*300, 300, fetch=fixture, pause=0)
        self.assertEqual(len(rows), 251)
        self.assertEqual(len(report["pages"]), 2)
        self.assertEqual(report["missing_starts_utc"], [])
        self.assertEqual([r[0] for r in rows], list(range(0,75300,300)))
    def test_gaps(self):
        rows, report = collect(0, 900, 300, fetch=lambda a,b,n: fixture(a,b,n)[:-1], pause=0)
        self.assertEqual(report["received"], 2)
        self.assertEqual(len(report["missing_starts_utc"]), 1)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            collect(0,300,300,fetch=lambda *args: [],pause=0)
    def test_export(self):
        rows, report = collect(0,600,300,fetch=fixture,pause=0)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/"dataset"
            save(rows, report, folder)
            with (folder/"candles.csv").open() as f:
                records = list(csv.DictReader(f))
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["available_at_utc"], "")
            manifest = json.loads((folder/"manifest.json").read_text())
            self.assertFalse(manifest["historical_publication_times_verified"])
            self.assertEqual(len(manifest["csv_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                save(rows, report, folder)
