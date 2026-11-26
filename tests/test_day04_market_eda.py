import csv
import hashlib
import json
import math
import random
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from examples.day04_market_eda import audit, analyze, save_report


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def fixture(folder, count=140, gap=None):
    rng = random.Random(42)
    path = folder / "candles.csv"
    price = 100.0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["provider", "symbol", "interval_seconds", "open_time_utc", "close_time_utc",
                    "available_at_utc", "open", "high", "low", "close", "volume"])
        for i in range(count):
            if i == gap:
                continue
            op = price
            price *= math.exp(rng.gauss(0, .002))
            w.writerow(["coinbase_exchange", "BTC-USD", 300, iso(i*300), iso((i+1)*300), "",
                        op, max(op, price)+1, min(op, price)-1, price, 1])
    (folder / "manifest.json").write_text(json.dumps({"provider": "coinbase_exchange", "symbol": "BTC-USD",
        "quote_type": "trade_ohlc", "interval_seconds": 300, "start_utc": iso(0), "end_exclusive_utc": iso(count*300),
        "expected": count, "received": count-int(gap is not None), "missing_starts_utc": [] if gap is None else [iso(gap*300)],
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))


class TestMarketEDA(unittest.TestCase):
    def test_gap_not_bridged(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder, gap=20)
            segment, report = audit(folder)
            self.assertEqual(len(segment), 119)
            self.assertEqual(report["segments"], 2)
            self.assertEqual(report["missing"], 1)

    def test_tampered_csv_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder)
            with (folder / "candles.csv").open("a") as f:
                f.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                audit(folder)

    def test_manifest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder)
            path = folder / "manifest.json"
            m = json.loads(path.read_text())
            m["received"] -= 1
            path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "received count"):
                audit(folder)

    def test_diagnostics_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder)
            _, r, report = analyze(folder)
            self.assertEqual(len(r), 139)
            self.assertEqual(len(report["acf_returns"]), 25)
            self.assertIn("adf_pvalue", report["stationarity"]["log_return"])
            save_report(folder, folder / "eda")
            self.assertEqual(len(list((folder / "eda").glob("*.png"))), 7)
            self.assertTrue((folder / "eda" / "report.json").exists())
