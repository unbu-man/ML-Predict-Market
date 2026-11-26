"""Bounded BTC-USD historical candle collector. Read-only; no trading."""
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from examples.day02_coinbase_probe import URL, normalize, parse_utc

def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()

def windows(start, end, interval):
    if interval not in (300, 3600) or start >= end or start % interval or end % interval:
        raise ValueError("Use aligned 5m or 1h UTC boundaries")
    for left in range(start, end, 250 * interval):
        yield left, min(left + 250 * interval, end)

def fetch_window(left, right, interval):
    query = urlencode({"start": iso(left), "end": iso(right), "granularity": interval})
    req = Request(URL + "?" + query, headers={"User-Agent": "ML-Predict-Market-research/0.1"})
    with urlopen(req, timeout=20) as response:
        return json.load(response)

def collect(start, end, interval, fetch=fetch_window, pause=1):
    rows, pages = [], []
    for left, right in windows(start, end, interval):
        payload = fetch(left, right, interval)
        normalized = normalize(payload, left, right, interval)
        rows.extend(normalized)
        pages.append({"start_utc": iso(left), "end_utc": iso(right),
                      "received": len(normalized),
                      "retrieved_at_utc": datetime.now(timezone.utc).isoformat()})
        if pause and right < end:
            time.sleep(pause)
    observed = {int(row[0]) for row in rows}
    missing = [iso(t) for t in range(start, end, interval) if t not in observed]
    report = {"provider": "coinbase_exchange", "symbol": "BTC-USD",
              "quote_type": "trade_ohlc", "interval_seconds": interval,
              "start_utc": iso(start), "end_exclusive_utc": iso(end),
              "expected": (end-start)//interval, "received": len(rows),
              "missing_starts_utc": missing, "pages": pages,
              "historical_publication_times_verified": False}
    if not rows:
        raise ValueError("No candles returned; no dataset written")
    return rows, report

def save(rows, report, folder):
    folder.mkdir(parents=True, exist_ok=False)
    csv_path = folder / "candles.csv"
    interval = report["interval_seconds"]
    with csv_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["provider", "symbol", "interval_seconds", "open_time_utc",
                         "close_time_utc", "available_at_utc", "open", "high",
                         "low", "close", "volume"])
        for t, low, high, op, close, volume in rows:
            writer.writerow(["coinbase_exchange", "BTC-USD", interval,
                             iso(t), iso(t+interval), "", op, high, low, close, volume])
    report["csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    # The manifest is written last; absence indicates an incomplete export.
    (folder / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval", type=int, choices=(300, 3600), default=300)
    args = parser.parse_args()
    start, end = parse_utc(args.start), parse_utc(args.end)
    a, b = int(start.timestamp()), int(end.timestamp())
    if (start.microsecond or end.microsecond or not start < end <= datetime.now(timezone.utc)
            or b-a > 7*86400 or a % args.interval or b % args.interval):
        parser.error("Choose aligned, completed UTC history of at most seven days")
    folder = Path("data") / "coinbase" / f"BTC-USD_{args.interval}_{a}_{b}"
    if folder.exists():
        parser.error("Dataset folder already exists; preserve it and choose another window")
    rows, report = collect(a, b, args.interval)
    save(rows, report, folder)
    print(json.dumps({k: report[k] for k in ("expected", "received", "historical_publication_times_verified")}, indent=2))
    print(f"Missing intervals: {len(report['missing_starts_utc'])}")
    print(f"Dataset and manifest: {folder}")

if __name__ == "__main__":
    main()
