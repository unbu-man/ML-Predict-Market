"""Read-only BTC-USD historical data connectivity probe. Python 3.10+.
No credentials, orders, or writes of market data. Run only after checking usage rights.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

def normalize(rows, start, end, interval=300):
    """Validate bounded historical trade candles; preserve gaps, never fill them."""
    if not isinstance(rows, list):
        raise ValueError("Expected a candle array")
    result = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != 6:
            raise ValueError("Unexpected candle schema")
        ts, low, high, op, close, vol = row
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in row):
            raise ValueError("Non-finite or non-numeric field")
        if ts % interval != 0 or min(low, high, op, close) <= 0 or vol < 0:
            raise ValueError("Invalid timestamp, price or volume")
        if low > min(op, close) or high < max(op, close):
            raise ValueError("Invalid OHLC")
        if ts < start or ts + interval > end:
            continue
        record = (ts, low, high, op, close, vol)
        if ts in result and result[ts] != record:
            raise ValueError("Conflicting duplicate")
        result[ts] = record
    return [result[t] for t in sorted(result)]

def parse_utc(value):
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None or t.utcoffset().total_seconds() != 0:
        raise ValueError("Explicit UTC required")
    return t

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval", type=int, choices=(300, 3600), default=300)
    args = parser.parse_args()
    start, end = parse_utc(args.start), parse_utc(args.end)
    a, b = int(start.timestamp()), int(end.timestamp())
    n = args.interval
    if not (start < end <= datetime.now(timezone.utc)) or a % n or b % n or start.microsecond or end.microsecond or (b-a)//n > 300:
        parser.error("Use an aligned historical UTC window of 1 to 300 complete candles")
    query = urlencode({"start": start.isoformat(), "end": end.isoformat(), "granularity": n})
    req = Request(URL+"?"+query, headers={"User-Agent":"ML-Predict-Market-research/0.1"})
    with urlopen(req, timeout=20) as response:
        rows = json.load(response)
    bars = normalize(rows, a, b, n)
    observed = {r[0] for r in bars}
    missing = [datetime.fromtimestamp(t, timezone.utc).isoformat() for t in range(a,b,n) if t not in observed]
    print(json.dumps({"provider":"coinbase_exchange","symbol":"BTC-USD","interval_seconds":n,
                      "expected":(b-a)//n,"received":len(bars),"missing_starts_utc":missing,
                      "historical_publication_times_verified":False}, indent=2))
    if not bars:
        raise SystemExit("No candles received; access or coverage unresolved")

if __name__ == "__main__":
    main()
