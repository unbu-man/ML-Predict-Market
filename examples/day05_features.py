"""Build a development-only 5m BTC feature/target pair, not a forecast."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from examples.day04_market_eda import audit

INTERVAL = 300
LAGS = (0, 1, 2, 3, 6, 12)
WINDOWS = (6, 12, 24)
FEATURE_COLUMNS = tuple([f"return_lag_{k}" for k in LAGS] +
    [f"{kind}_{w}" for w in WINDOWS for kind in ("mean_return", "std_return", "momentum")])
KEY = "anchor_close_utc"
META_COLUMNS = (KEY, "decision_time_assumed_utc")


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def build(segment, replay_delay_seconds):
    """Require contiguous 5m closes; replay latency is an assumption, not evidence."""
    if type(replay_delay_seconds) is not int or not 0 <= replay_delay_seconds < INTERVAL:
        raise ValueError("Replay delay must be an integer in [0, 299]")
    if len(segment) < 37:
        raise ValueError("Need at least 37 consecutive candles")
    for j, (t, close) in enumerate(segment):
        if t % INTERVAL or not math.isfinite(close) or close <= 0:
            raise ValueError("Invalid candle")
        if j and t != segment[j-1][0] + INTERVAL:
            raise ValueError("Cannot bridge missing or unordered candles")
    z = [math.log(close) for _, close in segment]
    r = [0.0] + [z[j]-z[j-1] for j in range(1, len(z))]
    features, targets = [], []
    # Index 24 has 24 actual past returns; reserve 12 future candles.
    for i in range(24, len(segment)-12):
        anchor = segment[i][0] + INTERVAL
        x = {KEY: iso(anchor), "decision_time_assumed_utc": iso(anchor+replay_delay_seconds)}
        for k in LAGS:
            x[f"return_lag_{k}"] = r[i-k]
        for w in WINDOWS:
            past = r[i-w+1:i+1]
            x[f"mean_return_{w}"] = statistics.mean(past)
            x[f"std_return_{w}"] = statistics.stdev(past)
            x[f"momentum_{w}"] = z[i]-z[i-w]
        y = {KEY: iso(anchor)}
        for h, name in ((1,"5m"), (12,"1h")):
            ret = z[i+h]-z[i]
            y[f"future_return_{name}"] = ret
            y[f"up_{name}"] = int(ret > 0)
            y[f"target_close_{name}_utc"] = iso(anchor+h*INTERVAL)
            y[f"label_available_{name}_assumed_utc"] = iso(anchor+h*INTERVAL+replay_delay_seconds)
        features.append(x)
        targets.append(y)
    return features, targets


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replay-delay-seconds", type=int, required=True)
    args = p.parse_args()
    segment, report = audit(args.dataset)
    if report["interval_seconds"] != INTERVAL:
        p.error("Use the 5m dataset, not the hourly export")
    features, targets = build(segment, args.replay_delay_seconds)
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {name: write_csv(args.output/name, rows)
              for name, rows in (("features.csv", features), ("targets.csv", targets))}
    manifest = {"source_sha256": report["source_sha256"], "scope": "development smoke test only",
                "feature_columns": list(FEATURE_COLUMNS), "join_key": KEY,
                "source_candles": len(segment), "rows": len(features),
                "warmup_bars": 24, "future_bars_reserved": 12,
                "replay_delay_seconds_assumed": args.replay_delay_seconds,
                "historical_publication_times_verified": False,
                "file_sha256": hashes}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("Feature/target dataset:", args.output)


if __name__ == "__main__":
    main()
