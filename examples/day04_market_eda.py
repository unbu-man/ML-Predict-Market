"""Audit a Day 3 export and run exploratory TSA; no training or trading."""
import argparse
import csv
import hashlib
import json
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path


def utc(value):
    t = datetime.fromisoformat(value)
    if t.utcoffset() != timedelta(0) or t.microsecond:
        raise ValueError("Expected whole-second UTC timestamp")
    return int(t.timestamp())


def audit(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / "candles.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["csv_sha256"]:
        raise ValueError("CSV checksum does not match manifest")
    n = manifest["interval_seconds"]
    a, b = utc(manifest["start_utc"]), utc(manifest["end_exclusive_utc"])
    if (manifest["provider"], manifest["symbol"], manifest["quote_type"]) != (
        "coinbase_exchange", "BTC-USD", "trade_ohlc"
    ) or n not in (300, 3600) or not a < b or a % n or b % n:
        raise ValueError("Unsupported manifest or time grid")
    if (b-a)//n != manifest["expected"]:
        raise ValueError("Manifest expected count mismatch")
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = utc(row["open_time_utc"])
            if (row["provider"] != manifest["provider"] or row["symbol"] != manifest["symbol"]
                    or int(row["interval_seconds"]) != n or not a <= t < b
                    or t % n or utc(row["close_time_utc"]) != t+n
                    or (rows and t <= rows[-1][0])):
                raise ValueError("Invalid, duplicate or unordered candle")
            op, hi, lo, cl, vol = (float(row[k]) for k in ("open", "high", "low", "close", "volume"))
            if (not all(map(math.isfinite, (op, hi, lo, cl, vol)))
                    or min(op, hi, lo, cl) <= 0 or vol < 0
                    or hi < max(op, cl) or lo > min(op, cl)):
                raise ValueError("Invalid OHLCV")
            rows.append((t, cl))
    seen = {t for t, _ in rows}
    missing = set(range(a, b, n)) - seen
    listed = [utc(t) for t in manifest["missing_starts_utc"]]
    if not rows or len(rows) != manifest["received"] or len(listed) != len(missing) or set(listed) != missing:
        raise ValueError("Missing intervals or received count do not match manifest")
    segments = []
    for row in rows:
        if not segments or row[0] != segments[-1][-1][0] + n:
            segments.append([])
        segments[-1].append(row)
    longest = max(segments, key=len)
    return longest, {"symbol": manifest["symbol"], "interval_seconds": n,
                     "expected": manifest["expected"], "received": len(rows),
                     "missing": len(missing), "segments": len(segments),
                     "longest_segment_candles": len(longest),
                     "source_sha256": manifest["csv_sha256"],
                     "historical_publication_times_verified": False}


def analyze(folder):
    import numpy as np
    from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss

    segment, report = audit(folder)
    close = np.array([c for _, c in segment], dtype=float)
    if len(close) < 80:
        raise ValueError("Need at least 80 consecutive candles for this exploratory report")
    returns = np.diff(np.log(close))
    if np.ptp(returns) < 1e-12 or np.ptp(returns**2) < 1e-20:
        raise ValueError("Insufficient return variation for TSA diagnostics")
    lags = min(24, len(returns)//2-1)
    report.update({"scope": "longest contiguous segment; exploratory only",
                   "return_count": len(returns), "mean_log_return": float(returns.mean()),
                   "std_log_return": float(returns.std(ddof=1)),
                   "quantiles": np.quantile(returns, [.01, .5, .99]).tolist(),
                   "acf_returns": acf(returns, nlags=lags, fft=True).tolist(),
                   "pacf_returns": pacf(returns, nlags=lags, method="ywm").tolist(),
                   "acf_squared_returns": acf(returns**2, nlags=lags, fft=True).tolist(),
                   "stationarity": {}})
    for name, values in (("log_price", np.log(close)), ("log_return", returns)):
        with warnings.catch_warnings(record=True) as notes:
            warnings.simplefilter("always")
            adf = adfuller(values, regression="c", autolag="AIC")
            kp = kpss(values, regression="c", nlags="auto")
        report["stationarity"][name] = {"adf_statistic": float(adf[0]), "adf_pvalue": float(adf[1]),
                                        "kpss_statistic": float(kp[0]), "kpss_pvalue": float(kp[1]),
                                        "warnings": [str(w.message) for w in notes]}
    return close, returns, report


def save_report(folder, output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    close, returns, report = analyze(folder)
    output.mkdir(parents=True, exist_ok=False)
    plots = {"close": close, "log_returns": returns,
             "rolling_volatility_12bars": [np.std(returns[i-12:i], ddof=1) for i in range(12, len(returns)+1)],
             "acf_returns": report["acf_returns"][1:], "pacf_returns": report["pacf_returns"][1:],
             "acf_squared_returns": report["acf_squared_returns"][1:]}
    for name, values in plots.items():
        fig, ax = plt.subplots(figsize=(9, 3.5))
        if name.startswith(("acf", "pacf")):
            ax.stem(range(1, len(values)+1), values)
            ax.set_xlabel("Lag (bars)")
        else:
            ax.plot(values)
            ax.set_xlabel("Observation index")
        ax.set_title(name.replace("_", " "))
        fig.tight_layout()
        fig.savefig(output / (name + ".png"), dpi=130)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(returns, bins=30)
    ax.set_title("Log return distribution")
    fig.tight_layout()
    fig.savefig(output / "return_histogram.png", dpi=130)
    plt.close(fig)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ("symbol", "received", "missing", "longest_segment_candles", "stationarity")}, indent=2))
    print("EDA report:", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    save_report(args.dataset, args.output)
