# Chapter XXI — First historical BTC/USD dataset

26 November 2026 — Collector starter and fixture tests. Real provider access and licensing remain unverified; do not mark the dataset complete yet.

After confirming permitted research use and passing the one-hour probe, run from repo root:

```bash
python3 -m unittest discover -s ./tests -t . -v
python3 -m examples.day03_collect_btc --start 2026-11-13T00:00:00Z --end 2026-11-20T00:00:00Z --interval 300
python3 -m examples.day03_collect_btc --start 2026-11-13T00:00:00Z --end 2026-11-20T00:00:00Z --interval 3600
```

The seven-day example is a pipeline smoke test, not adequate evidence for model skill. Expected 2016 five-minute and 168 hourly bars if uninterrupted; actual counts may differ. Requests use disjoint windows of at most 250 candles (below Coinbase's 300 maximum), with one-second pauses. On failure, stop rather than invent observations. No retry/backoff or resume yet; do not repeat requests aggressively. Use a distinct window to avoid overwriting previous exports.

Files are saved under gitignored data/coinbase/ in a separate directory per interval and window. candles.csv contains normalized chronological trade-based OHLCV and timestamps. manifest.json records expected/received counts, missing timestamps, page retrieval times and CSV SHA256; it is written last. Treat an export without a manifest as incomplete. Keep source data private until license permits distribution.

**Causality:** available_at_utc is blank, not fabricated from candle end or today's retrieval time. Historical publication latency remains unknown. Before feature joins or model training, document a conservative replay delay assumption and perform sensitivity checks. Do not fill missing prices or cross missing intervals when building labels.

**Next acceptance gate:** verify both real dataset manifests and tests; inspect gaps, chronological ordering, duplicate keys, price/volume validity and provenance. Then build a feature/label dataset, beginning with past-only returns and volatility, and evaluate zero-return and majority-class baselines. No XAU integration, model training, win rate or live trading claimed. XAU provider remains pending.

Source: https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles

[Chapter XX](20_day02_data_foundation.md) | [Chapter XIX](19_hybrid_selective_forecast_strategy.md)
