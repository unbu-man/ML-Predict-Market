# Chapter XXVII — Historical data expansion and licensing gate

**27 November 2026.** Data engineering before advanced modeling. The preceding seven-day experiment was inspected; it is not a final holdout.

## 1. Blocking issue: data permission

Coinbase's [Market Data Terms](https://www.coinbase.com/legal/market_data), updated August 7, 2026, section 3, restrict AI/ML research, training and benchmarking, as well as dissemination of derived works, without prior express written consent. We have not established such consent. Stop new Coinbase downloads, training, benchmarking, and dissemination of its derived results until applicable permissions are resolved. Review existing public documentation and repository visibility; making the repository private does not by itself cure prior dissemination or supply a license. No existing files or visibility settings changed by this chapter. Code-only synthetic testing can continue.

Potential candidate: [Zielak, Bitcoin Historical Data, Kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data), advertised CC BY-SA 4.0, minute candles with USD quotes from Bitstamp. However, [Bitstamp API documentation](https://www.bitstamp.net/api/) directs companies to obtain a commercial data license. A dataset uploader's license is not proof of upstream rights. Verify provenance, license coverage, attribution/share-alike obligations, permission for ML training and future commercial model/signal use, redistribution and Cambodia access before importing real data. Contact Coinbase marketdata@coinbase.com or Bitstamp partners@bitstamp.net as appropriate. Do not treat publicly accessible prices as automatically open-licensed. This is a conservative project decision, not legal advice.

## 2. Proposed frozen historical dates

One year ending before our previous November smoke test. UTC, start inclusive/end exclusive; date boundaries selected before inspecting new candidate data.

| Block | Start | End exclusive | Days | Expected 5m candles |
|---|---|---|---:|---:|
| Training | 2025-08-01 | 2026-04-01 | 243 | 69,984 |
| Validation | 2026-04-01 | 2026-06-01 | 61 | 17,568 |
| Reserved holdout | 2026-06-01 | 2026-08-01 | 61 | 17,568 |
| Total | 2025-08-01 | 2026-08-01 | 365 | 105,120 |

```math
N_{\mathrm{expected}}=365\times24\times\frac{60}{5}=105120
```

The candidate source's actual coverage remains unverified. If it lacks coverage, document a new plan before any analysis. Holdout dates are frozen in this chapter, not a prospective paper test. No final holdout scores until model, feature, hyperparameter, costs, risk and selection rules are fixed. Importer does not export holdout rows or parse their prices; a full-file hash reads bytes only for provenance. Keep the original data private. No claim of cryptographic access control.

## 3. Source-independent local import

Chapter XXVII intentionally replaces the proposed API expansion with an offline streaming CSV importer. No new market data are requested. The input contract is an authorized, sorted, single-venue BTC/USD spot OHLCV CSV with Unix-second UTC candle-open timestamps and header Timestamp,Open,High,Low,Close,Volume. No mixing exchanges, no USDT substitution, no fabricated availability timestamps. The permission reference is a local record, not an automated rights check. Do not input secrets.

### Aggregate five one-minute candles

```math
\begin{aligned}
O_t^{(5)}&=O_t^{(1)}\\
H_t^{(5)}&=\max_{i=0}^{4}H_{t+i}^{(1)}\\
L_t^{(5)}&=\min_{i=0}^{4}L_{t+i}^{(1)}\\
C_t^{(5)}&=C_{t+4}^{(1)}\\
V_t^{(5)}&=\sum_{i=0}^{4}V_{t+i}^{(1)}
\end{aligned}
```

Only five distinct, aligned, validated one-minute rows make a complete 5m bar. Missing minute means discard that bucket, not forward fill. Count missing 5m bars and longest contiguous runs per block. Do not bridge gaps in returns, features or targets. Require valid finite positive prices, OHLC envelope, nonnegative volume, strictly chronological unique timestamps. Source hash, output hashes and explicit date boundaries are recorded. Writes are staged before publishing a new directory; run one process per output. This is an offline importer, not a resumable network downloader.

## 4. Run safely now (no dependencies beyond Python 3.12)

```bash
git pull origin main
python3 -m unittest tests.test_day09_history -v
python3 -m examples.day09_history --synthetic --output data/history/synthetic_1y_plan
```

The synthetic CLI executed successfully in the assistant container: train 69,984, validation 17,568, zero missing; five unit tests passed (OHLCV arithmetic, missing-minute discard, duplicate rejection, holdout-price nonparsing and export/no-overwrite). Fake prices are strictly for software testing, never evidence of trading skill. The reserved holdout is not generated.

Output files: train.csv, validation.csv, manifest.json under gitignored data/history/. Existing directories are protected; use a new name for a rerun. If successful, read the manifest instead of rerunning.

For a real authorized file only, the importer accepts --csv, --provider and --permission-reference in place of --synthetic. Do not use this mode until the source and intended ML use are cleared. Permission text cannot establish rights. Source identity is supplied by the user and not independently verified.

## 5. Next gate

After confirming rights and actual coverage, add a provider-neutral audit and gap-aware causal feature pipeline for the separate development blocks; the older Coinbase-specific audit and relative-split examples are not directly compatible with this export. Keep chronological train and validation boundaries, purge label availability across boundaries, fit training-only transforms and use matched origins across models. Implement XGBoost on synthetic fixtures while permission is unresolved. Later repeat M0/M1/M2 and GARCH on authorized data, then fixed M3 XGBoost, purged walk-forward validation, and causal volatility/multitimeframe ablations. No reliable trading prediction, XAU access, live feed or commercial rights have been demonstrated.

[Chapter XXVI](26_supervised_baselines.md)
