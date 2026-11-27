"""Offline 1m BTC/USD importer; never fetches market data or opens a holdout."""
import argparse
import csv
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


def epoch(s):
    return int(datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp())


START = epoch('2025-08-01T00:00:00Z')
VALIDATION = epoch('2026-04-01T00:00:00Z')
HOLDOUT = epoch('2026-06-01T00:00:00Z')
END = epoch('2026-08-01T00:00:00Z')
HEADER = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_minutes(path, start, stop):
    """Read chronological Unix-second UTC candles; never parse holdout prices."""
    previous = None
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER:
            raise ValueError('Expected exact Timestamp,Open,High,Low,Close,Volume header')
        for row in reader:
            raw = float(row['Timestamp'])
            if not math.isfinite(raw) or raw % 60:
                raise ValueError('Expected aligned Unix-second timestamps')
            t = int(raw)
            if previous is not None and t <= previous:
                raise ValueError('Duplicate or unordered timestamp')
            previous = t
            if t >= stop:
                break
            if t < start:
                continue
            if None in row or any(row[k] is None for k in HEADER):
                raise ValueError('Malformed CSV row')
            yield t, *(float(row[k]) for k in HEADER[1:])


def synthetic(start, stop):
    """Deterministic fake prices for pipeline tests only, not market evidence."""
    rng = random.Random(42)
    close = 100000.
    for t in range(start, stop, 60):
        op = close
        close = op * math.exp(rng.gauss(0, .0005))
        yield t, op, max(op, close), min(op, close), close, rng.random()


def aggregate(rows, start, stop):
    """Emit only complete five-minute groups of distinct, valid one-minute rows."""
    group = []
    bucket = None
    prev = None
    for row in rows:
        t, op, hi, lo, cl, vol = row
        if (t % 60 or not start <= t < stop or (prev is not None and t <= prev)
                or not all(map(math.isfinite, (op, hi, lo, cl, vol)))
                or min(op, hi, lo, cl) <= 0 or lo > min(op, cl)
                or hi < max(op, cl) or vol < 0):
            raise ValueError('Invalid or nonchronological minute candle')
        prev = t
        b = t - t % 300
        if b != bucket:
            group, bucket = [], b
        group.append(row)
        if len(group) == 5:
            yield (b, group[0][1], max(x[2] for x in group),
                   min(x[3] for x in group), group[-1][4], sum(x[5] for x in group))


def export(rows, output, provider, permission_reference, source_hash=None,
           boundaries=(START, VALIDATION, HOLDOUT, END)):
    start, validation, holdout, end = boundaries
    if not start < validation < holdout < end or any(t % 300 for t in boundaries):
        raise ValueError('Invalid split boundaries')
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    blocks = {'train': (start, validation), 'validation': (validation, holdout)}
    report = {'scope': 'synthetic pipeline test only' if provider == 'synthetic' else 'authorized development data only',
              'provider': provider, 'symbol': 'BTC-USD', 'interval_seconds': 300,
              'permission_reference': permission_reference, 'source_sha256': source_hash,
              'historical_publication_times_verified': False,
              'holdout_start_utc': iso(holdout), 'holdout_end_exclusive_utc': iso(end),
              'holdout_exported': False, 'blocks': {}}
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/'export'
        stage.mkdir()
        for name, (a,b) in blocks.items():
            report['blocks'][name] = {'start_utc': iso(a), 'end_exclusive_utc': iso(b),
                                      'expected': (b-a)//300, 'received': 0, 'longest_contiguous': 0}
        last = {'train': None, 'validation': None}
        run = {'train': 0, 'validation': 0}
        with (stage/'train.csv').open('w', newline='') as f, (stage/'validation.csv').open('w', newline='') as g:
            writers = {'train': csv.writer(f), 'validation': csv.writer(g)}
            for w in writers.values():
                w.writerow(['open_time_utc','close_time_utc','available_at_utc','open','high','low','close','volume'])
            for t, op, hi, lo, cl, vol in aggregate(rows,start,holdout):
                name = 'train' if t < validation else 'validation'
                writers[name].writerow([iso(t),iso(t+300),'',op,hi,lo,cl,vol])
                b = report['blocks'][name]
                b['received'] += 1
                run[name] = run[name]+1 if last[name] == t-300 else 1
                b['longest_contiguous'] = max(b['longest_contiguous'],run[name])
                last[name] = t
        for name,b in report['blocks'].items():
            b['missing'] = b['expected']-b['received']
            b['csv_sha256'] = sha(stage/(name+'.csv'))
            if b['received'] == 0:
                raise ValueError(f'No complete candles in {name}; no export created')
        (stage/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        if output.exists():
            raise FileExistsError(str(output))
        stage.rename(output)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--synthetic', action='store_true')
    mode.add_argument('--csv', type=Path)
    p.add_argument('--provider')
    p.add_argument('--permission-reference')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.synthetic:
        report=export(synthetic(START,HOLDOUT),args.output,'synthetic','original simulated data')
    else:
        if not args.provider or args.provider.lower() == 'synthetic' or not args.permission_reference:
            p.error('Real CSV requires provider and documented permission for intended ML use')
        report=export(read_minutes(args.csv,START,HOLDOUT),args.output,args.provider,args.permission_reference,sha(args.csv))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
