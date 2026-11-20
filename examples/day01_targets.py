"""Day 1: validated candles and historical forward-return labels.

Research only. A future outcome may be used as a retrospective TRAINING LABEL,
never as a feature available at decision time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Sequence

FIVE_MINUTES = timedelta(minutes=5)


@dataclass(frozen=True)
class Candle:
    instrument: str
    session: str
    start_utc: datetime
    end_utc: datetime
    available_at_utc: datetime
    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        times = (self.start_utc, self.end_utc, self.available_at_utc)
        if any(t.tzinfo is None or t.utcoffset() != timedelta(0) for t in times):
            raise ValueError("Candle times must be timezone-aware UTC")
        if self.end_utc - self.start_utc != FIVE_MINUTES:
            raise ValueError("Day 1 fixture expects 5-minute candles")
        if self.available_at_utc < self.end_utc:
            raise ValueError("Publication cannot precede candle end")
        prices = (self.open, self.high, self.low, self.close)
        if not all(math.isfinite(p) and p > 0 for p in prices):
            raise ValueError("Prices must be finite and positive")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("OHLC envelope invalid")
        if not self.instrument or not self.session:
            raise ValueError("Instrument and session must be present")


def log_return(current_close: float, future_close: float) -> float:
    """Retrospective return label, not a prediction."""
    if not (math.isfinite(current_close) and math.isfinite(future_close)):
        raise ValueError("Closing prices must be finite")
    if current_close <= 0 or future_close <= 0:
        raise ValueError("Closing prices must be positive")
    return math.log(future_close / current_close)


def up_label(ret: float) -> int:
    """1=UP; 0=NOT_UP (down or unchanged)."""
    if not math.isfinite(ret):
        raise ValueError("Return must be finite")
    return int(ret > 0)


def target_at(candles: Sequence[Candle], index: int, horizon_bars: int) -> float | None:
    """Compute retrospective target for exactly consecutive 5m bars.

    Return None for missing data or a session boundary. A future close is
    NOT a feature and cannot be accessed before it has happened.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be positive")
    if index < 0 or index >= len(candles):
        raise IndexError("Invalid candle index")
    end = index + horizon_bars
    if end >= len(candles):
        return None
    current = candles[index]
    prior = current
    for nxt in candles[index + 1:end + 1]:
        if (nxt.instrument != current.instrument
                or nxt.session != current.session
                or nxt.start_utc != prior.end_utc):
            return None
        prior = nxt
    return log_return(current.close, prior.close)


def feature_is_available(candle: Candle, decision_time_utc: datetime) -> bool:
    """A completed candle is usable only after its publication time."""
    if decision_time_utc.tzinfo is None:
        raise ValueError("Decision time must be timezone-aware")
    return candle.available_at_utc <= decision_time_utc


if __name__ == "__main__":
    result = log_return(100_000, 100_100)
    print("Illustrative historical example — NOT real market data")
    print(f"Log return: {result:.10f}; log-return percent: {result*100:.6f}%")
    print(f"UP label: {up_label(result)}")
