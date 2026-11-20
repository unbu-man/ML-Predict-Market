"""Day 1 unit tests. Run: python3 -m unittest discover -s tests -p 'test_day01*.py' -v"""

from datetime import datetime, timedelta, timezone
import math
import unittest

from examples.day01_targets import Candle, log_return, up_label, target_at, feature_is_available

UTC = timezone.utc
BASE = datetime(2026, 11, 20, 0, 0, tzinfo=UTC)


def make_candle(index: int, *, session: str = "regular", start_offset_min: int = 0,
                publication_delay_seconds: int = 2, close: float | None = None) -> Candle:
    start = BASE + timedelta(minutes=index * 5 + start_offset_min)
    end = start + timedelta(minutes=5)
    price = 100 + index if close is None else close
    return Candle("BTC/USD", session, start, end,
                  end + timedelta(seconds=publication_delay_seconds),
                  price, price + 1, price - 1, price)


class Day01Tests(unittest.TestCase):
    def test_up_example(self):
        r = log_return(100_000, 100_100)
        self.assertAlmostEqual(r, math.log(1.001))
        self.assertEqual(up_label(r), 1)

    def test_down_example(self):
        r = log_return(100_000, 99_800)
        self.assertAlmostEqual(r, math.log(0.998))
        self.assertEqual(up_label(r), 0)

    def test_unchanged_is_not_up(self):
        self.assertEqual(log_return(42, 42), 0)
        self.assertEqual(up_label(0), 0)

    def test_invalid_price(self):
        for price in (0, -1, float("nan"), float("inf")):
            with self.subTest(price=price), self.assertRaises(ValueError):
                log_return(100, price)

    def test_invalid_ohlc(self):
        good = make_candle(0)
        with self.assertRaises(ValueError):
            Candle(good.instrument, good.session, good.start_utc, good.end_utc,
                   good.available_at_utc, 100, 99, 98, 101)

    def test_no_lookahead_until_published(self):
        candle = make_candle(0)
        self.assertFalse(feature_is_available(candle, BASE + timedelta(minutes=5)))
        self.assertTrue(feature_is_available(candle, BASE + timedelta(minutes=5, seconds=2)))

    def test_next_five_minute_and_one_hour_labels(self):
        candles = [make_candle(i) for i in range(13)]
        self.assertAlmostEqual(target_at(candles, 0, 1), math.log(101 / 100))
        self.assertAlmostEqual(target_at(candles, 0, 12), math.log(112 / 100))
        self.assertIsNone(target_at(candles, 2, 12))

    def test_missing_candle_rejects_horizon(self):
        self.assertIsNone(target_at([make_candle(0), make_candle(2)], 0, 1))

    def test_crossing_session_rejects_horizon(self):
        self.assertIsNone(target_at([make_candle(0), make_candle(1, session="reopened")], 0, 1))

    def test_invalid_horizon(self):
        with self.assertRaises(ValueError):
            target_at([make_candle(0)], 0, 0)

    def test_reject_naive_timestamp(self):
        with self.assertRaises(ValueError):
            feature_is_available(make_candle(0), datetime(2026, 11, 20, 0, 5))


if __name__ == "__main__":
    unittest.main()
