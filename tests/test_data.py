"""Data loading and quality checks: they must pass clean data and fail loudly on bad data."""
import copy

import numpy as np
import pandas as pd
import pytest

from s2p import data
from s2p.config import load_config

SYMS = ["AAAUSDT", "BBBUSDT"]
MONTHS = ["2021-01", "2021-02", "2021-03"]
UNIVERSE = {"2021-02": SYMS, "2021-03": SYMS}


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["data"].update(first_month="2021-01", last_month="2021-03", universe_size=2)
    c["splits"]["holdout"] = ["2021-03-20", "2021-03-31"]
    return c


def klines(month, start_price=100.0, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range(pd.Period(month).start_time, pd.Period(month).end_time, freq="h")
    ms = (t.asi8 // 10 ** 6)
    close = start_price * np.exp(np.cumsum(0.01 * rng.standard_normal(len(t))))
    df = pd.DataFrame({"open_time": ms, "open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": 1.0, "close_time": ms + data.HOUR_MS - 1,
                       "quote_volume": 1e6, "count": 1, "taker_buy_volume": 0.5,
                       "taker_buy_quote_volume": 5e5, "ignore": 0})
    df.attrs["microseconds"] = False
    return df


def funding(month):
    t = pd.date_range(pd.Period(month).start_time, pd.Period(month).end_time, freq="8h")
    return pd.DataFrame({"calc_time": t.asi8 // 10 ** 6 + 3, "interval_hours": 8, "rate": 1e-4})


def reader(modify=None):
    def read(kind, s, m):
        df = klines(m, seed=hash((s, m)) % 1000) if kind == "klines" else funding(m)
        return modify(kind, s, m, df) if modify else df
    return read


def test_clean_data_passes_and_holdout_is_cut(cfg):
    p = data.load_panel(cfg, universe=UNIVERSE, raw_reader=reader())
    assert p["close"].index.max() < pd.Timestamp("2021-03-20")       # holdout rows never loaded
    u = p["universe"]
    assert u.loc[pd.Timestamp("2021-02-10 05:00")].all() and not u.loc[pd.Timestamp("2021-01-10 05:00")].any()
    assert (p["funding"].loc[pd.Timestamp("2021-02-01 08:00")] == 1e-4).all()
    p_all = data.load_panel(cfg, holdout=True, universe=UNIVERSE, raw_reader=reader())
    assert p_all["close"].index.max() >= pd.Timestamp("2021-03-31 23:00")


def test_microseconds_are_detected_and_converted():
    t = np.array([1609459200000, 1609459200000 * 1000])
    ms, was_us = data.to_ms(t)
    assert was_us and (ms == 1609459200000).all()


@pytest.mark.parametrize("problem", ["conflicting_duplicates", "long_gap", "bad_ohlc", "missing_file", "off_hour"])
def test_quality_checks_fail_loudly(cfg, problem):
    def modify(kind, s, m, df):
        if kind != "klines" or s != "AAAUSDT" or m != "2021-02":
            return df
        if problem == "conflicting_duplicates":
            dup = df.iloc[[10]].copy()
            dup["close"] *= 1.5
            dup["high"] *= 1.5
            return pd.concat([df, dup])
        if problem == "long_gap":
            return df.drop(df.index[100:140])
        if problem == "bad_ohlc":
            df.loc[df.index[5], "low"] = df.loc[df.index[5], "high"] * 2
            return df
        if problem == "missing_file":
            return None
        if problem == "off_hour":
            df.loc[df.index[3], ["open_time", "close_time"]] += 1000
            return df
    with pytest.raises(data.DataQualityError):
        data.load_panel(cfg, universe=UNIVERSE, raw_reader=reader(modify))


def test_exact_duplicates_are_dropped_not_fatal(cfg):
    def modify(kind, s, m, df):
        return pd.concat([df, df.iloc[:5]]) if kind == "klines" else df
    p = data.load_panel(cfg, universe=UNIVERSE, raw_reader=reader(modify))
    assert "duplicate bars" in set(p["quality"]["check"])


def test_universe_is_point_in_time():
    st = pd.DataFrame([
        {"symbol": "A", "month": "2021-01", "qv": 10, "n_days": 31, "has_last": True},
        {"symbol": "B", "month": "2021-01", "qv": 20, "n_days": 31, "has_last": True},
        {"symbol": "C", "month": "2021-01", "qv": 99, "n_days": 31, "has_last": False},   # delisted
        {"symbol": "A", "month": "2021-02", "qv": 50, "n_days": 28, "has_last": True},
        {"symbol": "B", "month": "2021-02", "qv": 1, "n_days": 28, "has_last": True},
    ])
    u = data.rank_universe(st, ["2021-02", "2021-03"], size=1, min_days=25)
    assert u == {"2021-02": ["B"], "2021-03": ["A"]}       # Feb ranked on January only
    st.loc[st.month == "2021-02", "qv"] = [0, 1000]          # changing February data...
    assert data.rank_universe(st, ["2021-02"], 1, 25) == {"2021-02": ["B"]}   # ...cannot change Feb
