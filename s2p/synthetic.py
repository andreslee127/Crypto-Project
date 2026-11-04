"""Synthetic hourly market in the same format as data.load_panel. FOR TESTING CODE ONLY:
anything produced from it must be labelled SYNTHETIC.

A hidden AR(1) alpha a_it (half-life 24h) drives next-hour returns with IC ~ `ic` and is
partly visible in funding rates, so the funding signal should be found and the others not.
One symbol delists mid-sample and one lists late, to exercise the universe logic.
"""
import numpy as np
import pandas as pd


def rank_universe(qvol, close, size, min_days):
    """{month: members}: top-`size` by quote volume over the previous month, among symbols with
    >= min_days days of data in that month and data on its last day."""
    months = qvol.index.to_period("M")
    out = {}
    for m in pd.unique(months)[1:]:
        prev = (months == m - 1)
        vol = qvol[prev].sum()
        days = close[prev].notna().groupby(close.index[prev].normalize()).any().sum()
        alive = close[prev].iloc[-24:].notna().any()
        ok = (days >= min_days) & alive
        out[str(m)] = vol[ok].nlargest(size).index.tolist()
    return out


def make_panel(cfg, n_symbols=30, ic=0.03, seed=0, start=None, end=None):
    rng = np.random.default_rng(seed)
    d = cfg["data"]
    start = start or pd.Period(d["first_month"]).start_time + pd.Timedelta(hours=1)
    end = end or pd.Timestamp(cfg["splits"]["holdout"][0]) - pd.Timedelta(hours=1)
    index = pd.date_range(start, end, freq="h")
    T, N = len(index), n_symbols
    syms = [f"SYN{i:02d}USDT" for i in range(N)]

    rho = 0.5 ** (1 / 24)
    a = np.empty((T, N))
    a[0] = rng.standard_normal(N)
    shocks = rng.standard_normal((T, N)) * np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        a[t] = rho * a[t - 1] + shocks[t]
    sigma = rng.uniform(0.006, 0.015, N)
    beta = rng.uniform(0.7, 1.3, N)
    market = 0.006 * rng.standard_normal(T)
    eps = rng.standard_normal((T, N))
    r = np.zeros((T, N))
    r[1:] = beta * market[1:, None] + sigma * (ic * a[:-1] + np.sqrt(1 - ic ** 2) * eps[1:])
    close = 100 * np.exp(np.cumsum(np.log1p(r), axis=0))

    level = np.exp(rng.uniform(np.log(2e7), np.log(2e9), N))          # USD per hour
    qvol = level * np.exp(0.4 * rng.standard_normal((T, N)))
    funding = np.zeros((T, N))
    settle = index.hour % 8 == 0
    funding[settle] = 1e-4 - 1e-4 * (a[settle] + rng.standard_normal((settle.sum(), N)))

    close = pd.DataFrame(close, index=index, columns=syms)
    qvol = pd.DataFrame(qvol, index=index, columns=syms)
    funding = pd.DataFrame(funding, index=index, columns=syms)
    close.iloc[int(0.6 * T):, 0] = np.nan                 # SYN00 delists
    close.iloc[: int(0.3 * T), 1] = np.nan                # SYN01 lists late
    qvol[close.isna()] = np.nan
    funding[close.isna()] = np.nan

    members = rank_universe(qvol, close, d["universe_size"], d["universe_min_days"])
    uni = pd.DataFrame(False, index=index, columns=syms)
    months = index.to_period("M").astype(str)
    for m, mem in members.items():
        uni.loc[months == m, mem] = True
    quality = pd.DataFrame([{"level": "info", "check": "SYNTHETIC DATA", "count": 1, "examples": ""}])
    return {"close": close, "quote_volume": qvol, "funding": funding, "universe": uni, "quality": quality}
