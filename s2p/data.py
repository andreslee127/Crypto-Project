"""Parse Binance Vision files, build the point-in-time universe and the hourly panel,
and run data-quality checks that fail loudly.

Timestamps: a bar is indexed by its CLOSE time t (= open_time + 1h). A decision "at t" uses the
close at t and everything earlier, and earns the return from t to t+1h.
"""
import io
import json
import zipfile

import numpy as np
import pandas as pd

from s2p import download as dl

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
HOUR_MS = 3_600_000


class DataQualityError(RuntimeError):
    pass


def _read_csv_from_zip(path):
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode()
    has_header = not text[:1].isdigit()           # newer files start with a header row
    return pd.read_csv(io.StringIO(text), header=0 if has_header else None)


def to_ms(t):
    """Binance moved some files to microseconds. Values above 1e14 cannot be milliseconds
    (that would be the year 5138), so they are microseconds. Returns (ms, was_us)."""
    t = np.asarray(t, dtype=np.int64)
    us = t > 10 ** 14
    return np.where(us, t // 1000, t), bool(us.any())


def read_klines(path):
    df = _read_csv_from_zip(path)
    df.columns = KLINE_COLS
    df["open_time"], us1 = to_ms(df["open_time"])
    df["close_time"], us2 = to_ms(df["close_time"])
    df.attrs["microseconds"] = us1 or us2
    return df


def read_funding(path):
    df = _read_csv_from_zip(path)
    df.columns = ["calc_time", "interval_hours", "rate"]
    df["calc_time"], us = to_ms(df["calc_time"])
    df.attrs["microseconds"] = us
    return df


# ---------------------------------------------------------------- universe

def build_universe(cfg):
    """{month: [symbols]}: top-N by quote volume in the PREVIOUS calendar month.

    A symbol qualifies only if it has >= universe_min_days daily bars in that month and a bar on
    its last day (so it was still trading when the decision is made). Non-crypto contracts in
    data.exclude_symbols never qualify. Monthly statistics are cached; ranking is redone each call.
    """
    d = cfg["data"]
    stats_cache = cfg["paths"]["cache"] / "daily_stats.pkl"
    if stats_cache.exists():
        stats = pd.read_pickle(stats_cache)
    else:
        syms = dl.list_symbols(cfg)
        avail = dl.available_months(cfg, syms, "1d")
        rank_months = dl.months_between(d["first_month"], pd.Period(d["last_month"]) - 1)
        rows = []
        for s in syms:
            for m in avail[s]:
                p = cfg["paths"]["raw"] / dl.kline_key(s, "1d", m)
                if m not in rank_months or not p.exists():
                    continue
                k = read_klines(p)
                days = pd.to_datetime(k["open_time"], unit="ms").dt.normalize()
                rows.append({"symbol": s, "month": m, "qv": k["quote_volume"].sum(),
                             "n_days": days.nunique(),
                             "has_last": (days == pd.Period(m).end_time.normalize()).any()})
        stats = pd.DataFrame(rows)
        stats_cache.parent.mkdir(parents=True, exist_ok=True)
        stats.to_pickle(stats_cache)
    stats = stats[~stats.symbol.isin(d.get("exclude_symbols", []))]
    months = dl.months_between(pd.Period(d["first_month"]) + 1, d["last_month"])
    universe = rank_universe(stats, months, d["universe_size"], d["universe_min_days"])
    (cfg["paths"]["cache"] / "universe.json").write_text(json.dumps(universe, indent=0))
    return universe


def rank_universe(stats, months, size, min_days):
    """Universe of month m from the statistics of month m-1 only (point in time)."""
    ok = stats[(stats.n_days >= min_days) & stats.has_last]
    return {m: ok[ok.month == str(pd.Period(m) - 1)].nlargest(size, "qv").symbol.tolist() for m in months}


# ---------------------------------------------------------------- panel

def symbol_months(universe):
    """(symbol, month) pairs to load: universe months plus one warm-up month before each."""
    need = set()
    for m, members in universe.items():
        for s in members:
            need |= {(s, m), (s, str(pd.Period(m) - 1))}
    return sorted(need)


def _holdout_start(cfg):
    return pd.Timestamp(cfg["splits"]["holdout"][0])


def load_panel(cfg, holdout=False, universe=None, raw_reader=None):
    """Hourly panel as a dict of wide DataFrames (index = bar close time, columns = symbols):
    close, quote_volume, funding (rate at settlement hours, 0 otherwise, NaN where no funding
    file), universe (bool). Runs the quality checks and raises DataQualityError on failure.

    THE HOLDOUT IS CUT AWAY HERE, row by row, before any other computation, unless holdout=True.
    `raw_reader(kind, symbol, month)` can replace file reading (used by tests).
    """
    universe = universe if universe is not None else build_universe(cfg)
    reader = raw_reader or (lambda kind, s, m: _read_file(cfg, kind, s, m))
    cutoff = None if holdout else _holdout_start(cfg)
    d = cfg["data"]
    index = pd.date_range(pd.Period(d["first_month"]).start_time + pd.Timedelta(hours=1),
                          pd.Period(d["last_month"]).end_time.floor("h") + pd.Timedelta(hours=1),
                          freq="h")
    if cutoff is not None:
        index = index[index < cutoff]
    index = index[index <= pd.Timestamp(cfg["splits"]["holdout"][1]) + pd.Timedelta(days=1)]   # last usable day

    issues = Issues()
    klines, funding, have_funding = {}, {}, set()
    for s, m in symbol_months(universe):
        if cutoff is not None and pd.Period(m).start_time >= cutoff:
            continue                                      # month entirely in the holdout: never opened
        k = reader("klines", s, m)
        if k is None:
            issues.fail("missing kline file", f"{s} {m}")
            continue
        if k.attrs.get("microseconds"):
            issues.info("files in microseconds (converted)", f"{s} {m}")
        if k.attrs.get("repaired_days"):
            issues.info("truncated monthly file, days added from daily files", f"{s} {m}: {k.attrs['repaired_days']}")
        k = _cut(k, "close_time", cutoff)                 # holdout rows never get past here
        _check_file(k, s, m, issues)
        klines.setdefault(s, []).append(k)
        f = reader("funding", s, m)
        if f is not None:
            funding.setdefault(s, []).append(_cut(f, "calc_time", cutoff))
            have_funding.add((s, m))

    close, qvol = {}, {}
    for s, parts in klines.items():
        k = pd.concat(parts, ignore_index=True)
        k = _dedupe(k, s, issues)
        t = pd.to_datetime(k["close_time"] + 1, unit="ms")
        close[s] = pd.Series(k["close"].to_numpy(float), index=t)
        qvol[s] = pd.Series(k["quote_volume"].to_numpy(float), index=t)
    close = pd.DataFrame(close).reindex(index)
    qvol = pd.DataFrame(qvol).reindex(index)

    fund = pd.DataFrame(0.0, index=index, columns=close.columns)
    for s, parts in funding.items():
        f = pd.concat(parts, ignore_index=True).drop_duplicates("calc_time")
        t = pd.to_datetime(f["calc_time"], unit="ms").dt.floor("h")    # settles on the hour
        rates = pd.Series(f["rate"].to_numpy(float), index=t).groupby(level=0).last()
        fund.loc[rates.index.intersection(index), s] = rates.reindex(rates.index.intersection(index))
    months = index.to_period("M").astype(str)
    for s in fund.columns:                                  # no funding file -> unknown, not zero
        no_file = np.array([(s, m) not in have_funding for m in months])
        fund.loc[no_file, s] = np.nan

    uni = pd.DataFrame(False, index=index, columns=close.columns)
    for m, members in universe.items():
        rows = months == m
        cols = [s for s in members if s in uni.columns]
        uni.loc[rows, cols] = True

    panel = {"close": close, "quote_volume": qvol, "funding": fund, "universe": uni}
    quality_checks(cfg, panel, universe, have_funding, issues)
    return panel


def _read_file(cfg, kind, s, m):
    """Monthly file; for klines, daily files of the same month (repairs) are appended.
    Overlapping bars must agree exactly, or the duplicate check fails."""
    iv = cfg["data"]["interval"]
    key = dl.kline_key(s, iv, m) if kind == "klines" else dl.funding_key(s, m)
    p = cfg["paths"]["raw"] / key
    if not p.exists():
        return None
    if kind == "funding":
        return read_funding(p)
    k = read_klines(p)
    daily = sorted((p.parents[4] / "daily" / "klines" / s / iv).glob(f"{s}-{iv}-{m}-*.zip"))
    if daily:
        parts = [k] + [read_klines(f) for f in daily]
        k = pd.concat(parts, ignore_index=True)
        k.attrs["microseconds"] = any(x.attrs.get("microseconds") for x in parts)
        k.attrs["repaired_days"] = len(daily)
    return k


def _cut(df, col, cutoff):
    if cutoff is None:
        return df
    t = df[col] + (1 if col == "close_time" else 0)
    return df[pd.to_datetime(t, unit="ms") < cutoff]


# ---------------------------------------------------------------- quality checks

class Issues:
    """Collects failures (raise) and information items (print only)."""

    def __init__(self):
        self.items = {}

    def _add(self, level, check, detail):
        self.items.setdefault((level, check), []).append(detail)

    def fail(self, check, detail):
        self._add("FAIL", check, detail)

    def info(self, check, detail):
        self._add("info", check, detail)

    def summary(self):
        rows = [{"level": lv, "check": ck, "count": len(v), "examples": "; ".join(map(str, v[:4]))}
                for (lv, ck), v in sorted(self.items.items())]
        return pd.DataFrame(rows, columns=["level", "check", "count", "examples"])

    @property
    def failed(self):
        return any(lv == "FAIL" for lv, _ in self.items)


def _check_file(k, s, m, issues):
    if (k["open_time"] % HOUR_MS != 0).any():
        issues.fail("open_time not on the hour", f"{s} {m}")
    if (k["close_time"] - k["open_time"] != HOUR_MS - 1).any():
        issues.fail("bar length not 1h", f"{s} {m}")
    months = pd.to_datetime(k["open_time"], unit="ms").dt.to_period("M").astype(str)
    if (months != m).any():
        issues.fail("rows outside the file's month", f"{s} {m}")
    p = k[["open", "high", "low", "close"]]
    bad = ((p <= 0).any(axis=1) | (k.high < p.max(axis=1)) | (k.low > p.min(axis=1))
           | (k.volume < 0) | (k.quote_volume < 0))
    if bad.any():
        issues.fail("OHLC/volume inconsistent", f"{s} {m}: {int(bad.sum())} rows")


def _dedupe(k, s, issues):
    dup = k.duplicated("open_time", keep=False)
    if dup.any():
        exact = k[dup].duplicated(keep=False).all()
        (issues.info if exact else issues.fail)(
            "duplicate bars" + ("" if exact else " with different values"), f"{s}: {int(dup.sum())}")
    return k.drop_duplicates("open_time").sort_values("open_time")


def quality_checks(cfg, panel, universe, have_funding, issues):
    q, d = cfg["data"]["quality"], cfg["data"]
    close, uni = panel["close"], panel["universe"]
    months = close.index.to_period("M").astype(str)

    for m, members in universe.items():
        if (months == m).any() and len(members) != d["universe_size"]:
            issues.fail("universe size", f"{m}: {len(members)}")

    # Gaps, checked per symbol-month in the universe. Only universe months (+ warm-up) are loaded,
    # so time between two membership spells is not a gap. A series that stops inside a month and
    # never comes back is a delisting (kept, reported); data missing before then is a failure.
    for m, members in universe.items():
        rows = months == m
        if not rows.any():
            continue
        for s in members:
            if s not in close.columns:
                continue
            x = close.loc[rows, s]
            last_ever = close[s].last_valid_index()
            if last_ever is None:
                issues.fail("symbol without data", s)
                continue
            if last_ever < x.index[-1]:
                issues.info("delisted / data ends (kept: no survivorship bias)", f"{s} {last_ever:%Y-%m-%d %H}h")
                x = x.loc[:last_ever]
            if x.isna().mean() > q["max_missing_frac"]:
                issues.fail("missing hours in universe month", f"{s} {m}: {x.isna().mean():.1%}")
            run = x.isna().astype(int).groupby(x.notna().cumsum()).sum().max() if len(x) else 0
            if run > q["max_inner_gap_bars"]:
                issues.fail("long gap in universe month", f"{s} {m}: {run} bars")
            elif run > 0:
                issues.info("short gaps in universe months (bars)", f"{s} {m}: {run}")

    # funding coverage for in-universe symbol-months
    pairs = [(s, m) for m, mem in universe.items() if (months == m).any() for s in mem]
    miss = [p for p in pairs if p not in have_funding]
    if pairs and len(miss) / len(pairs) > q["max_funding_missing_frac"]:
        issues.fail("funding files missing", f"{len(miss)}/{len(pairs)} e.g. {miss[:3]}")
    elif miss:
        issues.info("funding files missing (signal NaN, no funding charged)", f"{len(miss)}/{len(pairs)}")
    big = panel["funding"].abs() > 0.01
    if big.any().any():
        issues.info("|funding rate| > 1% per settlement", f"{int(big.sum().sum())} settlements")

    r = close.pct_change(fill_method=None)
    ext = r.abs() > q["extreme_return"]
    for t, s in zip(*np.nonzero(ext.to_numpy())):
        issues.info("extreme 1h return", f"{close.columns[s]} {close.index[t]:%Y-%m-%d %H}h {r.iat[t, s]:+.0%}")

    summary = issues.summary()
    print("DATA QUALITY SUMMARY")
    print(summary.to_string(index=False) if len(summary) else "  no issues")
    panel["quality"] = summary
    if issues.failed:
        raise DataQualityError("data-quality checks FAILED:\n" + summary[summary.level == "FAIL"].to_string())
