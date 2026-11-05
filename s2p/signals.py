"""Signal library. Every quantity at row t uses only data with timestamp <= t.

Signals are cross-sectional ranks inside the point-in-time universe, scaled to unit variance,
and are paired with the return from t to t+1 (one bar later).
"""
import numpy as np
import pandas as pd


def prices(panel, cfg):
    """Close prices with short gaps forward-filled (at most max_ffill_bars)."""
    return panel["close"].ffill(limit=cfg["data"]["max_ffill_bars"])


def returns(close_ff):
    """Simple return over the bar ending at t."""
    return close_ff.pct_change(fill_method=None)


def ewma_vol(r, halflife):
    """Per-bar volatility from an EWMA of squared returns (data <= t)."""
    return np.sqrt((r ** 2).ewm(halflife=halflife, min_periods=halflife).mean())


def xs_rank(raw, mask):
    """Cross-sectional rank within `mask`, mapped to (-0.5, 0.5) and scaled to unit variance."""
    x = raw.where(mask)
    n = x.notna().sum(axis=1)
    u = (x.rank(axis=1) - 0.5).div(n, axis=0) - 0.5
    return u * np.sqrt(12.0)


def raw_signal(spec, logp, r, funding, qvol, vol, market):
    """Raw (un-ranked) value of one signal specified in advance in config.yaml."""
    L = spec["lookback"]
    kind = spec["kind"]
    if kind == "momentum":
        return logp - logp.shift(L)
    if kind == "reversal":
        return -(logp - logp.shift(L))
    if kind == "funding":                     # what a long paid over the last L hours
        return -funding.rolling(L, min_periods=L).sum()
    if kind == "volume":
        recent = qvol.rolling(L, min_periods=L).sum()
        base = qvol.rolling(spec["baseline"], min_periods=spec["baseline"]).mean() * L
        return np.log(recent / base)
    if kind == "residual":                    # minus the vol-scaled market-residual return
        W = spec["beta_window"]
        cov = r.rolling(W, min_periods=W).cov(market)
        beta = cov.div(market.rolling(W, min_periods=W).var(), axis=0)
        # residual over the last L bars, using the beta estimated at t
        resid = r.rolling(L, min_periods=L).sum() - beta.mul(market.rolling(L, min_periods=L).sum(), axis=0)
        return -resid / (vol * np.sqrt(L))
    raise ValueError(kind)


def build(panel, cfg):
    """Returns dict with: signals {name: DataFrame}, vol, fwd_ret, fwd_funding, tradeable."""
    close = prices(panel, cfg)
    r = returns(close)
    vol = ewma_vol(r, cfg["model"]["vol_halflife"])
    uni = panel["universe"]
    market = r.where(uni).mean(axis=1)                    # equal-weight universe return
    logp = np.log(close)
    tradeable = uni & close.notna() & vol.notna() & (vol > 0)
    signals = {}
    for name, spec in cfg["signals"].items():
        raw = raw_signal(spec, logp, r, panel["funding"], panel["quote_volume"], vol, market)
        raw = raw.replace([np.inf, -np.inf], np.nan)
        signals[name] = xs_rank(raw, tradeable & raw.notna())
    return {
        "signals": signals,
        "vol": vol,
        "ret": r,
        "fwd_ret": r.shift(-1),                           # return from t to t+1: target / PnL only
        "fwd_funding": panel["funding"].shift(-1),        # settles at t+1 on the position held over (t, t+1]
        "tradeable": tradeable,
        "quote_volume": panel["quote_volume"],
    }
