"""Layer 1: does each signal predict returns? IC, ICIR, decay, single-signal portfolios,
and the walk-forward ridge combination that turns signals into expected returns."""
import numpy as np
import pandas as pd
from scipy import stats as st

from s2p import stats


def ic_series(sig, fwd, min_assets=10):
    """Cross-sectional Spearman IC per bar: correlation of ranks of signal and next return."""
    m = sig.notna() & fwd.notna()
    s = sig.where(m).rank(axis=1)
    f = fwd.where(m).rank(axis=1)
    s = s.sub(s.mean(axis=1), axis=0)
    f = f.sub(f.mean(axis=1), axis=0)
    ic = (s * f).sum(axis=1) / np.sqrt((s ** 2).sum(axis=1) * (f ** 2).sum(axis=1))
    return ic.where(m.sum(axis=1) >= min_assets)


def pearson_ic(sig, y, min_assets=10):
    """Cross-sectional Pearson correlation per bar (the relation the ridge regression fits)."""
    m = sig.notna() & y.notna()
    s = sig.where(m)
    f = y.where(m)
    s = s.sub(s.mean(axis=1), axis=0)
    f = f.sub(f.mean(axis=1), axis=0)
    ic = (s * f).sum(axis=1) / np.sqrt((s ** 2).sum(axis=1) * (f ** 2).sum(axis=1))
    return ic.where(m.sum(axis=1) >= min_assets)


def ar1(sig):
    """Pooled lag-1 autocorrelation of a signal panel (same asset, consecutive bars)."""
    a, b = sig.to_numpy(), sig.shift(1).to_numpy()
    ok = ~np.isnan(a) & ~np.isnan(b)
    return np.corrcoef(a[ok], b[ok])[0, 1]


def half_life(rho):
    return np.log(0.5) / np.log(rho) if 0 < rho < 1 else np.nan


def single_signal_portfolio(sig, vol, fwd_ret):
    """Gross-leverage-1 portfolio with weights proportional to signal / vol.
    Returns per-bar gross return and turnover (sum |dw|)."""
    w = sig.div(vol).fillna(0.0)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    ret = (w * fwd_ret.fillna(0.0)).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    return ret, turnover


def signal_table(signals, d, window, cfg, rng):
    """One row per signal on the decision rows in `window` (a boolean index mask)."""
    bpy = cfg["portfolio"]["bars_per_year"]
    sc = cfg["stats"]
    c_lin = (cfg["costs"]["fee_bps"] + cfg["costs"]["half_spread_bps"]) * 1e-4
    rows = []
    for name, sig in signals.items():
        s = sig[window]
        ics = {}
        for h in sc["ic_horizons"]:                       # return over bar (t+h-1, t+h]
            ics[h] = ic_series(s, d["fwd_ret"].shift(-(h - 1))[window]).dropna()
        ic = ics[1]
        if len(ic) == 0:
            raise ValueError(f"signal {name} has no IC observations in this window")
        t = stats.newey_west_t(ic.to_numpy(), sc["nw_lags"])
        rho = ar1(s)
        h_last = sc["ic_horizons"][-1]
        ratio = ics[h_last].mean() / ic.mean()
        ic_hl = (h_last - 1) * np.log(0.5) / np.log(ratio) if 0 < ratio < 1 else np.nan
        ret, turn = single_signal_portfolio(s, d["vol"][window], d["fwd_ret"][window])
        sr, lo, hi = stats.sharpe_ci(ret.to_numpy(), bpy, sc["block"], sc["n_boot"], rng)
        net = ret - c_lin * turn
        # diagnostics: linear (not rank) IC on vol-scaled returns, and how much the gross PnL
        # depends on extreme hours (returns clipped at +-5%; NOT a tradeable number)
        pearson = pearson_ic(s, (d["fwd_ret"] / d["vol"])[window]).dropna().mean()
        ret_clip, _ = single_signal_portfolio(s, d["vol"][window], d["fwd_ret"][window].clip(-0.05, 0.05))
        row = {"signal": name, "mean_IC": ic.mean(), "IC_t_NW": t, "pearson_IC_volscaled": pearson,
               "p_value": 2 * st.norm.sf(abs(t)),
               "ICIR_annual": ic.mean() / ic.std() * np.sqrt(bpy),
               **{f"IC_h{h}": ics[h].mean() for h in sc["ic_horizons"][1:]},
               "signal_AR1": rho, "signal_half_life_h": half_life(rho), "IC_half_life_h": ic_hl,
               "gross_SR": sr, "gross_SR_lo": lo, "gross_SR_hi": hi,
               "turnover_per_year": turn.mean() * bpy,
               "breakeven_cost_bps": ret.mean() / turn.mean() * 1e4,
               "net_SR_1x_linear": stats.sharpe(net.to_numpy(), bpy),
               "diag_gross_SR_ret_clip5pct": stats.sharpe(ret_clip.to_numpy(), bpy),
               "n_bars": len(ic)}
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = stats.holm(out["p_value"].to_numpy())
    return out


def ic_robustness(signals, d, window, cfg, rng):
    """t-statistics of the mean IC under several dependence corrections, one row per signal.

    The IC series is hourly and its target (the next 1-hour return) does not overlap, but the
    signals use overlapping look-backs, so ICs can be autocorrelated. Methods:
      NW(L)      Newey-West (Bartlett) HAC standard error with L hourly lags
      boot_24h / boot_168h   circular block bootstrap of the mean IC (1-day / 1-week blocks)
      FM_daily   mean of daily-average ICs, Newey-West with 5 daily lags
      FM_weekly  mean of weekly-average ICs, plain t-test on weeks (weeks treated as independent)
    p_conservative is the largest p-value across methods; Holm is applied to it.
    """
    sc = cfg["stats"]
    rows = []
    for name, sig in signals.items():
        ic = ic_series(sig[window], d["fwd_ret"][window]).dropna()
        x = ic.to_numpy()
        row = {"signal": name, "mean_IC": x.mean(), "n_hours": len(x),
               "IC_autocorr_lag1": np.corrcoef(x[1:], x[:-1])[0, 1]}
        for L in [0, 4, 24, 48, 168, 336]:
            row[f"t_NW{L}"] = stats.newey_west_t(x, L)
        for b in [24, 168]:
            se = stats.bootstrap([x], np.mean, b, sc["n_boot"], rng).std(ddof=1)
            row[f"t_boot_{b}h"] = x.mean() / se
        daily = ic.groupby(ic.index.floor("D")).mean().to_numpy()
        weekly = ic.groupby(ic.index.to_period("W")).mean().to_numpy()
        row["t_FM_daily_NW5"] = stats.newey_west_t(daily, 5)
        row["t_FM_weekly"] = weekly.mean() / (weekly.std(ddof=1) / np.sqrt(len(weekly)))
        row["n_weeks"] = len(weekly)
        tcols = [k for k in row if k.startswith("t_")]
        # normal p for the large-sample methods, Student t (weeks - 1 df) for the weekly test
        pvals = [2 * (st.t.sf(abs(row[k]), len(weekly) - 1) if k == "t_FM_weekly" else st.norm.sf(abs(row[k])))
                 for k in tcols]
        row["min_abs_t"] = min(abs(row[k]) for k in tcols)
        row["p_conservative"] = max(pvals)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm_conservative"] = stats.holm(out["p_conservative"].to_numpy())
    return out


# ---------------------------------------------------------------- ridge combination

def stack(signals, names):
    """(T, N, K) array of signals; NaN -> 0 (a missing rank is treated as neutral)."""
    return np.nan_to_num(np.stack([signals[n].to_numpy() for n in names], axis=-1))


def ridge_target(d, clip):
    """Vol-scaled next-bar return, demeaned across the universe, winsorised at +-clip."""
    y = d["fwd_ret"] / d["vol"]
    y = y.where(d["tradeable"])
    y = y.sub(y.mean(axis=1), axis=0)
    return y.clip(-clip, clip)


def monthly_moments(X, y, valid, months):
    """Per decision-month sums X'X, X'y and counts (fitting data for later months)."""
    out = {}
    yv = np.nan_to_num(y)
    ok = valid & ~np.isnan(y)
    for m in pd.unique(months):
        rows = months == m
        Xm = X[rows][ok[rows]]
        ym = yv[rows][ok[rows]]
        out[m] = (Xm.T @ Xm, Xm.T @ ym, len(ym))
    return out


def walk_forward(moments, months_order, first_month, alpha):
    """Ridge coefficients for each month >= first_month, fitted on ALL earlier months only.

    Returns {month: (beta, kappa)}, kappa = in-sample std of the fitted score X beta.
    """
    K = next(iter(moments.values()))[0].shape[0]
    XtX, Xty, n = np.zeros((K, K)), np.zeros(K), 0
    fits = {}
    for m in months_order:
        if m >= first_month and n > 0:
            S = XtX / n
            beta = np.linalg.solve(S + alpha * np.eye(K), Xty / n)
            fits[m] = (beta, np.sqrt(beta @ S @ beta))
        A, b, k = moments[m]                      # month m joins the fit only after it is over
        XtX, Xty, n = XtX + A, Xty + b, n + k
    return fits


def predict(X, months, fits):
    """Score (expected vol-scaled return) and per-row beta/kappa; NaN before the first fit."""
    T, N, K = X.shape
    beta = np.full((T, K), np.nan)
    kappa = np.full(T, np.nan)
    for m, (b, k) in fits.items():
        rows = months == m
        beta[rows], kappa[rows] = b, k
    score = np.einsum("tnk,tk->tn", X, beta)
    return score, beta, kappa
