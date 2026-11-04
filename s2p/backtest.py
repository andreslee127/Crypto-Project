"""Layer 2: trading policies and PnL accounting.

Every policy is written as one rule, applied bar by bar to the (drifted) position x in USD:
    x_new = clip( x + M * (aim - x), lo, hi ),  and x_new = 0 where the asset is not tradeable.
    naive : M = 1,   aim = Markowitz target, no clip
    EMA   : M = eta, aim = Markowitz target, no clip
    GP    : M and aim from the Riccati solution (per asset, closed form, see lq.scalar_policy)
    band  : M = 0, clip into the no-trade band [lo, hi] from the proportional-cost DP
Accounting per bar (USD):  net = gross - linear cost - impact cost - funding.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from s2p import dp, lq


def simulate(M, aim, lo, hi, tradeable, fwd_ret, fwd_fund, c_lin, lam):
    """Run one policy. All inputs are (T, N) arrays; returns a per-bar DataFrame of USD flows."""
    T, N = M.shape
    x = np.zeros(N)
    out = np.zeros((T, 7))
    for t in range(T):
        x_new = np.clip(x + M[t] * (aim[t] - x), lo[t], hi[t])
        x_new = np.where(tradeable[t], x_new, 0.0)
        dx = x_new - x
        lin = c_lin[t] @ np.abs(dx)
        imp = 0.5 * (lam[t] * dx * dx).sum()
        gross = x_new @ fwd_ret[t]
        fund = x_new @ fwd_fund[t]                 # longs pay positive funding
        out[t] = gross, lin, imp, fund, gross - lin - imp - fund, np.abs(dx).sum(), np.abs(x_new).sum()
        x = x_new * (1.0 + fwd_ret[t])             # position drifts with the price
    return pd.DataFrame(out, columns=["gross", "cost_linear", "cost_impact", "funding", "net",
                                      "traded", "gross_exposure"])


# ---------------------------------------------------------------- policy inputs

def band_tables(cfg, rho_g, beta, c_grid):
    """No-trade band edges L(c, g), U(c, g) in normalised units (one DP per cost level)."""
    b = cfg["band_dp"]
    x = np.linspace(-b["x_max"], b["x_max"], b["n_x"])
    f = np.linspace(-b["f_max"], b["f_max"], b["n_f"])
    L, U = [], []
    for c in c_grid:
        l, u, _ = dp.solve_band(1.0, 1.0, c, rho_g, np.sqrt(1 - rho_g ** 2), beta, x, f, b["gh_nodes"])
        L.append(l)
        U.append(u)
    return {"c_grid": np.asarray(c_grid, float), "f_grid": f, "L": np.array(L), "U": np.array(U)}


def band_lookup(tables, c_hat, g):
    """Bilinear interpolation of the band edges in (log c, g); inputs clipped to the grid."""
    lc = np.log(tables["c_grid"])
    pts = np.stack([np.clip(np.log(np.maximum(c_hat, 1e-12)), lc[0], lc[-1]),
                    np.clip(g, tables["f_grid"][0], tables["f_grid"][-1])], axis=-1)
    Li = RegularGridInterpolator((lc, tables["f_grid"]), tables["L"])
    Ui = RegularGridInterpolator((lc, tables["f_grid"]), tables["U"])
    return Li(pts), Ui(pts)


def policy_inputs(kind, param, ctx):
    """(M, aim, lo, hi) arrays for a policy. ctx holds the model inputs (see run_all.context)."""
    vol, score, gamma = ctx["vol"], ctx["score"], ctx["gamma"]
    shape = vol.shape
    ones, inf = np.ones(shape), np.full(shape, np.inf)
    mkw = score / (gamma * vol)                    # Markowitz: mu / (gamma sigma^2), mu = sigma * score
    if kind == "naive":
        return ones, mkw, -inf, inf
    if kind == "ema":
        return param * ones, mkw, -inf, inf
    if kind == "gp":
        s = gamma * vol ** 2
        lam_p = ctx["lam"] + param * 2 * ctx["c_lin"] / ctx["q_ref"]     # quadratic proxy of all costs
        rho = ctx["rho"] ** (1.0 / ctx.get("decay_mult", 1.0))          # assumed decay (sensitivity)
        M, w = lq.scalar_policy(s, lam_p, ctx["discount"], rho)
        mkw_k = ctx["X"] * ctx["beta"][:, None, :] / (gamma * vol)[..., None]   # per-signal Markowitz
        aim = (w * mkw_k).sum(axis=-1)
        return M, aim, -inf, inf
    if kind == "band":
        kappa = ctx["kappa"][:, None]
        c_hat = param * ctx["c_lin"] / (vol * kappa)                   # normalised cost
        Lz, Uz = band_lookup(ctx["band_tables"], c_hat, score / kappa)
        scale = kappa / (gamma * vol)                                    # x = scale * z
        return np.zeros(shape), np.zeros(shape), scale * Lz, scale * Uz
    raise ValueError(kind)


def run_policy(kind, param, ctx):
    """Simulate a policy. The costs in ctx are both what the policy assumes and what is charged."""
    M, aim, lo, hi = policy_inputs(kind, param, ctx)
    ok = ctx["tradeable"]
    zero_off = lambda a: np.where(ok, np.nan_to_num(a, nan=0.0, posinf=np.inf, neginf=-np.inf), 0.0)
    res = simulate(zero_off(M), zero_off(aim), np.where(ok, lo, -np.inf), np.where(ok, hi, np.inf), ok,
                   np.nan_to_num(ctx["fwd_ret"]), np.nan_to_num(ctx["fwd_fund"]),
                   np.nan_to_num(ctx["c_lin"]), np.nan_to_num(ctx["lam"]))
    res.index = ctx["index"]
    if not np.isfinite(res.to_numpy()).all():
        raise FloatingPointError(f"non-finite PnL in policy {kind}({param})")
    return res
