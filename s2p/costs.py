"""Transaction-cost model. ALL LEVELS ARE ASSUMPTIONS: kline data contains no quotes, so spreads
and impact cannot be measured; only volume and volatility (used to scale impact) are observed.

Cost of trading dx dollars of asset i at bar t:
    c * |dx|  +  1/2 * lam_i,t * dx^2
with c = fee + half-spread (fraction of notional), and lam calibrated so that the quadratic
matches the square-root law  cost/notional = Y * sigma_daily * sqrt(|dx| / ADV)  at the
reference trade size q_ref:  1/2 lam q_ref^2 = Y sigma q_ref^1.5 / sqrt(ADV).
"""
import numpy as np
import pandas as pd


def adv(quote_volume, days):
    """Trailing average daily quote volume in USD (data <= t)."""
    return quote_volume.rolling(24 * days, min_periods=12 * days).sum() / days


def cost_inputs(d, cfg, fee=1.0, spread=1.0, impact=1.0):
    """Per-bar, per-asset linear cost c (fraction) and quadratic impact lam (1/USD).
    fee, spread, impact are multipliers on the configured levels (for sensitivity tables)."""
    k = cfg["costs"]
    c = (fee * k["fee_bps"] + spread * k["half_spread_bps"]) * 1e-4
    q_ref = k["q_ref_frac_aum"] * cfg["portfolio"]["aum_usd"]
    sigma_daily = d["vol"] * np.sqrt(24)
    a = adv(d["quote_volume"], k["adv_days"])
    lam = impact * 2 * k["impact_coef"] * sigma_daily / np.sqrt(a.where(a > 0) * q_ref)
    # Zero volume = halted market (not tradeable, see run_all). A forced exit there is charged
    # the last valid impact coefficient (forward fill uses only data <= t).
    lam = lam.ffill()
    c_lin = pd.DataFrame(c, index=lam.index, columns=lam.columns)
    return c_lin, lam, q_ref


def sqrt_law_cost(dx, sigma_daily, adv_usd, Y=1.0):
    """The square-root-law cost the quadratic approximates (used in tests and documentation)."""
    return Y * sigma_daily * np.abs(dx) ** 1.5 / np.sqrt(adv_usd)
