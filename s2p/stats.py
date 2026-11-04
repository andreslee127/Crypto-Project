"""Statistics: Sharpe ratios with block-bootstrap confidence intervals, Newey-West t-stats,
Holm multiple-testing correction, and the selection-bias haircut (deflated Sharpe ratio)."""
import numpy as np
import pandas as pd
from scipy import stats as st

EULER_GAMMA = 0.5772156649015329


def sharpe(r, bars_per_year):
    r = np.asarray(r, float)
    sd = r.std(ddof=1)
    return np.nan if sd == 0 else r.mean() / sd * np.sqrt(bars_per_year)


def block_indices(n, block, rng):
    """Circular block bootstrap: indices of one resample of length n."""
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n


def bootstrap(series, stat, block, n_boot, rng):
    """Bootstrap distribution of stat(*series) using the SAME blocks for every series
    (so differences between strategies are paired). series: list of equal-length arrays."""
    arrs = [np.asarray(s, float) for s in series]
    n = len(arrs[0])
    return np.array([stat(*[a[idx] for a in arrs]) for idx in
                     (block_indices(n, block, rng) for _ in range(n_boot))])


def sharpe_ci(r, bars_per_year, block, n_boot, rng, level=0.95):
    draws = bootstrap([r], lambda x: sharpe(x, bars_per_year), block, n_boot, rng)
    a = (1 - level) / 2
    return sharpe(r, bars_per_year), np.nanquantile(draws, a), np.nanquantile(draws, 1 - a)


def newey_west_t(x, lags):
    """t-statistic of the mean of x with a Newey-West (Bartlett) HAC standard error."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(var / n)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def expected_max_sharpe(n_trials, sr_std):
    """Expected maximum of n_trials Sharpe ratios that are pure noise with std sr_std
    (Bailey & Lopez de Prado 2014). Same units as sr_std."""
    if n_trials < 2:
        return 0.0
    z1 = st.norm.ppf(1 - 1 / n_trials)
    z2 = st.norm.ppf(1 - 1 / (n_trials * np.e))
    return sr_std * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(r, sr0_per_bar):
    """Probability that the true per-bar Sharpe exceeds sr0, allowing for skew and fat tails.
    With sr0 = expected maximum noise Sharpe this is the deflated Sharpe ratio."""
    r = np.asarray(r, float)
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = st.skew(r), st.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return st.norm.cdf((sr - sr0_per_bar) * np.sqrt(n - 1) / denom)


def effective_n(r, max_lag=168):
    """Number of independent observations: n / (1 + 2 sum of autocorrelations), lags <= max_lag
    (truncated at the first non-positive autocorrelation)."""
    r = np.asarray(r, float) - np.mean(r)
    n = len(r)
    denom = r @ r
    s = 0.0
    for k in range(1, max_lag + 1):
        rho = (r[k:] @ r[:-k]) / denom
        if rho <= 0:
            break
        s += rho
    return n / (1 + 2 * s)


class TrialLog:
    """Every variant evaluated is recorded here, with the period it was evaluated on."""

    def __init__(self):
        self.rows = []

    def add(self, family, name, period, sharpe_annual, **extra):
        self.rows.append({"family": family, "name": name, "period": period,
                          "sharpe": sharpe_annual, **extra})

    def frame(self):
        return pd.DataFrame(self.rows)
