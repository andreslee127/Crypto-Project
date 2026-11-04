"""Monte Carlo of trading policies on simulated AR(1) signals (data-free, theory layer).

A policy is a function (x_prev, f) -> x. Utility per period is the model objective
x'Bf - gamma/2 x'Sigma x - 1/2 dx'Lambda dx - c'|dx|, evaluated with the *expected* return Bf
(return noise has mean zero and only adds variance to the estimate).
"""
import numpy as np


def simulate_signals(T, rho, n_paths, rng):
    """Stationary AR(1) signals with unit variance: f' = rho f + sqrt(1-rho^2) eps. Shape (paths, T, K)."""
    rho = np.asarray(rho, float)
    K = len(rho)
    f = np.empty((n_paths, T, K))
    f[:, 0] = rng.standard_normal((n_paths, K))
    e = rng.standard_normal((n_paths, T, K)) * np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        f[:, t] = rho * f[:, t - 1] + e[:, t]
    return f


def run_policy(policy, f, B, Sigma, gamma, Lam=None, c=None, burn=200):
    """Average per-period utility of `policy` on each simulated path (vectorised over paths).

    Returns (utility per path, mean |dx| per path). The first `burn` periods are discarded so the
    starting position x = 0 does not matter.
    """
    n_paths, T, _ = f.shape
    n = B.shape[0]
    Lam = np.zeros((n, n)) if Lam is None else Lam
    c = np.zeros(n) if c is None else np.asarray(c, float)
    x = np.zeros((n_paths, n))
    util = np.zeros(n_paths)
    turn = np.zeros(n_paths)
    for t in range(T):
        x_new = policy(x, f[:, t])
        dx = x_new - x
        u = (np.einsum("pi,ik,pk->p", x_new, B, f[:, t])
             - 0.5 * gamma * np.einsum("pi,ij,pj->p", x_new, Sigma, x_new)
             - 0.5 * np.einsum("pi,ij,pj->p", dx, Lam, dx)
             - np.abs(dx) @ c)
        if t >= burn:
            util += u
            turn += np.abs(dx).sum(axis=1)
        x = x_new
    return util / (T - burn), turn / (T - burn)


def linear_policy(M, C):
    """x = x_prev + M (C f - x_prev). Naive: M = I. EMA: M = eta I with C = Markowitz."""
    return lambda x, f: x + (f @ C.T - x) @ M.T


def band_policy(L, U, f_grid):
    """Single asset: clip yesterday's position into [L(f), U(f)] (linear interpolation in f)."""
    def pol(x, f):
        lo = np.interp(f[:, 0], f_grid, L)[:, None]
        hi = np.interp(f[:, 0], f_grid, U)[:, None]
        return np.clip(x, lo, hi)
    return pol
