"""Linear-quadratic dynamic trading (Garleanu-Pedersen 2013), derived in THEORY.md.

Model, per period t (n assets, K signals):
    f_{t+1} = (I - Phi) f_t + eps_{t+1},   Cov(eps) = Omega
    expected return of holding x_t over (t, t+1]  =  B f_t          (B is n x K)
    reward_t = x_t'B f_t - gamma/2 x_t'Sigma x_t - 1/2 dx_t'Lambda dx_t,  dx_t = x_t - x_{t-1}
    maximise  E sum_t beta^t reward_t.

The value function is quadratic:  V(x_{t-1}, f_t) = -1/2 x'Axx x + x'Axf f + 1/2 f'Aff f + const,
and the optimal rule is  x_t = x_{t-1} + M (aim_t - x_{t-1}),  aim_t = C f_t.
"""
import numpy as np
from scipy.linalg import solve_discrete_are


def _sym(a):
    return 0.5 * (a + a.T)


def bellman_step(V, B, Sigma, Lam, Phi, Omega, gamma, beta):
    """One backward step: value coefficients of period t+1 -> period t, plus the policy.

    V = (Axx, Axf, Aff, const).  Returns (V_new, G, H) with optimal x_t = G x_{t-1} + H f_t.
    """
    Axx, Axf, Aff, const = V
    R = np.eye(len(Phi)) - Phi                    # E_t f_{t+1} = R f_t
    J = gamma * Sigma + Lam + beta * Axx          # curvature of the period-t problem in x_t
    D = B + beta * Axf @ R                        # linear term in f_t (today's + future alpha)
    G = np.linalg.solve(J, Lam)                   # weight on yesterday's position
    H = np.linalg.solve(J, D)                     # weight on today's signal
    Axx_new = _sym(Lam - Lam @ G)
    Axf_new = Lam @ H
    Aff_new = _sym(D.T @ H + beta * R.T @ Aff @ R)
    const_new = beta * (const + 0.5 * np.trace(Aff @ Omega))
    return (Axx_new, Axf_new, Aff_new, const_new), G, H


def finite_horizon(T, B, Sigma, Lam, Phi, Omega, gamma, beta):
    """Backward Riccati recursion over periods t = 0..T-1 with V_T = 0 (no terminal value).

    Returns lists (indexed by t) of value coefficients V_t and policies (G_t, H_t).
    """
    n, K = B.shape
    V = (np.zeros((n, n)), np.zeros((n, K)), np.zeros((K, K)), 0.0)
    values, policies = [None] * T, [None] * T
    for t in reversed(range(T)):
        V, G, H = bellman_step(V, B, Sigma, Lam, Phi, Omega, gamma, beta)
        values[t], policies[t] = V, (G, H)
    return values, policies


def infinite_horizon(B, Sigma, Lam, Phi, gamma, beta):
    """Stationary solution, computed independently of the backward recursion.

    Axx solves a discrete algebraic Riccati equation (scipy), Axf a linear (Sylvester-type)
    equation. Returns dict with M (trading rate), C (aim = C f), Axx, Axf.
    """
    n, K = B.shape
    I = np.eye(n)
    # Axx = Lam - Lam J^{-1} Lam is the DARE of the LQR problem with state x_{t-1}, control dx:
    #   cost 1/2 [x'gS x + 2 x'gS u + u'(gS+Lam)u],  x_t = x_{t-1} + u,  discount beta.
    gS = gamma * Sigma
    sb = np.sqrt(beta)
    Axx = _sym(solve_discrete_are(sb * I, sb * I, gS, gS + Lam, s=gS))
    J = gS + Lam + beta * Axx
    L = Lam @ np.linalg.inv(J)
    R = np.eye(K) - Phi
    # Axf = L (B + beta Axf R)  ->  vec(Axf) = (I - beta R' kron L)^{-1} vec(L B)   (column-major vec)
    lhs = np.eye(n * K) - beta * np.kron(R.T, L)
    Axf = np.linalg.solve(lhs, (L @ B).reshape(-1, order="F")).reshape(n, K, order="F")
    W = gS + beta * Axx
    M = np.linalg.solve(J, W)                     # = I - J^{-1} Lam
    C = np.linalg.solve(W, B + beta * Axf @ R)    # aim portfolio per unit of signal
    return {"M": M, "C": C, "Axx": Axx, "Axf": Axf}


def markowitz(B, Sigma, gamma):
    """Static mean-variance portfolio per unit of signal: x = (gamma Sigma)^{-1} B f."""
    return np.linalg.solve(gamma * Sigma, B)


# ---- single asset, diagonal signal decay: closed forms used by the backtester ----

def scalar_riccati(s, lam, beta):
    """Positive root A of  beta A^2 + (s + lam (1-beta)) A - lam s = 0,  with s = gamma sigma^2.

    Vectorised over arrays. Trading rate M = A / lam, aim weight z = s / (s + beta A).
    """
    q = s + lam * (1.0 - beta)
    return (-q + np.sqrt(q * q + 4.0 * beta * lam * s)) / (2.0 * beta)


def scalar_policy(s, lam, beta, rho):
    """Trading rate M and per-signal aim weights w_k for one asset.

    aim = sum_k w_k * markowitz_k,  w_k = z / (1 - (1-z) rho_k),  rho_k = AR(1) coefficient.
    Shapes: s, lam broadcast together; rho has shape (K,) and is put on the last axis.
    """
    s, lam = np.asarray(s, float), np.asarray(lam, float)
    A = scalar_riccati(s, lam, beta)
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(lam > 0, A / np.where(lam > 0, lam, 1.0), 1.0)   # lam -> 0 gives M -> 1
    z = s / (s + beta * A)
    w = z[..., None] / (1.0 - (1.0 - z[..., None]) * np.asarray(rho))
    return M, w


def gp_published_a(gamma, lam, rho_disc):
    """Closed form for 'a' with Lambda = lam*Sigma in GP's discounting, as I recall it from the
    paper (NOT verified against the text; see THEORY.md section 2.5). Used only in a test that
    checks it against our own solver after mapping GP's discounting onto ours."""
    q = gamma * (1 - rho_disc) + lam * rho_disc
    return (-q + np.sqrt(q * q + 4 * gamma * lam * (1 - rho_disc) ** 2)) / (2 * (1 - rho_disc))
