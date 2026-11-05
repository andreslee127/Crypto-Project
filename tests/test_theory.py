"""Validation of the Garleanu-Pedersen solution, in the order required:
(i) finite-horizon Riccati vs brute-force grid DP, (ii) infinite horizon vs long finite horizon,
(iii) Monte Carlo: optimal beats naive and EMA, (iv) limiting cases.
"""
import numpy as np
import pytest

from s2p import dp, lq, sim


def random_problem(rng, n=3, K=2):
    A = rng.standard_normal((n, n))
    Sigma = A @ A.T / n + 0.5 * np.eye(n)
    Lc = rng.standard_normal((n, n))
    Lam = Lc @ Lc.T / n + 0.2 * np.eye(n)          # deliberately NOT proportional to Sigma
    B = rng.standard_normal((n, K))
    Phi = np.diag(rng.uniform(0.02, 0.5, K))
    return B, Sigma, Lam, Phi


# ---------- (i) finite horizon: Riccati recursion == brute-force dynamic programming ----------

def test_riccati_matches_bruteforce_dp():
    b, gamma, sig2, Lam, rho, beta, T = 1.0, 2.0, 0.5, 2.0, 0.8, 0.95, 4
    sig_eps = np.sqrt(1 - rho ** 2)
    x_grid = np.linspace(-3, 3, 301)
    f_grid = np.linspace(-4, 4, 81)
    P = dp.transition_matrix(f_grid, rho, sig_eps, 15)
    s = gamma * sig2

    V = np.zeros((len(f_grid), len(x_grid)))
    for _ in range(T):
        V, idx = dp.bellman_bruteforce(V, x_grid, f_grid, P, b, s, beta, lambda d: 0.5 * Lam * d ** 2)

    args = (np.array([[b]]), np.array([[sig2]]), np.array([[Lam]]), np.array([[1 - rho]]),
            np.array([[sig_eps ** 2]]), gamma, beta)
    values, policies = lq.finite_horizon(T, *args)
    G, H = policies[0]
    (Axx, Axf, Aff, const) = values[0]
    X, Fm = np.meshgrid(x_grid, f_grid)                 # shape (F, N) like V
    x_star = G[0, 0] * X + H[0, 0] * Fm
    V_ric = -0.5 * Axx[0, 0] * X ** 2 + Axf[0, 0] * X * Fm + 0.5 * Aff[0, 0] * Fm ** 2 + const

    inner = (np.abs(Fm) <= 2) & (np.abs(X) <= 1.5)
    step = x_grid[1] - x_grid[0]
    assert np.max(np.abs(x_grid[idx] - x_star)[inner]) <= step       # policy within one grid step
    assert np.max(np.abs(V - V_ric)[inner]) < 1e-2 * np.max(np.abs(V_ric[inner]))


# ---------- (ii) infinite horizon == limit of a long finite horizon ----------

def test_infinite_horizon_is_limit_of_finite():
    rng = np.random.default_rng(0)
    B, Sigma, Lam, Phi = random_problem(rng)
    gamma, beta = 1.5, 0.99
    _, policies = lq.finite_horizon(3000, B, Sigma, Lam, Phi, np.eye(2), gamma, beta)
    G0, H0 = policies[0]
    sol = lq.infinite_horizon(B, Sigma, Lam, Phi, gamma, beta)
    I = np.eye(3)
    assert np.allclose(G0, I - sol["M"], atol=1e-9)
    assert np.allclose(H0, sol["M"] @ sol["C"], atol=1e-9)


def test_aim_is_weighted_average_of_current_and_future_markowitz():
    # aim_t = Z markowitz_t + (I - Z) E_t aim_{t+1},  Z = (gamma Sigma + beta Axx)^{-1} gamma Sigma
    rng = np.random.default_rng(1)
    B, Sigma, Lam, Phi = random_problem(rng)
    gamma, beta = 1.5, 0.99
    sol = lq.infinite_horizon(B, Sigma, Lam, Phi, gamma, beta)
    W = gamma * Sigma + beta * sol["Axx"]
    Z = np.linalg.solve(W, gamma * Sigma)
    C_rhs = Z @ lq.markowitz(B, Sigma, gamma) + (np.eye(3) - Z) @ sol["C"] @ (np.eye(2) - Phi)
    assert np.allclose(sol["C"], C_rhs, atol=1e-10)


def test_scalar_closed_form_matches_general_solver():
    # the backtester's per-asset closed form == the general matrix solution for n = 1, K = 3
    s, lam, beta = 0.7, 3.0, 0.999
    rho = np.array([0.5, 0.9, 0.99])
    B = np.array([[0.3, -0.2, 0.1]])
    sol = lq.infinite_horizon(B, np.array([[s]]), np.array([[lam]]), np.diag(1 - rho), 1.0, beta)
    M, w = lq.scalar_policy(s, lam, beta, rho)
    assert np.isclose(M, sol["M"][0, 0])
    assert np.allclose(w * B[0] / s, sol["C"][0])


def test_matches_gp2013_closed_form():
    # Garleanu & Pedersen (2013, JF 68(6)), eq. (4): (1-r)^{t+1} on alpha and risk, (1-r)^t on
    # costs; Assumption 1: Lambda = lam*Sigma. Dividing by (1-r)^t maps it to our problem with
    # beta = 1-r, gamma -> (1-r) gamma, B -> (1-r) B. Three correlated assets, two signals.
    rng = np.random.default_rng(3)
    B, Sigma, _, Phi = random_problem(rng)
    gamma, lam, r = 2.0, 5.0, 0.02
    n, K = B.shape
    sol = lq.infinite_horizon((1 - r) * B, Sigma, lam * Sigma, Phi, (1 - r) * gamma, 1 - r)
    a = lq.gp_published_a(gamma, lam, r)
    assert np.allclose(sol["M"], a / lam * np.eye(n))                # Prop. 2(ii), eqs (9)-(10)
    W = (1 - r) * gamma * Sigma + (1 - r) * sol["Axx"]
    Z = np.linalg.solve(W, (1 - r) * gamma * Sigma)
    assert np.allclose(Z, gamma / (gamma + a) * np.eye(n))           # Prop. 3, z = gamma/(gamma+a)
    aim_gp = np.linalg.solve(gamma * Sigma, B) @ np.linalg.inv(np.eye(K) + a * Phi / gamma)
    assert np.allclose(sol["C"], aim_gp)                             # Prop. 4(i), eq. (14)


def test_gp2013_general_structure():
    # Prop. 2(i), eqs (7)-(8), for general Lambda: trading rate Lambda^{-1} Axx, aim Axx^{-1} Axf f
    rng = np.random.default_rng(4)
    B, Sigma, Lam, Phi = random_problem(rng)
    sol = lq.infinite_horizon(B, Sigma, Lam, Phi, 1.5, 0.99)
    assert np.allclose(sol["M"], np.linalg.solve(Lam, sol["Axx"]))
    assert np.allclose(sol["C"], np.linalg.solve(sol["Axx"], sol["Axf"]))


# ---------- (iii) Monte Carlo: optimal policy has higher net utility ----------

def test_monte_carlo_gp_beats_naive_and_ema():
    rng = np.random.default_rng(2)
    rho = np.array([0.5, 0.98])                      # a fast and a slow signal
    B = np.array([[0.10, 0.05], [0.04, 0.08]])
    Sigma = np.array([[1.0, 0.5], [0.5, 1.5]])
    Lam = np.diag([4.0, 8.0])
    gamma, beta = 1.0, 0.999
    f = sim.simulate_signals(2200, rho, 200, rng)
    sol = lq.infinite_horizon(B, Sigma, Lam, np.diag(1 - rho), gamma, beta)
    mkw = lq.markowitz(B, Sigma, gamma)
    run = lambda pol: sim.run_policy(pol, f, B, Sigma, gamma, Lam=Lam)[0]

    u_gp = run(sim.linear_policy(sol["M"], sol["C"]))
    u_naive = run(sim.linear_policy(np.eye(2), mkw))
    u_ema = {eta: run(sim.linear_policy(eta * np.eye(2), mkw)) for eta in [0.02, 0.05, 0.1, 0.2, 0.4]}
    best_eta = max(u_ema, key=lambda e: u_ema[e].mean())

    for other in (u_naive, u_ema[best_eta]):
        d = u_gp - other                              # common random numbers: paired difference
        assert d.mean() > 3 * d.std(ddof=1) / np.sqrt(len(d))


# ---------- (iv) limiting cases ----------

def test_zero_cost_gives_markowitz_and_huge_cost_gives_no_trading():
    rng = np.random.default_rng(3)
    B, Sigma, _, Phi = random_problem(rng)
    gamma, beta = 1.0, 0.99
    tiny = lq.infinite_horizon(B, Sigma, 1e-9 * np.eye(3), Phi, gamma, beta)
    assert np.allclose(tiny["M"], np.eye(3), atol=1e-6)
    assert np.allclose(tiny["C"], lq.markowitz(B, Sigma, gamma), atol=1e-6)
    huge = lq.infinite_horizon(B, Sigma, 1e9 * np.eye(3), Phi, gamma, beta)
    assert np.max(np.abs(huge["M"])) < 1e-3


@pytest.mark.parametrize("lam", [1e-9, 1e9])
def test_scalar_limits(lam):
    M, w = lq.scalar_policy(1.0, lam, 0.999, np.array([0.9]))
    if lam < 1:
        assert np.isclose(M, 1, atol=1e-4) and np.isclose(w, 1, atol=1e-4)
    else:
        assert M < 1e-3
