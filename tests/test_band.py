"""Proportional-cost (no-trade band) dynamic programme."""
import numpy as np

from s2p import dp, lq, sim

RHO, BETA = 0.95, 0.99
SIG_EPS = np.sqrt(1 - RHO ** 2)
X = np.linspace(-4, 4, 201)
F = np.linspace(-4, 4, 61)


def test_band_shortcut_equals_bruteforce():
    c = 0.1
    P = dp.transition_matrix(F, RHO, SIG_EPS, 15)
    V = np.zeros((len(F), len(X)))
    for _ in range(5):                                   # a non-trivial concave continuation value
        V, _, _ = dp.bellman_band(V, X, F, P, 1.0, 1.0, BETA, c)
    V_fast, _, _ = dp.bellman_band(V, X, F, P, 1.0, 1.0, BETA, c)
    V_slow, _ = dp.bellman_bruteforce(V, X, F, P, 1.0, 1.0, BETA, lambda d: c * np.abs(d))
    assert np.allclose(V_fast, V_slow, atol=1e-10)


def test_band_structure():
    L, U, _ = dp.solve_band(1.0, 1.0, 0.1, RHO, SIG_EPS, BETA, X, F)
    inner = np.abs(F) <= 3
    assert np.all(L <= U)
    assert np.all(np.diff(L[inner]) >= 0) and np.all(np.diff(U[inner]) >= 0)   # buy more when f is higher
    mid = len(F) // 2
    assert np.isclose(L[mid], -U[mid])                   # symmetric around zero at f = 0
    assert U[mid] - L[mid] > 0.1                          # a genuine no-trade region


def test_band_collapses_to_markowitz_without_costs():
    L, U, _ = dp.solve_band(1.0, 1.0, 0.0, RHO, SIG_EPS, BETA, X, F)
    inner = np.abs(F) <= 3
    step = X[1] - X[0]
    assert np.max(np.abs(L - F)[inner]) <= step and np.max(np.abs(U - F)[inner]) <= step


def test_band_beats_tuned_gp_under_proportional_cost():
    c = 0.2
    L, U, _ = dp.solve_band(1.0, 1.0, c, RHO, SIG_EPS, BETA, X, F)
    f = sim.simulate_signals(3000, [RHO], 100, np.random.default_rng(4))
    one = np.ones((1, 1))
    run = lambda pol: sim.run_policy(pol, f, one, one, 1.0, c=[c])[0]
    u_band = run(sim.band_policy(L, U, F))
    u_gp = {}
    for lam in [0.5, 1, 2, 4, 8, 16]:                    # give GP its best quadratic proxy
        sol = lq.infinite_horizon(one, one, lam * one, np.array([[1 - RHO]]), 1.0, BETA)
        u_gp[lam] = run(sim.linear_policy(sol["M"], sol["C"]))
    best = max(u_gp, key=lambda k: u_gp[k].mean())
    d = u_band - u_gp[best]
    assert d.mean() > 0
