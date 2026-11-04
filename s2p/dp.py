"""Grid dynamic programmes for one asset and one AR(1) signal.

State (x_{t-1}, f_t). Reward of choosing x:  b f x - s/2 x^2 - cost(x - x_{t-1}),
signal f' = rho f + sig_eps * eps. Arrays V[k, j] are indexed [signal point k, position point j].
Used (1) as a brute-force check of the Riccati solution, (2) to solve the proportional-cost
(no-trade band) problem, which has no closed form.
"""
import numpy as np


def transition_matrix(f_grid, rho, sig_eps, n_nodes):
    """P[k, k'] = probability weight that f' lands on grid point k' given f = f_grid[k].

    Gauss-Hermite nodes for eps, linear interpolation between grid points, clamped at the edges.
    """
    z, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()
    F = len(f_grid)
    P = np.zeros((F, F))
    rows = np.arange(F)
    for zi, wi in zip(z, w):
        fp = np.clip(rho * f_grid + sig_eps * zi, f_grid[0], f_grid[-1])
        j = np.clip(np.searchsorted(f_grid, fp) - 1, 0, F - 2)
        t = (fp - f_grid[j]) / (f_grid[j + 1] - f_grid[j])
        np.add.at(P, (rows, j), wi * (1 - t))
        np.add.at(P, (rows, j + 1), wi * t)
    return P


def hold_value(V_next, x_grid, f_grid, P, b, s, beta):
    """W[k, j] = value of ending period with position x_grid[j] when the signal is f_grid[k]."""
    return b * f_grid[:, None] * x_grid[None, :] - 0.5 * s * x_grid[None, :] ** 2 + beta * P @ V_next


def bellman_bruteforce(V_next, x_grid, f_grid, P, b, s, beta, trade_cost):
    """One Bellman step by exhaustive search over the next position (O(F N^2)).

    trade_cost(dx) gives the cost of a trade of size dx. Returns V and argmax indices.
    """
    W = hold_value(V_next, x_grid, f_grid, P, b, s, beta)
    cost = trade_cost(x_grid[None, :] - x_grid[:, None])        # cost[j, j'] for x_j -> x_j'
    Q = W[:, None, :] - cost[None, :, :]
    idx = Q.argmax(axis=2)
    return Q.max(axis=2), idx


def bellman_band(V_next, x_grid, f_grid, P, b, s, beta, c):
    """One Bellman step for proportional cost c|dx| using the band structure (O(F N)).

    W is concave in x, so the optimum is: buy up to L = argmax W(x) - c x if x < L,
    sell down to U = argmax W(x) + c x if x > U, otherwise hold.
    """
    W = hold_value(V_next, x_grid, f_grid, P, b, s, beta)
    k = np.arange(len(f_grid))
    iL = (W - c * x_grid).argmax(axis=1)
    iU = (W + c * x_grid).argmax(axis=1)
    L, U = x_grid[iL], x_grid[iU]
    buy = W[k, iL][:, None] - c * (L[:, None] - x_grid[None, :])
    sell = W[k, iU][:, None] - c * (x_grid[None, :] - U[:, None])
    x = x_grid[None, :]
    V = np.where(x < L[:, None], buy, np.where(x > U[:, None], sell, W))
    return V, L, U


def solve_band(b, s, c, rho, sig_eps, beta, x_grid, f_grid, n_nodes=15, tol=1e-8, max_iter=50000):
    """Infinite-horizon proportional-cost problem by value iteration.

    Stops when the span of the value update (max - min) is below tol: the policy depends only
    on differences of V, which converge even when the level still drifts.
    Returns (L, U, V): buy and sell boundaries for each signal grid point, and the value.
    """
    P = transition_matrix(f_grid, rho, sig_eps, n_nodes)
    V = np.zeros((len(f_grid), len(x_grid)))
    for it in range(max_iter):
        V_new, L, U = bellman_band(V, x_grid, f_grid, P, b, s, beta, c)
        d = V_new - V
        V = V_new
        if d.max() - d.min() < tol:
            return L, U, V
    raise RuntimeError(f"band DP did not converge in {max_iter} iterations")
