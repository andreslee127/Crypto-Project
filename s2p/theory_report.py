"""Theory layer results (data-free): band vs quadratic policy, decay sensitivity.

Everything is in normalised single-asset units: signal g ~ AR(1) with unit variance, reward
z g - z^2/2 - cost(dz). The frictionless optimum z = g earns 0.5 per period, so utilities are
reported as a share of that ("% of frictionless utility").
"""
import numpy as np
import pandas as pd

from s2p import dp, lq, sim
from s2p.plotstyle import INK2, MUTED, POLICY_COLOR, POLICY_LABEL, plt, save

ONE = np.ones((1, 1))
FRICTIONLESS = 0.5


def grids(cfg):
    g = cfg["band_dp"]
    return np.linspace(-g["x_max"], g["x_max"], g["n_x"]), np.linspace(-g["f_max"], g["f_max"], g["n_f"])


def band_rule(c, rho, beta, cfg):
    x, f = grids(cfg)
    L, U, _ = dp.solve_band(1.0, 1.0, c, rho, np.sqrt(1 - rho ** 2), beta, x, f, cfg["band_dp"]["gh_nodes"])
    return L, U, f


def compare_policies(rho, beta, c, lam_true, f, cfg, lam_grid, eta_grid, c_grid):
    """Utility (% of frictionless), turnover and share of periods with a trade for each policy,
    under true costs c|dz| + lam_true/2 dz^2. Every policy gets its own best tuning parameter."""
    run = lambda pol: sim.run_policy(pol, f, ONE, ONE, 1.0, Lam=lam_true * ONE, c=[c])
    rows = {}

    def record(name, param, pol):
        u, turn = run(pol)
        if name not in rows or u.mean() > rows[name]["u"].mean():
            rows[name] = {"param": param, "u": u, "turnover": turn.mean()}

    record("naive", None, sim.linear_policy(ONE, ONE))
    for eta in eta_grid:
        record("ema", eta, sim.linear_policy(eta * ONE, ONE))
    for lam in lam_grid:
        sol = lq.infinite_horizon(ONE, ONE, lam * ONE, np.array([[1 - rho]]), 1.0, beta)
        record("gp", lam, sim.linear_policy(sol["M"], sol["C"]))
    for cb in c_grid:
        L, U, fg = band_rule(cb, rho, beta, cfg)
        record("band", cb, sim.band_policy(L, U, fg))
    base = rows["naive"]["u"]
    out = []
    for name, r in rows.items():
        d = r["u"] - base
        out.append({"policy": name, "tuned_param": r["param"],
                    "utility_pct": 100 * r["u"].mean() / FRICTIONLESS,
                    "gain_vs_naive_pct": 100 * d.mean() / FRICTIONLESS,
                    # common random numbers: the paired difference has a much smaller error
                    "gain_se_pct": 100 * d.std(ddof=1) / np.sqrt(len(d)) / FRICTIONLESS,
                    "turnover_per_period": r["turnover"]})
    return pd.DataFrame(out)


def band_figure(cfg, rho, beta, c, lam_gp, path):
    """Buy / hold / sell regions of the proportional-cost DP, with the GP rule's aim line."""
    L, U, f = band_rule(c, rho, beta, cfg)
    sol = lq.infinite_horizon(ONE, ONE, lam_gp * ONE, np.array([[1 - rho]]), 1.0, beta)
    inner = np.abs(f) <= 3
    f, L, U = f[inner], L[inner], U[inner]
    aim = sol["C"][0, 0] * f
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4))
    # (a) regions in (signal, position) space
    lo, hi = -3.5, 3.5
    ax.fill_between(f, lo, L, color=POLICY_COLOR["naive"], alpha=0.12, lw=0)
    ax.fill_between(f, U, hi, color="#e34948", alpha=0.10, lw=0)
    ax.fill_between(f, L, U, color=POLICY_COLOR["band"], alpha=0.35, lw=0)
    ax.plot(f, f, color=MUTED, lw=1.2, ls="--")
    ax.text(0.9, -2.6, "BUY up to the band", color=INK2)
    ax.text(-2.9, 2.7, "SELL down to the band", color=INK2)
    ax.annotate("HOLD: inside the band", xy=(-1.5, -1.5), xytext=(-2.9, 0.8), color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.text(2.1, 2.9, "Markowitz", color=MUTED)
    ax.set_xlim(-3, 3), ax.set_ylim(lo, hi)
    ax.set_xlabel("signal g (standard deviations)")
    ax.set_ylabel("position before trading (Markowitz units)")
    ax.set_title("(a) Buy / hold / sell regions")
    # (b) the same rules relative to the Markowitz target, where differences are visible
    bx.axhline(0, color=MUTED, lw=1.2, ls="--")
    bx.fill_between(f, L - f, U - f, color=POLICY_COLOR["band"], alpha=0.35, lw=0)
    bx.plot(f, L - f, color=POLICY_COLOR["band"], lw=2)
    bx.plot(f, U - f, color=POLICY_COLOR["band"], lw=2)
    bx.plot(f, aim - f, color=POLICY_COLOR["gp"], lw=2)
    bx.text(-2.9, (U - f)[2] + 0.04, "band (no trade inside)", color=INK2)
    bx.text(1.0, (aim - f)[-1] - 0.03, f"GP aim (lambda = {lam_gp:.1f})", color=POLICY_COLOR["gp"], va="top")
    bx.text(-2.9, 0.02, "Markowitz", color=MUTED, va="bottom")
    bx.set_xlim(-3, 3)
    bx.set_xlabel("signal g (standard deviations)")
    bx.set_ylabel("position minus Markowitz target")
    bx.set_title("(b) Relative to Markowitz")
    hl = np.log(0.5) / np.log(rho)
    fig.suptitle(f"Proportional cost c = {c} (normalised), signal half-life {hl:.0f} periods",
                 fontweight="bold")
    save(fig, path)


def run(cfg, out_dir, n_paths=100, T=4000):
    rng = np.random.default_rng(cfg["seed"])
    beta = cfg["portfolio"]["discount"]
    lam_grid = list(np.logspace(-1, 3, 13))
    eta_grid = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    tables = {}

    # 1. Proportional costs (band's home turf) and quadratic costs (GP's home turf).
    #    c is in normalised units c / (sigma * kappa); realistic crypto values are O(1).
    rho = 0.5 ** (1 / 24)
    f = sim.simulate_signals(T, [rho], n_paths, rng)
    for label, c, lam in [("prop_c0.5", 0.5, 0.0), ("prop_c2", 2.0, 0.0), ("quad_lam20", 0.0, 20.0)]:
        c_grid = [c * k for k in (0.5, 1, 2)] if c > 0 else [0.1, 0.3, 1.0]
        tables[label] = compare_policies(rho, beta, c, lam, f, cfg, lam_grid, eta_grid, c_grid)
    gp_lam = float(tables["prop_c0.5"].set_index("policy").loc["gp", "tuned_param"])
    band_figure(cfg, rho, beta, 0.5, gp_lam, out_dir / "fig_theory_band.png")

    # 2. Decay sensitivity: same proportional cost, signal half-life varies.
    rows = []
    for hl in [3, 12, 48, 192]:
        r = 0.5 ** (1 / hl)
        f_hl = sim.simulate_signals(T, [r], n_paths, rng)
        t = compare_policies(r, beta, 1.0, 0.0, f_hl, cfg, lam_grid, eta_grid, [0.5, 1.0, 2.0])
        t.insert(0, "half_life", hl)
        rows.append(t)
    tables["decay"] = pd.concat(rows, ignore_index=True)
    return tables
