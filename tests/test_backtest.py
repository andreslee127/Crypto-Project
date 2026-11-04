"""PnL accounting reconciles against a hand-computed example; policy rules behave as specified."""
import numpy as np

from s2p import backtest, costs, lq

INF = np.inf


def test_pnl_reconciles_with_hand_computation():
    # 2 assets, 3 bars. Costs: linear 10 bp, impact lam = (2e-4, 4e-4) per USD.
    aim = np.array([[100.0, -50.0], [110.0, -40.0], [110.0, 0.0]])
    M = np.ones((3, 2))
    tradeable = np.array([[1, 1], [1, 1], [1, 0]], bool)
    fwd_ret = np.array([[0.10, -0.20], [0.0, 0.5], [-0.10, 0.0]])
    fwd_fund = np.array([[0.001, 0.0], [0.0, -0.002], [0.0, 0.0]])
    c_lin = np.full((3, 2), 0.001)
    lam = np.tile([2e-4, 4e-4], (3, 1))
    res = backtest.simulate(M, aim, -INF * np.ones((3, 2)), INF * np.ones((3, 2)), tradeable,
                            fwd_ret, fwd_fund, c_lin, lam)
    # bar 0: buy 100, sell 50.  linear 0.15, impact 0.5*(2e-4*1e4 + 4e-4*2500) = 1.5
    #        gross 100*0.1 + (-50)*(-0.2) = 20, funding 100*0.001 = 0.1 -> net 18.25
    # bar 1: drifted (110, -40) = aim, no trade. gross -40*0.5 = -20,
    #        funding (-40)*(-0.002) = +0.08 paid (shorts pay when funding < 0) -> net -20.08
    # bar 2: asset 2 not tradeable -> forced to 0 from -60: linear 0.06, impact 0.5*4e-4*3600 = 0.72
    #        gross 110*(-0.1) = -11 -> net -11.78
    expected = {"gross": [20, -20, -11], "cost_linear": [0.15, 0, 0.06], "cost_impact": [1.5, 0, 0.72],
                "funding": [0.1, 0.08, 0], "net": [18.25, -20.08, -11.78], "traded": [150, 0, 60]}
    for col, vals in expected.items():
        assert np.allclose(res[col], vals), col
    assert np.allclose(res.net, res.gross - res.cost_linear - res.cost_impact - res.funding)
    assert np.isclose(res.net.sum(), -13.61)


def test_band_policy_holds_inside_and_trades_to_edge():
    lo = np.array([[-1.0], [-1.0], [2.0]])
    hi = np.array([[1.0], [1.0], [3.0]])
    z = np.zeros((3, 1))
    res = backtest.simulate(z, z, lo, hi, np.ones((3, 1), bool), z, z, z, z)
    # starts at 0 (inside [-1, 1]): no trade; then the band moves to [2, 3]: buy exactly 2
    assert np.allclose(res.traded, [0, 0, 2])


def test_scalar_gp_limits_inside_backtest_inputs():
    s = np.array([1.0, 1.0])
    M, w = lq.scalar_policy(s, np.array([0.0, 1e12]), 0.999, np.array([0.9]))
    assert np.allclose(M, [1.0, 0.0], atol=1e-6)


def test_quadratic_impact_matches_sqrt_law_at_reference_size():
    sigma_d, adv, q = 0.04, 5e8, 25_000.0
    lam = 2 * 1.0 * sigma_d / np.sqrt(adv * q)
    assert np.isclose(0.5 * lam * q ** 2, costs.sqrt_law_cost(q, sigma_d, adv))
