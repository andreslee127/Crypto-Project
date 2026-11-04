"""No look-ahead: change every datum after t*, and nothing decided at or before t* may change.
Covers signals, universe, walk-forward ridge, costs and all four policies (SYNTHETIC data)."""
import copy

import numpy as np
import pandas as pd
import pytest

from s2p import backtest, run_all, signals, stats, synthetic
from s2p.config import load_config

T_STAR = pd.Timestamp("2021-07-15 12:00")
PARAMS = [("naive", None), ("ema", 0.1), ("gp", 1.0), ("band", 1.0)]


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    c = copy.deepcopy(load_config())
    c["data"].update(first_month="2021-01", last_month="2021-09", universe_size=12)
    c["splits"] = {"train": ["2021-01-01", "2021-06-30"], "test": ["2021-07-01", "2021-08-31"],
                   "holdout": ["2021-09-01", "2021-09-30"], "burn_in_months": 2}
    c["band_dp"].update(n_x=201, n_f=41, c_grid=[0.1, 1.0, 10.0])
    c["paths"]["cache"] = tmp_path_factory.mktemp("cache")
    return c


def perturb(panel, cfg, seed=1):
    """Scramble everything strictly after T_STAR and recompute the universe from it."""
    rng = np.random.default_rng(seed)
    p = {k: v.copy() for k, v in panel.items()}
    after = p["close"].index > T_STAR
    shape = (after.sum(), p["close"].shape[1])
    p["close"].loc[after] *= np.exp(np.cumsum(0.05 * rng.standard_normal(shape), axis=0))
    p["quote_volume"].loc[after] *= np.exp(rng.standard_normal(shape))
    p["funding"].loc[after] += 0.001 * rng.standard_normal(shape)
    members = synthetic.rank_universe(p["quote_volume"], p["close"], cfg["data"]["universe_size"],
                                      cfg["data"]["universe_min_days"])
    months = p["close"].index.to_period("M").astype(str)
    p["universe"][:] = False
    for m, mem in members.items():
        p["universe"].loc[months == m, mem] = True
    return p


def pipeline(cfg, panel):
    d = signals.build(panel, cfg)
    wins = {"train": run_all.window(d["vol"].index, "2021-03-01", "2021-06-30")}
    selected = {}
    ctx = run_all.build_context(cfg, d, wins, selected, stats.TrialLog())
    runs = {k: backtest.run_policy(k, p, ctx) for k, p in PARAMS}
    return d, ctx, selected, runs


def same(a, b):
    # 1e-12 relative tolerance: BLAS dot products can differ in the last bit with memory
    # alignment; any real look-ahead changes values by many orders of magnitude more.
    return np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.fixture(scope="module")
def both(cfg):
    panel = synthetic.make_panel(cfg, n_symbols=15, seed=3)
    return pipeline(cfg, panel), pipeline(cfg, perturb(panel, cfg)), panel


def test_perturbation_changes_the_future(both):
    (d1, ctx1, _, runs1), (d2, ctx2, _, runs2), _ = both
    later = d1["vol"].index > T_STAR + pd.Timedelta(days=2)
    assert not same(ctx1["score"][later], ctx2["score"][later])      # the test is not vacuous


def test_signals_and_scores_do_not_look_ahead(both):
    (d1, ctx1, s1, _), (d2, ctx2, s2, _), _ = both
    past = d1["vol"].index <= T_STAR
    for name in d1["signals"]:
        assert same(d1["signals"][name][past], d2["signals"][name][past]), name
    assert same(d1["tradeable"][past], d2["tradeable"][past])
    assert same(ctx1["score"][past], ctx2["score"][past])
    assert same(ctx1["lam"][past], ctx2["lam"][past])
    assert s1 == s2                                                   # train-only estimates unchanged


@pytest.mark.parametrize("kind", [k for k, _ in PARAMS])
def test_decisions_do_not_look_ahead(both, kind):
    (d1, _, _, runs1), (_, _, _, runs2), _ = both
    r1, r2 = runs1[kind], runs2[kind]
    before = r1.index < T_STAR
    assert same(r1[before], r2[before])
    # the decision AT t* is also unchanged (only its realised return, which ends after t*, differs)
    for col in ["traded", "cost_linear", "cost_impact", "gross_exposure"]:
        assert same(r1.loc[T_STAR, col], r2.loc[T_STAR, col]), col


def test_harness_detects_a_planted_leak(both, cfg):
    # negative control: a signal that peeks one bar ahead must be caught by the same comparison
    (d1, *_), (d2, *_), panel = both
    leak = lambda d: d["fwd_ret"]
    past = d1["vol"].index <= T_STAR
    assert not same(leak(d1)[past], leak(d2)[past])
