"""IC significance under dependence corrections (SYNTHETIC panels)."""
import copy

import numpy as np
import pandas as pd

from s2p import research
from s2p.config import load_config


def panel_with_ic(true_ic, T=24 * 7 * 60, N=25, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=T, freq="h")
    sig = rng.standard_normal((T, N))
    ret = true_ic * sig + np.sqrt(1 - true_ic ** 2) * rng.standard_normal((T, N))
    cols = [f"A{i}" for i in range(N)]
    return (pd.DataFrame(sig, idx, cols),
            {"fwd_ret": pd.DataFrame(ret, idx, cols)}, np.ones(T, bool))


def test_ic_robustness_detects_signal_and_controls_noise():
    cfg = copy.deepcopy(load_config())
    cfg["stats"]["n_boot"] = 200
    rng = np.random.default_rng(1)
    sig, d, w = panel_with_ic(0.05)
    noise, d0, _ = panel_with_ic(0.0, seed=2)
    out = research.ic_robustness({"real": sig}, d, w, cfg, rng)
    out0 = research.ic_robustness({"noise": noise}, d0, w, cfg, rng)
    assert out.loc[0, "p_holm_conservative"] < 1e-6
    assert out0.loc[0, "p_conservative"] > 0.01
    tcols = [c for c in out.columns if c.startswith("t_")]
    t = out.loc[0, tcols].astype(float)
    assert (t > 0).all() and t.max() / t.min() < 2       # iid data: all methods agree roughly
