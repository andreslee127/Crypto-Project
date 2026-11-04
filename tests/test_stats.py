import numpy as np

from s2p import stats


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(0)
    n, sd = 50, 0.3
    sim = rng.normal(0, sd, size=(20000, n)).max(axis=1).mean()
    assert abs(stats.expected_max_sharpe(n, sd) - sim) / sim < 0.03


def test_holm():
    adj = stats.holm([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])


def test_newey_west_reduces_to_iid_t_without_lags():
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1, 5000)
    t_iid = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert np.isclose(stats.newey_west_t(x, 0), t_iid)


def test_bootstrap_ci_covers_true_sharpe_and_widens_with_autocorrelation():
    rng = np.random.default_rng(2)
    r = rng.normal(0.02, 1, 8760)
    sr, lo, hi = stats.sharpe_ci(r, 8760, 168, 500, rng)
    assert lo < 0.02 * np.sqrt(8760) < hi
    e = rng.normal(0, 1, 8760)
    ar = np.empty_like(e)
    ar[0] = e[0]
    for t in range(1, len(e)):
        ar[t] = 0.9 * ar[t - 1] + e[t]
    _, lo2, hi2 = stats.sharpe_ci(ar + 0.1, 8760, 168, 500, rng)
    _, lo1, hi1 = stats.sharpe_ci(ar + 0.1, 8760, 1, 500, rng)
    assert hi2 - lo2 > 1.5 * (hi1 - lo1)            # blocks keep the dependence; iid draws do not


def test_effective_n():
    rng = np.random.default_rng(3)
    assert stats.effective_n(rng.normal(size=10000)) > 9000
