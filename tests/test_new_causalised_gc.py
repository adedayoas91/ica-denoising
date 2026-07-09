from __future__ import annotations

import numpy as np

from csl.core.causalised_gc import GcStar
from csl.core.new_causalised_GC import FastGcStar, _fft_cross_null


def _var_data(n=8, T=1200, seed=0):
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    A[1, 0] = 0.5
    A[2, 1] = 0.5
    A[5, 4] = 0.45
    x = np.zeros((n, T))
    for t in range(1, T):
        x[:, t] = A @ x[:, t - 1] + rng.normal(0, 0.3, n)
    return x


def test_fft_cross_null_matches_bruteforce():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = 0.6 * x + rng.normal(size=400) * 0.5
    obs, p = _fft_cross_null(x[None, :], y[None, :])
    # brute-force all circular shifts
    base = abs(np.corrcoef(x, y)[0, 1])
    count = 0
    for s in range(1, x.size):
        rolled = np.r_[x[s:], x[:s]]
        if abs(np.corrcoef(rolled, y)[0, 1]) >= base - 1e-12:
            count += 1
    expected_p = (count + 1) / x.size
    assert abs(obs[0] - base) < 1e-9
    assert abs(p[0] - expected_p) < 1e-9


def test_output_shapes_match_reference():
    x = _var_data()
    ref = GcStar(n_perm=100, n_pasts=2, n_lags=2, method="fcgc", random_state=0).fit(x)
    fast = FastGcStar(n_perm=100, n_pasts=2, n_lags=2, method="fcgc", random_state=0).fit(x)
    assert fast.inv_corr_.shape == ref.inv_corr_.shape
    assert fast.corr_.shape == ref.corr_.shape
    assert fast.pVal_inv_corr_.shape == ref.pVal_inv_corr_.shape


def test_fcgc_partial_corr_equals_residual_corr_offdiag():
    """Precision-matrix partial correlations equal regression-residual corr."""
    x = _var_data()
    ref = GcStar(n_perm=50, n_pasts=2, n_lags=2, method="fcgc", random_state=0).fit(x)
    fast = FastGcStar(n_perm=50, n_pasts=2, n_lags=2, method="fcgc", random_state=0).fit(x)
    n = x.shape[0]
    rows = ref.inv_corr_.shape[0]
    # compare only genuine cross pairs (exclude self entries i == j within any block)
    mask = np.ones((rows, n), dtype=bool)
    for j in range(n):
        for lag in range(ref.n_pasts + 1):
            mask[j + lag * n, j] = False
    r = np.corrcoef(ref.inv_corr_[mask].ravel(), fast.inv_corr_[mask].ravel())[0, 1]
    assert r > 0.99


def test_edge_agreement_with_reference():
    x = _var_data(seed=2)
    ref = GcStar(n_perm=300, n_pasts=2, n_lags=2, method="fcgc", random_state=0).fit(x)
    fast = FastGcStar(n_perm=300, n_pasts=2, n_lags=2, method="fcgc", random_state=0).fit(x)
    gr = (ref.get_result(alpha=0.05, beta=0.05).lag_only != 0).astype(int)
    gf = (fast.get_result(alpha=0.05, beta=0.05).lag_only != 0).astype(int)
    off = ~np.eye(x.shape[0], dtype=bool)
    agreement = (gr[off] == gf[off]).mean()
    assert agreement >= 0.95


def test_determinism():
    x = _var_data(seed=3)
    g1 = FastGcStar(n_perm=200, n_pasts=2, method="fcgc", random_state=0).fit(x).get_result()
    g2 = FastGcStar(n_perm=200, n_pasts=2, method="fcgc", random_state=0).fit(x).get_result()
    assert np.array_equal(g1.lag_only, g2.lag_only)


def test_parametric_mode_runs():
    x = _var_data(seed=4)
    fast = FastGcStar(
        n_perm=100, n_pasts=2, method="fcgc", random_state=0, exact_permutation=False
    ).fit(x)
    res = fast.get_result(alpha=0.05, beta=0.05)
    assert res.lag_only.shape == (x.shape[0], x.shape[0])


def test_cgc_path_runs_and_finds_edges():
    x = _var_data(seed=5)
    fast = FastGcStar(n_perm=200, n_pasts=2, n_lags=2, method="cgc", random_state=0).fit(x)
    graph = fast.get_result(alpha=0.05, beta=0.05).lag_only
    assert graph.shape == (x.shape[0], x.shape[0])
    assert (graph != 0).sum() >= 1
