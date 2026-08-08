"""Tests for the hardened c-GC estimator (Section 5.2)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from csl.core import CGCResult, GcStar, benjamini_hochberg, fit_cgc


def _load_path_module():
    """Load the hyphenated shim exactly as ``adapters.py`` does."""

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "csl"
        / "core"
        / "causalised-GC.py"
    )
    spec = importlib.util.spec_from_file_location(
        "csl.core.causalised_gc_pathload", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _var_one_edge(t: int = 300, coef: float = 0.85, seed: int = 7) -> np.ndarray:
    """One-edge VAR: variable 0 drives variable 1 at lag 1.

    Returns array shaped ``(2, T)`` (variables x time).
    """

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(t)
    y = np.zeros(t)
    for s in range(1, t):
        y[s] = coef * x[s - 1] + 0.3 * rng.standard_normal()
    return np.vstack([x, y])


# --------------------------------------------------------------------------- #
# Import paths
# --------------------------------------------------------------------------- #
def test_package_import_yields_gcstar():
    assert isinstance(GcStar(), GcStar)


def test_path_load_yields_gcstar():
    module = _load_path_module()
    assert hasattr(module, "GcStar")
    estimator = module.GcStar(n_perm=10, n_pasts=1)
    assert isinstance(estimator, module.GcStar)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_determinism_same_random_state():
    data = _var_one_edge()
    a = GcStar(n_perm=80, n_pasts=2, random_state=123).fit(data)
    b = GcStar(n_perm=80, n_pasts=2, random_state=123).fit(data)
    np.testing.assert_array_equal(a.pVal_inv_corr_, b.pVal_inv_corr_)
    np.testing.assert_array_equal(a.pVal_corr_, b.pVal_corr_)
    np.testing.assert_array_equal(
        a.get_connectivity_matrix(), b.get_connectivity_matrix()
    )


def test_different_random_state_changes_pvalues():
    data = _var_one_edge()
    a = GcStar(n_perm=80, n_pasts=2, random_state=1).fit(data)
    b = GcStar(n_perm=80, n_pasts=2, random_state=2).fit(data)
    assert not np.array_equal(a.pVal_inv_corr_, b.pVal_inv_corr_)


# --------------------------------------------------------------------------- #
# p-value bounds
# --------------------------------------------------------------------------- #
def test_pvalue_bounds():
    data = _var_one_edge()
    n_perm = 60
    est = GcStar(n_perm=n_perm, n_pasts=2, random_state=0).fit(data)
    for pvals in (est.pVal_corr_, est.pVal_inv_corr_):
        assert np.all(pvals > 0.0)
        assert np.all(pvals <= 1.0)
        assert np.all(pvals >= 1.0 / (n_perm + 1) - 1e-12)


# --------------------------------------------------------------------------- #
# FDR
# --------------------------------------------------------------------------- #
def test_bh_q_ge_p_and_monotone():
    rng = np.random.default_rng(0)
    pvals = rng.uniform(0, 1, size=50)
    q = benjamini_hochberg(pvals)
    assert np.all(q >= pvals - 1e-12)
    assert np.all(q <= 1.0)
    order = np.argsort(pvals)
    assert np.all(np.diff(q[order]) >= -1e-12)  # monotone step-up in p order


def test_bh_known_values():
    pvals = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    q = benjamini_hochberg(pvals)
    # q_i = p_i * n / rank, then monotone min from the top -> all 0.05 here.
    np.testing.assert_allclose(q, 0.05, atol=1e-12)


def test_result_has_qvalues():
    data = _var_one_edge()
    result = fit_cgc(data, n_perm=40, n_pasts=2, random_state=0)
    assert isinstance(result, CGCResult)
    assert result.qvalues.shape == result.pvalues.shape
    assert np.all(result.qvalues >= result.pvalues - 1e-12)


# --------------------------------------------------------------------------- #
# Lag collapse & contemporaneous exclusion
# --------------------------------------------------------------------------- #
def test_lag_collapse_shapes():
    data = _var_one_edge()
    result = fit_cgc(data, n_perm=40, n_pasts=3, random_state=0)
    n = data.shape[0]
    assert result.lag_specific.shape == (3, n, n)
    assert result.lag_only.shape == (n, n)
    assert result.lags == (1, 2, 3)
    # Collapsed lag-only is the element-wise max over lag-specific blocks.
    np.testing.assert_array_equal(
        result.lag_only, result.lag_specific.max(axis=0)
    )


def test_contemporaneous_excluded_by_default():
    data = _var_one_edge()
    excluded = fit_cgc(data, n_perm=40, n_pasts=2, random_state=0)
    assert excluded.contemporaneous is None
    included = fit_cgc(
        data, n_perm=40, n_pasts=2, random_state=0, include_contemporaneous=True
    )
    assert included.contemporaneous is not None
    assert included.contemporaneous.shape == (2, 2)
    # lag_only never contains the contemporaneous block.
    np.testing.assert_array_equal(excluded.lag_only, included.lag_only)


# --------------------------------------------------------------------------- #
# Graph orientation
# --------------------------------------------------------------------------- #
def test_orientation_recovered():
    data = _var_one_edge(t=400, coef=0.9, seed=3)
    result = fit_cgc(
        data, n_perm=120, n_pasts=1, random_state=0, alpha=0.05, beta=0.05
    )
    lag_only = result.lag_only
    # Edge 0 -> 1 recovered, reverse edge 1 -> 0 absent.
    assert lag_only[0, 1] > 0.0
    assert lag_only[1, 0] == 0.0


def test_connectivity_matrix_orientation_backcompat():
    data = _var_one_edge(t=400, coef=0.9, seed=3)
    est = GcStar(n_perm=120, n_pasts=1, random_state=0).fit(data)
    conn = est.get_connectivity_matrix(simulation=True, alpha=0.05, beta=0.05)
    assert conn.shape == (2, 2)
    assert conn[0, 1] > 0.0
    assert conn[1, 0] == 0.0


# --------------------------------------------------------------------------- #
# Robustness: zero-variance and low-rank inputs
# --------------------------------------------------------------------------- #
def test_zero_variance_warns_and_pvalues_high():
    data = _var_one_edge()
    data[0, :] = 1.0  # constant first variable
    est = GcStar(n_perm=40, n_pasts=1, random_state=0).fit(data)
    result = est.get_result()
    assert any("Zero-variance" in w for w in result.warnings)
    # Correlations involving the constant row collapse to non-significant.
    assert np.all(est.pVal_corr_ > 0.0)
    assert np.all(np.isfinite(est.corr_))


def test_low_rank_input_no_crash():
    base = _var_one_edge()
    # Append a perfectly collinear variable to force rank deficiency.
    collinear = base[0] + base[1]
    data = np.vstack([base, collinear])
    result = fit_cgc(data, n_perm=30, n_pasts=2, random_state=0)
    assert np.all(np.isfinite(result.lag_only))
    assert result.numerical_rank is not None


def test_underdetermined_warns():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 6))  # tiny T relative to conditioning dim
    est = GcStar(n_perm=10, n_pasts=2, random_state=0).fit(data)
    assert any("too small" in w for w in est._warnings)


def test_underdetermined_raises_when_configured():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 6))
    est = GcStar(
        n_perm=10, n_pasts=2, random_state=0, raise_on_underdetermined=True
    )
    with pytest.raises(ValueError):
        est.fit(data)
