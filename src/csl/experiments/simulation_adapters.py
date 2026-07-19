"""Connectivity estimators for the simulation benchmark (Section 5.1).

Each adapter consumes traces shaped ``(n_neurons, T)`` and returns a
:class:`GraphEstimate` with a weighted score matrix and a binary adjacency in
the ``[source, target]`` convention. The simulation c-GC family can use either
the vectorized :class:`csl.core.new_causalised_GC.FastGcStar` implementation or
the normal hardened :mod:`csl.core.causalised_gc` module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "GraphEstimate",
    "estimate_cgc",
    "estimate_var",
    "estimate_pcmci",
    "estimate_jpcmciplus",
    "ESTIMATORS",
]


def _load_fit_cgc():
    """Load ``fit_cgc`` from the hardened module, falling back to path load."""

    try:
        from csl.core.causalised_gc import fit_cgc

        return fit_cgc
    except Exception:  # pragma: no cover - exotic environments
        import importlib.util
        import sys
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "core" / "causalised_gc.py"
        spec = importlib.util.spec_from_file_location("csl.core.causalised_gc", path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load fit_cgc.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.fit_cgc


def _load_fast_gcstar():
    """Load the vectorized c-GC estimator, returning ``None`` if unavailable."""

    try:
        from csl.core.new_causalised_GC import FastGcStar

        return FastGcStar
    except Exception:  # pragma: no cover - fallback for exotic installs
        return None


@dataclass(frozen=True)
class GraphEstimate:
    """A connectivity estimate in ``[source, target]`` orientation."""

    estimator: str
    scores: np.ndarray  # continuous weighted evidence (n, n)
    binary: np.ndarray  # binary adjacency (n, n)
    binary_fdr: Optional[np.ndarray]
    warnings: tuple[str, ...]


def estimate_cgc(
    traces: np.ndarray,
    *,
    method: str = "cgc",
    backend: str = "fast",
    n_pasts: int = 2,
    n_lags: int = 2,
    n_perm: int = 200,
    alpha: float = 0.01,
    beta: float = 0.001,
    compute_fdr: bool = False,
    random_state: int = 0,
) -> GraphEstimate:
    """Run the c-GC (``method='cgc'``) or c-GC* (``method='fcgc'``) estimator.

    Contemporaneous edges are excluded from the primary graph; the collapsed
    lag-only matrix is returned as the score. Set ``compute_fdr=True`` only for
    legacy benchmark outputs that explicitly request FDR-scored graphs.
    """

    data = np.asarray(traces, dtype=float)
    backend = str(backend).lower()
    if backend not in {"fast", "normal"}:
        raise ValueError("backend must be 'fast' or 'normal'.")

    fast_gcstar = _load_fast_gcstar() if backend == "fast" else None
    if fast_gcstar is not None:
        estimator = fast_gcstar(
            n_perm=n_perm,
            n_pasts=n_pasts,
            n_lags=n_lags,
            method=method,
            random_state=random_state,
        ).fit(data)
        result = estimator.get_result(
            alpha=alpha,
            beta=beta,
            include_contemporaneous=False,
        )
        fdr_result = None
        if compute_fdr:
            fdr_result = estimator.get_result(
                alpha=alpha,
                beta=beta,
                include_contemporaneous=False,
                use_fdr=True,
            )
    else:
        fit_cgc = _load_fit_cgc()
        result = fit_cgc(
            data,
            n_perm=n_perm,
            n_pasts=n_pasts,
            n_lags=n_lags,
            method=method,
            random_state=random_state,
            alpha=alpha,
            beta=beta,
            include_contemporaneous=False,
        )
        fdr_result = None
        if compute_fdr:
            fdr_result = fit_cgc(
                data,
                n_perm=n_perm,
                n_pasts=n_pasts,
                n_lags=n_lags,
                method=method,
                random_state=random_state,
                alpha=alpha,
                beta=beta,
                include_contemporaneous=False,
                use_fdr=True,
            )
    scores = np.asarray(result.lag_only, dtype=float)
    binary = (scores != 0).astype(int)
    np.fill_diagonal(binary, 0)

    binary_fdr = None
    if fdr_result is not None:
        binary_fdr = (np.asarray(fdr_result.lag_only) != 0).astype(int)
        np.fill_diagonal(binary_fdr, 0)

    return GraphEstimate(
        estimator=method,
        scores=scores,
        binary=binary,
        binary_fdr=binary_fdr,
        warnings=tuple(result.warnings),
    )


def _var_design(x: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    t, d = x.shape
    rows = [np.concatenate([x[s - lag] for lag in range(1, p + 1)]) for s in range(p, t)]
    return np.asarray(rows), x[p:]


def estimate_var(
    traces: np.ndarray,
    *,
    orders: tuple[int, ...] = (1, 2, 3),
    threshold: float = 0.05,
) -> GraphEstimate:
    """Linear VAR/Granger baseline with BIC order selection (training info only)."""

    x = np.asarray(traces, dtype=float).T  # (T, d)
    t, d = x.shape
    best_order, best_bic = orders[0], np.inf
    for p in orders:
        if t - p <= d * p + 1:
            continue
        z, y = _var_design(x, p)
        beta, *_ = np.linalg.lstsq(z, y, rcond=None)
        resid = y - z @ beta
        sigma = max(float(np.mean(resid**2)), 1e-12)
        k = d * d * p
        bic = (t - p) * np.log(sigma) + k * np.log(t - p)
        if bic < best_bic:
            best_bic, best_order = bic, p

    z, y = _var_design(x, best_order)
    beta, *_ = np.linalg.lstsq(z, y, rcond=None)
    coeffs = beta.reshape(best_order, d, d)  # [lag, source, target]
    scores = np.max(np.abs(coeffs), axis=0)  # [source, target]
    binary = (scores > threshold).astype(int)
    np.fill_diagonal(binary, 0)
    np.fill_diagonal(scores, 0.0)
    return GraphEstimate(
        estimator="var",
        scores=scores,
        binary=binary,
        binary_fdr=None,
        warnings=(),
    )


def estimate_pcmci(
    traces: np.ndarray,
    *,
    tau_max: int = 2,
    alpha: float = 0.05,
) -> GraphEstimate:
    """PCMCI+ estimator via Tigramite (smoke/core 10-neuron conditions only)."""

    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    x = np.asarray(traces, dtype=float).T
    dataframe = pp.DataFrame(x)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=alpha)
    graph = results["graph"]  # (d, d, tau+1)
    d = x.shape[1]
    scores = np.zeros((d, d))
    binary = np.zeros((d, d), dtype=int)
    val = results.get("val_matrix")
    for i in range(d):
        for j in range(d):
            for lag in range(1, graph.shape[2]):
                if graph[i, j, lag] in ("-->", "o->"):
                    binary[i, j] = 1
                    if val is not None:
                        scores[i, j] = max(scores[i, j], abs(val[i, j, lag]))
    np.fill_diagonal(binary, 0)
    return GraphEstimate(
        estimator="pcmci",
        scores=scores,
        binary=binary,
        binary_fdr=None,
        warnings=(),
    )


def estimate_jpcmciplus(
    traces: np.ndarray,
    *,
    tau_max: int = 2,
    alpha: float = 0.05,
) -> GraphEstimate:
    """JPCMCI+ estimator via Tigramite with all observed nodes as system nodes."""

    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr_mult import ParCorrMult
    from tigramite.jpcmciplus import JPCMCIplus

    x = np.asarray(traces, dtype=float).T
    dataframe = pp.DataFrame(x)
    node_classification = {i: "system" for i in range(x.shape[1])}
    jpcmciplus = JPCMCIplus(
        dataframe=dataframe,
        cond_ind_test=ParCorrMult(significance="analytic"),
        node_classification=node_classification,
        verbosity=0,
    )
    results = jpcmciplus.run_jpcmciplus(
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=alpha,
    )
    graph = results["graph"]
    d = x.shape[1]
    scores = np.zeros((d, d))
    binary = np.zeros((d, d), dtype=int)
    val = results.get("val_matrix")
    for i in range(d):
        for j in range(d):
            for lag in range(1, graph.shape[2]):
                if graph[i, j, lag] in ("-->", "o->"):
                    binary[i, j] = 1
                    if val is not None:
                        scores[i, j] = max(scores[i, j], abs(val[i, j, lag]))
    np.fill_diagonal(binary, 0)
    return GraphEstimate(
        estimator="jpcmciplus",
        scores=scores,
        binary=binary,
        binary_fdr=None,
        warnings=(),
    )


ESTIMATORS = {
    "cgc": lambda tr, **kw: estimate_cgc(tr, method="cgc", **kw),
    "cgc_star": lambda tr, **kw: estimate_cgc(tr, method="fcgc", **kw),
    "var": estimate_var,
    "pcmci": estimate_pcmci,
    "jpcmciplus": estimate_jpcmciplus,
}
