"""Adapters that connect estimators to the experiment harness."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from pathlib import Path
import sys

import numpy as np


def _load_gcstar_class() -> type:
    """Load GcStar from the notebook-friendly causalised module."""

    module_path = Path(__file__).resolve().parents[1] / "core" / "causalised-GC.py"
    spec = importlib.util.spec_from_file_location(
        "csl.core.causalised_gc",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GcStar from {module_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GcStar


GcStar = _load_gcstar_class()


def _design_matrix(X: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a stacked autoregressive design matrix from ``X``."""

    T, d = X.shape
    Y = X[p:]
    rows = []
    for t in range(p, T):
        rows.append(np.concatenate([X[t - lag] for lag in range(1, p + 1)]))
    Z = np.asarray(rows)
    return Z, Y


def _binary_adjacency_from_lstsq(
    X: np.ndarray,
    p: int,
    *,
    threshold: float = 0.08,
) -> np.ndarray:
    """Baseline least-squares adjacency estimator used for smoke tests."""

    _, d = X.shape
    Z, Y = _design_matrix(X, p)
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    beta = beta.reshape(p, d, d)

    adjacency = np.zeros((d, d), dtype=int)
    for lag in range(p):
        adjacency = np.logical_or(adjacency, np.abs(beta[lag]) > threshold)

    adjacency = adjacency.astype(int)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def analyze_with_baseline_lstsq(
    X: np.ndarray,
    p_values: list[int],
) -> dict[int, np.ndarray]:
    """Run the baseline least-squares estimator for each depth in ``p_values``."""

    return {
        int(p_value): _binary_adjacency_from_lstsq(X, int(p_value))
        for p_value in p_values
    }


def _run_gcstar_single_depth(
    X: np.ndarray,
    p: int,
    *,
    method: str,
    alpha: float,
    beta: float,
    n_perm: int,
    n_lags: int,
    temporal: bool,
    verbose: int,
    simulation: bool,
) -> np.ndarray:
    """Run ``GcStar`` for one conditioning depth and return a binary graph."""

    estimator = GcStar(
        n_perm=n_perm,
        n_pasts=int(p),
        n_lags=n_lags,
        temporal=temporal,
        method=method,
    )
    estimator.fit(np.asarray(X).T, verbose=verbose)
    connectivity = estimator.get_connectivity_matrix(
        simulation=simulation,
        alpha=alpha,
        beta=beta,
    )
    adjacency = (np.asarray(connectivity) != 0).astype(int)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def make_gcstar_analyzer(
    method: str,
    *,
    alpha: float = 0.01,
    beta: float = 0.001,
    n_perm: int = 200,
    n_lags: int = 1,
    temporal: bool = True,
    verbose: int = 0,
    simulation: bool = True,
) -> Callable[[np.ndarray, list[int]], dict[int, np.ndarray]]:
    """Create an analyzer function backed by ``GcStar``."""

    if method not in {"cgc", "fcgc"}:
        raise ValueError("method must be 'cgc' or 'fcgc'.")

    def analyze(X: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
        return {
            int(p_value): _run_gcstar_single_depth(
                X,
                int(p_value),
                method=method,
                alpha=alpha,
                beta=beta,
                n_perm=n_perm,
                n_lags=n_lags,
                temporal=temporal,
                verbose=verbose,
                simulation=simulation,
            )
            for p_value in p_values
        }

    return analyze


def analyze_with_gcstar_cgc(
    X: np.ndarray,
    p_values: list[int],
) -> dict[int, np.ndarray]:
    """Run ``GcStar`` with the conventional c-GC conditioning set."""

    return make_gcstar_analyzer("cgc")(X, p_values)


def analyze_with_gcstar_fcgc(
    X: np.ndarray,
    p_values: list[int],
) -> dict[int, np.ndarray]:
    """Run ``GcStar`` with the full-conditioning fcGC variant."""

    return make_gcstar_analyzer("fcgc")(X, p_values)


def analyze_with_user_method(
    X: np.ndarray, p_values: list[int]
) -> dict[int, np.ndarray]:
    """Placeholder for user-supplied methods.

    Replace this function body if you want to keep the external-method hook
    inside the package instead of using ``--user-method module:function``.
    """

    raise NotImplementedError(
        "Replace analyze_with_user_method with your custom analyzer or use --user-method."
    )


def normalize_adjacency_output(
    raw: object,
    p_values: list[int],
) -> dict[int, np.ndarray]:
    """Normalize user-method outputs into ``dict[int, adjacency]`` form."""

    if isinstance(raw, dict):
        output = {int(key): np.asarray(value).astype(int) for key, value in raw.items()}
    elif isinstance(raw, (list, tuple)):
        if len(raw) != len(p_values):
            raise ValueError("List/tuple method output must match len(p_values).")
        output = {
            int(p_value): np.asarray(value).astype(int)
            for p_value, value in zip(p_values, raw, strict=True)
        }
    else:
        raise TypeError(
            "User method must return dict[p, adjacency] or a list of adjacency matrices."
        )

    required = {int(p_value) for p_value in p_values}
    missing = sorted(required.difference(output))
    if missing:
        raise ValueError(f"User method output is missing p-values: {missing}")

    d = next(iter(output.values())).shape[0]
    for p_value, adjacency in output.items():
        if adjacency.shape != (d, d):
            raise ValueError(
                f"Adjacency for p={p_value} has shape {adjacency.shape}, expected {(d, d)}."
            )
        adjacency = (adjacency != 0).astype(int)
        np.fill_diagonal(adjacency, 0)
        output[p_value] = adjacency
    return output


def load_external_method(
    spec: str,
) -> Callable[[np.ndarray, list[int]], dict[int, np.ndarray]]:
    """Load an analyzer from ``module:function`` notation."""

    if ":" not in spec:
        raise ValueError("--user-method must be formatted as 'module:function'.")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)

    def wrapped(X: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
        return normalize_adjacency_output(function(X, p_values), p_values)

    return wrapped


METHODS = {
    "baseline_lstsq": analyze_with_baseline_lstsq,
    "gcstar_cgc": analyze_with_gcstar_cgc,
    "gcstar_fcgc": analyze_with_gcstar_fcgc,
    "user_method": analyze_with_user_method,
}
