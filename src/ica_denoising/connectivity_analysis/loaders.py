"""Loaders for already-saved connectivity artifacts.

The estimator directory layout is::

    outputs/connectivity/<group>/<recording>/<estimator>/
        <variant>.pkl              # n_past -> matrix (v2a-RSNs)
        connectivity_matrices.pkl  # variant -> matrix (motorneurons)
        weighted/<variant>.npy     # collapsed weighted matrix per variant
        metadata/<variant>.json
        run_summary_p2.{csv,json}

These loaders never rerun an estimator. They prefer the ``weighted/`` arrays
(present for both groups) and fall back to the pickles when needed.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Mapping

import numpy as np

from .config import ConnectivityAnalysisConfig

logger = logging.getLogger(__name__)

__all__ = ["load_variant_matrices", "load_trace_array"]

_COMBINED_PICKLE = "connectivity_matrices.pkl"


def _collapse_value(value: object, n_past: int) -> np.ndarray | None:
    """Collapse a stored pickle value into a single 2D matrix.

    Args:
        value: Either an array or a mapping of ``n_past`` -> array.
        n_past: Conditioning order to select from a mapping.

    Returns:
        A 2D ``ndarray`` or ``None`` if the value cannot be interpreted.
    """

    if isinstance(value, np.ndarray):
        return value if value.ndim == 2 else None
    if isinstance(value, Mapping):
        if n_past in value:
            inner = value[n_past]
        elif len(value) == 1:
            inner = next(iter(value.values()))
        else:
            # Fall back to the smallest available key for determinism.
            key = sorted(value.keys(), key=lambda k: (not isinstance(k, int), k))[0]
            inner = value[key]
        return _collapse_value(inner, n_past)
    return None


def _load_pickle(path: Path) -> object:
    """Load a pickle file.

    Args:
        path: Path to the pickle file.

    Returns:
        The unpickled object.
    """

    with path.open("rb") as handle:
        return pickle.load(handle)


def _matrices_from_weighted(weighted_dir: Path) -> dict[str, np.ndarray]:
    """Load every ``weighted/<variant>.npy`` matrix.

    Args:
        weighted_dir: Path to the ``weighted`` subdirectory.

    Returns:
        Mapping of variant name to weighted matrix.
    """

    matrices: dict[str, np.ndarray] = {}
    for npy_path in sorted(weighted_dir.glob("*.npy")):
        matrices[npy_path.stem] = np.load(npy_path)
    return matrices


def _matrices_from_pickles(
    estimator_dir: Path, n_past: int
) -> dict[str, np.ndarray]:
    """Reconstruct variant matrices from connectivity pickles.

    Args:
        estimator_dir: Estimator directory.
        n_past: Conditioning order to select from per-variant pickles.

    Returns:
        Mapping of variant name to matrix.
    """

    matrices: dict[str, np.ndarray] = {}
    combined_path = estimator_dir / _COMBINED_PICKLE
    if combined_path.exists():
        combined = _load_pickle(combined_path)
        if isinstance(combined, Mapping):
            for variant, value in combined.items():
                matrix = _collapse_value(value, n_past)
                if matrix is not None:
                    matrices[str(variant)] = matrix
            return matrices

    for pkl_path in sorted(estimator_dir.glob("*.pkl")):
        if pkl_path.name == _COMBINED_PICKLE:
            continue
        value = _load_pickle(pkl_path)
        matrix = _collapse_value(value, n_past)
        if matrix is not None:
            matrices[pkl_path.stem] = matrix
        else:
            logger.warning("Could not collapse pickle %s into a matrix.", pkl_path)
    return matrices


def load_variant_matrices(
    estimator_dir: str | Path,
    config: ConnectivityAnalysisConfig | None = None,
) -> dict[str, np.ndarray]:
    """Load the weighted connectivity matrix for each variant.

    Prefers the ``weighted/`` arrays and falls back to the connectivity pickles
    (per-variant or combined) when ``weighted/`` is missing or empty.

    Args:
        estimator_dir: Path to an ``.../<estimator>/`` directory.
        config: Analysis configuration (only ``n_past`` is used here). Defaults
            are used when ``None``.

    Returns:
        Mapping of variant name (e.g. ``"raw"``, ``"fastica_cleaned"``) to its
        weighted connectivity matrix.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no variant matrices can be loaded.
    """

    cfg = config or ConnectivityAnalysisConfig()
    estimator_path = Path(estimator_dir)
    if not estimator_path.is_dir():
        raise FileNotFoundError(f"Estimator directory not found: {estimator_path}")

    weighted_dir = estimator_path / "weighted"
    matrices: dict[str, np.ndarray] = {}
    if weighted_dir.is_dir():
        matrices = _matrices_from_weighted(weighted_dir)
    if not matrices:
        matrices = _matrices_from_pickles(estimator_path, cfg.n_past)
    if not matrices:
        raise ValueError(f"No variant matrices found under {estimator_path}.")
    return matrices


def load_trace_array(path: str | Path) -> np.ndarray:
    """Load a trace matrix saved as ``.npy``.

    Args:
        path: Path to a ``(n_neurons, n_frames)`` ``.npy`` file supplied by the
            caller (the ``source_path`` values in run summaries are external and
            stale, so callers must provide a valid path).

    Returns:
        The trace array.

    Raises:
        FileNotFoundError: If the file does not exist.
    """

    trace_path = Path(path)
    if not trace_path.is_file():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    return np.load(trace_path)
