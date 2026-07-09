"""Hardened c-GC / fcGC estimator implementation.

This module is the source of truth for the ``GcStar`` estimator. It exposes a
deterministic, FDR-aware estimator together with a structured :class:`CGCResult`
container. The hyphenated ``causalised-GC.py`` file is a thin compatibility shim
that re-exports the public names defined here so that path-based loaders keep
working.

Determinism is guaranteed by drawing all circular-shift permutation indices from
an explicit :class:`numpy.random.Generator` seeded with ``random_state``: the
same ``random_state`` yields identical output.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from numba import jit
except Exception:  # pragma: no cover - numba is optional at runtime

    def jit(*_args, **_kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


__all__ = [
    "GcStar",
    "CGCResult",
    "regression_residual",
    "benjamini_hochberg",
    "fit_cgc",
]


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values (q-values).

    The step-up procedure is implemented with NumPy only (no extra
    dependency). Adjusted values are monotone-corrected and clipped to
    ``[0, 1]``; they always satisfy ``q >= p`` element-wise.

    Args:
        pvals: Array of raw p-values of arbitrary shape.

    Returns:
        Array of adjusted p-values with the same shape as ``pvals``.
    """

    flat = np.asarray(pvals, dtype=float).ravel()
    n = flat.size
    if n == 0:
        return flat.reshape(pvals.shape)

    order = np.argsort(flat)
    ranked = flat[order]
    factors = n / np.arange(1, n + 1)
    adjusted = ranked * factors
    # Enforce monotonic non-increasing values scanning from largest p-value.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out.reshape(np.asarray(pvals).shape)


@jit(nopython=True)
def _perm_count_numba(x: np.ndarray, y: np.ndarray, shifts: np.ndarray) -> int:
    """Count circular-shift permutations at least as extreme as the observed.

    Degenerate inputs (length-one or zero-variance series) return
    ``shifts.size`` so the corrected p-value collapses to ``1``.
    """

    if x.size <= 1 or y.size <= 1 or shifts.size == 0:
        return shifts.size
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return shifts.size

    abs_obs = np.abs(np.corrcoef(x, y)[1, 0])
    count = 0
    for k in range(shifts.size):
        shift = shifts[k]
        rolled = np.hstack((x[shift:], x[:shift]))
        if np.abs(np.corrcoef(rolled, y)[1, 0]) >= abs_obs:
            count += 1
    return count


@jit(nopython=True)
def _perm_test_numba(x: np.ndarray, y: np.ndarray, n_perm: int) -> float:
    """Back-compatible permutation p-value using internal RNG.

    Retained for callers that imported the original helper. New code should use
    the deterministic generator-driven path in :class:`GcStar`.
    """

    if x.size <= 1 or y.size <= 1 or n_perm <= 0:
        return 1.0
    abs_obs = np.abs(np.corrcoef(x, y)[1, 0])
    count = 0
    for _ in range(n_perm):
        shift = np.random.randint(1, x.size)
        rolled = np.hstack((x[shift:], x[:shift]))
        if np.abs(np.corrcoef(rolled, y)[1, 0]) >= abs_obs:
            count += 1
    return count / n_perm


def _residual_and_rank(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, int]:
    """Regress ``z`` out of ``x`` and return residual plus design rank."""

    if z.size == 0:
        return x - np.mean(x), 1
    if z.ndim == 1:
        z = z[np.newaxis, :]

    design = np.vstack([z, np.ones(z.shape[1])]).T
    try:
        coef, _res, rank, _sv = np.linalg.lstsq(design, x, rcond=None)
    except np.linalg.LinAlgError:
        # NumPy's gelsd (divide-and-conquer SVD) can raise "SVD did not
        # converge" on rank-deficient designs (e.g. cluster-cleaned low-rank
        # traces). Fall back to the gelsy driver (complete orthogonal
        # factorization), which is robust to rank deficiency.
        from scipy.linalg import lstsq as _scipy_lstsq

        coef, _res, rank, _sv = _scipy_lstsq(design, x, lapack_driver="gelsy")
    fitted = design @ coef
    return x - fitted, int(rank)


def regression_residual(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return residuals after regressing ``z`` out of ``x`` (back-compat API)."""

    residual, _rank = _residual_and_rank(x, z)
    return residual


@dataclass(frozen=True)
class CGCResult:
    """Structured output of a fitted :class:`GcStar` estimator.

    Attributes:
        lag_specific: Weighted adjacency per lag, shape ``(n_lags, n, n)`` where
            entry ``[k, src, tgt]`` is the significant weight of the lag-``k+1``
            edge ``src -> tgt``.
        lag_only: Collapsed lag-only weighted adjacency ``(n, n)`` obtained by
            element-wise max over ``lag_specific`` (contemporaneous excluded).
        contemporaneous: Optional weighted contemporaneous adjacency ``(n, n)``;
            ``None`` unless ``include_contemporaneous`` was requested.
        weights: Significance-masked ``|correlation|`` weights, shape
            ``(n_rows, n)`` across every lag block.
        pvalues: Conditional permutation p-values, shape ``(n_rows, n)``.
        qvalues: Benjamini-Hochberg adjusted q-values, shape ``(n_rows, n)``.
        lags: Tuple of lag indices represented in ``lag_specific``.
        numerical_rank: Rank of the stacked lagged design, if recorded.
        residual_variance: Mean conditional residual variance, if recorded.
        warnings: Conditioning / data warnings raised during the fit.
    """

    lag_specific: np.ndarray
    lag_only: np.ndarray
    contemporaneous: Optional[np.ndarray]
    weights: np.ndarray
    pvalues: np.ndarray
    qvalues: np.ndarray
    lags: tuple[int, ...]
    numerical_rank: Optional[int]
    residual_variance: Optional[float]
    warnings: tuple[str, ...]


@dataclass
class GcStar:
    """Granger-causality estimator from a causal-Bayesian-network viewpoint.

    Args:
        n_perm: Number of circular-shift permutations for p-value estimation.
        n_pasts: Conditioning depth used to construct the shifted state.
        n_lags: Number of lag blocks merged by :meth:`get_connectivity_matrix`.
        temporal: Reserved compatibility flag.
        method: Either ``"cgc"`` or ``"fcgc"``.
        parallel: If ``True``, run the unconditional and conditional passes in
            separate worker threads.
        random_state: Seed for the permutation generator (determinism).
        raise_on_underdetermined: If ``True``, raise instead of warning when the
            sample count is too small for the conditioning dimension.
    """

    n_perm: int = 1000
    n_pasts: int = 1
    n_lags: int = 1
    temporal: bool = True
    method: str = "cgc"
    parallel: bool = False
    random_state: int = 0
    raise_on_underdetermined: bool = False
    corr_: Optional[np.ndarray] = None
    pVal_corr_: Optional[np.ndarray] = None
    inv_corr_: Optional[np.ndarray] = None
    pVal_inv_corr_: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate basic configuration."""

        if self.method not in {"cgc", "fcgc"}:
            raise ValueError("method must be 'cgc' or 'fcgc'.")
        if self.n_pasts < 0:
            raise ValueError("n_pasts must be non-negative.")
        if self.n_lags < 1:
            raise ValueError("n_lags must be at least 1.")
        self.logger = logger
        self._warnings: list[str] = []
        self.numerical_rank_: Optional[int] = None
        self.residual_variance_: Optional[float] = None

    def _configure_logging(self, verbose: int) -> None:
        """Configure module logging for interactive runs."""

        level = logging.WARNING
        if verbose == 1:
            level = logging.INFO
        elif verbose >= 2:
            level = logging.DEBUG
        logging.basicConfig(level=level, force=True)

    def _warn(self, message: str) -> None:
        """Record and emit a conditioning / data warning."""

        self._warnings.append(message)
        self.logger.warning(message)

    def _make_shifts(self, length: int) -> np.ndarray:
        """Draw a fixed circular-shift index array from the generator."""

        rng = getattr(self, "_rng", None)
        if rng is None:
            rng = np.random.default_rng(self.random_state)
            self._rng = rng
        high = max(length, 2)
        return rng.integers(1, high, size=self.n_perm).astype(np.int64)

    def shift_data(self, arr: np.ndarray) -> np.ndarray:
        """Create stacked lagged views of ``arr`` shaped ``(n_variables, T)``."""

        self.n_neur = arr.shape[0]
        if self.n_pasts == 0:
            return arr.copy()

        trimmed = arr[:, self.n_pasts :]
        for index in range(self.n_pasts):
            start = self.n_pasts - 1 - index
            stop = -index - 1
            trimmed = np.r_[trimmed, arr[:, start:stop]]
        return trimmed

    def get_conditioning_set(self, data: np.ndarray, i: int, j: int) -> np.ndarray:
        """Build the c-GC conditioning set for a single directed pair."""

        self.shifted_data = self.shift_data(data.copy())
        source_index = i % self.n_neur
        excluded_history = [
            source_index + lag * self.n_neur for lag in range(i // self.n_neur)
        ]
        excluded = np.r_[np.array(excluded_history, dtype=int), [i, j]]
        return np.delete(self.shifted_data, excluded, axis=0)

    def correlation_func(
        self, data: np.ndarray, shifts: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute unconditional dependence and corrected permutation p-values."""

        self.n_neur = data.shape[0]
        shifted = self.shift_data(data.copy())
        n_rows = shifted.shape[0]
        if shifts is None:
            shifts = self._make_shifts(shifted.shape[1])

        corr = np.nan_to_num(np.abs(np.corrcoef(shifted)))
        pvals = np.ones((n_rows, self.n_neur))
        for i in range(n_rows):
            for j in range(self.n_neur):
                count = _perm_count_numba(shifted[i, :], shifted[j, :], shifts)
                pvals[i, j] = (count + 1) / (self.n_perm + 1)
        return corr[:, : self.n_neur], pvals

    def inv_correlation_func(
        self, data: np.ndarray, shifts: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute conditional dependence using residual correlations."""

        self.n_neur = data.shape[0]
        shifted = self.shift_data(data.copy())
        n_rows = shifted.shape[0]
        if shifts is None:
            shifts = self._make_shifts(shifted.shape[1])

        inv_corr = np.zeros((n_rows, self.n_neur))
        pvals = np.ones((n_rows, self.n_neur))
        residual_vars: list[float] = []
        min_rank: Optional[int] = None

        for i in range(n_rows):
            for j in range(self.n_neur):
                x = shifted[i]
                y = shifted[j]
                if self.method == "fcgc":
                    z = np.delete(shifted.copy(), [i, j], axis=0)
                else:
                    z = self.get_conditioning_set(data, i, j)

                x_res, rank = _residual_and_rank(x, z)
                y_res, _ = _residual_and_rank(y, z)
                residual_vars.append(float(np.var(x_res)))
                min_rank = rank if min_rank is None else min(min_rank, rank)

                inv_corr[i, j] = np.nan_to_num(
                    np.abs(np.corrcoef(x_res, y_res)[1, 0])
                )
                count = _perm_count_numba(x_res, y_res, shifts)
                pvals[i, j] = (count + 1) / (self.n_perm + 1)

        if residual_vars:
            self.residual_variance_ = float(np.mean(residual_vars))
        if min_rank is not None and min_rank < (n_rows - 1):
            self._warn(
                f"Conditioning design is rank-deficient (rank={min_rank} < "
                f"{n_rows - 1}); residuals fall back to a robust solver."
            )
        return inv_corr, pvals

    def _check_inputs(self, data: np.ndarray, n_rows: int, n_samples: int) -> None:
        """Validate sample size and variance, recording warnings."""

        constant_rows = [
            int(idx) for idx in range(data.shape[0]) if np.std(data[idx]) == 0.0
        ]
        if constant_rows:
            self._warn(f"Zero-variance input variables detected: {constant_rows}.")
        if n_samples <= n_rows:
            message = (
                f"Sample count T={n_samples} is too small for conditioning "
                f"dimension {n_rows}; estimates are unreliable."
            )
            if self.raise_on_underdetermined:
                raise ValueError(message)
            self._warn(message)

    def fit(self, data: np.ndarray, verbose: int = 0) -> "GcStar":
        """Fit the estimator on an array of shape ``(n_variables, T)``."""

        self.data = np.asarray(data, dtype=float).copy()
        self._warnings = []
        self._rng = np.random.default_rng(self.random_state)
        self._configure_logging(verbose)

        self.shifted_data = self.shift_data(self.data)
        n_rows, n_samples = self.shifted_data.shape
        self.numerical_rank_ = int(np.linalg.matrix_rank(self.shifted_data))
        self._check_inputs(self.data, n_rows, n_samples)
        shifts = self._make_shifts(n_samples)

        if self.parallel:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                corr_future = executor.submit(self.correlation_func, data, shifts)
                inv_future = executor.submit(self.inv_correlation_func, data, shifts)
                self.corr_, self.pVal_corr_ = corr_future.result()
                self.inv_corr_, self.pVal_inv_corr_ = inv_future.result()
        else:
            self.corr_, self.pVal_corr_ = self.correlation_func(data, shifts)
            self.inv_corr_, self.pVal_inv_corr_ = self.inv_correlation_func(
                data, shifts
            )
        return self

    def _significance_mask(
        self, alpha: float, beta: float, use_fdr: bool
    ) -> np.ndarray:
        """Return the combined unconditional/conditional significance mask."""

        if use_fdr:
            corr_sig = benjamini_hochberg(self.pVal_corr_) <= alpha
            inv_sig = benjamini_hochberg(self.pVal_inv_corr_) <= beta
        else:
            corr_sig = self.pVal_corr_ <= alpha
            inv_sig = self.pVal_inv_corr_ <= beta
        return corr_sig & inv_sig

    def get_result(
        self,
        *,
        alpha: float = 0.01,
        beta: float = 0.001,
        include_contemporaneous: bool = False,
        use_fdr: bool = False,
    ) -> CGCResult:
        """Assemble a structured :class:`CGCResult`.

        Lagged edge weights are taken from the corresponding lagged correlation
        block (never the contemporaneous block). The collapsed ``lag_only``
        matrix aggregates lags by element-wise maximum.
        """

        if self.corr_ is None or self.inv_corr_ is None:
            raise RuntimeError("fit must be called before get_result.")

        n = self.n_neur
        mask = self._significance_mask(alpha, beta, use_fdr)
        weights = self.corr_ * mask

        lag_blocks = [
            weights[k * n : (k + 1) * n, :n] for k in range(1, self.n_pasts + 1)
        ]
        if lag_blocks:
            lag_specific = np.stack(lag_blocks, axis=0)
            lag_only = lag_specific.max(axis=0)
        else:
            lag_specific = np.empty((0, n, n))
            lag_only = np.zeros((n, n))

        contemporaneous = weights[0:n, :n].copy() if include_contemporaneous else None
        qvalues = benjamini_hochberg(self.pVal_inv_corr_)

        return CGCResult(
            lag_specific=lag_specific,
            lag_only=lag_only,
            contemporaneous=contemporaneous,
            weights=weights,
            pvalues=self.pVal_inv_corr_.copy(),
            qvalues=qvalues,
            lags=tuple(range(1, self.n_pasts + 1)),
            numerical_rank=self.numerical_rank_,
            residual_variance=self.residual_variance_,
            warnings=tuple(self._warnings),
        )

    def get_connectivity_matrix(
        self,
        *,
        simulation: bool = True,
        alpha: float = 0.01,
        beta: float = 0.001,
    ) -> np.ndarray:
        """Compatibility API returning a single weighted connectivity matrix.

        With ``simulation=True`` (the adapter default) this returns the
        collapsed lag-only matrix (contemporaneous block excluded), equivalent
        to :attr:`CGCResult.lag_only` restricted to lags ``1..n_lags``. With
        ``simulation=False`` the contemporaneous block is merged in. Lagged
        edges are weighted by their own lag correlation, fixing the previous
        contemporaneous-weighting bug.
        """

        if self.corr_ is None or self.inv_corr_ is None:
            raise RuntimeError("fit must be called before get_connectivity_matrix.")

        n = self.n_neur
        mask = (self.pVal_corr_ <= alpha) & (self.pVal_inv_corr_ <= beta)
        n_blocks = self.n_pasts + 1
        blocks_bool = [mask[k * n : (k + 1) * n, :n] for k in range(n_blocks)]
        blocks_corr = [self.corr_[k * n : (k + 1) * n, :n] for k in range(n_blocks)]

        if simulation:
            if n_blocks == 1:
                included = [0]
            else:
                max_lag = min(self.n_lags, n_blocks - 1)
                included = list(range(1, max_lag + 1))
        else:
            if self.n_lags == 1:
                included = [0, 1] if n_blocks > 1 else [0]
            else:
                max_lag = min(self.n_lags, n_blocks - 1)
                included = list(range(0, max_lag + 1))

        weight = np.zeros((n, n))
        for k in included:
            weight = np.maximum(weight, blocks_corr[k] * blocks_bool[k])
        self.conn_mat = weight
        return self.conn_mat

    def compute_confusion_matrix(
        self, truth: np.ndarray, *, simulation: bool = True
    ) -> np.ndarray:
        """Compute the confusion matrix against a ground-truth adjacency."""

        if not hasattr(self, "conn_mat"):
            raise RuntimeError("get_connectivity_matrix must be called first.")

        target = truth.T if simulation else truth
        tp = np.sum(np.logical_and(target != 0, self.conn_mat != 0))
        fn = np.sum(np.logical_and(target != 0, self.conn_mat == 0))
        fp = np.sum(np.logical_and(target == 0, self.conn_mat != 0))
        tn = np.sum(np.logical_and(target == 0, self.conn_mat == 0))
        self.confusion_matrix = np.array([[tp, fp], [fn, tn]])
        return self.confusion_matrix

    def compute_metrics(self) -> np.ndarray:
        """Return accuracy, precision, recall, FPR, balanced accuracy, and F1."""

        if not hasattr(self, "confusion_matrix"):
            raise RuntimeError("compute_confusion_matrix must be called first.")

        tp, fp, fn, tn = self.confusion_matrix.flatten()
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        balanced_accuracy = (specificity + recall) / 2
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return np.array([accuracy, precision, recall, fpr, balanced_accuracy, f1])

    def compute_shd_sid(
        self, truth: np.ndarray, inferred: np.ndarray, *, simulation: bool = True
    ) -> np.ndarray:
        """Compute SHD, importing ``cdt`` only when needed."""

        from importlib import import_module

        try:
            metrics = import_module("cdt.metrics")
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "cdt.metrics is required for SHD/SID computations."
            ) from exc

        target = truth.T if simulation else truth
        self.shd_ = metrics.SHD(target=target, pred=inferred, double_for_anticausal=False)
        return self.shd_


def fit_cgc(
    data: np.ndarray,
    *,
    n_perm: int = 1000,
    n_pasts: int = 1,
    n_lags: int = 1,
    method: str = "cgc",
    random_state: int = 0,
    alpha: float = 0.01,
    beta: float = 0.001,
    include_contemporaneous: bool = False,
    use_fdr: bool = False,
    verbose: int = 0,
) -> CGCResult:
    """Fit a :class:`GcStar` estimator and return a :class:`CGCResult`.

    Args:
        data: Array shaped ``(n_variables, T)``.
        n_perm: Number of circular-shift permutations.
        n_pasts: Conditioning depth.
        n_lags: Lag-collapse depth (compatibility parameter).
        method: ``"cgc"`` or ``"fcgc"``.
        random_state: Seed for the permutation generator.
        alpha: Unconditional significance threshold.
        beta: Conditional significance threshold.
        include_contemporaneous: Populate the contemporaneous matrix when True.
        use_fdr: Apply Benjamini-Hochberg thresholds instead of raw p-values.
        verbose: Logging verbosity.

    Returns:
        A populated :class:`CGCResult`.
    """

    estimator = GcStar(
        n_perm=n_perm,
        n_pasts=n_pasts,
        n_lags=n_lags,
        method=method,
        random_state=random_state,
    )
    estimator.fit(data, verbose=verbose)
    return estimator.get_result(
        alpha=alpha,
        beta=beta,
        include_contemporaneous=include_contemporaneous,
        use_fdr=use_fdr,
    )
