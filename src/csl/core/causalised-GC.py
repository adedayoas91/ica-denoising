"""Core c-GC/fcGC estimator implementation.

This module contains a cleaned-up version of the original ``GcStar``
implementation. It is intended to be imported both from the experiment harness
and from interactive notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np

try:
    from numba import jit
except Exception:  # pragma: no cover - numba is optional at runtime

    def jit(*_args, **_kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


SHD = None
SID = None


@jit(nopython=True)
def _perm_test_numba(x: np.ndarray, y: np.ndarray, n_perm: int) -> float:
    """Compute a circular-shift permutation p-value for correlation.

    The implementation mirrors the original estimator logic but guards against
    degenerate short series by using the largest valid shift range available.
    """

    if x.size <= 1 or y.size <= 1 or n_perm <= 0:
        return 1.0

    count = 0
    corr_obs = np.corrcoef(x, y)[1, 0]
    x_copy = x.copy()
    low = 1
    high = x.size

    for _ in range(n_perm):
        shift = np.random.randint(low, high)
        rolled = np.hstack((x_copy[shift:], x_copy[:shift]))
        corr_perm = np.corrcoef(rolled, y)[1, 0]
        if np.abs(corr_perm) >= np.abs(corr_obs):
            count += 1

    return count / n_perm


def regression_residual(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return residuals after regressing ``z`` out of ``x``.

    Parameters
    ----------
    x:
        One-dimensional target series of shape ``(T,)``.
    z:
        Conditioning set shaped ``(n_covariates, T)``.
    """

    if z.size == 0:
        return x - np.mean(x)

    if z.ndim == 1:
        z = z[np.newaxis, :]

    design = np.vstack([z, np.ones(z.shape[1])]).T
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    fitted = design @ coef
    return x - fitted


@dataclass
class GcStar:
    """Granger-causality estimator from a causal-Bayesian-network viewpoint.

    Parameters
    ----------
    n_perm:
        Number of circular-shift permutations used for p-value estimation.
    n_pasts:
        Conditioning depth used to construct the shifted state.
    n_lags:
        Number of lag blocks merged into the final connectivity matrix.
    temporal:
        Reserved compatibility flag retained from the original implementation.
    method:
        Either ``"cgc"`` or ``"fcgc"``.
    parallel:
        If ``True``, run unconditional and conditional dependence passes in
        separate worker threads.
    """

    n_perm: int = 1000
    n_pasts: int = 1
    n_lags: int = 1
    temporal: bool = True
    method: str = "cgc"
    parallel: bool = False
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
        self.logger = logging.getLogger(__name__)

    def _configure_logging(self, verbose: int) -> None:
        """Configure module logging for interactive runs."""

        level = logging.WARNING
        if verbose == 1:
            level = logging.INFO
        elif verbose >= 2:
            level = logging.DEBUG
        logging.basicConfig(level=level, force=True)

    def shift_data(self, arr: np.ndarray) -> np.ndarray:
        """Create stacked lagged views of the data.

        Parameters
        ----------
        arr:
            Array shaped ``(n_variables, T)``.
        """

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

    def correlation_func(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute unconditional dependence and permutation p-values."""

        self.n_neur = data.shape[0]
        shifted = self.shift_data(data.copy())
        n_rows = shifted.shape[0]
        corr = np.abs(np.corrcoef(shifted))
        pvals = np.zeros((n_rows, self.n_neur))

        for i in range(n_rows):
            for j in range(self.n_neur):
                pvals[i, j] = _perm_test_numba(
                    shifted[i, :], shifted[j, :], self.n_perm
                )

        return corr[:, : self.n_neur], pvals

    def inv_correlation_func(
        self,
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute conditional dependence using residual correlations."""

        self.n_neur = data.shape[0]
        shifted = self.shift_data(data.copy())
        n_rows = shifted.shape[0]
        inv_corr = np.zeros((n_rows, self.n_neur))
        pvals = np.zeros((n_rows, self.n_neur))

        for i in range(n_rows):
            for j in range(self.n_neur):
                x = shifted[i]
                y = shifted[j]
                if self.method == "fcgc":
                    z = np.delete(shifted.copy(), [i, j], axis=0)
                else:
                    z = self.get_conditioning_set(data, i, j)

                x_res = regression_residual(x, z)
                y_res = regression_residual(y, z)
                inv_corr[i, j] = np.abs(np.corrcoef(x_res, y_res)[1, 0])
                pvals[i, j] = _perm_test_numba(x_res, y_res, self.n_perm)

        return inv_corr, pvals

    def fit(self, data: np.ndarray, verbose: int = 0) -> "GcStar":
        """Fit the estimator on an array of shape ``(n_variables, T)``."""

        self.data = data.copy()
        self.shifted_data = self.shift_data(self.data)
        self._configure_logging(verbose)

        if self.parallel:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                corr_future = executor.submit(self.correlation_func, data)
                inv_future = executor.submit(self.inv_correlation_func, data)
                self.corr_, self.pVal_corr_ = corr_future.result()
                self.inv_corr_, self.pVal_inv_corr_ = inv_future.result()
        else:
            self.corr_, self.pVal_corr_ = self.correlation_func(data)
            self.inv_corr_, self.pVal_inv_corr_ = self.inv_correlation_func(data)

        return self

    def get_connectivity_matrix(
        self,
        *,
        simulation: bool = True,
        alpha: float = 0.01,
        beta: float = 0.001,
    ) -> np.ndarray:
        """Construct a weighted connectivity matrix from significance masks."""

        if self.corr_ is None or self.inv_corr_ is None:
            raise RuntimeError("fit must be called before get_connectivity_matrix.")

        sig_corr = np.multiply(self.corr_, self.pVal_corr_ <= alpha)
        sig_inv = np.multiply(self.inv_corr_, self.pVal_inv_corr_ <= beta)
        inferred = np.logical_and(sig_corr, sig_inv)

        all_: list[np.ndarray] = []
        n_neur = inferred.shape[1]
        for lag in range(self.n_pasts + 1):
            start = lag * n_neur
            stop = (lag + 1) * n_neur
            all_.append(inferred[start:stop, 0:n_neur])

        if simulation:
            self.conn_mat = all_[1] if len(all_) > 1 else all_[0]
            if self.n_lags > 1:
                max_lag = min(self.n_lags, len(all_) - 1)
                for i in range(2, max_lag + 1):
                    self.conn_mat = np.logical_or(self.conn_mat, all_[i])
        elif self.n_lags == 1:
            self.conn_mat = (
                np.logical_or(all_[0], all_[1]) if len(all_) > 1 else all_[0]
            )
        elif self.n_lags > 1:
            self.conn_mat = all_[0]
            max_lag = min(self.n_lags, len(all_) - 1)
            for i in range(1, max_lag + 1):
                self.conn_mat = np.logical_or(self.conn_mat, all_[i])

        self.conn_mat = np.multiply(self.corr_[:n_neur, :], self.conn_mat)
        return self.conn_mat

    def compute_confusion_matrix(
        self,
        truth: np.ndarray,
        *,
        simulation: bool = True,
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
        """Return accuracy, precision, recall, FPR, BA, and F1."""

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
        self,
        truth: np.ndarray,
        inferred: np.ndarray,
        *,
        simulation: bool = True,
    ) -> np.ndarray:
        """Compute SHD, importing ``cdt`` only when needed."""

        global SHD, SID
        if SHD is None:
            try:
                from cdt.metrics import SHD as _SHD, SID as _SID
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "cdt.metrics is required for SHD/SID computations."
                ) from exc
            SHD, SID = _SHD, _SID

        target = truth.T if simulation else truth
        self.shd_ = SHD(target=target, pred=inferred, double_for_anticausal=False)
        # self.sid_ = SID(target=target, pred=inferred)
        return self.shd_  # , self.sid_
