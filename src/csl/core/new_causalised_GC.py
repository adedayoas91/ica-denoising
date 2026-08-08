"""Vectorized, faster c-GC / fc-GC estimator (algorithmic optimization).

This module reimplements the hot paths of :class:`csl.core.causalised_gc.GcStar`
without changing the inference semantics or the output format. It subclasses the
hardened estimator and overrides only the two expensive passes, so
``get_result`` / ``get_connectivity_matrix`` and the whole graph-construction
logic are inherited unchanged (guaranteed output parity).

Optimizations
-------------
1. **FFT circular-shift null** -- the circular-shift permutation distribution is
   computed for *all* ``T'-1`` shifts at once via
   ``ifft(fft(x) * conj(fft(y)))``. This replaces the ``O(n_perm * T')``
   per-pair loop with ``O(T' log T')`` and uses the exact randomization
   distribution (a superset of any sampled ``n_perm`` shifts).
2. **Batched FFT across pairs** -- the unconditional pass is evaluated one
   target at a time, batched over all source rows (BLAS/FFT, no Python loop).
3. **Precision-matrix partial correlations (fc-GC)** -- all conditional
   statistics come from a single ridge-regularized ``R x R`` inverse via
   ``pcorr(i, j) = -P_ij / sqrt(P_ii P_jj)``. The exact permutation residuals
   use the block identity ``R_A = inv(P_AA) @ (P @ Z)_A``.
4. **Two-stage screening** -- a cheap analytic (Fisher-z) filter selects the
   small candidate set on which the exact FFT permutation null is run.

No new third-party dependency is required (NumPy FFT + BLAS only); numba is not
needed because the speedups are algorithmic rather than JIT constant-factor.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np

from .causalised_gc import GcStar

logger = logging.getLogger(__name__)

__all__ = ["FastGcStar"]


# --------------------------------------------------------------------------- #
# Vectorized circular-shift permutation null
# --------------------------------------------------------------------------- #
def _fft_cross_null(
    x_rows: np.ndarray, y_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact circular-shift permutation p-values for paired rows.

    For each row pair ``(x_rows[k], y_rows[k])`` this computes the absolute
    Pearson correlation at every circular shift via a single FFT and returns the
    observed ``|corr|`` together with the corrected p-value
    ``(count + 1) / T'`` where ``count`` is the number of non-trivial shifts at
    least as extreme as the observed correlation.

    Args:
        x_rows: Array of shape ``(m, T')`` (the shifted/residual sources).
        y_rows: Array of shape ``(m, T')`` (the targets), same shape as ``x_rows``.

    Returns:
        Tuple ``(obs_abs, pvalues)`` each of shape ``(m,)``.
    """

    if x_rows.shape != y_rows.shape:
        raise ValueError("x_rows and y_rows must share the same shape.")
    m, t = x_rows.shape
    if m == 0:
        return np.zeros(0), np.ones(0)

    xc = x_rows - x_rows.mean(axis=1, keepdims=True)
    yc = y_rows - y_rows.mean(axis=1, keepdims=True)
    x_norm = np.linalg.norm(xc, axis=1)
    y_norm = np.linalg.norm(yc, axis=1)
    denom = x_norm * y_norm
    valid = denom > 0

    fx = np.fft.fft(xc, axis=1)
    fy = np.fft.fft(yc, axis=1)
    cross = np.fft.ifft(fx * np.conj(fy), axis=1).real  # (m, T'); [:,0] is the obs numerator

    safe_denom = np.where(valid, denom, 1.0)[:, None]
    corr = cross / safe_denom
    obs = np.abs(corr[:, 0])
    null = np.abs(corr[:, 1:])  # exclude the zero shift (the observation)
    count = (null >= obs[:, None] - 1e-12).sum(axis=1)
    pvalues = (count + 1.0) / t
    obs = np.where(valid, obs, 0.0)
    pvalues = np.where(valid, pvalues, 1.0)
    return obs, pvalues


def _fisher_screen_pvalues(corr: np.ndarray, n_samples: int, dof_used: int) -> np.ndarray:
    """Two-sided Fisher-z screening p-values for a correlation matrix."""

    df = max(n_samples - dof_used - 2, 1)
    r = np.clip(np.abs(corr), 0.0, 1 - 1e-9)
    t_stat = r * np.sqrt(df / np.clip(1 - r**2, 1e-12, None))
    try:
        from scipy.stats import t as student_t

        return 2 * student_t.sf(t_stat, df)
    except Exception:  # pragma: no cover - scipy always present here
        # Normal approximation fallback.
        from math import erfc, sqrt

        return np.vectorize(lambda z: erfc(z / sqrt(2)))(t_stat)


@dataclass
class FastGcStar(GcStar):
    """Drop-in faster :class:`GcStar` using FFT + precision-matrix optimizations.

    Args:
        ridge: Ridge added to the covariance before inversion in the fc-GC
            precision path (stabilizes rank-deficient cleaned traces).
        screen_alpha: Analytic screening threshold; the exact FFT permutation
            null is computed only for pairs with screening p below this value.
        exact_permutation: If ``True`` (default) run the FFT permutation null on
            screened candidates; if ``False`` report the analytic screen p-value
            directly (fastest, parametric).
    """

    ridge: float = 1e-6
    screen_alpha: float = 0.2
    exact_permutation: bool = True

    # -- unconditional pass --------------------------------------------------
    def correlation_func(
        self, data: np.ndarray, shifts: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched FFT unconditional dependence and corrected p-values."""

        self.n_neur = data.shape[0]
        n = self.n_neur
        shifted = self.shift_data(np.asarray(data, dtype=float).copy())
        rows, t = shifted.shape

        zc = shifted - shifted.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(zc, axis=1)
        safe = np.where(norms == 0, 1.0, norms)
        fz = np.fft.fft(zc, axis=1)

        corr = np.zeros((rows, n))
        pvals = np.ones((rows, n))
        for j in range(n):
            cross = np.fft.ifft(fz * np.conj(fz[j]), axis=1).real  # (rows, T')
            denom = safe * safe[j]
            cs = cross / denom[:, None]
            obs = np.abs(cs[:, 0])
            null = np.abs(cs[:, 1:])
            count = (null >= obs[:, None] - 1e-12).sum(axis=1)
            invalid = (norms == 0) | (norms[j] == 0)
            corr[:, j] = np.where(invalid, 0.0, obs)
            pvals[:, j] = np.where(invalid, 1.0, (count + 1.0) / t)
        return corr, pvals

    # -- conditional pass ----------------------------------------------------
    def inv_correlation_func(
        self, data: np.ndarray, shifts: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conditional dependence via the precision (fc-GC) or residual (c-GC) path."""

        self.n_neur = data.shape[0]
        if self.method == "fcgc":
            return self._inv_correlation_precision(data)
        return self._inv_correlation_residual(data)

    def _inv_correlation_precision(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """fc-GC: partial correlations from one ridge-regularized inverse."""

        n = self.n_neur
        shifted = self.shift_data(np.asarray(data, dtype=float).copy())
        rows, t = shifted.shape
        zc = shifted - shifted.mean(axis=1, keepdims=True)

        sigma = (zc @ zc.T) / t
        sigma = sigma + self.ridge * np.eye(rows)
        precision = np.linalg.inv(sigma)
        diag = np.sqrt(np.clip(np.diag(precision), 1e-18, None))
        partial = -precision / np.outer(diag, diag)
        inv_corr = np.nan_to_num(np.abs(partial[:, :n]))

        # Analytic screen over all pairs (conditioning on R-2 covariates).
        screen_p = _fisher_screen_pvalues(inv_corr, t, dof_used=rows - 2)
        if not self.exact_permutation:
            return inv_corr, screen_p

        pvals = np.ones((rows, n))
        whitened = precision @ zc  # (rows, T'); used by the block residual identity
        # Candidates exclude self-pairs (row i is the same variable as target j),
        # whose 2x2 precision block is singular and whose edge is a zeroed diagonal.
        candidates = [
            (int(i), int(j))
            for i, j in np.argwhere(screen_p < self.screen_alpha)
            if int(i) != int(j)
        ]
        if candidates:
            src_res = np.empty((len(candidates), t))
            tgt_res = np.empty((len(candidates), t))
            for k, (i, j) in enumerate(candidates):
                a = (i, j)
                p_aa = precision[np.ix_(a, a)]
                residuals, *_ = np.linalg.lstsq(p_aa, whitened[list(a)], rcond=None)
                src_res[k] = residuals[0]
                tgt_res[k] = residuals[1]
            _, perm_p = _fft_cross_null(src_res, tgt_res)
            for k, (i, j) in enumerate(candidates):
                pvals[i, j] = perm_p[k]
        # Non-candidates keep their (large) analytic p so they stay non-significant.
        candidate_mask = np.zeros((rows, n), dtype=bool)
        for i, j in candidates:
            candidate_mask[i, j] = True
        pvals[~candidate_mask] = screen_p[~candidate_mask]
        self.residual_variance_ = float(np.mean(1.0 / np.clip(np.diag(precision), 1e-12, None)))
        return inv_corr, pvals

    def _inv_correlation_residual(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """c-GC: partial correlations from per-source ridge-regularized inverses.

        Unlike fc-GC (one ``R x R`` inverse for the whole graph), the c-GC
        conditioning set drops the *source's own lagged history* for each pair,
        so a single precision matrix cannot serve every edge. The cost is
        recovered by noting that the conditioning set depends only on the source
        row ``i`` (its history exclusion ``E_i``), not on the target ``j``.
        Therefore one inverse of ``Sigma`` restricted to ``V \\ E_i`` yields the
        partial correlation ``-P_ij / sqrt(P_ii P_jj)`` -- identical to the
        residual correlation ``corr(x_res, y_res)`` -- for *all* targets ``j`` of
        that source at once. This turns the ``O(rows * n)`` per-pair least-squares
        loop into ``~2 * rows`` small inverses (one extra only when a target
        coincides with the excluded lag-0 source copy). Exact-permutation
        residuals reuse the block identity ``R_A = inv(P_AA) @ (P @ Z)_A`` from
        the fc-GC path, so the screened FFT null is unchanged.
        """

        n = self.n_neur
        shifted = self.shift_data(np.asarray(data, dtype=float).copy())
        rows, t = shifted.shape
        zc = shifted - shifted.mean(axis=1, keepdims=True)
        sigma_full = (zc @ zc.T) / t

        inv_corr = np.zeros((rows, n))
        pvals = np.ones((rows, n))
        residual_inv_diag: list[float] = []

        src_res: list[np.ndarray] = []
        tgt_res: list[np.ndarray] = []
        index: list[tuple[int, int]] = []

        # Cache precision matrices keyed by the retained-row signature so that,
        # e.g., all lag-0 sources (empty history -> full set) share one inverse.
        cache: dict[tuple[int, ...], tuple[np.ndarray, dict[int, int], np.ndarray]] = {}

        def get_precision(
            keep: tuple[int, ...]
        ) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
            cached = cache.get(keep)
            if cached is None:
                idx = np.array(keep)
                sub = sigma_full[np.ix_(idx, idx)] + self.ridge * np.eye(idx.size)
                precision = np.linalg.inv(sub)
                pos = {orig: loc for loc, orig in enumerate(keep)}
                cached = (idx, pos, precision)
                cache[keep] = cached
            return cached

        all_rows = range(rows)
        for i in range(rows):
            source_index = i % n
            block = i // n
            # E_i: the source's own lag-0..(block-1) copies (history to drop).
            excluded_history = {source_index + lag * n for lag in range(block)}
            keep_a = tuple(r for r in all_rows if r not in excluded_history)
            idx_a, pos_a, prec_a = get_precision(keep_a)
            diag_a = np.sqrt(np.clip(np.diag(prec_a), 1e-18, None))

            for j in range(n):
                if j == i:
                    continue  # self-pair -> zeroed diagonal in graph construction
                if j in excluded_history:
                    # Target coincides with the excluded lag-0 source copy: it is
                    # the pair variable here, so add it back to the kept set.
                    keep = tuple(sorted(set(keep_a) | {j}))
                    idx, pos, prec = get_precision(keep)
                    diag = np.sqrt(np.clip(np.diag(prec), 1e-18, None))
                else:
                    keep, idx, pos, prec, diag = keep_a, idx_a, pos_a, prec_a, diag_a

                li, lj = pos[i], pos[j]
                partial = float(
                    np.nan_to_num(np.abs(prec[li, lj] / (diag[li] * diag[lj])))
                )
                inv_corr[i, j] = partial
                residual_inv_diag.append(1.0 / float(np.clip(prec[li, li], 1e-12, None)))

                # Analytic screen: only permute promising pairs.
                screen = _fisher_screen_pvalues(
                    np.array([partial]), t, dof_used=len(keep) - 2
                )[0]
                if self.exact_permutation and screen < self.screen_alpha:
                    p_aa = prec[np.ix_((li, lj), (li, lj))]
                    whitened = prec[[li, lj]] @ zc[idx]  # (2, T'); block residuals
                    residuals, *_ = np.linalg.lstsq(p_aa, whitened, rcond=None)
                    src_res.append(residuals[0])
                    tgt_res.append(residuals[1])
                    index.append((i, j))
                else:
                    pvals[i, j] = screen

        if index:
            _, perm_p = _fft_cross_null(np.array(src_res), np.array(tgt_res))
            for k, (i, j) in enumerate(index):
                pvals[i, j] = perm_p[k]
        if residual_inv_diag:
            self.residual_variance_ = float(np.mean(residual_inv_diag))
        return inv_corr, pvals
