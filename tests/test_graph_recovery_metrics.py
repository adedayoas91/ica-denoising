from __future__ import annotations

import numpy as np

from csl.experiments.graph_metrics import (
    direction_reversal_count,
    edge_density_bias,
    graph_recovery_metrics,
    jaccard_overlap,
    matthews_corrcoef,
    structural_hamming_distance,
)


def _truth():
    # 0 -> 1, 1 -> 2 (source, target)
    a = np.zeros((3, 3), dtype=int)
    a[0, 1] = 1
    a[1, 2] = 1
    return a


def test_perfect_recovery():
    truth = _truth()
    metrics = graph_recovery_metrics(truth, truth)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["shd"] == 0.0


def test_diagonal_excluded():
    truth = _truth()
    pred = truth.copy()
    np.fill_diagonal(pred, 1)  # diagonal must be ignored
    metrics = graph_recovery_metrics(pred, truth)
    assert metrics["f1"] == 1.0


def test_structural_hamming_distance_known():
    truth = _truth()
    pred = truth.copy()
    pred[0, 1] = 0  # drop one edge
    pred[2, 0] = 1  # add one edge
    assert structural_hamming_distance(pred, truth) == 2


def test_direction_reversal_detected():
    truth = _truth()
    pred = np.zeros((3, 3), dtype=int)
    pred[1, 0] = 1  # reversed 0->1
    assert direction_reversal_count(pred, truth) >= 1


def test_edge_density_bias_sign():
    truth = _truth()
    dense = np.ones((3, 3), dtype=int)
    assert edge_density_bias(dense, truth) > 0


def test_jaccard_overlap_known():
    a = _truth()
    b = a.copy()
    b[1, 2] = 0
    # intersection 1 edge, union 2 edges
    assert jaccard_overlap(a, b) == 0.5


def test_mcc_bounds():
    truth = _truth()
    assert matthews_corrcoef(truth, truth) == 1.0


def test_scores_auroc_present():
    truth = _truth()
    scores = truth.astype(float) * 0.9
    metrics = graph_recovery_metrics(truth, truth, scores=scores)
    assert "auroc" in metrics
