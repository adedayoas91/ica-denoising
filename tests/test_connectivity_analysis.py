"""Unit tests for the derived connectivity analysis module (Section 5.7).

Tests use small hand-constructed matrices with known answers. One guarded smoke
test loads a real saved artifact only when it exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ica_denoising.connectivity_analysis import (
    ConnectivityAnalysisConfig,
    NodeAnnotations,
    compare_estimators,
    condition_number,
    degree_changes,
    directional_enrichment_test,
    edge_count,
    edge_density,
    effective_rank,
    group_enrichment_test,
    ipsilateral_enrichment_test,
    jaccard_overlap,
    normalized_hamming,
    numerical_rank,
    reciprocal_fraction,
    stable_rank,
    summarize_recording,
)

_REAL_ARTIFACT = Path(
    "outputs/connectivity/v2a-RSNs/220119_F2_run11/c-GC"
)
_REAL_CGC_STAR = Path(
    "outputs/connectivity/v2a-RSNs/220119_F2_run11/c-GC-star"
)


def test_edge_count_and_density_excludes_diagonal() -> None:
    w = np.array([[9.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    assert edge_count(w) == 2
    assert edge_density(w) == pytest.approx(2.0 / 6.0)


def test_jaccard_and_hamming_known_graphs() -> None:
    a = np.zeros((3, 3))
    a[0, 1] = 1.0
    a[1, 2] = 1.0
    b = np.zeros((3, 3))
    b[0, 1] = 1.0
    b[2, 0] = 1.0
    assert jaccard_overlap(a, b) == pytest.approx(1.0 / 3.0)
    assert normalized_hamming(a, b) == pytest.approx(2.0 / 6.0)


def test_jaccard_empty_graphs_is_one() -> None:
    z = np.zeros((4, 4))
    assert jaccard_overlap(z, z) == 1.0
    assert normalized_hamming(z, z) == 0.0


def test_reciprocal_fraction_known_pairs() -> None:
    w = np.zeros((3, 3))
    w[0, 1] = 1.0
    w[1, 0] = 1.0  # reciprocal pair
    w[0, 2] = 1.0  # one-way
    assert reciprocal_fraction(w) == pytest.approx(2.0 / 3.0)


def test_reciprocal_fraction_no_edges() -> None:
    assert reciprocal_fraction(np.zeros((3, 3))) == 0.0


def test_degree_changes_vs_raw() -> None:
    raw = np.zeros((3, 3))
    raw[0, 1] = 1.0
    variant = np.zeros((3, 3))
    variant[0, 1] = 1.0
    variant[0, 2] = 1.0
    change = degree_changes(variant, raw)
    # node 0 gains one out-edge.
    assert change.out_degree_delta[0] == 1
    # node 2 gains one in-edge.
    assert change.in_degree_delta[2] == 1
    assert change.out_degree_delta[1] == 0


def test_numerical_and_effective_rank_low_rank() -> None:
    mat = np.diag([5.0, 4.0, 3.0, 0.0, 0.0])
    assert numerical_rank(mat) == 3
    eff = effective_rank(mat)
    assert 1.0 < eff < 3.0
    assert eff == pytest.approx(2.938, abs=1e-2)


def test_numerical_rank_relative_tolerance() -> None:
    mat = np.diag([5.0, 4.0, 1e-12])
    assert numerical_rank(mat, rank_tolerance=1e-6) == 2
    assert numerical_rank(mat) == 3  # default tolerance keeps the tiny value


def test_stable_rank_and_condition_number() -> None:
    mat = np.diag([5.0, 4.0, 3.0])
    assert stable_rank(mat) == pytest.approx((25 + 16 + 9) / 25.0)
    assert condition_number(mat) == pytest.approx(5.0 / 3.0)
    assert condition_number(np.diag([5.0, 0.0])) == float("inf")


def _emitter_receiver_complete_graph(
    n_per_group: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = 2 * n_per_group
    labels = np.array(
        ["emitter"] * n_per_group + ["receiver"] * n_per_group
    )
    return np.zeros((n, n)), labels


def test_group_enrichment_significant_when_all_emitter_to_receiver() -> None:
    n_per = 5
    w, labels = _emitter_receiver_complete_graph(n_per)
    for i in range(n_per):
        for j in range(n_per, 2 * n_per):
            w[i, j] = 1.0  # every edge emitter -> receiver
    result = group_enrichment_test(
        w,
        labels,
        "emitter",
        "receiver",
        n_permutations=1000,
        rng=np.random.default_rng(0),
    )
    assert result.observed == pytest.approx(1.0)
    assert result.p_value < 0.05


def test_group_enrichment_not_significant_for_balanced_graph() -> None:
    # Complete off-diagonal graph: every label permutation yields the same
    # statistic, so the permutation p-value is exactly 1.0.
    n_per = 5
    w, labels = _emitter_receiver_complete_graph(n_per)
    n = 2 * n_per
    w[:] = 1.0
    np.fill_diagonal(w, 0.0)
    result = group_enrichment_test(
        w,
        labels,
        "emitter",
        "receiver",
        n_permutations=500,
        rng=np.random.default_rng(0),
    )
    assert result.p_value > 0.5
    assert result.n_edges == n * (n - 1)


def test_directional_enrichment_rostro_caudal_chain() -> None:
    n = 5
    coord = np.arange(n, dtype=float)
    w = np.zeros((n, n))
    for i in range(n - 1):
        w[i, i + 1] = 1.0  # forward chain, all rostral -> caudal
    result = directional_enrichment_test(
        w, coord, n_permutations=1000, rng=np.random.default_rng(0)
    )
    assert result.observed == pytest.approx(1.0)
    assert result.p_value < 0.05


def test_ipsilateral_enrichment_observed() -> None:
    side = np.array(["L", "L", "L", "R", "R", "R"])
    w = np.zeros((6, 6))
    # all edges within the same side
    for pair in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        w[pair] = 1.0
    result = ipsilateral_enrichment_test(
        w, side, n_permutations=500, rng=np.random.default_rng(0)
    )
    assert result.observed == pytest.approx(1.0)
    assert 0.0 < result.p_value <= 1.0


def test_permutation_tests_are_deterministic() -> None:
    n_per = 4
    w, labels = _emitter_receiver_complete_graph(n_per)
    for i in range(n_per):
        for j in range(n_per, 2 * n_per):
            w[i, j] = 1.0
    r1 = group_enrichment_test(
        w, labels, "emitter", "receiver", n_permutations=200, rng=np.random.default_rng(7)
    )
    r2 = group_enrichment_test(
        w, labels, "emitter", "receiver", n_permutations=200, rng=np.random.default_rng(7)
    )
    assert r1.p_value == r2.p_value
    assert r1.null_mean == r2.null_mean


@pytest.mark.skipif(
    not (_REAL_ARTIFACT / "weighted").is_dir(),
    reason="Real connectivity artifact not available.",
)
def test_smoke_real_artifact() -> None:
    cfg = ConnectivityAnalysisConfig(n_permutations=50, random_state=0)
    summary = summarize_recording(_REAL_ARTIFACT, config=cfg)
    per_variant = summary["per_variant"]
    assert "raw" in set(per_variant["variant"])
    assert (per_variant["n_nodes"] > 0).all()
    assert {"edge_density", "reciprocal_fraction"}.issubset(per_variant.columns)
    # enrichment without annotations is empty.
    assert summary["enrichment"].empty

    if (_REAL_CGC_STAR / "weighted").is_dir():
        comparison = compare_estimators(_REAL_ARTIFACT, _REAL_CGC_STAR, cfg)
        assert "jaccard_cgc_vs_cgcstar" in comparison.columns
        assert not comparison.empty


def test_summarize_with_annotations_and_traces(tmp_path: Path) -> None:
    # Build a tiny fake estimator directory with weighted matrices.
    weighted = tmp_path / "weighted"
    weighted.mkdir()
    rng = np.random.default_rng(1)
    n = 6
    raw = (rng.random((n, n)) > 0.5).astype(float)
    np.fill_diagonal(raw, 0.0)
    variant = raw.copy()
    variant[0, 3] = 1.0
    np.save(weighted / "raw.npy", raw)
    np.save(weighted / "fastica_cleaned.npy", variant)

    traces = {
        "raw": rng.random((n, 50)),
        "fastica_cleaned": rng.random((n, 50)),
    }
    annotations = NodeAnnotations(
        group_labels=np.array(["emitter"] * 3 + ["receiver"] * 3),
        side=np.array(["L", "L", "L", "R", "R", "R"]),
        rostro_caudal=np.arange(n, dtype=float),
    )
    cfg = ConnectivityAnalysisConfig(n_permutations=50, random_state=0)
    summary = summarize_recording(
        tmp_path, traces=traces, node_metadata=annotations, config=cfg
    )
    per_variant = summary["per_variant"]
    assert "numerical_rank" in per_variant.columns
    # three enrichment tests x two variants.
    assert set(summary["enrichment"]["test"]) == {"group", "ipsilateral", "directional"}
    assert len(summary["enrichment"]) == 6
