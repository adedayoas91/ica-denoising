from __future__ import annotations

import numpy as np

from ica_denoising.simulation.artifacts import inject_artifacts
from ica_denoising.simulation.config import ArtifactConfig
from ica_denoising.simulation.variants import ORACLE_VARIANTS, build_variants


def _clean(rng):
    return rng.normal(0.0, 1.0, size=(6, 400))


def test_no_artifact_passthrough():
    rng = np.random.default_rng(0)
    clean = _clean(rng)
    bundle = inject_artifacts(clean, ArtifactConfig(types=()), rng)
    assert np.allclose(bundle.corrupted, clean)
    assert bundle.achieved_asr == 0.0


def test_artifact_strength_monotonic():
    rng = np.random.default_rng(1)
    clean = _clean(rng)
    achieved = []
    for asr in (0.2, 0.5, 1.0):
        cfg = ArtifactConfig(types=("drift", "neuropil"), artifact_to_signal=asr)
        bundle = inject_artifacts(clean, cfg, np.random.default_rng(2))
        achieved.append(bundle.achieved_asr)
    assert achieved[0] < achieved[1] < achieved[2]


def test_artifact_preserves_shape():
    rng = np.random.default_rng(3)
    clean = _clean(rng)
    cfg = ArtifactConfig(types=("motion",), artifact_to_signal=0.5)
    bundle = inject_artifacts(clean, cfg, rng)
    assert bundle.corrupted.shape == clean.shape


def test_variants_oracle_isolation():
    rng = np.random.default_rng(4)
    clean = _clean(rng)
    cfg = ArtifactConfig(types=("drift",), artifact_to_signal=0.5)
    bundle = inject_artifacts(clean, cfg, rng)
    variants = build_variants(
        clean=clean,
        corrupted=bundle.corrupted,
        artifact=bundle.artifact,
        methods=("fastica", "pca"),
        keep_top=4,
        rank_target=4,
        random_state=0,
        sample_rate_hz=5.0,
    )
    for v in variants:
        if v.uses_ground_truth:
            assert v.variant_id in ORACLE_VARIANTS or v.variant_id.startswith("pca")
        # non-oracle variants must not be labelled clean/artifact_oracle/oracle_selection
        if v.variant_id in ORACLE_VARIANTS:
            assert v.uses_ground_truth


def test_variant_grid_expands_ranks_restarts_and_sobi_lags():
    rng = np.random.default_rng(44)
    clean = _clean(rng)
    cfg = ArtifactConfig(types=("drift",), artifact_to_signal=0.5)
    bundle = inject_artifacts(clean, cfg, rng)

    variants = build_variants(
        clean=clean,
        corrupted=bundle.corrupted,
        artifact=bundle.artifact,
        methods=("fastica", "sobi"),
        keep_top=4,
        rank_target=4,
        random_state=0,
        sample_rate_hz=5.0,
        rank_targets=(("full", 6), ("ev90", 3)),
        random_states=(0, 1),
        sobi_lag_sets=((1, 2), (2, 4)),
    )

    ids = {variant.variant_id for variant in variants}
    assert "fastica/cluster_keep_top_04_full/rank_6/seed_0" in ids
    assert "fastica/cluster_keep_top_04_ev90/rank_3/seed_1" in ids
    assert "sobi/cluster_keep_top_04_lags_1_2_full/rank_6/seed_0" in ids
    assert "sobi/cluster_keep_top_04_lags_2_4_ev90/rank_3/seed_1" in ids


def test_infomax_variant_uses_infomax_dec(monkeypatch):
    from ica_denoising.simulation import variants as variant_mod

    rng = np.random.default_rng(45)
    clean = rng.normal(0.0, 1.0, size=(4, 80))
    corrupted = clean + 0.05 * rng.normal(size=clean.shape)
    artifact = corrupted - clean
    calls = {"infomax": [], "fastica_fun": []}

    def fake_infomax_dec(data, n_comps, max_iter, random_state, **_kwargs):
        calls["infomax"].append(
            {
                "shape": data.shape,
                "n_comps": n_comps,
                "max_iter": max_iter,
                "random_state": random_state,
            }
        )
        sources = np.column_stack(
            [
                np.linspace(-1.0, 1.0, data.shape[1]),
                np.sin(np.linspace(0.0, 2.0 * np.pi, data.shape[1])),
                np.zeros(data.shape[1]),
            ]
        )
        mixing = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        mean = np.zeros(data.shape[0])
        return sources, np.empty((n_comps, data.shape[1])), mixing, mean

    def fake_ica_reconstruct(traces, keep_top, rng_seed, fun, *, n_components=None):
        calls["fastica_fun"].append(fun)
        return traces.copy()

    monkeypatch.setattr(variant_mod, "infomax_dec", fake_infomax_dec)
    monkeypatch.setattr(variant_mod, "_ica_reconstruct", fake_ica_reconstruct)
    monkeypatch.setattr(variant_mod, "_oracle_select", lambda traces, _artifact, _seed: traces.copy())

    variants = build_variants(
        clean=clean,
        corrupted=corrupted,
        artifact=artifact,
        methods=("infomax",),
        keep_top=2,
        rank_target=3,
        random_state=7,
        sample_rate_hz=5.0,
        rank_targets=(("rank3", 3),),
        random_states=(7,),
    )

    ids = {variant.variant_id for variant in variants}
    assert "infomax/cluster_keep_top_02_rank3/rank_3/seed_7" in ids
    assert calls["infomax"] == [
        {
            "shape": corrupted.shape,
            "n_comps": 3,
            "max_iter": 500,
            "random_state": 7,
        }
    ]
    assert "exp" not in calls["fastica_fun"]


def test_artifact_oracle_recovers_clean():
    rng = np.random.default_rng(5)
    clean = _clean(rng)
    cfg = ArtifactConfig(types=("drift",), artifact_to_signal=0.5)
    bundle = inject_artifacts(clean, cfg, rng)
    variants = build_variants(
        clean=clean,
        corrupted=bundle.corrupted,
        artifact=bundle.artifact,
        methods=(),
        keep_top=4,
        rank_target=4,
        random_state=0,
        sample_rate_hz=5.0,
    )
    oracle = next(v for v in variants if v.variant_id == "artifact_oracle")
    assert np.allclose(oracle.traces, clean, atol=1e-6)
