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
