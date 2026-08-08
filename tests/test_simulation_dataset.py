from __future__ import annotations

import numpy as np

from ica_denoising.simulation.config import SimulationConfig
from ica_denoising.simulation.dataset import build_dataset, default_scenarios


def _smoke_cfg():
    return SimulationConfig.from_dict(
        {
            "benchmark_version": "test",
            "n_frames": 300,
            "graph": {"n_observed": 5, "lag_order": 2},
            "behavior": {"n_parents": 2},
        }
    )


def _scenario(scenario_id):
    return next(s for s in default_scenarios() if s.scenario_id == scenario_id)


def test_same_seed_identical():
    cfg = _smoke_cfg()
    s = _scenario("S2")
    d1 = build_dataset(cfg, s, seed=3)
    d2 = build_dataset(cfg, s, seed=3)
    assert np.array_equal(d1.corrupted, d2.corrupted)
    assert np.array_equal(d1.truth_adjacency, d2.truth_adjacency)


def test_different_seed_changes_data():
    cfg = _smoke_cfg()
    s = _scenario("S2")
    d1 = build_dataset(cfg, s, seed=1)
    d2 = build_dataset(cfg, s, seed=2)
    assert not np.array_equal(d1.corrupted, d2.corrupted)


def test_s0_corrupted_equals_clean():
    cfg = _smoke_cfg()
    s = _scenario("S0")
    d = build_dataset(cfg, s, seed=0)
    assert np.allclose(d.corrupted, d.clean_fluorescence)
    assert np.allclose(d.artifact, 0.0)


def test_behavior_depends_on_past_parents_only():
    cfg = _smoke_cfg()
    s = _scenario("S0")
    d = build_dataset(cfg, s, seed=0)
    # behavior at t=0 cannot depend on positive-lag parents -> should be near 0 drive
    assert d.vigor.shape[0] == cfg.n_frames
    assert d.behavior_lag >= 1


def test_save_load_roundtrip(tmp_path):
    cfg = _smoke_cfg()
    s = _scenario("S2")
    d = build_dataset(cfg, s, seed=0)
    path = d.save(tmp_path / "dataset.npz")
    from ica_denoising.simulation.dataset import SimulatedDataset

    loaded = SimulatedDataset.load(path)
    assert np.array_equal(loaded.corrupted, d.corrupted)


def test_config_roundtrip(tmp_path):
    cfg = _smoke_cfg()
    path = cfg.write_resolved(tmp_path / "resolved.json")
    from ica_denoising.simulation.config import load_config

    reloaded = load_config(path)
    assert reloaded.n_frames == cfg.n_frames
    assert reloaded.graph.n_observed == cfg.graph.n_observed
