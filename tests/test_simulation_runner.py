from __future__ import annotations

import json

import numpy as np
import pandas as pd

from csl.experiments.simulation_adapters import GraphEstimate
from ica_denoising.simulation.config import (
    ArtifactConfig,
    CalciumConfig,
    DynamicsConfig,
    ScenarioConfig,
    SimulationConfig,
)
from ica_denoising.simulation.dataset import build_dataset
from ica_denoising.simulation.runner import (
    _select_scored_graph,
    replicate_dir,
    run_benchmark,
)


def _smoke_cfg(tmp_path):
    return SimulationConfig.from_dict(
        {
            "benchmark_version": "smoke_test",
            "seeds": [0, 1],
            "n_frames": 250,
            "graph": {"n_observed": 5, "lag_order": 2},
            "behavior": {"n_parents": 2},
            "bss": {"methods": ["fastica", "pca"], "keep_top": 3},
            "estimator": {
                "estimators": ["cgc", "var"],
                "n_pasts": 2,
                "n_lags": 2,
                "n_perm": 10,
                "alpha": 0.1,
                "beta": 0.1,
            },
            "run": {"output_dir": str(tmp_path / "outputs"), "resume": True},
            "scenarios": [
                {
                    "scenario_id": "S0",
                    "calcium": {"enabled": False},
                    "artifacts": {"types": []},
                }
            ],
        }
    )


def test_dry_run_plans_only(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    planned = run_benchmark(cfg, dry_run=True)
    assert len(planned) == 2
    for path in planned:
        assert not (path / "manifest.json").exists()


def test_smoke_run_end_to_end(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    paths = run_benchmark(cfg)
    assert len(paths) == 2
    for path in paths:
        for name in (
            "dataset.npz",
            "manifest.json",
            "variants.csv",
            "trace_metrics.csv",
            "graph_metrics.csv",
        ):
            assert (path / name).exists(), name
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["complete"] is True
        assert "resolved_config" in manifest
        # canonical variant id + trace hash present on rows
        graph_rows = pd.read_csv(path / "graph_metrics.csv")
        assert "variant_id" in graph_rows.columns
        assert "trace_hash" in graph_rows.columns


def test_resume_skips_completed(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    run_benchmark(cfg)
    manifest = replicate_dir(cfg, "S0", 0) / "manifest.json"
    first_mtime = manifest.stat().st_mtime_ns
    # second run with resume should not rewrite the manifest
    run_benchmark(cfg, resume=True)
    assert manifest.stat().st_mtime_ns == first_mtime


def test_reporting_regenerates_from_csv(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    run_benchmark(cfg)
    from ica_denoising.simulation.reporting import (
        collect_metric_table,
        summarize_graph_recovery,
    )

    root = tmp_path / "outputs" / "smoke_test"
    table = collect_metric_table(root, "graph")
    assert not table.empty
    summary = summarize_graph_recovery(root)
    assert not summary.empty


def test_manifest_validation_passes(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    run_benchmark(cfg)
    from ica_denoising.simulation.validate import validate_benchmark

    root = tmp_path / "outputs" / "smoke_test"
    report = validate_benchmark(root)
    assert report.ok, report.errors
    assert report.checked == 2


def test_fdr_correction_selects_fdr_binary_graph():
    est = GraphEstimate(
        estimator="cgc",
        scores=np.array([[0.0, 0.9], [0.1, 0.0]]),
        binary=np.array([[0, 1], [1, 0]]),
        binary_fdr=np.array([[0, 0], [1, 0]]),
        warnings=(),
    )

    scored, correction = _select_scored_graph(est, "fdr")

    assert correction == "fdr"
    assert scored[0, 1] == 0
    assert scored[1, 0] == 1


def test_failed_estimator_marks_manifest_incomplete(tmp_path, monkeypatch):
    from ica_denoising.simulation import runner

    def _fail(_traces):
        raise RuntimeError("forced estimator failure")

    monkeypatch.setitem(runner.ESTIMATORS, "forced_failure", _fail)
    cfg = _smoke_cfg(tmp_path)
    cfg = SimulationConfig.from_dict(
        {
            **cfg.to_dict(),
            "seeds": [0],
            "estimator": {**cfg.to_dict()["estimator"], "estimators": ["forced_failure"]},
        }
    )

    paths = run_benchmark(cfg)
    manifest = json.loads((paths[0] / "manifest.json").read_text())
    failures = pd.read_csv(paths[0] / "failures.csv")

    assert manifest["complete"] is False
    assert not failures.empty


def test_behavior_feedback_changes_generated_dynamics(tmp_path):
    base_cfg = SimulationConfig.from_dict(
        {
            "benchmark_version": "feedback",
            "n_frames": 120,
            "graph": {"n_observed": 4, "lag_order": 2},
        }
    )
    no_feedback = ScenarioConfig(
        "S0",
        dynamics=DynamicsConfig(model="linear_var", behavior_feedback=False),
        calcium=CalciumConfig(enabled=False),
        artifacts=ArtifactConfig(types=()),
    )
    feedback = ScenarioConfig(
        "S7",
        dynamics=DynamicsConfig(model="linear_var", behavior_feedback=True),
        calcium=CalciumConfig(enabled=False),
        artifacts=ArtifactConfig(types=()),
    )

    a = build_dataset(base_cfg, no_feedback, seed=0)
    b = build_dataset(base_cfg, feedback, seed=0)

    assert not np.allclose(a.clean_fluorescence, b.clean_fluorescence)
