from __future__ import annotations

import json

import pandas as pd

from ica_denoising.simulation.config import SimulationConfig
from ica_denoising.simulation.runner import replicate_dir, run_benchmark


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
