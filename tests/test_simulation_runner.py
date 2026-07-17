from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from csl.experiments.simulation_adapters import GraphEstimate, estimate_jpcmciplus
from ica_denoising.simulation.config import (
    ArtifactConfig,
    CalciumConfig,
    DynamicsConfig,
    ScenarioConfig,
    SimulationConfig,
)
from ica_denoising.simulation.dataset import build_dataset
from ica_denoising.simulation.runner import (
    _assert_no_unrequested_bss_variants,
    _manifest_variant_implementation_current,
    _select_scored_graph,
    import_completed_replicates_from_benchmark,
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


def test_import_completed_replicate_filters_to_target_method(tmp_path):
    source_root = tmp_path / "source" / "core"
    source_dir = source_root / "S0" / "seed_0"
    source_dir.mkdir(parents=True)
    np.savez_compressed(source_dir / "dataset.npz", corrupted=np.zeros((2, 5)))
    variant_ids = [
        "clean",
        "raw",
        "artifact_oracle",
        "fastica/cluster_keep_top_03_full/rank_2/seed_0",
        "infomax/cluster_keep_top_03_full/rank_2/seed_0",
        "sobi/cluster_keep_top_03_lags_1_2_3_5_full/rank_2/seed_0",
        "jade/cluster_keep_top_03_full/rank_2/seed_0",
        "pca/rank_matched_full/rank_2/seed_0",
        "random_subspace/rank_matched_full/rank_2/seed_0",
        "fastica/all/rank_2/seed_0",
        "causal_lowpass/cutoff_1.25hz/rank_2/seed_0",
        "oracle_selection",
    ]
    pd.DataFrame(
        {
            "variant_id": variant_ids,
            "trace_hash": [f"hash_{idx}" for idx, _ in enumerate(variant_ids)],
        }
    ).to_csv(source_dir / "variants.csv", index=False)
    for csv_name in ("trace_metrics", "behavior_metrics", "state_metrics"):
        pd.DataFrame(
            {
                "variant_id": variant_ids,
                "trace_hash": [f"hash_{idx}" for idx, _ in enumerate(variant_ids)],
                "value": np.arange(len(variant_ids)),
            }
        ).to_csv(source_dir / f"{csv_name}.csv", index=False)
    graph_rows = []
    for variant_id in variant_ids:
        graph_rows.append(
            {
                "variant_id": variant_id,
                "estimator": "cgc",
                "trace_hash": "hash",
                "f1": 0.0,
            }
        )
        graph_path = source_dir / "graphs" / "cgc" / f"{variant_id.replace('/', '__')}.npz"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            graph_path,
            scores=np.zeros((2, 2)),
            binary=np.zeros((2, 2), dtype=int),
            scored_binary=np.zeros((2, 2), dtype=int),
        )
    pd.DataFrame(graph_rows).to_csv(source_dir / "graph_metrics.csv", index=False)
    pd.DataFrame().to_csv(source_dir / "failures.csv", index=False)
    cfg = SimulationConfig.from_dict(
        {
            **_smoke_cfg(tmp_path).to_dict(),
            "benchmark_version": "core_sobi",
            "seeds": [0],
            "bss": {"methods": ["sobi"], "keep_top": 3},
            "run": {"output_dir": str(tmp_path / "target"), "resume": True},
        }
    )
    manifest = {
        "benchmark_version": "core",
        "schema_version": cfg.schema_version,
        "scenario_id": "S0",
        "seed": 0,
        "resolved_config": _smoke_cfg(tmp_path).to_dict(),
        "git_revision": "test",
        "versions": {},
        "dataset_hash": "hash",
        "achieved_asr": 0.0,
        "start": "2026-01-01T00:00:00+00:00",
        "end": "2026-01-01T00:00:00+00:00",
        "runtime_seconds": 1.0,
        "variant_implementation_version": "core_bss_method_specific_controls_v2",
        "n_variants": len(variant_ids),
        "failures": [],
        "complete": True,
    }
    (source_dir / "manifest.json").write_text(json.dumps(manifest))

    report = import_completed_replicates_from_benchmark(
        source_root,
        cfg,
        scenarios=["S0"],
        seeds=[0],
    )

    assert report["imported"] == 1
    target_dir = replicate_dir(cfg, "S0", 0)
    copied_variants = set(pd.read_csv(target_dir / "variants.csv")["variant_id"])
    assert "sobi/cluster_keep_top_03_lags_1_2_3_5_full/rank_2/seed_0" in copied_variants
    assert "fastica/cluster_keep_top_03_full/rank_2/seed_0" not in copied_variants
    assert "infomax/cluster_keep_top_03_full/rank_2/seed_0" not in copied_variants
    assert "jade/cluster_keep_top_03_full/rank_2/seed_0" not in copied_variants
    assert "raw" in copied_variants
    assert "fastica/all/rank_2/seed_0" not in copied_variants
    target_manifest = json.loads((target_dir / "manifest.json").read_text())
    assert target_manifest["benchmark_version"] == "core_sobi"
    assert target_manifest["complete"] is True
    assert (target_dir / "graphs" / "cgc" / "raw.npz").exists()


def test_method_split_guard_rejects_unrequested_fastica_variant():
    variants = [
        SimpleNamespace(variant_id="clean"),
        SimpleNamespace(variant_id="infomax/cluster_keep_top_03_full/rank_2/seed_0"),
        SimpleNamespace(variant_id="fastica/all/rank_2/seed_0"),
    ]

    with pytest.raises(RuntimeError, match="unrequested BSS variants"):
        _assert_no_unrequested_bss_variants(variants, ("infomax",))


def test_legacy_fastica_manifest_without_version_can_resume(tmp_path):
    cfg = SimulationConfig.from_dict(
        {
            **_smoke_cfg(tmp_path).to_dict(),
            "bss": {"methods": ["fastica"], "keep_top": 3},
        }
    )

    assert _manifest_variant_implementation_current({"complete": True}, cfg)


def test_legacy_nonfastica_manifest_without_version_is_stale(tmp_path):
    cfg = SimulationConfig.from_dict(
        {
            **_smoke_cfg(tmp_path).to_dict(),
            "bss": {"methods": ["infomax"], "keep_top": 3},
        }
    )

    assert not _manifest_variant_implementation_current({"complete": True}, cfg)


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


def test_resume_extends_missing_estimator_outputs(tmp_path, monkeypatch):
    from ica_denoising.simulation import runner

    cfg = SimulationConfig.from_dict(
        {
            **_smoke_cfg(tmp_path).to_dict(),
            "seeds": [0],
            "n_frames": 120,
            "estimator": {
                **_smoke_cfg(tmp_path).to_dict()["estimator"],
                "estimators": ["var"],
            },
        }
    )
    run_benchmark(cfg)
    out_dir = replicate_dir(cfg, "S0", 0)
    graph_before = pd.read_csv(out_dir / "graph_metrics.csv")
    assert set(graph_before["estimator"]) == {"var"}

    def _new_estimator(traces):
        n = traces.shape[0]
        scores = np.zeros((n, n), dtype=float)
        if n > 1:
            scores[0, 1] = 1.0
        binary = (scores != 0).astype(int)
        return GraphEstimate(
            estimator="new_estimator",
            scores=scores,
            binary=binary,
            binary_fdr=None,
            warnings=(),
        )

    monkeypatch.setitem(runner.ESTIMATORS, "new_estimator", _new_estimator)
    extended_cfg = SimulationConfig.from_dict(
        {
            **cfg.to_dict(),
            "estimator": {
                **cfg.to_dict()["estimator"],
                "estimators": ["var", "new_estimator"],
            },
        }
    )

    run_benchmark(extended_cfg, resume=True)

    graph_after = pd.read_csv(out_dir / "graph_metrics.csv")
    assert {"var", "new_estimator"} == set(graph_after["estimator"])
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert "new_estimator" in manifest["resolved_config"]["estimator"]["estimators"]
    assert (out_dir / "graphs" / "new_estimator" / "raw.npz").exists()


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


def test_jpcmciplus_adapter_runs_on_small_trace_matrix():
    traces = np.random.default_rng(0).normal(size=(4, 80))

    estimate = estimate_jpcmciplus(traces, tau_max=2, alpha=0.05)

    assert estimate.estimator == "jpcmciplus"
    assert estimate.binary.shape == (4, 4)
    assert estimate.scores.shape == (4, 4)


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
