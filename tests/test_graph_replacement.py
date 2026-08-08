from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ica_denoising.simulation.graph_replacement import (
    replace_graph_estimator_outputs,
)


def _write_manifest(path: Path, *, beta: float = 0.001) -> None:
    manifest = {
        "benchmark_version": "test",
        "schema_version": "1",
        "scenario_id": "S0",
        "seed": 0,
        "resolved_config": {
            "estimator": {
                "estimators": ["cgc", "cgc_star"],
                "n_perm": 200,
                "beta": beta,
            }
        },
        "git_revision": "test",
        "versions": {},
        "dataset_hash": "hash",
        "start": "2026-01-01T00:00:00+00:00",
        "end": "2026-01-01T00:00:00+00:00",
        "runtime_seconds": 1.0,
        "complete": True,
        "failures": [],
    }
    path.write_text(json.dumps(manifest))


def _write_graph(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.array([[0, value], [0, 0]], dtype=int)
    np.savez_compressed(path, scores=binary.astype(float), binary=binary)


def _safe(variant_id: str) -> str:
    return variant_id.replace("/", "__")


def test_replace_graph_estimator_outputs_keeps_other_estimators(tmp_path):
    source = tmp_path / "source" / "S0" / "seed_0"
    target = tmp_path / "target" / "S0" / "seed_0"
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    _write_manifest(source / "manifest.json", beta=0.005)
    _write_manifest(target / "manifest.json", beta=0.001)

    variant_id = "raw"
    source_metrics = pd.DataFrame(
        [
            {
                "variant_id": variant_id,
                "estimator": "cgc",
                "trace_hash": "hash",
                "graph_correction": "none",
                "f1": 1.0,
            },
            {
                "variant_id": variant_id,
                "estimator": "cgc_star",
                "trace_hash": "hash",
                "graph_correction": "none",
                "f1": 0.9,
            },
        ]
    )
    target_metrics = pd.DataFrame(
        [
            {
                "variant_id": variant_id,
                "estimator": "cgc",
                "trace_hash": "hash",
                "graph_correction": "fdr",
                "f1": 0.0,
            },
            {
                "variant_id": variant_id,
                "estimator": "var",
                "trace_hash": "hash",
                "graph_correction": "fdr_unavailable",
                "f1": 0.5,
            },
        ]
    )
    source_metrics.to_csv(source / "graph_metrics.csv", index=False)
    target_metrics.to_csv(target / "graph_metrics.csv", index=False)
    pd.DataFrame().to_csv(source / "failures.csv", index=False)
    pd.DataFrame(
        [{"variant": variant_id, "estimator": "var", "error": "keep"}]
    ).to_csv(target / "failures.csv", index=False)

    for estimator, value in (("cgc", 1), ("cgc_star", 1)):
        _write_graph(source / "graphs" / estimator / f"{_safe(variant_id)}.npz", value)
    _write_graph(target / "graphs" / "cgc" / f"{_safe(variant_id)}.npz", 0)
    _write_graph(target / "graphs" / "var" / f"{_safe(variant_id)}.npz", 1)

    report = replace_graph_estimator_outputs(source.parent.parent, target.parent.parent)

    assert report.replaced == 1
    merged = pd.read_csv(target / "graph_metrics.csv")
    by_estimator = {row.estimator: row.f1 for row in merged.itertuples(index=False)}
    assert by_estimator["cgc"] == 1.0
    assert by_estimator["cgc_star"] == 0.9
    assert by_estimator["var"] == 0.5
    updated = np.load(target / "graphs" / "cgc" / "raw.npz")
    assert updated["binary"][0, 1] == 1
    failures = pd.read_csv(target / "failures.csv")
    assert failures["estimator"].tolist() == ["var"]
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["graph_replacement_history"][0]["source_estimator_config"]["beta"] == 0.005
