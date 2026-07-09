"""Benchmark runner: generates datasets, variants, graphs, and metrics.

Implements the per-replicate output contract and manifest of Section 5.1, with
``--resume`` support and per-method failure isolation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import subprocess
from typing import Optional

import numpy as np
import pandas as pd

from csl.experiments import graph_metrics as gm
from csl.experiments.simulation_adapters import ESTIMATORS

from .config import ScenarioConfig, SimulationConfig
from .dataset import build_dataset, default_scenarios
from .metrics import behavior_metrics, state_metrics, trace_metrics
from .variants import build_variants

logger = logging.getLogger(__name__)

__all__ = ["run_replicate", "run_benchmark", "replicate_dir"]


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        rev = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        return f"{rev}{'-dirty' if dirty else ''}"
    except Exception:  # pragma: no cover - git optional
        return "unknown"


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "sklearn", "pandas"):
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except Exception:  # pragma: no cover
            versions[pkg] = "missing"
    return versions


def replicate_dir(cfg: SimulationConfig, scenario_id: str, seed: int) -> Path:
    """Return the output directory for one replicate."""

    return (
        Path(cfg.run.output_dir)
        / cfg.benchmark_version
        / scenario_id
        / f"seed_{seed}"
    )


def _resolve_scenarios(
    cfg: SimulationConfig, names: Optional[list[str]]
) -> list[ScenarioConfig]:
    scenarios = list(cfg.scenarios) or list(default_scenarios())
    if names:
        wanted = set(names)
        scenarios = [s for s in scenarios if s.scenario_id in wanted]
    return scenarios


def _graph_estimate_for_variant(
    estimator: str, traces: np.ndarray, cfg: SimulationConfig
):
    est_cfg = cfg.estimator
    fn = ESTIMATORS[estimator]
    if estimator in ("cgc", "cgc_star"):
        return fn(
            traces,
            n_pasts=est_cfg.n_pasts,
            n_lags=est_cfg.n_lags,
            n_perm=est_cfg.n_perm,
            alpha=est_cfg.alpha,
            beta=est_cfg.beta,
            random_state=est_cfg.estimator_seed,
        )
    if estimator == "var":
        return fn(traces, orders=est_cfg.var_orders)
    return fn(traces)


def run_replicate(
    cfg: SimulationConfig,
    scenario: ScenarioConfig,
    seed: int,
) -> Path:
    """Generate one replicate and write the full output contract."""

    started = datetime.now(timezone.utc)
    out_dir = replicate_dir(cfg, scenario.scenario_id, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graphs").mkdir(exist_ok=True)

    dataset = build_dataset(cfg, scenario, seed)
    dataset.save(out_dir / "dataset.npz")

    rank_target = max(2, int(0.8 * dataset.clean_fluorescence.shape[0]))
    variants = build_variants(
        clean=dataset.clean_fluorescence,
        corrupted=dataset.corrupted,
        artifact=dataset.artifact,
        methods=cfg.bss.methods,
        keep_top=cfg.bss.keep_top,
        rank_target=rank_target,
        random_state=cfg.bss.random_states[0],
        sample_rate_hz=cfg.sample_rate_hz,
        lowpass_cutoff_hz=cfg.bss.lowpass_cutoff_hz,
    )

    variant_rows = []
    trace_rows = []
    behavior_rows = []
    state_rows = []
    graph_rows = []
    failures: list[dict] = []

    estimators = list(cfg.estimator.estimators)
    if cfg.estimator.run_pcmci and "pcmci" not in estimators:
        estimators.append("pcmci")

    truth = dataset.truth_adjacency

    # estimate clean-oracle graphs first for reference Jaccard overlap.
    clean_variant = next(v for v in variants if v.variant_id == "clean")
    clean_refs: dict[str, np.ndarray] = {}
    for estimator in estimators:
        try:
            clean_refs[estimator] = _graph_estimate_for_variant(
                estimator, clean_variant.traces, cfg
            ).binary
        except Exception as exc:  # pragma: no cover
            failures.append({"variant": "clean", "estimator": estimator, "error": str(exc)})

    for variant in variants:
        trace_hash = _hash_array(variant.traces)
        variant_rows.append(
            {
                "variant_id": variant.variant_id,
                "trace_hash": trace_hash,
                "reconstruction_rank": variant.reconstruction_rank,
                "explained_variance_ratio": variant.explained_variance_ratio,
                "uses_ground_truth": variant.uses_ground_truth,
            }
        )
        tm = trace_metrics(variant.traces, dataset.clean_fluorescence, dataset.artifact)
        tm.update({"variant_id": variant.variant_id, "trace_hash": trace_hash})
        trace_rows.append(tm)

        try:
            bm = behavior_metrics(variant.traces, dataset.vigor, dataset.bout)
            bm.update({"variant_id": variant.variant_id, "trace_hash": trace_hash})
            behavior_rows.append(bm)
        except Exception as exc:  # pragma: no cover
            failures.append({"variant": variant.variant_id, "stage": "behavior", "error": str(exc)})

        sm = state_metrics(variant.traces, dataset.artifact)
        sm.update({"variant_id": variant.variant_id, "trace_hash": trace_hash})
        state_rows.append(sm)

        for estimator in estimators:
            try:
                est = _graph_estimate_for_variant(estimator, variant.traces, cfg)
            except Exception as exc:
                failures.append(
                    {"variant": variant.variant_id, "estimator": estimator, "error": str(exc)}
                )
                continue
            gpath = out_dir / "graphs" / estimator / f"{_safe(variant.variant_id)}.npz"
            gpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                gpath, scores=est.scores, binary=est.binary, trace_hash=trace_hash
            )
            metrics = gm.graph_recovery_metrics(
                est.binary,
                truth,
                scores=est.scores,
                reference=clean_refs.get(estimator),
            )
            metrics.update(
                {
                    "variant_id": variant.variant_id,
                    "estimator": estimator,
                    "trace_hash": trace_hash,
                }
            )
            graph_rows.append(metrics)

    _write_csv(out_dir / "variants.csv", variant_rows)
    _write_csv(out_dir / "trace_metrics.csv", trace_rows)
    _write_csv(out_dir / "behavior_metrics.csv", behavior_rows)
    _write_csv(out_dir / "state_metrics.csv", state_rows)
    _write_csv(out_dir / "graph_metrics.csv", graph_rows)

    ended = datetime.now(timezone.utc)
    manifest = {
        "benchmark_version": cfg.benchmark_version,
        "schema_version": cfg.schema_version,
        "scenario_id": scenario.scenario_id,
        "seed": seed,
        "resolved_config": cfg.to_dict(),
        "git_revision": _git_revision(),
        "versions": _package_versions(),
        "dataset_hash": _hash_array(dataset.corrupted),
        "achieved_asr": dataset.achieved_asr,
        "start": started.isoformat(),
        "end": ended.isoformat(),
        "runtime_seconds": (ended - started).total_seconds(),
        "n_variants": len(variant_rows),
        "failures": failures,
        "complete": True,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Replicate %s/seed_%s complete in %.1fs", scenario.scenario_id, seed, manifest["runtime_seconds"])
    return out_dir


def run_benchmark(
    cfg: SimulationConfig,
    *,
    scenarios: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    resume: Optional[bool] = None,
    dry_run: bool = False,
) -> list[Path]:
    """Run the benchmark across scenarios and seeds."""

    resume = cfg.run.resume if resume is None else resume
    seed_list = seeds if seeds is not None else list(cfg.seeds)
    scenario_list = _resolve_scenarios(cfg, scenarios)

    planned: list[Path] = []
    for scenario in scenario_list:
        for seed in seed_list:
            out_dir = replicate_dir(cfg, scenario.scenario_id, seed)
            manifest = out_dir / "manifest.json"
            if resume and manifest.exists():
                try:
                    if json.loads(manifest.read_text()).get("complete"):
                        logger.info("Skipping complete replicate %s", out_dir)
                        planned.append(out_dir)
                        continue
                except json.JSONDecodeError:  # pragma: no cover
                    pass
            if dry_run:
                logger.info("[dry-run] would run %s seed %s", scenario.scenario_id, seed)
                planned.append(out_dir)
                continue
            planned.append(run_replicate(cfg, scenario, seed))
    return planned


def _safe(name: str) -> str:
    return name.replace("/", "__")


def _write_csv(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
