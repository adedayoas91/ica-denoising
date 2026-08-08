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
import shutil
import subprocess
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from csl.experiments import graph_metrics as gm
from csl.experiments.simulation_adapters import ESTIMATORS

from .config import ScenarioConfig, SimulationConfig
from .dataset import SimulatedDataset, build_dataset, default_scenarios
from .metrics import behavior_metrics, state_metrics, trace_metrics
from .variants import build_variants

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[Mapping[str, object]], None]
_CORE_BSS_VARIANT_IMPLEMENTATION_VERSION = "core_bss_method_specific_controls_v2"

__all__ = [
    "import_completed_replicates_from_benchmark",
    "run_replicate",
    "run_benchmark",
    "replicate_dir",
]


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


def _estimator_list(cfg: SimulationConfig) -> list[str]:
    """Return configured estimators, including legacy PCMCI+ toggle support."""

    estimators = list(dict.fromkeys(str(e) for e in cfg.estimator.estimators))
    if cfg.estimator.run_pcmci and "pcmci" not in estimators:
        estimators.append("pcmci")
    return estimators


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
            backend=est_cfg.cgc_backend,
            compute_fdr=str(est_cfg.correction).lower() == "fdr",
            random_state=est_cfg.estimator_seed,
        )
    if estimator == "var":
        return fn(traces, orders=est_cfg.var_orders)
    if estimator in ("pcmci", "jpcmciplus"):
        return fn(
            traces,
            tau_max=est_cfg.n_lags,
            alpha=est_cfg.pcmci_alpha,
        )
    return fn(traces)


def _rank_targets(cfg: SimulationConfig, traces: np.ndarray) -> tuple[tuple[str, int], ...]:
    """Resolve configured BSS rank modes to concrete reconstruction ranks."""

    n_max = min(traces.shape)
    rows: list[tuple[str, int]] = []
    x = np.asarray(traces, dtype=float).T
    for mode in cfg.bss.ranks:
        token = str(mode).lower()
        if token == "full":
            rank = n_max
        elif token.startswith("ev"):
            threshold = float(token.removeprefix("ev")) / 100.0
            if not 0.0 < threshold <= 1.0:
                raise ValueError(f"Invalid BSS rank mode: {mode!r}")
            pca = PCA(n_components=n_max).fit(x)
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            rank = int(np.searchsorted(cumulative, threshold, side="left") + 1)
        elif token.startswith("rank"):
            rank = int(token.removeprefix("rank").strip("_"))
        else:
            raise ValueError(f"Unknown BSS rank mode: {mode!r}")
        rows.append((token, max(1, min(int(rank), n_max))))
    if not rows:
        rows.append(("full", n_max))
    return tuple(dict.fromkeys(rows))


def _build_variants_for_dataset(cfg: SimulationConfig, dataset: SimulatedDataset):
    """Build configured trace variants for an already generated dataset."""

    rank_target = max(2, int(0.8 * dataset.clean_fluorescence.shape[0]))
    bss_random_states = cfg.bss.random_states or (cfg.seed,)
    variants = build_variants(
        clean=dataset.clean_fluorescence,
        corrupted=dataset.corrupted,
        artifact=dataset.artifact,
        methods=cfg.bss.methods,
        keep_top=cfg.bss.keep_top,
        rank_target=rank_target,
        random_state=bss_random_states[0],
        sample_rate_hz=cfg.sample_rate_hz,
        lowpass_cutoff_hz=cfg.bss.lowpass_cutoff_hz,
        rank_targets=_rank_targets(cfg, dataset.corrupted),
        random_states=bss_random_states,
        sobi_lag_sets=cfg.bss.sobi_lag_sets,
    )
    _assert_no_unrequested_bss_variants(variants, cfg.bss.methods)
    return variants


def _assert_no_unrequested_bss_variants(variants, methods: tuple[str, ...]) -> None:
    requested = {str(method).lower() for method in methods}
    bss_methods = {"fastica", "infomax", "sobi", "jade"}
    unexpected = sorted(
        variant.variant_id
        for variant in variants
        for method in bss_methods - requested
        if variant.variant_id.startswith(f"{method}/")
    )
    if unexpected:
        raise RuntimeError(
            "Method-specific benchmark produced unrequested BSS variants: "
            f"{unexpected[:10]}"
        )


def _variant_implementation_version(cfg: SimulationConfig) -> str:
    return _CORE_BSS_VARIANT_IMPLEMENTATION_VERSION


def _manifest_variant_implementation_current(
    manifest: Mapping[str, object],
    cfg: SimulationConfig,
) -> bool:
    expected = _variant_implementation_version(cfg)
    if not expected:
        return True
    actual = manifest.get("variant_implementation_version")
    if actual == expected:
        return True
    return actual is None and _legacy_fastica_manifest_compatible(cfg)


def _legacy_fastica_manifest_compatible(cfg: SimulationConfig) -> bool:
    requested_bss = {
        str(method).lower()
        for method in cfg.bss.methods
        if str(method).lower() != "pca"
    }
    return requested_bss == {"fastica"}


def _select_scored_graph(est, correction: str) -> tuple[np.ndarray, str]:
    correction = str(correction).lower()
    if correction == "none":
        return np.asarray(est.binary, dtype=int), "none"
    if correction == "fdr":
        if est.binary_fdr is not None:
            return np.asarray(est.binary_fdr, dtype=int), "fdr"
        return np.asarray(est.binary, dtype=int), "fdr_unavailable"
    raise ValueError(f"Unknown graph correction: {correction!r}")


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

    variants = _build_variants_for_dataset(cfg, dataset)

    variant_rows = []
    trace_rows = []
    behavior_rows = []
    state_rows = []
    graph_rows = []
    failures: list[dict] = []

    estimators = _estimator_list(cfg)

    n_observed = dataset.clean_fluorescence.shape[0]
    truth = dataset.truth_adjacency[:n_observed, :n_observed]

    # estimate clean-oracle graphs first for reference Jaccard overlap.
    clean_variant = next(v for v in variants if v.variant_id == "clean")
    clean_refs: dict[str, np.ndarray] = {}
    for estimator in estimators:
        try:
            clean_est = _graph_estimate_for_variant(
                estimator, clean_variant.traces, cfg
            )
            clean_refs[estimator] = _select_scored_graph(
                clean_est, cfg.estimator.correction
            )[0]
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
                "rank_mode": variant.rank_mode,
                "random_state": variant.random_state,
                "sobi_lags": " ".join(str(lag) for lag in variant.sobi_lags),
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
            scored_binary, graph_correction = _select_scored_graph(
                est, cfg.estimator.correction
            )
            gpath = out_dir / "graphs" / estimator / f"{_safe(variant.variant_id)}.npz"
            gpath.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "scores": est.scores,
                "binary": est.binary,
                "scored_binary": scored_binary,
                "trace_hash": trace_hash,
                "graph_correction": graph_correction,
            }
            if est.binary_fdr is not None:
                payload["binary_fdr"] = est.binary_fdr
            np.savez_compressed(gpath, **payload)
            metrics = gm.graph_recovery_metrics(
                scored_binary,
                truth,
                scores=est.scores,
                reference=clean_refs.get(estimator),
            )
            metrics.update(
                {
                    "variant_id": variant.variant_id,
                    "estimator": estimator,
                    "trace_hash": trace_hash,
                    "graph_correction": graph_correction,
                }
            )
            graph_rows.append(metrics)

    _write_csv(out_dir / "variants.csv", variant_rows)
    _write_csv(out_dir / "trace_metrics.csv", trace_rows)
    _write_csv(out_dir / "behavior_metrics.csv", behavior_rows)
    _write_csv(out_dir / "state_metrics.csv", state_rows)
    _write_csv(out_dir / "graph_metrics.csv", graph_rows)
    _write_csv(out_dir / "failures.csv", failures)

    ended = datetime.now(timezone.utc)
    manifest = {
        "benchmark_version": cfg.benchmark_version,
        "schema_version": cfg.schema_version,
        "scenario_id": scenario.scenario_id,
        "seed": seed,
        "resolved_config": cfg.to_dict(),
        "git_revision": _git_revision(),
        "versions": _package_versions(),
        "variant_implementation_version": _variant_implementation_version(cfg),
        "dataset_hash": _hash_array(dataset.corrupted),
        "achieved_asr": dataset.achieved_asr,
        "start": started.isoformat(),
        "end": ended.isoformat(),
        "runtime_seconds": (ended - started).total_seconds(),
        "n_variants": len(variant_rows),
        "failures": failures,
        "complete": len(failures) == 0,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Replicate %s/seed_%s complete in %.1fs", scenario.scenario_id, seed, manifest["runtime_seconds"])
    return out_dir


def _graph_path(out_dir: Path, estimator: str, variant_id: str) -> Path:
    return out_dir / "graphs" / estimator / f"{_safe(variant_id)}.npz"


def _graph_pair_exists(out_dir: Path, estimator: str, variant_id: str) -> bool:
    return _graph_path(out_dir, estimator, variant_id).exists()


def _replicate_outputs_complete_for_config(
    out_dir: Path,
    cfg: SimulationConfig,
) -> bool:
    required = (
        "dataset.npz",
        "variants.csv",
        "trace_metrics.csv",
        "behavior_metrics.csv",
        "state_metrics.csv",
        "graph_metrics.csv",
    )
    if any(not (out_dir / name).exists() for name in required):
        return False
    try:
        variants = pd.read_csv(out_dir / "variants.csv")
        graph_metrics = pd.read_csv(out_dir / "graph_metrics.csv")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return False
    if variants.empty or graph_metrics.empty:
        return False
    variant_ids = set(variants["variant_id"].astype(str))
    if not _variant_ids_belong_to_config(variant_ids, cfg):
        return False
    expected = {
        (estimator, variant_id)
        for estimator in _estimator_list(cfg)
        for variant_id in variant_ids
    }
    present = {
        (str(row.estimator), str(row.variant_id))
        for row in graph_metrics[["estimator", "variant_id"]].drop_duplicates().itertuples(
            index=False
        )
    }
    if not expected.issubset(present):
        return False
    return all(
        _graph_pair_exists(out_dir, estimator, variant_id)
        for estimator, variant_id in expected
    )


def _replicate_complete_for_config(
    out_dir: Path,
    cfg: SimulationConfig,
) -> bool:
    manifest = out_dir / "manifest.json"
    try:
        manifest_data = json.loads(manifest.read_text())
        if not bool(manifest_data.get("complete")):
            return False
        if not _manifest_variant_implementation_current(manifest_data, cfg):
            return False
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return _replicate_outputs_complete_for_config(out_dir, cfg)


def _load_scored_graph(path: Path) -> np.ndarray:
    data = np.load(path)
    if "scored_binary" in data:
        return np.asarray(data["scored_binary"], dtype=int)
    return np.asarray(data["binary"], dtype=int)


def _write_graph_result(
    out_dir: Path,
    estimator: str,
    variant,
    trace_hash: str,
    cfg: SimulationConfig,
    truth: np.ndarray,
    reference: np.ndarray | None,
) -> tuple[dict, np.ndarray]:
    est = _graph_estimate_for_variant(estimator, variant.traces, cfg)
    scored_binary, graph_correction = _select_scored_graph(
        est, cfg.estimator.correction
    )
    gpath = _graph_path(out_dir, estimator, variant.variant_id)
    gpath.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scores": est.scores,
        "binary": est.binary,
        "scored_binary": scored_binary,
        "trace_hash": trace_hash,
        "graph_correction": graph_correction,
    }
    if est.binary_fdr is not None:
        payload["binary_fdr"] = est.binary_fdr
    np.savez_compressed(gpath, **payload)
    if variant.variant_id == "clean" and reference is None:
        reference = scored_binary
    metrics = gm.graph_recovery_metrics(
        scored_binary,
        truth,
        scores=est.scores,
        reference=reference,
    )
    metrics.update(
        {
            "variant_id": variant.variant_id,
            "estimator": estimator,
            "trace_hash": trace_hash,
            "graph_correction": graph_correction,
        }
    )
    return metrics, scored_binary


def _filter_resolved_failures(out_dir: Path, failures: pd.DataFrame) -> pd.DataFrame:
    if failures.empty or not {"variant", "estimator"}.issubset(failures.columns):
        return failures
    keep = []
    for row in failures.itertuples(index=False):
        variant = str(getattr(row, "variant"))
        estimator = str(getattr(row, "estimator"))
        keep.append(not _graph_pair_exists(out_dir, estimator, variant))
    return failures.loc[keep].copy()


def _extend_replicate_graphs(
    cfg: SimulationConfig,
    scenario: ScenarioConfig,
    seed: int,
) -> Path:
    """Append missing graph-estimator outputs to an existing replicate."""

    started = datetime.now(timezone.utc)
    out_dir = replicate_dir(cfg, scenario.scenario_id, seed)
    if not (out_dir / "dataset.npz").exists():
        return run_replicate(cfg, scenario, seed)
    try:
        manifest_data = json.loads((out_dir / "manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        manifest_data = {}
    if not _manifest_variant_implementation_current(manifest_data, cfg):
        return run_replicate(cfg, scenario, seed)

    dataset = SimulatedDataset.load(out_dir / "dataset.npz")
    variants = _build_variants_for_dataset(cfg, dataset)
    variant_by_id = {variant.variant_id: variant for variant in variants}
    trace_hashes = {
        variant_id: _hash_array(variant.traces)
        for variant_id, variant in variant_by_id.items()
    }

    required_csvs = (
        "variants.csv",
        "trace_metrics.csv",
        "behavior_metrics.csv",
        "state_metrics.csv",
    )
    if any(not (out_dir / name).exists() for name in required_csvs):
        return run_replicate(cfg, scenario, seed)

    try:
        graph_metrics = pd.read_csv(out_dir / "graph_metrics.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        graph_metrics = pd.DataFrame()
    existing_pairs = set()
    if not graph_metrics.empty:
        keep_rows = []
        for row in graph_metrics[["estimator", "variant_id"]].drop_duplicates().itertuples(
            index=False
        ):
            estimator = str(row.estimator)
            variant_id = str(row.variant_id)
            if _graph_pair_exists(out_dir, estimator, variant_id):
                existing_pairs.add((estimator, variant_id))
        for row in graph_metrics.itertuples(index=False):
            keep_rows.append(
                (str(row.estimator), str(row.variant_id)) in existing_pairs
            )
        graph_metrics = graph_metrics.loc[keep_rows].copy()
    graph_rows = graph_metrics.to_dict(orient="records")

    failures = _filter_resolved_failures(
        out_dir, _read_csv_or_empty(out_dir / "failures.csv")
    )
    failure_rows = failures.to_dict(orient="records")

    n_observed = dataset.clean_fluorescence.shape[0]
    truth = dataset.truth_adjacency[:n_observed, :n_observed]
    clean_variant = variant_by_id["clean"]
    estimators = _estimator_list(cfg)
    clean_refs: dict[str, np.ndarray] = {}

    for estimator in estimators:
        clean_path = _graph_path(out_dir, estimator, "clean")
        if clean_path.exists():
            clean_refs[estimator] = _load_scored_graph(clean_path)
            continue
        try:
            metrics, scored_binary = _write_graph_result(
                out_dir,
                estimator,
                clean_variant,
                trace_hashes["clean"],
                cfg,
                truth,
                None,
            )
            graph_rows.append(metrics)
            existing_pairs.add((estimator, "clean"))
            clean_refs[estimator] = scored_binary
        except Exception as exc:
            failure_rows.append(
                {"variant": "clean", "estimator": estimator, "error": str(exc)}
            )

    for variant_id, variant in variant_by_id.items():
        for estimator in estimators:
            if (estimator, variant_id) in existing_pairs:
                continue
            try:
                metrics, _ = _write_graph_result(
                    out_dir,
                    estimator,
                    variant,
                    trace_hashes[variant_id],
                    cfg,
                    truth,
                    clean_refs.get(estimator),
                )
                graph_rows.append(metrics)
                existing_pairs.add((estimator, variant_id))
            except Exception as exc:
                failure_rows.append(
                    {
                        "variant": variant_id,
                        "estimator": estimator,
                        "error": str(exc),
                    }
                )

    _write_csv(out_dir / "graph_metrics.csv", graph_rows)
    _write_csv(out_dir / "failures.csv", failure_rows)

    ended = datetime.now(timezone.utc)
    manifest_path = out_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        manifest = {}
    manifest.update(
        {
            "benchmark_version": cfg.benchmark_version,
            "schema_version": cfg.schema_version,
            "scenario_id": scenario.scenario_id,
            "seed": seed,
            "resolved_config": cfg.to_dict(),
            "git_revision": _git_revision(),
            "versions": _package_versions(),
            "variant_implementation_version": _variant_implementation_version(cfg),
            "dataset_hash": _hash_array(dataset.corrupted),
            "achieved_asr": dataset.achieved_asr,
            "end": ended.isoformat(),
            "runtime_seconds": float(manifest.get("runtime_seconds", 0.0))
            + (ended - started).total_seconds(),
            "n_variants": len(variant_by_id),
            "failures": failure_rows,
            "complete": False,
        }
    )
    manifest.setdefault("start", started.isoformat())
    manifest["complete"] = (
        len(failure_rows) == 0
        and _replicate_outputs_complete_for_config(out_dir, cfg)
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "Replicate %s/seed_%s extended in %.1fs",
        scenario.scenario_id,
        seed,
        (ended - started).total_seconds(),
    )
    return out_dir


def run_benchmark(
    cfg: SimulationConfig,
    *,
    scenarios: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    resume: Optional[bool] = None,
    dry_run: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> list[Path]:
    """Run the benchmark across scenarios and seeds."""

    resume = cfg.run.resume if resume is None else resume
    seed_list = seeds if seeds is not None else list(cfg.seeds)
    scenario_list = _resolve_scenarios(cfg, scenarios)
    total = len(seed_list) * len(scenario_list)
    completed = 0

    planned: list[Path] = []
    for scenario in scenario_list:
        for seed in seed_list:
            out_dir = replicate_dir(cfg, scenario.scenario_id, seed)
            manifest = out_dir / "manifest.json"
            if resume and manifest.exists():
                if _replicate_complete_for_config(out_dir, cfg):
                    logger.info("Skipping complete replicate %s", out_dir)
                    planned.append(out_dir)
                    completed += 1
                    _notify_progress(
                        progress_callback,
                        event="skipped",
                        stage="run",
                        scenario_id=scenario.scenario_id,
                        seed=seed,
                        completed=completed,
                        total=total,
                        path=out_dir,
                    )
                    continue
            if dry_run:
                logger.info("[dry-run] would run %s seed %s", scenario.scenario_id, seed)
                planned.append(out_dir)
                completed += 1
                _notify_progress(
                    progress_callback,
                    event="planned",
                    stage="run",
                    scenario_id=scenario.scenario_id,
                    seed=seed,
                    completed=completed,
                    total=total,
                    path=out_dir,
                )
                continue
            _notify_progress(
                progress_callback,
                event="start",
                stage="run",
                scenario_id=scenario.scenario_id,
                seed=seed,
                completed=completed,
                total=total,
                path=out_dir,
            )
            if resume and manifest.exists():
                planned.append(_extend_replicate_graphs(cfg, scenario, seed))
            else:
                planned.append(run_replicate(cfg, scenario, seed))
            completed += 1
            _notify_progress(
                progress_callback,
                event="done",
                stage="run",
                scenario_id=scenario.scenario_id,
                seed=seed,
                completed=completed,
                total=total,
                path=out_dir,
            )
    return planned


def import_completed_replicates_from_benchmark(
    source_root: str | Path,
    target_cfg: SimulationConfig,
    *,
    scenarios: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    resume: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    """Copy completed all-method replicates into a method-specific benchmark.

    The copied replicate is filtered to the target configuration's BSS methods
    plus the shared controls that :func:`build_variants` always emits. A
    complete target manifest is written so ``run_benchmark(..., resume=True)``
    skips the imported replicate.
    """

    source_root = Path(source_root)
    seed_list = seeds if seeds is not None else list(target_cfg.seeds)
    scenario_list = _resolve_scenarios(target_cfg, scenarios)
    total = len(seed_list) * len(scenario_list)
    report = {
        "checked": 0,
        "imported": 0,
        "skipped_existing": 0,
        "missing_source": 0,
        "incomplete_source": 0,
        "failed": 0,
    }
    for scenario in scenario_list:
        for seed in seed_list:
            report["checked"] += 1
            target_dir = replicate_dir(target_cfg, scenario.scenario_id, seed)
            target_manifest = target_dir / "manifest.json"
            source_dir = source_root / scenario.scenario_id / f"seed_{seed}"
            event = "imported"
            if resume and _replicate_complete_for_config(target_dir, target_cfg):
                report["skipped_existing"] += 1
                event = "skipped_existing"
            elif not (source_dir / "manifest.json").exists():
                report["missing_source"] += 1
                event = "missing_source"
            elif not _manifest_complete(source_dir / "manifest.json"):
                report["incomplete_source"] += 1
                event = "incomplete_source"
            else:
                try:
                    _import_replicate_subset(source_dir, target_dir, target_cfg)
                    report["imported"] += 1
                except Exception as exc:  # pragma: no cover - defensive notebook path
                    report["failed"] += 1
                    event = "failed"
                    logger.exception(
                        "Failed to import %s into %s: %s",
                        source_dir,
                        target_dir,
                        exc,
                    )
            _notify_progress(
                progress_callback,
                event=event,
                stage="import",
                scenario_id=scenario.scenario_id,
                seed=seed,
                completed=report["checked"],
                total=total,
                source=source_dir,
                path=target_dir,
            )
    return report


def _safe(name: str) -> str:
    return name.replace("/", "__")


def _write_csv(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def _notify_progress(
    progress_callback: ProgressCallback | None,
    **payload: object,
) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _manifest_complete(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text()).get("complete"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def _import_replicate_subset(
    source_dir: Path,
    target_dir: Path,
    target_cfg: SimulationConfig,
) -> None:
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    if not _manifest_variant_implementation_current(source_manifest, target_cfg):
        raise ValueError(
            f"{source_dir}: source variants were produced by an older BSS implementation."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "graphs").mkdir(exist_ok=True)
    shutil.copy2(source_dir / "dataset.npz", target_dir / "dataset.npz")

    variants = pd.read_csv(source_dir / "variants.csv")
    selected_ids = set(
        variants.loc[
            variants["variant_id"].map(
                lambda variant_id: _variant_belongs_to_methods(
                    str(variant_id),
                    target_cfg.bss.methods,
                )
            ),
            "variant_id",
        ].astype(str)
    )
    if not selected_ids:
        raise ValueError(
            f"{source_dir}: no variants matched methods {target_cfg.bss.methods!r}."
        )
    missing_methods = _missing_method_specific_variants(
        selected_ids, target_cfg.bss.methods
    )
    if missing_methods:
        raise ValueError(
            f"{source_dir}: missing method-specific variants for {missing_methods!r}."
        )

    for csv_name in (
        "variants.csv",
        "trace_metrics.csv",
        "behavior_metrics.csv",
        "state_metrics.csv",
        "graph_metrics.csv",
    ):
        frame = pd.read_csv(source_dir / csv_name)
        filtered = frame[frame["variant_id"].astype(str).isin(selected_ids)]
        filtered.to_csv(target_dir / csv_name, index=False)

    failures = _read_csv_or_empty(source_dir / "failures.csv")
    if not failures.empty and "variant" in failures:
        failures = failures[failures["variant"].astype(str).isin(selected_ids)]
    failures.to_csv(target_dir / "failures.csv", index=False)

    graph_metrics = pd.read_csv(target_dir / "graph_metrics.csv")
    for row in graph_metrics[["estimator", "variant_id"]].drop_duplicates().itertuples(
        index=False
    ):
        source_graph = (
            source_dir / "graphs" / str(row.estimator) / f"{_safe(str(row.variant_id))}.npz"
        )
        if not source_graph.exists():
            raise FileNotFoundError(source_graph)
        target_graph = (
            target_dir / "graphs" / str(row.estimator) / f"{_safe(str(row.variant_id))}.npz"
        )
        target_graph.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_graph, target_graph)

    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        **source_manifest,
        "benchmark_version": target_cfg.benchmark_version,
        "resolved_config": target_cfg.to_dict(),
        "variant_implementation_version": _variant_implementation_version(target_cfg),
        "start": now,
        "end": now,
        "runtime_seconds": 0.0,
        "source_runtime_seconds": source_manifest.get("runtime_seconds"),
        "n_variants": int(len(selected_ids)),
        "failures": failures.to_dict(orient="records"),
        "complete": failures.empty,
        "imported_from": str(source_dir),
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _variant_belongs_to_methods(
    variant_id: str,
    methods: tuple[str, ...],
) -> bool:
    if variant_id in {"clean", "raw", "artifact_oracle", "oracle_selection"}:
        return True
    if variant_id.startswith(("pca/", "random_subspace/", "causal_lowpass/")):
        return True
    for method in methods:
        method = str(method).lower()
        if variant_id.startswith(f"{method}/"):
            return True
    return False


def _variant_ids_belong_to_config(
    variant_ids: set[str],
    cfg: SimulationConfig,
) -> bool:
    return all(
        _variant_belongs_to_methods(variant_id, cfg.bss.methods)
        for variant_id in variant_ids
    ) and not _missing_method_specific_variants(variant_ids, cfg.bss.methods)


def _missing_method_specific_variants(
    variant_ids: set[str],
    methods: tuple[str, ...],
) -> list[str]:
    missing = []
    for method in methods:
        method = str(method).lower()
        if method == "pca":
            continue
        if not any(variant_id.startswith(f"{method}/") for variant_id in variant_ids):
            missing.append(method)
    return missing


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
