"""Immutable configuration for the ground-truth simulation benchmark.

Every run writes its fully resolved configuration so that defaults are never the
only record of executed settings (Section 5.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import json
from pathlib import Path
from typing import Any

__all__ = [
    "GraphConfig",
    "DynamicsConfig",
    "CalciumConfig",
    "ArtifactConfig",
    "BehaviorConfig",
    "BssConfig",
    "EstimatorConfig",
    "RunConfig",
    "ScenarioConfig",
    "SimulationConfig",
    "load_config",
]


@dataclass(frozen=True)
class GraphConfig:
    """Ground-truth directed graph settings."""

    family: str = "sparse_random"  # sparse_random | feedforward_modular | bilateral_chain
    n_observed: int = 10
    n_hidden: int = 0
    edge_density: float = 0.2
    lag_order: int = 2
    weight_low: float = 0.1
    weight_high: float = 0.4
    spectral_radius: float = 0.9
    n_modules: int = 2


@dataclass(frozen=True)
class DynamicsConfig:
    """Neural dynamics settings."""

    model: str = "linear_var"  # linear_var | rate_spike
    innovation: str = "gaussian"
    snr: float = 5.0
    hidden_confounding: bool = False
    behavior_feedback: bool = False
    burn_in: int = 200


@dataclass(frozen=True)
class CalciumConfig:
    """Calcium observation-model settings."""

    enabled: bool = True
    decay_seconds: float = 1.5
    decay_jitter: float = 0.3
    gain_low: float = 0.8
    gain_high: float = 1.2
    measurement_noise: float = 0.05
    poisson_noise: bool = False
    saturation: float = 0.0  # 0 disables saturation


@dataclass(frozen=True)
class ArtifactConfig:
    """Artifact-model settings; strength is an achieved artifact-to-signal ratio."""

    types: tuple[str, ...] = ()  # drift | motion | neuropil | neural_band | crosstalk
    artifact_to_signal: float = 0.5
    severity: float = 1.0
    drift_timescale: float = 200.0
    motion_rate: float = 0.01
    motion_decay: float = 2.0
    crosstalk_strength: float = 0.05
    nonfinite_fraction: float = 0.0


@dataclass(frozen=True)
class BehaviorConfig:
    """Behavior-generation settings."""

    n_parents: int = 3
    behavior_lag: int = 1
    noise: float = 0.1
    bout_quantile: float = 0.75
    smooth_window: int = 5


@dataclass(frozen=True)
class BssConfig:
    """BSS variant / control settings."""

    methods: tuple[str, ...] = ("fastica", "pca")
    ranks: tuple[str, ...] = ("full",)  # full | ev90 | ev95 | ev99
    random_states: tuple[int, ...] = (0,)
    sobi_lag_sets: tuple[tuple[int, ...], ...] = ((1, 2, 3, 5),)
    keep_top: int = 4
    lowpass_cutoff_hz: float = 0.0  # 0 disables


@dataclass(frozen=True)
class EstimatorConfig:
    """Connectivity-estimator settings."""

    estimators: tuple[str, ...] = ("cgc", "cgc_star", "var")
    n_pasts: int = 2
    n_lags: int = 2
    n_perm: int = 200
    alpha: float = 0.01
    beta: float = 0.001
    correction: str = "fdr"  # fdr | none
    estimator_seed: int = 0
    var_orders: tuple[int, ...] = (1, 2, 3)
    run_pcmci: bool = False


@dataclass(frozen=True)
class RunConfig:
    """Runner / IO settings."""

    output_dir: str = "outputs/simulation"
    resume: bool = True
    workers: int = 1


@dataclass(frozen=True)
class ScenarioConfig:
    """A single named simulation scenario."""

    scenario_id: str
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    calcium: CalciumConfig = field(default_factory=CalciumConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level immutable benchmark configuration."""

    benchmark_version: str = "v0"
    schema_version: str = "1"
    seed: int = 0
    seeds: tuple[int, ...] = (0, 1)
    sample_rate_hz: float = 5.0
    n_frames: int = 600
    graph: GraphConfig = field(default_factory=GraphConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    bss: BssConfig = field(default_factory=BssConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    run: RunConfig = field(default_factory=RunConfig)
    scenarios: tuple[ScenarioConfig, ...] = ()

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the resolved configuration."""

        return _to_jsonable(self)

    def to_json(self, indent: int = 2) -> str:
        """Return the resolved configuration as a JSON string."""

        return json.dumps(self.to_dict(), indent=indent)

    def write_resolved(self, path: str | Path) -> Path:
        """Write the fully resolved configuration to ``path``."""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json())
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SimulationConfig":
        """Build a configuration from a (possibly partial) dict."""

        data = dict(payload)
        graph = GraphConfig(**data.pop("graph", {}))
        behavior = BehaviorConfig(**data.pop("behavior", {}))
        bss_raw = data.pop("bss", {})
        bss = _build_bss(bss_raw)
        estimator = _build_estimator(data.pop("estimator", {}))
        run = RunConfig(**data.pop("run", {}))
        scenarios = tuple(_build_scenario(s) for s in data.pop("scenarios", []))
        seeds = tuple(int(s) for s in data.pop("seeds", (0, 1)))
        return cls(
            graph=graph,
            behavior=behavior,
            bss=bss,
            estimator=estimator,
            run=run,
            scenarios=scenarios,
            seeds=seeds,
            **data,
        )


def _build_bss(raw: dict[str, Any]) -> BssConfig:
    data = dict(raw)
    for key in ("methods", "ranks", "random_states"):
        if key in data:
            data[key] = tuple(data[key])
    if "sobi_lag_sets" in data:
        data["sobi_lag_sets"] = tuple(tuple(int(x) for x in s) for s in data["sobi_lag_sets"])
    return BssConfig(**data)


def _build_estimator(raw: dict[str, Any]) -> EstimatorConfig:
    data = dict(raw)
    for key in ("estimators", "var_orders"):
        if key in data:
            data[key] = tuple(data[key])
    return EstimatorConfig(**data)


def _build_scenario(raw: dict[str, Any]) -> ScenarioConfig:
    data = dict(raw)
    dyn = DynamicsConfig(**data.pop("dynamics", {}))
    cal = CalciumConfig(**data.pop("calcium", {}))
    art_raw = data.pop("artifacts", {})
    if "types" in art_raw:
        art_raw = {**art_raw, "types": tuple(art_raw["types"])}
    art = ArtifactConfig(**art_raw)
    return ScenarioConfig(dynamics=dyn, calcium=cal, artifacts=art, **data)


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def load_config(path: str | Path) -> SimulationConfig:
    """Load a :class:`SimulationConfig` from a JSON file."""

    payload = json.loads(Path(path).read_text())
    return SimulationConfig.from_dict(payload)
