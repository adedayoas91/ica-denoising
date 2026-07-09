"""Dataset orchestration: graph -> dynamics -> calcium -> artifacts -> behavior.

A :class:`SimulatedDataset` is fully reproducible from ``(config, scenario,
seed)`` and serializes to a portable ``dataset.npz`` (Section 5.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zlib

import numpy as np

from .artifacts import inject_artifacts
from .behavior import simulate_behavior
from .calcium import simulate_calcium
from .config import (
    ArtifactConfig,
    CalciumConfig,
    DynamicsConfig,
    ScenarioConfig,
    SimulationConfig,
)
from .dynamics import simulate_dynamics
from .graphs import build_graph

__all__ = ["SimulatedDataset", "build_dataset", "default_scenarios"]


@dataclass(frozen=True)
class SimulatedDataset:
    """A fully generated simulation replicate."""

    scenario_id: str
    seed: int
    truth_adjacency: np.ndarray  # [source, target]
    truth_coefficients: np.ndarray  # [lag, target, source]
    node_groups: np.ndarray
    positions: np.ndarray
    clean_fluorescence: np.ndarray  # oracle (n, T)
    corrupted: np.ndarray  # raw observed (n, T)
    artifact: np.ndarray  # injected artifact (n, T)
    vigor: np.ndarray
    angle: np.ndarray
    bout: np.ndarray
    behavior_parents: np.ndarray
    behavior_lag: int
    achieved_asr: float

    def save(self, path: str | Path) -> Path:
        """Save the dataset to a compressed ``.npz`` file."""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            scenario_id=self.scenario_id,
            seed=self.seed,
            truth_adjacency=self.truth_adjacency,
            truth_coefficients=self.truth_coefficients,
            node_groups=self.node_groups,
            positions=self.positions,
            clean_fluorescence=self.clean_fluorescence,
            corrupted=self.corrupted,
            artifact=self.artifact,
            vigor=self.vigor,
            angle=self.angle,
            bout=self.bout,
            behavior_parents=self.behavior_parents,
            behavior_lag=self.behavior_lag,
            achieved_asr=self.achieved_asr,
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "SimulatedDataset":
        """Load a dataset from a ``.npz`` file."""

        data = np.load(path, allow_pickle=False)
        return cls(
            scenario_id=str(data["scenario_id"]),
            seed=int(data["seed"]),
            truth_adjacency=data["truth_adjacency"],
            truth_coefficients=data["truth_coefficients"],
            node_groups=data["node_groups"],
            positions=data["positions"],
            clean_fluorescence=data["clean_fluorescence"],
            corrupted=data["corrupted"],
            artifact=data["artifact"],
            vigor=data["vigor"],
            angle=data["angle"],
            bout=data["bout"],
            behavior_parents=data["behavior_parents"],
            behavior_lag=int(data["behavior_lag"]),
            achieved_asr=float(data["achieved_asr"]),
        )


def build_dataset(
    cfg: SimulationConfig,
    scenario: ScenarioConfig,
    seed: int,
) -> SimulatedDataset:
    """Build a reproducible :class:`SimulatedDataset`.

    The full chain is driven by a single seeded generator so that the same
    ``(cfg, scenario, seed)`` reproduces identical arrays.
    """

    scenario_code = zlib.crc32(scenario.scenario_id.encode("utf-8"))
    rng = np.random.default_rng((seed, scenario_code))
    graph = build_graph(cfg.graph, rng)
    activity = simulate_dynamics(graph, scenario.dynamics, cfg.n_frames, rng)
    clean = simulate_calcium(
        activity.observed, scenario.calcium, cfg.sample_rate_hz, rng
    )
    artifacts = inject_artifacts(clean.fluorescence, scenario.artifacts, rng)
    behavior = simulate_behavior(activity.states, cfg.behavior, rng)

    return SimulatedDataset(
        scenario_id=scenario.scenario_id,
        seed=seed,
        truth_adjacency=graph.adjacency,
        truth_coefficients=graph.coefficients,
        node_groups=graph.node_groups,
        positions=graph.positions,
        clean_fluorescence=clean.fluorescence,
        corrupted=artifacts.corrupted,
        artifact=artifacts.artifact,
        vigor=behavior.vigor,
        angle=behavior.angle,
        bout=behavior.bout,
        behavior_parents=behavior.parents,
        behavior_lag=behavior.behavior_lag,
        achieved_asr=artifacts.achieved_asr,
    )


def default_scenarios() -> tuple[ScenarioConfig, ...]:
    """Return the canonical S0-S7 scenario presets (Section 5.1)."""

    linear = dict(model="linear_var")
    rate = dict(model="rate_spike")
    no_cal = CalciumConfig(enabled=False)
    cal = CalciumConfig(enabled=True)
    return (
        ScenarioConfig(
            "S0", DynamicsConfig(**linear), no_cal, ArtifactConfig(types=())
        ),
        ScenarioConfig("S1", DynamicsConfig(**linear), cal, ArtifactConfig(types=())),
        ScenarioConfig(
            "S2", DynamicsConfig(**linear), cal, ArtifactConfig(types=("drift", "neuropil"))
        ),
        ScenarioConfig(
            "S3", DynamicsConfig(**linear), cal, ArtifactConfig(types=("motion",))
        ),
        ScenarioConfig(
            "S4", DynamicsConfig(**linear), cal, ArtifactConfig(types=("neural_band",))
        ),
        ScenarioConfig(
            "S5",
            DynamicsConfig(**rate),
            cal,
            ArtifactConfig(types=("drift", "motion", "neuropil")),
        ),
        ScenarioConfig(
            "S6",
            DynamicsConfig(model="linear_var", hidden_confounding=True),
            cal,
            ArtifactConfig(types=("drift",)),
        ),
        ScenarioConfig(
            "S7",
            DynamicsConfig(model="linear_var", behavior_feedback=True),
            cal,
            ArtifactConfig(types=("drift",)),
        ),
    )
