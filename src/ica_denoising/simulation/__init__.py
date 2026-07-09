"""Ground-truth simulation benchmark for BSS denoising validation (Section 5.1)."""

from __future__ import annotations

from .config import SimulationConfig, load_config
from .dataset import SimulatedDataset, build_dataset, default_scenarios
from .runner import run_benchmark, run_replicate

__all__ = [
    "SimulationConfig",
    "load_config",
    "SimulatedDataset",
    "build_dataset",
    "default_scenarios",
    "run_benchmark",
    "run_replicate",
]
