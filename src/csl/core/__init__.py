"""Core estimator implementations."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_gcstar() -> type:
    """Load ``GcStar`` from the causalised module file."""

    module_path = Path(__file__).resolve().with_name("causalised-GC.py")
    spec = importlib.util.spec_from_file_location(
        "csl.core.causalised_gc",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GcStar from {module_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GcStar


GcStar = _load_gcstar()

__all__ = ["GcStar"]
