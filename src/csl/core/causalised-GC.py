"""Compatibility shim for the hyphenated c-GC module path.

The real implementation lives in the importable :mod:`causalised_gc` sibling.
This file exists only so that path-based loaders (``importlib`` against
``causalised-GC.py``) keep yielding ``GcStar`` and friends. It loads the sibling
module by absolute path and re-exports its public names, avoiding any reliance
on relative imports that break under path loading.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_CANONICAL = "csl.core.causalised_gc"
_existing = sys.modules.get(_CANONICAL)
if _existing is not None and hasattr(_existing, "GcStar"):
    # The canonical module is already imported normally; reuse it.
    _impl = _existing
else:
    try:
        _impl = importlib.import_module(_CANONICAL)
        if not hasattr(_impl, "GcStar"):
            raise ImportError("canonical module not fully initialised")
    except Exception:  # pragma: no cover - exotic path-only loads
        # Path-load the implementation under a PRIVATE name and register it in
        # sys.modules so dataclasses defined in it resolve their ``__module__``
        # (otherwise dataclass field introspection raises AttributeError).
        _IMPL_PATH = Path(__file__).resolve().with_name("causalised_gc.py")
        _PRIVATE = "csl.core._causalised_gc_impl"
        _spec = importlib.util.spec_from_file_location(_PRIVATE, _IMPL_PATH)
        if _spec is None or _spec.loader is None:
            raise ImportError(f"Could not load implementation from {_IMPL_PATH}.")
        _impl = importlib.util.module_from_spec(_spec)
        sys.modules[_PRIVATE] = _impl
        _spec.loader.exec_module(_impl)

GcStar = _impl.GcStar
CGCResult = _impl.CGCResult
regression_residual = _impl.regression_residual
benjamini_hochberg = _impl.benjamini_hochberg
fit_cgc = _impl.fit_cgc
_perm_test_numba = _impl._perm_test_numba

__all__ = [
    "GcStar",
    "CGCResult",
    "regression_residual",
    "benjamini_hochberg",
    "fit_cgc",
]
