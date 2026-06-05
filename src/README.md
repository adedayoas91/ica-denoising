# Source Layout

Canonical imports now go through the `ica_denoising` package namespace. The
legacy flat modules in `src/`, `src/core/`, and `src/ic_quality/` are temporary
compatibility shims for external notebooks that have not migrated yet.

## Active Areas

- `ica_denoising/bss_notebook.py` - dataset discovery, notebook import setup,
  BSS artifact paths, decomposition metadata, and cleaned-trace save/load
  helpers.
- `ica_denoising/core/` - numerical BSS primitives and plotting helpers used by
  notebooks.
- `ica_denoising/behavior_decoding.py` - chained full-recording behavior
  decoding utilities.
- `ica_denoising/causal_behavior_decoding.py` - causal-state behavior decoding
  utilities.
- `ica_denoising/evaluation_pipeline.py`, `ica_denoising/evaluation_runner.py`,
  `ica_denoising/evaluation_diagnostics.py` - leakage-safe fold-local evaluation
  and aggregate reporting.
- `ica_denoising/ic_quality/` - optional IC-audit package. Its active modules are
  covered by `tests/test_ic_quality.py`; do not delete them as dead code.

## Restructure Direction

New source should follow these boundaries:

- Notebook and artifact IO stays in `ica_denoising/bss_notebook.py`; import it as
  `ica_denoising.bss_notebook`.
- Strict, manuscript-facing metrics belong in the `ica_denoising.evaluation_*`
  modules.
- Full-recording descriptive behavior analysis belongs in `behavior_decoding.py`
  and `causal_behavior_decoding.py`.
- Low-level numerical routines stay in `ica_denoising/core/`; notebook-only
  plotting should not leak into strict evaluation modules.
- Optional human IC audit logic stays in `ica_denoising/ic_quality/`.

Remove the legacy flat shims only after external notebooks are checked.

See `DELETION_CANDIDATES.md` before deleting source. It separates safe generated
files from source symbols that need migration before removal.
