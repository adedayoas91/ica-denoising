# Deletion Candidates

This file marks redundant source surfaces and generated artifacts. It is a
cleanup register, not an instruction to delete all entries immediately.

## Safe Generated-File Cleanup

These are not source modules and can be deleted locally whenever outputs are
regenerated:

- `src/.DS_Store`
- `src/core/.DS_Store`
- `src/**/__pycache__/`
- stale `src/__pycache__/review_*.pyc`
- stale `src/core/__pycache__/IC_inspection*.pyc`
- stale `src/ic_quality/__pycache__/model_selection*.pyc`

## Source Already Removed

Keep these deleted unless a concrete import failure proves otherwise:

- `src/core/IC_inspection.py` - legacy manual IC-inspection surface; no current
  source/test/notebook import remains.
- `src/ic_quality/model_selection.py` - superseded by
  `ica_denoising/ic_quality/scoring.py` and
  `ica_denoising/ic_quality/validation.py`; no current source/test/notebook import
  remains.

## Source Marked For Later Deletion

These are still present because deleting them now would widen the diff or require
checking external notebook copies.

- legacy compatibility shims: `src/behavior_decoding.py`, `src/bss_notebook.py`,
  `src/causal_behavior_decoding.py`, `src/evaluation_diagnostics.py`,
  `src/evaluation_pipeline.py`, `src/evaluation_runner.py`, `src/core/ica_utils.py`,
  `src/core/visualization.py`, and `src/ic_quality/*.py`. Delete these after all
  external notebooks import through `ica_denoising.*`.
- `src/ica_denoising/core/ica_utils.py::compute` - legacy one-shot demo workflow. No repository
  caller remains, and the helper currently calls `eig_dec(traces)` without the
  required variance argument. Delete after any external notebook copies are
  checked.
- `src/ica_denoising/core/visualization.py::plott_ics`, `plot_FT_spectrals`,
  `plottings_spectrals`, `plottings_logSpectral`, `plottings_group_spectrals` -
  legacy notebook plotting helpers. `plottings_spectrals` is only retained by
  `ica_utils.compute`. Keep `plot_clusters`, which is used by the canonical V2a
  clustering notebook.
- `src/ica_denoising/ic_quality/reporting.py::write_html_review_report` -
  deprecated alias for `write_html_quality_report`. Remove after older notebooks
  stop importing it.

## Not Redundant

Do not delete these active `ica_denoising.ic_quality` modules in the current
codebase:

- `ica_denoising/ic_quality/features.py`
- `ica_denoising/ic_quality/scoring.py`
- `ica_denoising/ic_quality/validation.py`
- `ica_denoising/ic_quality/reporting.py`
- `ica_denoising/ic_quality/pipeline.py`

They are exercised by `tests/test_ic_quality.py` and support the optional IC-audit
notebook.
