# IC Quality Package

`ic_quality` is optional but not dead. It provides a human-auditable IC quality
workflow around saved decompositions:

- `features.py` computes temporal, spectral, loading, spatial, and behavior-link
  features per component.
- `scoring.py` converts those features into keep/review/drop recommendations.
- `validation.py` reconstructs traces after component removal and computes
  leave-one-IC-out checks.
- `reporting.py` writes CSV, JSON, and HTML quality artifacts.
- `pipeline.py` orchestrates the end-to-end IC-quality run.

Deletion status:

- `model_selection.py` has already been removed and should stay deleted.
- `write_html_review_report` is a deprecated compatibility alias. New code should
  call `write_html_quality_report`.
