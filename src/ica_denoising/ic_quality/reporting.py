from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

import pandas as pd


def save_ic_quality_outputs(
    output_dir: Path,
    feature_table: pd.DataFrame,
    scored_table: pd.DataFrame,
    *,
    validation_table: pd.DataFrame | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Write CSV/JSON artifacts for an IC-quality run."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": output_dir / "ic_quality_table.csv",
        "recommendations": output_dir / "ic_rejection_recommendations.csv",
        "metadata": output_dir / "ic_quality_metadata.json",
        "html_report": output_dir / "ic_quality_report.html",
    }
    feature_table.to_csv(paths["features"], index=False)
    scored_table.to_csv(paths["recommendations"], index=False)
    if validation_table is not None:
        paths["validation"] = output_dir / "ic_leave_one_out_validation.csv"
        validation_table.to_csv(paths["validation"], index=False)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        **dict(metadata or {}),
    }
    paths["metadata"].write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    write_html_quality_report(
        paths["html_report"],
        scored_table=scored_table,
        validation_table=validation_table,
        metadata=payload,
    )
    return paths


def write_html_quality_report(
    path: Path,
    *,
    scored_table: pd.DataFrame,
    validation_table: pd.DataFrame | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write a compact auditable HTML report for human IC-quality checks."""

    path = Path(path)
    summary = _recommendation_summary(scored_table)
    sections = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>IC Quality Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;line-height:1.4}"
        "table{border-collapse:collapse;margin:16px 0;width:100%}"
        "th,td{border:1px solid #ddd;padding:6px;font-size:12px}"
        "th{background:#f5f5f5;text-align:left}"
        ".drop{background:#ffecec}.review{background:#fff8dc}.keep{background:#eef8ee}"
        "code{background:#f5f5f5;padding:2px 4px}</style></head><body>",
        "<h1>IC Quality Report</h1>",
        "<p>Use this report to approve or override recommended IC rejection decisions.</p>",
        "<h2>Run metadata</h2>",
        f"<pre>{json.dumps(dict(metadata or {}), indent=2, default=str)}</pre>",
        "<h2>Recommendation summary</h2>",
        summary.to_html(index=False, escape=True),
        "<h2>Ranked IC recommendations</h2>",
        _styled_table(scored_table),
    ]
    if validation_table is not None:
        sections.extend(
            [
                "<h2>Leave-one-IC-out validation</h2>",
                validation_table.to_html(index=False),
            ]
        )
    sections.extend(
        [
            "<h2>Reviewer notes</h2>",
            "<p>Approved rejected ICs:</p>",
            "<pre>APPROVED_REJECT_COMPONENTS = []</pre>",
            "</body></html>",
        ]
    )
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def write_html_review_report(
    path: Path,
    *,
    scored_table: pd.DataFrame,
    validation_table: pd.DataFrame | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Deprecated compatibility alias for ``write_html_quality_report``."""

    return write_html_quality_report(
        path,
        scored_table=scored_table,
        validation_table=validation_table,
        metadata=metadata,
    )


def _recommendation_summary(scored_table: pd.DataFrame) -> pd.DataFrame:
    if "recommendation" not in scored_table.columns:
        return pd.DataFrame({"recommendation": [], "count": []})
    return (
        scored_table["recommendation"]
        .value_counts()
        .rename_axis("recommendation")
        .reset_index(name="count")
        .sort_values("recommendation")
    )


def _styled_table(table: pd.DataFrame) -> str:
    if "recommendation" not in table.columns:
        return table.to_html(index=False, escape=True)
    html = table.to_html(index=False, escape=True)
    rows = html.split("<tbody>")
    if len(rows) != 2:
        return html
    head, body = rows
    body_parts = body.split("<tr>")
    rebuilt = [body_parts[0]]
    for (_, row), fragment in zip(table.iterrows(), body_parts[1:]):
        class_name = str(row.get("recommendation", "")).lower()
        rebuilt.append(f"<tr class='{class_name}'>{fragment}")
    return head + "<tbody>" + "".join(rebuilt)
