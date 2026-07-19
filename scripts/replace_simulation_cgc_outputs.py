#!/usr/bin/env python
"""Replace c-GC/c-GC* graph outputs in existing simulation split roots.

Example:
    python scripts/replace_simulation_cgc_outputs.py \\
        --source-prefix core_1000_cgc_fast \\
        --target-prefix core_1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ica_denoising.simulation.graph_replacement import (  # noqa: E402
    replace_graph_estimator_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace selected graph-estimator outputs between split roots."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/simulation",
        help="Directory containing benchmark-version roots.",
    )
    parser.add_argument(
        "--source-prefix",
        default="core_1000_cgc_fast",
        help="Prefix for replacement split roots, e.g. core_1000_cgc_fast.",
    )
    parser.add_argument(
        "--target-prefix",
        default="core_1000",
        help="Prefix for existing split roots to update, e.g. core_1000.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["fastica", "infomax", "jade", "sobi"],
        help="Method split suffixes to process.",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=["cgc", "cgc_star"],
        help="Graph estimators to replace.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check source/target replicate layout without writing target files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for method in args.methods:
        source_root = output_dir / f"{args.source_prefix}_{method}"
        target_root = output_dir / f"{args.target_prefix}_{method}"
        report = replace_graph_estimator_outputs(
            source_root,
            target_root,
            estimators=tuple(args.estimators),
            dry_run=args.dry_run,
        )
        verb = "Would replace" if args.dry_run else "Replaced"
        print(
            f"{verb} {report.replaced}/{report.checked} replicate(s) "
            f"for {method}: {', '.join(report.estimators)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
