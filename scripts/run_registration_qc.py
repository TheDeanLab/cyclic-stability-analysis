# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cy-exm[workflow] @ git+https://github.com/TheDeanLab/Cy-ExM.git@c8b6a9f4a2f5f5fab3296fdc81190a7324da2a52",
#   "nd2==0.11.3",
#   "numpy>=1.26,<3",
#   "scipy>=1.12,<2",
#   "scikit-image>=0.22,<1",
#   "tifffile>=2025.1.10",
#   "PyYAML>=6,<7",
#   "pandas>=2.2,<3",
#   "matplotlib>=3.8,<4",
#   "seaborn>=0.13,<1",
#   "Jinja2>=3.1,<4",
#   "psutil>=5.9,<8",
# ]
# ///
"""Create target-blind physical registration QC without opening target images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cyclic_stability.config import load_analysis_config
from cyclic_stability.cyexm_artifacts import resolve_round_artifacts
from cyclic_stability.registration_qc import analyze_registration_qc, publish_qc_outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--correspondence", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate read-only artifacts and write target-blind QC for every loaded cell."""
    args = _parser().parse_args(argv)
    decisions = args.analysis_config.parent / "qc-decisions.csv"
    config = load_analysis_config(args.analysis_config, args.correspondence, decisions)
    validated = tuple(
        (cell, tuple(resolve_round_artifacts(round_spec) for round_spec in cell.rounds))
        for cell in config.cells
    )
    results = tuple(
        analyze_registration_qc(
            cell,
            artifacts,
            args.run_dir / "cells" / cell.cohort / cell.cell_id,
        )
        for cell, artifacts in validated
    )
    registration_rows = tuple(
        row for result in results for row in result.registration_rows
    )
    deformation_rows = tuple(
        row for result in results for row in result.deformation_rows
    )
    decision_rows = tuple(row for result in results for row in result.decision_requests)
    publish_qc_outputs(
        args.run_dir / "qc", registration_rows, deformation_rows, decision_rows
    )
    print(
        f"wrote {len(decision_rows)} blank target-blind moving-round decision requests"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
