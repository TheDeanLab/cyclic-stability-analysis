# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cy-exm[workflow] @ git+https://github.com/TheDeanLab/Cy-ExM.git@f6e0de36cd7f6b5e85f093884c2cf224225f4ec2",
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
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, TypeVar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

T = TypeVar("T")
R = TypeVar("R")


def _ordered_process_map(
    worker: Callable[[T], R], items: tuple[T, ...], *, max_workers: int
) -> tuple[R, ...]:
    """Apply one bounded process per item while retaining deterministic order."""
    if isinstance(max_workers, bool) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    if max_workers == 1:
        return tuple(map(worker, items))
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=get_context("spawn"),
    ) as executor:
        return tuple(executor.map(worker, items, chunksize=1))


def _run_qc_job(job: tuple[Any, tuple[Any, ...], Path]) -> Any:
    """Analyze one independently publishable marker-cell QC unit."""
    from cyclic_stability.registration_qc import analyze_registration_qc

    cell, artifacts, output_dir = job
    return analyze_registration_qc(cell, artifacts, output_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--correspondence", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--cohort", action="append", default=[], help="limit QC to these cohorts"
    )
    parser.add_argument("--cell", help="limit QC to one configured cell")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate read-only artifacts and write target-blind QC for every loaded cell."""
    args = _parser().parse_args(argv)
    from cyexm.runtime import NativeThreadConfig, configure_native_threads

    configure_native_threads(NativeThreadConfig(itk_threads=1, blas_threads=1))

    from cyclic_stability.config import load_analysis_config
    from cyclic_stability.cyexm_artifacts import resolve_round_artifacts
    from cyclic_stability.registration_qc import publish_qc_outputs

    decisions = args.analysis_config.parent / "qc-decisions.csv"
    config = load_analysis_config(args.analysis_config, args.correspondence, decisions)
    selected_cells = tuple(
        cell
        for cell in config.cells
        if (not args.cohort or cell.cohort in args.cohort)
        and (args.cell is None or cell.cell_id == args.cell)
    )
    if not selected_cells:
        print("unavailable: no configured cells match the selection", file=sys.stderr)
        return 1
    validated = tuple(
        (cell, tuple(resolve_round_artifacts(round_spec) for round_spec in cell.rounds))
        for cell in selected_cells
    )
    jobs = tuple(
        (
            cell,
            artifacts,
            args.run_dir / "cells" / cell.cohort / cell.cell_id,
        )
        for cell, artifacts in validated
    )
    print(
        f"progress: queued registration QC for {len(jobs)} cells with "
        f"{args.max_workers} worker(s)",
        file=sys.stderr,
        flush=True,
    )
    results = _ordered_process_map(_run_qc_job, jobs, max_workers=args.max_workers)
    print(
        "progress: registration QC cell analyses completed", file=sys.stderr, flush=True
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
