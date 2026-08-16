# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cy-exm[workflow] @ git+https://github.com/TheDeanLab/Cy-ExM.git@c8b6a9f4a2f5f5fab3296fdc81190a7324da2a52",
#   "nd2==0.11.3",
#   "PyYAML>=6,<7",
#   "psutil>=5.9,<8",
#   "numpy>=1.26,<3",
#   "scipy>=1.12,<2",
#   "scikit-image>=0.22,<1",
#   "tifffile>=2025.1.10",
#   "pandas>=2.2,<3",
#   "matplotlib>=3.8,<4",
#   "seaborn>=0.13,<1",
#   "Jinja2>=3.1,<4",
# ]
# ///
"""Run restart-safe, target-blind-gated cyclic marker-cell analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--correspondence", type=Path, required=True)
    parser.add_argument("--qc-decisions", type=Path, required=True)
    parser.add_argument("--execution-profile", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--run-dir", type=Path)
    destination.add_argument("--results-root", type=Path)
    parser.add_argument("--cohort", action="append", default=[])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cell")
    selection.add_argument("--pilot-first-accepted", action="store_true")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--force-task", action="store_true")
    return parser


def _pilot_cell(config, cohorts: tuple[str, ...]):
    if len(cohorts) != 1:
        raise ValueError("--pilot-first-accepted requires exactly one --cohort")
    decisions = {
        (value.cohort, value.cell_id, value.acquisition_round): value.decision
        for value in config.qc_decisions
    }
    eligible = []
    for cell in config.cells:
        if cohorts and cell.cohort not in cohorts:
            continue
        moving = [item for item in cell.rounds if item.acquisition_round != 2]
        if moving and all(
            decisions.get((item.cohort, item.cell_id, item.acquisition_round))
            == "accept"
            for item in moving
        ):
            eligible.append(cell)
    if not eligible:
        raise ValueError("no cell has accept decisions for every moving round")
    return min(eligible, key=lambda value: (value.cohort, value.cell_id))


def _write_selection(run_dir: Path, payload: dict[str, object]) -> None:
    """Durably publish the optional selection record without partial JSON."""
    target = run_dir / "selection.json"
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory = os.open(run_dir, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _software_identity() -> str:
    """Return pinned interpreter, analysis-tree, Cy-ExM, and runtime versions."""
    from importlib.metadata import version

    dependencies = (
        "nd2",
        "numpy",
        "scipy",
        "scikit-image",
        "tifffile",
        "PyYAML",
        "psutil",
    )
    versions = {name: version(name) for name in dependencies}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    tree = hashlib.sha256()
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix().encode()
        tree.update(len(relative).to_bytes(8, "big"))
        tree.update(relative)
        payload = path.read_bytes()
        tree.update(len(payload).to_bytes(8, "big"))
        tree.update(payload)
    return json.dumps(
        {
            "analysis_commit": commit,
            "analysis_tree_sha256": tree.hexdigest(),
            "cyexm_commit": "c8b6a9f4a2f5f5fab3296fdc81190a7324da2a52",
            "python": platform.python_version(),
            "dependencies": versions,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def main(argv: list[str] | None = None) -> int:
    """Initialize one immutable run and execute its selected deterministic task set."""
    args = _parser().parse_args(argv)
    # These imports do not import a scientific backend.  Native threads are set
    # immediately after profile validation and before any task analyzer is imported.
    from cyclic_stability.config import load_analysis_config, load_execution_profile
    from cyclic_stability.runner import initialize_run, run_tasks
    from cyclic_stability.task import (
        build_tasks,
        configure_native_threads,
        run_cell_task,
    )
    from cyexm.registration.artifacts import FileIdentity

    config = load_analysis_config(
        args.analysis_config, args.correspondence, args.qc_decisions
    )
    profile = load_execution_profile(args.execution_profile)
    configure_native_threads(profile.native_threads)
    cohorts = tuple(args.cohort)
    known_cohorts = {"golgi", "lamin", "lamp1"}
    if any(cohort not in known_cohorts for cohort in cohorts):
        raise ValueError("--cohort must name a configured cohort")
    if profile.mode == "slurm" and args.task_index is None:
        raise ValueError("SLURM execution requires --task-index")
    selected = None
    if args.cell is not None:
        selected = next(
            (
                cell
                for cell in config.cells
                if cell.cell_id == args.cell and (not cohorts or cell.cohort in cohorts)
            ),
            None,
        )
    if args.cell is not None and selected is None:
        raise KeyError(
            "--cell does not match a configured cell in the requested cohort"
        )
    if args.pilot_first_accepted:
        selected = _pilot_cell(config, cohorts)
    tasks = build_tasks(
        config,
        cohorts,
        cells=(selected,) if selected is not None else None,
    )
    run_input_identities = tuple(
        FileIdentity.from_path(path)
        for path in (
            args.analysis_config,
            args.correspondence,
            args.qc_decisions,
            args.execution_profile,
        )
    )
    software_identity = _software_identity()
    tasks = tuple(
        replace(
            task,
            input_identities=(*task.input_identities, *run_input_identities),
            software_identity=software_identity,
        )
        for task in tasks
    )
    if args.task_index is not None:
        if profile.mode != "slurm":
            raise ValueError("--task-index requires a slurm execution profile")
        if args.cell is not None or args.pilot_first_accepted:
            raise ValueError("--task-index cannot be combined with cell selection")
        if args.task_index < 0 or args.task_index >= len(tasks):
            raise IndexError("--task-index is outside deterministic task inventory")
        tasks = (tasks[args.task_index],)
    if args.force_task:
        tasks = tuple(replace(task, force=True) for task in tasks)
    if args.run_dir is not None:
        run_dir = args.run_dir
        results_root = None
    else:
        assert args.results_root is not None
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.results_root / f"{timestamp}-{config.config_digest[:12]}"
        results_root = args.results_root
    initialize_run(
        run_dir,
        analysis_config=args.analysis_config,
        correspondence=args.correspondence,
        qc_decisions=args.qc_decisions,
        execution_profile=args.execution_profile,
        config_digest=config.config_digest,
        results_root=results_root,
    )
    if selected is not None:
        _write_selection(
            run_dir,
            {
                "mode": "pilot-first-accepted",
                "cohort": selected.cohort,
                "cell_id": selected.cell_id,
            },
        )
    if not tasks:
        _write_selection(
            run_dir, {"status": "unavailable", "reason": "no schedulable tasks"}
        )
        print("unavailable: no schedulable tasks")
        return 0
    if profile.mode == "slurm":
        record = run_cell_task(
            replace(
                tasks[0],
                native_threads=profile.native_threads,
                scratch_root=profile.scratch_root,
            ),
            run_dir,
        )
        print(f"{record.status}: {record.key}")
        return 0 if record.status == "complete" else 1
    record = run_tasks(tasks, profile, run_dir)
    failures = sum(item.status != "complete" for item in record.tasks)
    print(
        f"ran {len(record.tasks)} tasks; failed={failures}; compact={record.compact_table}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
