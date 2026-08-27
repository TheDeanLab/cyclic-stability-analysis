"""Deterministic bounded execution for marker-cell tasks."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import multiprocessing
import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from cyclic_stability.models import ExecutionProfile
from cyclic_stability.task import (
    CellTask,
    TaskRecord,
    configure_native_threads,
    run_cell_task,
)

OUTPUT_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Deterministically ordered task records and their compact table."""

    tasks: tuple[TaskRecord, ...]
    compact_table: Path


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _worker(task: CellTask, run_dir: Path) -> TaskRecord:
    configure_native_threads(task.native_threads)
    return run_cell_task(task, run_dir)


def _write_compact(records: tuple[TaskRecord, ...], run_dir: Path) -> Path:
    path = run_dir / "compact-tasks.csv"
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("key", "status", "config_digest"), lineterminator="\n"
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "key": record.key,
                    "status": record.status,
                    "config_digest": record.config_digest,
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def run_tasks(
    tasks: tuple[CellTask, ...], profile: ExecutionProfile, run_dir: Path
) -> RunRecord:
    """Run tasks sequentially or in one bounded spawned process pool.

    SLURM submission is intentionally absent: its array index invokes exactly one
    task through the CLI.
    """
    if profile.mode == "slurm":
        raise ValueError(
            "SLURM mode is an index-only CLI operation; it does not submit jobs"
        )
    configured = tuple(
        replace(
            task,
            native_threads=profile.native_threads,
            scratch_root=profile.scratch_root,
        )
        for task in tasks
    )
    if profile.mode == "sequential":
        records = tuple(_worker(task, run_dir) for task in configured)
    else:
        with ProcessPoolExecutor(
            max_workers=profile.max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = {
                task.key: executor.submit(_worker, task, run_dir) for task in configured
            }
            records = tuple(futures[key].result() for key in sorted(futures))
    records = tuple(sorted(records, key=lambda record: record.key))
    return RunRecord(records, _write_compact(records, run_dir))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def initialize_run(
    run_dir: Path,
    *,
    analysis_config: Path,
    correspondence: Path,
    qc_decisions: Path,
    execution_profile: Path,
    config_digest: str,
    results_root: Path | None = None,
) -> None:
    """Snapshot immutable run configuration and atomically publish latest pointer."""
    copies = {
        "analysis.yaml": analysis_config,
        "correspondence.csv": correspondence,
        "qc-decisions.csv": qc_decisions,
        "execution-profile.yaml": execution_profile,
    }
    # Read each mutable source once: this is the exact byte sequence both hashed
    # and persisted, so a writer cannot create a digest/copy TOCTOU split.
    snapshots = {name: source.read_bytes() for name, source in copies.items()}
    initialize_run_from_bytes(
        run_dir,
        snapshots=snapshots,
        config_digest=config_digest,
        results_root=results_root,
    )


def initialize_run_from_bytes(
    run_dir: Path,
    *,
    snapshots: Mapping[str, bytes],
    config_digest: str,
    results_root: Path | None = None,
) -> None:
    """Initialize or verify a run from exact caller-owned configuration bytes.

    Parameters
    ----------
    run_dir : pathlib.Path
        Immutable run root. A generic artifact may already have created only its
        ``tasks`` scaffold while holding the corresponding task lock.
    snapshots : Mapping[str, bytes]
        Exact four frozen configuration payloads under their canonical run names.
    config_digest : str
        Scientific digest parsed from these exact bytes.
    results_root : pathlib.Path or None, default=None
        Optional root receiving the atomic latest-run pointer.

    Raises
    ------
    ValueError
        If names are incomplete, an existing uninitialized root contains anything
        other than the current task scaffold, or frozen state does not match.
    """
    expected_names = {
        "analysis.yaml",
        "correspondence.csv",
        "qc-decisions.csv",
        "execution-profile.yaml",
    }
    if set(snapshots) != expected_names or any(
        not isinstance(payload, bytes) for payload in snapshots.values()
    ):
        raise ValueError("run snapshots must contain the exact frozen input set")
    if not config_digest:
        raise ValueError("run config_digest must be nonempty")
    config_dir = run_dir / "config"
    digests = {name: _sha256_bytes(payload) for name, payload in snapshots.items()}
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir.parent / f".{run_dir.name}.initialize.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (run_dir / "run.json").exists():
            try:
                existing = json.loads(
                    (run_dir / "run.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    "existing run has no valid immutable run record"
                ) from error
            if (
                set(existing)
                != {
                    "config_digest",
                    "config_sha256",
                    "cyexm_commit",
                    "initialized_at",
                    "output_schema_version",
                }
                or existing.get("config_digest") != config_digest
                or existing.get("config_sha256") != digests
                or existing.get("cyexm_commit")
                != "c8b6a9f4a2f5f5fab3296fdc81190a7324da2a52"
                or existing.get("output_schema_version") != OUTPUT_SCHEMA_VERSION
                or any(
                    not (config_dir / name).is_file()
                    or _sha256_bytes((config_dir / name).read_bytes()) != digest
                    for name, digest in digests.items()
                )
            ):
                raise ValueError(
                    "existing frozen run identity does not match immutable inputs"
                )
            return
        if run_dir.exists():
            entries = {path.name for path in run_dir.iterdir()}
            if entries - {"tasks"}:
                raise ValueError(
                    "existing uninitialized run contains non-task artifacts"
                )
            tasks = run_dir / "tasks"
            if tasks.exists() and (tasks.is_symlink() or not tasks.is_dir()):
                raise ValueError("existing run task scaffold is invalid")
        else:
            run_dir.mkdir()
        for name, payload in snapshots.items():
            _atomic_bytes(config_dir / name, payload)
        _atomic_bytes(
            run_dir / "run.json",
            (
                json.dumps(
                    {
                        "config_digest": config_digest,
                        "config_sha256": digests,
                        "cyexm_commit": "c8b6a9f4a2f5f5fab3296fdc81190a7324da2a52",
                        "initialized_at": datetime.now(UTC).isoformat(),
                        "output_schema_version": OUTPUT_SCHEMA_VERSION,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode(),
        )
    if results_root is not None:
        _atomic_bytes(
            results_root / "latest-run.txt", (str(run_dir.resolve()) + "\n").encode()
        )


__all__ = [
    "RunRecord",
    "initialize_run",
    "initialize_run_from_bytes",
    "run_tasks",
]
