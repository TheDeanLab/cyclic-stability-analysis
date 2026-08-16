"""Bounded, deterministic task runner behavior."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cyclic_stability.runner import run_tasks
from synthetic import normalized_table_bytes, profile, synthetic_tasks


def test_sequential_and_process_modes_produce_identical_compact_tables(
    tmp_path: Path,
) -> None:
    tasks = synthetic_tasks(tmp_path)
    sequential = run_tasks(tasks, profile("sequential"), tmp_path / "sequential")
    processes = run_tasks(
        tasks, profile("process", max_workers=2), tmp_path / "process"
    )
    assert normalized_table_bytes(sequential) == normalized_table_bytes(processes)


def test_runner_records_operational_resource_fields(tmp_path: Path) -> None:
    record = run_tasks(synthetic_tasks(tmp_path), profile("sequential"), tmp_path)
    assert all(item.elapsed_seconds >= 0.0 for item in record.tasks)
    assert all(item.peak_rss_bytes > 0 for item in record.tasks)
    assert all(item.process_read_bytes >= 0 for item in record.tasks)


def test_cli_rejects_run_directory_and_results_root_together(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_analysis.py",
            "--analysis-config",
            str(tmp_path / "analysis.yaml"),
            "--correspondence",
            str(tmp_path / "correspondence.csv"),
            "--qc-decisions",
            str(tmp_path / "qc.csv"),
            "--execution-profile",
            str(tmp_path / "execution.yaml"),
            "--run-dir",
            str(tmp_path / "run"),
            "--results-root",
            str(tmp_path / "results"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr
