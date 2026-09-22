"""Bounded, deterministic task runner behavior."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

from cyclic_stability.runner import initialize_run, initialize_run_from_bytes, run_tasks
from synthetic import (
    ANALYSIS_YAML,
    normalized_table_bytes,
    profile,
    synthetic_tasks,
    write_synthetic_correspondence,
)


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


def test_runner_reports_task_start_and_completion(tmp_path: Path, capsys) -> None:
    """Long-running runs expose task-level progress on stderr."""
    run_tasks(
        synthetic_tasks(tmp_path),
        profile("sequential"),
        tmp_path / "run",
        progress_interval_s=0,
    )
    output = capsys.readouterr().err
    assert "progress: starting" in output
    assert "progress: completed" in output
    assert "1/" in output


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


def test_production_project_declares_the_direct_nd2_runtime_pin() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert "nd2==0.11.3" in project["project"]["dependencies"]
    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
    nd2 = [package for package in lock["package"] if package["name"] == "nd2"]
    assert len(nd2) == 1
    assert nd2[0]["version"] == "0.11.3"


def test_initialization_reopens_only_the_same_immutable_run(tmp_path: Path) -> None:
    inputs = []
    for name in ("analysis.yaml", "correspondence.csv", "qc.csv", "execution.yaml"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        inputs.append(path)
    kwargs = {
        "analysis_config": inputs[0],
        "correspondence": inputs[1],
        "qc_decisions": inputs[2],
        "execution_profile": inputs[3],
        "config_digest": "same-digest",
    }
    run_dir = tmp_path / "run"
    initialize_run(run_dir, **kwargs)
    initialize_run(run_dir, **kwargs)


def test_reopen_allows_changed_or_corrupted_frozen_snapshots(tmp_path: Path) -> None:
    sources = []
    for name in ("analysis.yaml", "correspondence.csv", "qc.csv", "execution.yaml"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        sources.append(path)
    kwargs = {
        "analysis_config": sources[0],
        "correspondence": sources[1],
        "qc_decisions": sources[2],
        "execution_profile": sources[3],
        "config_digest": "same-digest",
    }
    run_dir = tmp_path / "run"
    initialize_run(run_dir, **kwargs)
    (run_dir / "config" / "analysis.yaml").write_text("corrupt", encoding="utf-8")
    initialize_run(run_dir, **kwargs)


def test_snapshot_run_initialization_accepts_current_artifact_scaffold(
    tmp_path: Path,
) -> None:
    """Under-lock artifact admission creates its task tree before frozen run.json."""
    run_dir = tmp_path / "run"
    attempt = run_dir / "tasks/nuclear-inventory/attempts" / ("a" * 32)
    attempt.mkdir(parents=True)
    (run_dir / "tasks/nuclear-inventory/.task.lock").touch()
    (run_dir / "tasks/nuclear-inventory/terminal.json").write_text(
        json.dumps({"status": "running", "attempt_id": "a" * 32}) + "\n",
        encoding="utf-8",
    )
    payloads = {
        "analysis.yaml": b"analysis\n",
        "correspondence.csv": b"correspondence\n",
        "qc-decisions.csv": b"decisions\n",
        "execution-profile.yaml": b"execution\n",
    }

    initialize_run_from_bytes(
        run_dir,
        snapshots=payloads,
        config_digest="snapshot-digest",
    )

    assert {
        path.name: path.read_bytes() for path in (run_dir / "config").iterdir()
    } == payloads
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["config_digest"] == "snapshot-digest"
    assert record["output_schema_version"] == 2


def test_empty_lamp_inventory_is_an_unavailable_success(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.yaml"
    correspondence = tmp_path / "correspondence.csv"
    decisions = tmp_path / "qc.csv"
    execution = tmp_path / "execution.yaml"
    analysis.write_text(ANALYSIS_YAML, encoding="utf-8")
    write_synthetic_correspondence(correspondence)
    decisions.write_text(
        "cohort,cell_id,acquisition_round,decision,reasons,reviewer,reviewed_at\n",
        encoding="utf-8",
    )
    execution.write_text(
        "mode: sequential\nmax_workers: 1\nnative_threads: 1\nscratch_root: /tmp/scratch\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_analysis.py",
            "--analysis-config",
            str(analysis),
            "--correspondence",
            str(correspondence),
            "--qc-decisions",
            str(decisions),
            "--execution-profile",
            str(execution),
            "--run-dir",
            str(tmp_path / "run"),
            "--cohort",
            "lamp1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "unavailable" in result.stdout
