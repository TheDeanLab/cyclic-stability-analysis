"""Restart-safe per-cell task behavior."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cyclic_stability.task import run_cell_task
from synthetic import cell_task, different_identity


def test_completed_task_is_reused_only_for_matching_inputs(tmp_path: Path) -> None:
    task = cell_task(tmp_path)
    first = run_cell_task(task, tmp_path)
    second = run_cell_task(task, tmp_path)
    assert first.attempt_id == second.attempt_id
    assert second.reused is True
    changed = replace(task, config_digest="changed")
    third = run_cell_task(changed, tmp_path)
    assert third.attempt_id != first.attempt_id
    assert third.reused is False


def test_changed_input_identity_invalidates_task_reuse(tmp_path: Path) -> None:
    task = cell_task(tmp_path)
    first = run_cell_task(task, tmp_path)
    changed = replace(
        task, input_identities=(*task.input_identities[:-1], different_identity())
    )
    second = run_cell_task(changed, tmp_path)
    assert second.attempt_id != first.attempt_id
    assert second.reused is False


def test_partial_attempt_never_authorizes_reuse(tmp_path: Path) -> None:
    task = cell_task(tmp_path)
    partial = tmp_path / "tasks" / task.key / "attempts" / "interrupted"
    partial.mkdir(parents=True)
    (partial / "summary.csv").write_text("partial", encoding="utf-8")
    result = run_cell_task(task, tmp_path)
    assert result.status == "complete"
    assert result.attempt_id != "interrupted"


def test_output_mutation_invalidates_completed_pointer(tmp_path: Path) -> None:
    task = cell_task(tmp_path)
    first = run_cell_task(task, tmp_path)
    output = Path(first.output_identities[0].path)
    output.write_text("changed", encoding="utf-8")
    second = run_cell_task(task, tmp_path)
    assert second.attempt_id != first.attempt_id


def test_record_keeps_the_exact_software_version_identity(tmp_path: Path) -> None:
    task = cell_task(tmp_path)
    record = run_cell_task(task, tmp_path)
    assert record.software_versions == (task.software_identity,)
