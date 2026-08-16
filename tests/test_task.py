"""Restart-safe per-cell task behavior."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from cyclic_stability.task import _record_from_payload, _record_payload, run_cell_task
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


def test_symlinked_attempts_root_is_rejected_without_writing_outside(
    tmp_path: Path,
) -> None:
    task = cell_task(tmp_path)
    task_dir = tmp_path / "tasks" / task.key
    task_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (task_dir / "attempts").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_cell_task(task, tmp_path)
    assert not list(outside.iterdir())


def test_shrunk_record_cannot_hide_a_deleted_required_output(tmp_path: Path) -> None:
    task = cell_task(tmp_path)
    first = run_cell_task(task, tmp_path)
    record_path = first.attempt_dir / "record.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    deleted = Path(payload["output_identities"].pop()["path"])
    deleted.unlink()
    record_path.write_text(json.dumps(payload), encoding="utf-8")
    second = run_cell_task(task, tmp_path)
    assert second.attempt_id != first.attempt_id
    assert second.reused is False


def test_extra_nested_output_invalidates_reuse(tmp_path: Path) -> None:
    task = cell_task(tmp_path)
    first = run_cell_task(task, tmp_path)
    extra = first.attempt_dir / "diagnostics" / "injected.json"
    extra.write_text("{}\n", encoding="utf-8")
    second = run_cell_task(task, tmp_path)
    assert second.attempt_id != first.attempt_id


def test_record_decoder_rejects_coerced_or_nonfinite_scalars(tmp_path: Path) -> None:
    record = run_cell_task(cell_task(tmp_path), tmp_path)
    payload = _record_payload(record)
    payload["elapsed_seconds"] = "0"
    with pytest.raises(ValueError, match="numeric"):
        _record_from_payload(payload)
    payload = _record_payload(record)
    payload["peak_rss_bytes"] = float("nan")
    with pytest.raises(ValueError, match="numeric"):
        _record_from_payload(payload)
