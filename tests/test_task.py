"""Restart-safe per-cell task behavior."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from cyclic_stability.models import MarkerResult, QcDecision
from cyclic_stability.nuclear import NuclearComponentQc
from cyclic_stability.signal import SignalRow
from cyclic_stability.task import (
    CellTask,
    PreflightSnapshot,
    _record_from_payload,
    _record_payload,
    default_marker_analyzer,
    run_cell_task,
)
from synthetic import cell_task, different_identity


def test_default_analyzer_publishes_target_blind_nuclear_component_qc(
    analysis_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accepted and rejected DAPI components must remain visible without target fields."""
    from cyclic_stability import golgi, images, morphology, provenance, registration_qc
    from cyclic_stability import signal as signal_module

    cell = analysis_config.cell("golgi", "Cell1")
    decisions = tuple(
        QcDecision(
            "golgi",
            cell.cell_id,
            number,
            "accept",
            ("target-blind",),
            "reviewer",
            "2026-08-26T00:00:00Z",
        )
        for number in (1, 3, 4, 5)
    )
    config = replace(analysis_config, qc_decisions=decisions)
    component_qc_rows = (
        NuclearComponentQc(
            "N001",
            True,
            (),
            800.0,
            100.0,
            0.9,
            (1.0, 2.0, 3.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        NuclearComponentQc(
            "C002",
            False,
            ("volume-below-minimum",),
            8.0,
            1.0,
            0.8,
            (4.0, 5.0, 6.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )
    component_qc = tuple(MappingProxyType(asdict(row)) for row in component_qc_rows)
    marker_result = MarkerResult("golgi", cell.cell_id, (), (), (), (), component_qc)
    monkeypatch.setattr(
        registration_qc,
        "analyze_registration_qc",
        lambda *_args: SimpleNamespace(
            registration_rows=(), deformation_rows=(), overlay_paths=()
        ),
    )
    monkeypatch.setattr(golgi, "analyze_golgi_cell", lambda *_args: marker_result)
    monkeypatch.setattr(
        provenance,
        "channel_tiff_paths",
        lambda _round: (
            ("dapi", tmp_path / "dapi.tif"),
            ("target", tmp_path / "target.tif"),
        ),
    )
    monkeypatch.setattr(images, "load_volume", lambda _path: np.ones((3, 3, 3)))
    monkeypatch.setattr(
        signal_module,
        "raw_structure_roi",
        lambda *_args: np.ones((3, 3, 3), dtype=bool),
    )
    monkeypatch.setattr(morphology, "threshold_ensemble", lambda *_args: ())
    signal_row = SignalRow(
        "golgi",
        cell.cell_id,
        2,
        "matched-settings",
        True,
        0.5,
        3.0,
        "measured",
        10.0,
        4.0,
        2.0,
        MappingProxyType({50: 5.0}),
        MappingProxyType({50: 4.0}),
    )
    monkeypatch.setattr(signal_module, "measure_signal", lambda *_args: (signal_row,))
    task = CellTask(
        cell,
        config,
        config.config_digest,
        (),
        default_marker_analyzer,
        preflight=PreflightSnapshot((), ()),
    )
    output = default_marker_analyzer(task, tmp_path / "output")
    rows = output.tables["nuclear-component-qc"]
    assert [row["nucleus_id"] for row in rows] == ["N001", "C002"]
    assert [row["accepted"] for row in rows] == [True, False]
    expected_fields = set(NuclearComponentQc.__dataclass_fields__)
    assert all(set(row) == expected_fields for row in rows)
    assert all(
        not any(
            "target" in field or "intensity" in field or "mask" in field
            for field in row
        )
        for row in rows
    )
    assert output.tables["raw-signal"][0]["nucleus_id"] == "aggregate"


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
