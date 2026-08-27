"""Restart-safe per-cell task behavior."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from cyclic_stability import task as task_module
from cyclic_stability.models import CellSpec, MarkerResult, QcDecision
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
from cyexm.registration._request_policy import channel_transform_parameters
from cyexm.registration.artifacts import (
    ArtifactStore,
    CompletionRecord,
    FileIdentity,
    StageRequest,
)
from cyexm.spatial import ImageGeometry
from synthetic import cell_task, different_identity


def _artifact_task(
    source: Path,
    producer,
    *,
    config_digest: str = "config-v1",
    software_identity: str = "repository-v1",
) -> task_module.ArtifactTask:
    """Adapt concise lifecycle fixtures to the mandatory snapshot contract."""

    def snapshot():
        identity = task_module.ContentIdentity.from_path(source)
        return task_module.ArtifactSnapshot(
            context=Path(identity.path),
            config_digest=config_digest,
            input_identities=(identity,),
        )

    def admitted_producer(output_dir: Path, _context: object) -> tuple[Path, ...]:
        return producer(output_dir)

    return task_module.ArtifactTask(
        key="nuclear-inventory",
        software_identity=software_identity,
        snapshot=snapshot,
        producer=admitted_producer,
    )


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


def test_production_preflight_failure_supersedes_prior_success_terminal(
    analysis_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed production preflight must belong to a newly admitted attempt."""
    decisions = tuple(
        QcDecision(
            "golgi",
            "Cell1",
            number,
            "accept",
            ("target-blind",),
            "reviewer",
            "2026-08-27T00:00:00Z",
        )
        for number in (1, 3, 4, 5)
    )
    task = replace(
        cell_task(tmp_path),
        config=replace(analysis_config, qc_decisions=decisions),
    )
    task_dir = tmp_path / "tasks" / task.key
    preflight_calls = 0
    observed_terminal_status: list[str] = []
    observed_lock_held: list[bool] = []

    def preflight(value: CellTask) -> CellTask:
        nonlocal preflight_calls
        preflight_calls += 1
        if preflight_calls == 1:
            return value
        terminal = json.loads((task_dir / "terminal.json").read_text())
        observed_terminal_status.append(str(terminal["status"]))
        with (task_dir / ".task.lock").open("a+b") as competing_lock:
            try:
                task_module.fcntl.flock(
                    competing_lock,
                    task_module.fcntl.LOCK_EX | task_module.fcntl.LOCK_NB,
                )
            except BlockingIOError:
                observed_lock_held.append(True)
            else:
                observed_lock_held.append(False)
                task_module.fcntl.flock(competing_lock, task_module.fcntl.LOCK_UN)
        raise ValueError("synthetic production preflight failure")

    monkeypatch.setattr(task_module, "_preflight_task", preflight)
    first = run_cell_task(task, tmp_path)
    failed = run_cell_task(replace(task, config_digest="changed"), tmp_path)

    assert first.status == "complete"
    assert failed.status == "failed"
    assert failed.attempt_id != first.attempt_id
    assert observed_terminal_status == ["running"]
    assert observed_lock_held == [True]
    terminal = json.loads((task_dir / "terminal.json").read_text())
    assert terminal["status"] == "failed"
    assert terminal["attempt_id"] == failed.attempt_id
    failure = json.loads((failed.attempt_dir / "failure.json").read_text())
    assert failure["error_type"] == "ValueError"
    assert failure["message"] == "synthetic production preflight failure"
    assert (first.attempt_dir / "record.json").is_file()


def test_production_reuse_recomputes_current_preflight_under_running_authority(
    analysis_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persisted preflight snapshot must not select stale production artifacts."""
    decisions = tuple(
        QcDecision(
            "golgi",
            "Cell1",
            number,
            "accept",
            ("target-blind",),
            "reviewer",
            "2026-08-27T00:00:00Z",
        )
        for number in (1, 3, 4, 5)
    )
    base = cell_task(tmp_path)
    analyzer_calls = 0

    def analyze(value: CellTask, output_dir: Path):
        nonlocal analyzer_calls
        analyzer_calls += 1
        return base.analyzer(value, output_dir)

    task = replace(
        base,
        config=replace(analysis_config, qc_decisions=decisions),
        analyzer=analyze,
    )
    discovered = {
        "selected": tmp_path / "discovery" / "profile-a" / "registered-target.tif"
    }
    discovered["selected"].parent.mkdir(parents=True)
    discovered["selected"].write_bytes(b"selected-a")
    replacement = tmp_path / "discovery" / "profile-b" / "registered-target.tif"
    replacement.parent.mkdir(parents=True)
    replacement.write_bytes(b"selected-b")
    task_dir = tmp_path / "tasks" / task.key
    terminal_statuses: list[str] = []
    preflight_calls = 0

    def preflight(value: CellTask) -> CellTask:
        nonlocal preflight_calls
        preflight_calls += 1
        terminal_statuses.append(
            str(json.loads((task_dir / "terminal.json").read_text())["status"])
        )
        identity = task_module.ContentIdentity.from_path(discovered["selected"])
        snapshot = PreflightSnapshot((), (identity,))
        return replace(
            value,
            input_identities=(*value.input_identities, identity),
            preflight=snapshot,
        )

    monkeypatch.setattr(task_module, "_preflight_task", preflight)

    first = run_cell_task(task, tmp_path)
    discovered["selected"] = replacement
    second = run_cell_task(task, tmp_path)
    third = run_cell_task(task, tmp_path)

    assert first.status == second.status == third.status == "complete"
    assert second.attempt_id != first.attempt_id
    assert second.reused is False
    assert third.attempt_id == second.attempt_id
    assert third.reused is True
    assert preflight_calls == 3
    assert analyzer_calls == 2
    assert terminal_statuses == ["running", "running", "running"]


def test_preflight_replaces_parent_symlink_aliases_before_producer_reads(
    analysis_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retargeting a parent alias cannot split hashed and consumed source bytes."""
    from cyclic_stability import cyexm_artifacts, provenance
    from cyclic_stability.cyexm_artifacts import RoundArtifacts

    root_a = tmp_path / "source-a"
    root_b = tmp_path / "source-b"
    alias = tmp_path / "selected-source"
    for root, marker in ((root_a, "A"), (root_b, "B")):
        (root / "raw").mkdir(parents=True)
        (root / "registered").mkdir()
        for relative in (
            "source.nd2",
            "dataset.yaml",
            "execution.yaml",
            "registered/dapi.tif",
            "registered/tubulin.tif",
            "registered/target.tif",
            "registered/target.tif.complete.json",
        ):
            path = root / relative
            path.write_text(f"{relative}-{marker}\n", encoding="utf-8")
        for name in (
            "round_2_647_lamin.tif",
            "round_2_488_tubulin.tif",
            "round_2_405_DAPI.tif",
        ):
            (root / "raw" / name).write_text(f"{name}-{marker}\n", encoding="utf-8")
    alias.symlink_to(root_a, target_is_directory=True)
    base = cell_task(tmp_path)
    round_spec = replace(
        base.cell.rounds[1],
        cohort="lamin",
        acquisition_round=2,
        workflow_round=2,
        nd2_path=alias / "source.nd2",
        tiff_directory=alias / "raw",
        dataset_manifest=alias / "dataset.yaml",
        execution_profile=alias / "execution.yaml",
    )
    cell = CellSpec("lamin", "Cell1", (round_spec,))
    artifact = RoundArtifacts(
        acquisition_round=2,
        workflow_round=2,
        workflow_output_root=alias / "registered",
        registered_dapi=alias / "registered/dapi.tif",
        registered_tubulin=alias / "registered/tubulin.tif",
        registered_target=alias / "registered/target.tif",
        affine=None,
        warp=None,
        pre_metrics=None,
        linear_metrics=None,
        nonlinear_metrics=None,
        chunk_manifest=None,
        completion_records=(SimpleNamespace(),),
        completion_record_paths=(alias / "registered/target.tif.complete.json",),
        completion_record_identities=(
            task_module.ContentIdentity.from_path(
                alias / "registered/target.tif.complete.json"
            ),
        ),
    )
    monkeypatch.setattr(
        cyexm_artifacts,
        "validate_cell_inputs",
        lambda _cell: SimpleNamespace(rounds=(artifact,)),
    )
    observed: dict[str, object] = {}

    def analyze(value: CellTask, _output_dir: Path):
        assert value.preflight is not None
        canonical_round = value.cell.rounds[0]
        assert value.config is not None
        config_round = value.config.cell("lamin", "Cell1").rounds[0]
        canonical_artifact = value.preflight.rounds[0]
        observed["round"] = canonical_round
        observed["config_round"] = config_round
        observed["artifact"] = canonical_artifact
        alias.unlink()
        alias.symlink_to(root_b, target_is_directory=True)
        raw_target = dict(provenance.channel_tiff_paths(canonical_round))["target"]
        observed["payloads"] = (
            canonical_round.nd2_path.read_text(encoding="utf-8"),
            raw_target.read_text(encoding="utf-8"),
            canonical_round.dataset_manifest.read_text(encoding="utf-8"),
            canonical_round.execution_profile.read_text(encoding="utf-8"),
            config_round.nd2_path.read_text(encoding="utf-8"),
            canonical_artifact.registered_target.read_text(encoding="utf-8"),
            canonical_artifact.completion_record_paths[0].read_text(encoding="utf-8"),
        )
        return task_module.TaskOutput(tables={"summary": ({"source": "A"},)})

    task = replace(
        base,
        cell=cell,
        config=replace(analysis_config, cells=(cell,), qc_decisions=()),
        input_identities=(),
        analyzer=analyze,
    )

    record = run_cell_task(task, tmp_path / "run")

    assert record.status == "complete"
    assert observed["payloads"] == (
        "source.nd2-A\n",
        "round_2_647_lamin.tif-A\n",
        "dataset.yaml-A\n",
        "execution.yaml-A\n",
        "source.nd2-A\n",
        "registered/target.tif-A\n",
        "registered/target.tif.complete.json-A\n",
    )
    canonical_round = observed["round"]
    config_round = observed["config_round"]
    canonical_artifact = observed["artifact"]
    assert canonical_round.nd2_path == root_a / "source.nd2"
    assert canonical_round.tiff_directory == root_a / "raw"
    assert canonical_round.dataset_manifest == root_a / "dataset.yaml"
    assert canonical_round.execution_profile == root_a / "execution.yaml"
    assert config_round == canonical_round
    assert canonical_artifact.registered_target == root_a / "registered/target.tif"
    assert canonical_artifact.completion_record_paths == (
        root_a / "registered/target.tif.complete.json",
    )


def test_artifact_task_changed_input_or_repository_preserves_prior_evidence(
    tmp_path: Path,
) -> None:
    """Reusing or overwriting an old attempt would erase its provenance."""
    source = tmp_path / "registered-dapi.tif"
    source.write_bytes(b"dapi-v1")

    def publish(output_dir: Path) -> tuple[Path, ...]:
        output = output_dir / "nuclear-inventory.json"
        output.write_text('{\n    "status": "complete"\n}\n', encoding="utf-8")
        return (output,)

    task = _artifact_task(source, publish)
    first = task_module.run_artifact_task(task, tmp_path / "run")
    reopened = task_module.run_artifact_task(task, tmp_path / "run")
    assert reopened.attempt_id == first.attempt_id
    assert reopened.reused is True

    source.write_bytes(b"dapi-v2")
    second = task_module.run_artifact_task(task, tmp_path / "run")
    changed_repository = replace(task, software_identity="repository-v2")
    third = task_module.run_artifact_task(changed_repository, tmp_path / "run")

    assert len({first.attempt_id, second.attempt_id, third.attempt_id}) == 3
    assert first.attempt_dir.is_dir()
    assert (first.attempt_dir / "nuclear-inventory.json").is_file()
    pointer = json.loads(
        (tmp_path / "run" / "tasks" / task.key / "terminal.json").read_text()
    )
    assert pointer["status"] == "complete"
    assert pointer["attempt_id"] == third.attempt_id
    assert (third.attempt_dir / "record.json").is_file()
    manifest = json.loads(
        (third.attempt_dir / "output-manifest.json").read_text(encoding="utf-8")
    )
    assert [item["relative"] for item in manifest["outputs"]] == [
        "nuclear-inventory.json"
    ]
    Path(third.output_identities[0].path).write_text(
        '{"status":"tampered"}\n', encoding="utf-8"
    )
    after_tamper = task_module.run_artifact_task(changed_repository, tmp_path / "run")
    assert after_tamper.attempt_id != third.attempt_id
    assert after_tamper.reused is False


def test_artifact_task_failure_has_failed_terminal_pointer(
    tmp_path: Path,
) -> None:
    """Publishing complete.json for a failed producer would authorize bad reuse."""
    source = tmp_path / "registered-dapi.tif"
    source.write_bytes(b"dapi")

    def fail(_output_dir: Path) -> tuple[Path, ...]:
        raise ValueError("synthetic inventory failure")

    task = _artifact_task(source, fail)
    record = task_module.run_artifact_task(task, tmp_path / "run")

    assert record.status == "failed"
    failure = json.loads(
        (record.attempt_dir / "failure.json").read_text(encoding="utf-8")
    )
    assert failure["error_type"] == "ValueError"
    assert failure["message"] == "synthetic inventory failure"
    assert failure["attempt_id"] == record.attempt_id
    assert failure["key"] == task.key
    assert failure["config_digest"] == "config-v1"
    assert failure["input_identities"] == [
        dict(task_module.ContentIdentity.from_path(source).to_dict())
    ]
    assert failure["software_identity"] == task.software_identity
    terminal = json.loads(
        (tmp_path / "run" / "tasks" / task.key / "terminal.json").read_text()
    )
    assert terminal["status"] == "failed"
    assert terminal["attempt_id"] == record.attempt_id
    assert not (tmp_path / "run" / "tasks" / task.key / "complete.json").exists()


def test_artifact_snapshot_runs_under_lock_after_running_authority(
    tmp_path: Path,
) -> None:
    """Moving snapshot work before admission would expose stale success authority."""
    source = tmp_path / "registered-dapi.tif"
    source.write_bytes(b"dapi-snapshot")
    observed: list[object] = []
    run_dir = tmp_path / "run"

    def snapshot():
        terminal = run_dir / "tasks" / "nuclear-inventory" / "terminal.json"
        observed.append(json.loads(terminal.read_text(encoding="utf-8"))["status"])
        identity, payload = task_module.ContentIdentity.read_bytes(source)
        return task_module.ArtifactSnapshot(
            context=payload,
            config_digest="config-from-snapshot",
            input_identities=(identity,),
        )

    def publish(output_dir: Path, context: object) -> tuple[Path, ...]:
        observed.append(context)
        output = output_dir / "nuclear-inventory.json"
        output.write_text(json.dumps({"payload": bytes(context).decode()}))
        return (output,)

    record = task_module.run_artifact_task(
        task_module.ArtifactTask(
            key="nuclear-inventory",
            software_identity="repository-v1",
            snapshot=snapshot,
            producer=publish,
        ),
        run_dir,
    )

    assert record.status == "complete"
    assert record.config_digest == "config-from-snapshot"
    assert observed == ["running", b"dapi-snapshot"]


def test_artifact_snapshot_input_rewrite_during_parse_fails_before_producer(
    tmp_path: Path,
) -> None:
    """Parsed bytes whose authoritative entry changed must never reach production."""
    source = tmp_path / "analysis.yaml"
    source.write_text("version: one\n", encoding="utf-8")
    producer_called = False

    def snapshot():
        identity, payload = task_module.ContentIdentity.read_bytes(source)
        parsed = payload.decode("utf-8").strip()
        source.write_text("version: two\n", encoding="utf-8")
        return task_module.ArtifactSnapshot(
            context=parsed,
            config_digest="digest-one",
            input_identities=(identity,),
        )

    def publish(_output_dir: Path, _context: object) -> tuple[Path, ...]:
        nonlocal producer_called
        producer_called = True
        raise AssertionError("producer must not run from a stale snapshot")

    record = task_module.run_artifact_task(
        task_module.ArtifactTask(
            key="nuclear-inventory",
            software_identity="repository-v1",
            snapshot=snapshot,
            producer=publish,
        ),
        tmp_path / "run",
    )

    assert record.status == "failed"
    assert producer_called is False
    failure = json.loads((record.attempt_dir / "failure.json").read_text())
    assert failure["message"] == (
        "one or more immutable task inputs changed before execution"
    )


def test_artifact_snapshot_canonical_context_survives_parent_alias_retarget(
    tmp_path: Path,
) -> None:
    """A producer must consume the canonical snapshot path, never a retargeted alias."""
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "input.txt").write_text("root-a", encoding="utf-8")
    (root_b / "input.txt").write_text("root-b", encoding="utf-8")
    alias = tmp_path / "current"
    alias.symlink_to(root_a, target_is_directory=True)

    def snapshot():
        identity, payload = task_module.ContentIdentity.read_bytes(alias / "input.txt")
        alias.unlink()
        alias.symlink_to(root_b, target_is_directory=True)
        return task_module.ArtifactSnapshot(
            context=(Path(identity.path), payload),
            config_digest="canonical-a",
            input_identities=(identity,),
        )

    def publish(output_dir: Path, context: object) -> tuple[Path, ...]:
        canonical, payload = context
        output = output_dir / "nuclear-inventory.json"
        output.write_text(
            json.dumps(
                {
                    "canonical": str(canonical),
                    "snapshotted": payload.decode(),
                    "consumed": canonical.read_text(encoding="utf-8"),
                }
            ),
            encoding="utf-8",
        )
        return (output,)

    record = task_module.run_artifact_task(
        task_module.ArtifactTask(
            key="nuclear-inventory",
            software_identity="repository-v1",
            snapshot=snapshot,
            producer=publish,
        ),
        tmp_path / "run",
    )

    assert record.status == "complete"
    payload = json.loads(Path(record.output_identities[0].path).read_text())
    assert payload == {
        "canonical": str((root_a / "input.txt").resolve()),
        "snapshotted": "root-a",
        "consumed": "root-a",
    }


def test_group_snapshot_keeps_one_parent_directory_when_alias_retargets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Related configuration files must not be mixed across alias generations."""
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    root_a.mkdir()
    root_b.mkdir()
    for root, marker in ((root_a, "A"), (root_b, "B")):
        (root / "analysis.yaml").write_text(marker, encoding="utf-8")
        (root / "correspondence.csv").write_text(marker, encoding="utf-8")
    alias = tmp_path / "current"
    alias.symlink_to(root_a, target_is_directory=True)
    original_read = os.read
    retargeted = False

    def retarget_after_first_payload(descriptor: int, size: int) -> bytes:
        nonlocal retargeted
        payload = original_read(descriptor, size)
        if not retargeted and payload == b"A":
            alias.unlink()
            alias.symlink_to(root_b, target_is_directory=True)
            retargeted = True
        return payload

    monkeypatch.setattr(os, "read", retarget_after_first_payload)

    snapshots = task_module.ContentIdentity.read_bytes_many(
        (alias / "analysis.yaml", alias / "correspondence.csv")
    )

    assert retargeted is True
    assert [payload for _identity, payload in snapshots] == [b"A", b"A"]
    assert {Path(identity.path).parent for identity, _payload in snapshots} == {
        root_a.resolve()
    }


def test_artifact_discovery_failure_supersedes_prior_success(
    tmp_path: Path,
) -> None:
    """Every invocation must rediscover inputs before considering prior reuse."""
    source = tmp_path / "registered-dapi.tif"
    source.write_bytes(b"dapi")
    discovery_available = True
    producer_calls = 0

    def snapshot():
        if not discovery_available:
            raise ValueError("synthetic DAPI discovery failure")
        return task_module.ArtifactSnapshot(
            context="discovered",
            config_digest="config-v1",
            input_identities=(task_module.ContentIdentity.from_path(source),),
        )

    def publish(output_dir: Path, context: object) -> tuple[Path, ...]:
        nonlocal producer_calls
        producer_calls += 1
        output = output_dir / "nuclear-inventory.json"
        output.write_text(json.dumps({"context": context}), encoding="utf-8")
        return (output,)

    task = task_module.ArtifactTask(
        key="nuclear-inventory",
        software_identity="repository-v1",
        snapshot=snapshot,
        producer=publish,
    )
    first = task_module.run_artifact_task(task, tmp_path / "run")
    discovery_available = False
    failed = task_module.run_artifact_task(task, tmp_path / "run")

    assert first.status == "complete"
    assert failed.status == "failed"
    assert failed.attempt_id != first.attempt_id
    assert producer_calls == 1
    terminal = json.loads(
        (tmp_path / "run/tasks/nuclear-inventory/terminal.json").read_text()
    )
    assert terminal["status"] == "failed"
    assert terminal["attempt_id"] == failed.attempt_id


def test_producer_input_mutation_never_publishes_completion(tmp_path: Path) -> None:
    """An input changed during analysis must fail before any success metadata exists."""
    source = tmp_path / "registered-dapi.tif"
    source.write_bytes(b"input-v1")
    original_mtime = source.stat().st_mtime_ns

    def mutate_input(output_dir: Path) -> tuple[Path, ...]:
        output = output_dir / "nuclear-inventory.json"
        output.write_text('{"value":"complete"}\n', encoding="utf-8")
        source.write_bytes(b"input-v2")
        os.utime(source, ns=(source.stat().st_atime_ns, original_mtime))
        return (output,)

    task = _artifact_task(source, mutate_input)

    failed = task_module.run_artifact_task(task, tmp_path / "run")

    assert failed.status == "failed"
    terminal = json.loads(
        (tmp_path / "run" / "tasks" / task.key / "terminal.json").read_text()
    )
    assert terminal["status"] == "failed"
    assert terminal["attempt_id"] == failed.attempt_id
    failure = json.loads((failed.attempt_dir / "failure.json").read_text())
    assert (
        failure["message"]
        == "one or more immutable task inputs changed during execution"
    )
    assert not (failed.attempt_dir / "record.json").exists()
    assert not (failed.attempt_dir / "output-manifest.json").exists()
    assert not (tmp_path / "run" / "tasks" / task.key / "complete.json").exists()


@pytest.mark.parametrize("mutation_stage", ("output-validation", "record-construction"))
def test_late_input_mutation_is_rejected_at_final_success_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_stage: str,
) -> None:
    """Fallible success preparation must finish before the last input revalidation."""
    source = tmp_path / "registered-dapi.tif"
    source.write_bytes(b"input-v1")
    original_mtime = source.stat().st_mtime_ns

    def publish(output_dir: Path) -> tuple[Path, ...]:
        output = output_dir / "nuclear-inventory.json"
        output.write_text('{"value":"complete"}\n', encoding="utf-8")
        return (output,)

    mutated = False

    def mutate_input() -> None:
        nonlocal mutated
        if mutated:
            return
        mutated = True
        source.write_bytes(b"input-v2")
        os.utime(source, ns=(source.stat().st_atime_ns, original_mtime))

    if mutation_stage == "output-validation":
        original_validate = task_module._validate_output

        def validate_then_mutate(path: Path) -> None:
            original_validate(path)
            mutate_input()

        monkeypatch.setattr(task_module, "_validate_output", validate_then_mutate)
    else:
        original_record_payload = task_module._record_payload

        def construct_record_then_mutate(record):
            payload = original_record_payload(record)
            mutate_input()
            return payload

        monkeypatch.setattr(
            task_module, "_record_payload", construct_record_then_mutate
        )

    task = _artifact_task(source, publish)

    failed = task_module.run_artifact_task(task, tmp_path / "run")

    assert failed.status == "failed"
    terminal = json.loads(
        (tmp_path / "run" / "tasks" / task.key / "terminal.json").read_text()
    )
    assert terminal["status"] == "failed"
    assert terminal["attempt_id"] == failed.attempt_id
    failure = json.loads((failed.attempt_dir / "failure.json").read_text())
    assert "immutable task inputs changed" in failure["message"]
    assert not (failed.attempt_dir / "record.json").exists()
    assert not (failed.attempt_dir / "output-manifest.json").exists()


def _production_provenance_task(
    analysis_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CellTask, Path, Path]:
    """Create one production-preflight-shaped task with record-derived source input."""
    from cyclic_stability import cyexm_artifacts, provenance
    from cyclic_stability.cyexm_artifacts import RoundArtifacts

    base = cell_task(tmp_path)
    round_spec = base.cell.rounds[2]
    root = tmp_path / "production-provenance"
    execution_profile = root / "execution.yaml"
    execution_profile.parent.mkdir(parents=True)
    execution_profile.write_text("executor: synthetic\n", encoding="utf-8")
    dataset_manifest = root / "dataset.yaml"
    dataset_manifest.write_text("dataset: synthetic\n", encoding="utf-8")
    nd2 = root / "source.nd2"
    nd2.write_bytes(b"nd2")
    raw_paths = tuple(
        (name, root / "raw" / f"round-3-{name}.tif")
        for name in ("dapi", "tubulin", "target")
    )
    for _name, path in raw_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    fixed = root / "registered-dapi.tif"
    native_source = root / "native-target.tif"
    affine = root / "affine.mat"
    warp = root / "warp.nii.gz"
    registered_tubulin = root / "registered-tubulin.tif"
    registered_target = root / "registered-target.tif"
    for path, payload in (
        (fixed, b"fixed"),
        (native_source, b"source-v1"),
        (affine, b"affine"),
        (warp, b"warp"),
        (registered_tubulin, b"tubulin"),
    ):
        path.write_bytes(payload)
    spacing = (1.0, 1.0, 1.0)
    request = StageRequest.for_files(
        stage="channel-transform-v1",
        inputs=(fixed, native_source, affine, warp),
        geometry=ImageGeometry(
            shape=(1, 1, 1),
            origin=(0.0, 0.0, 0.0),
            spacing=spacing,
            direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        method="warp-affine",
        accuracy="not-applicable",
        native_threads={"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 1},
        upstream=FileIdentity.from_path(warp),
        parameters=channel_transform_parameters(
            pixel_size_xyz=spacing, interpolation="linear"
        ),
    )
    store = ArtifactStore()

    def write_registered(path: Path) -> None:
        path.write_bytes(b"registered")

    store.commit(
        registered_target,
        request,
        write_registered,
        lambda path: path.read_bytes() == b"registered",
    )
    completion_record_path = store.record_path(registered_target)
    completion_record = CompletionRecord.read(completion_record_path)
    artifact = RoundArtifacts(
        acquisition_round=round_spec.acquisition_round,
        workflow_round=round_spec.workflow_round,
        workflow_output_root=root,
        registered_dapi=fixed,
        registered_tubulin=registered_tubulin,
        registered_target=registered_target,
        affine=affine,
        warp=warp,
        pre_metrics=None,
        linear_metrics=None,
        nonlinear_metrics=None,
        chunk_manifest=None,
        completion_records=(completion_record,),
        completion_record_paths=(completion_record_path,),
        completion_record_identities=(
            task_module.ContentIdentity.from_path(completion_record_path),
        ),
        native_target=Path(completion_record.request.inputs[1].path),
        native_target_identity=task_module.ContentIdentity.from_path(native_source),
        moving_crop=None,
    )
    cell = replace(
        base.cell,
        rounds=(
            replace(
                round_spec,
                nd2_path=nd2,
                tiff_directory=root / "raw",
                dataset_manifest=dataset_manifest,
                execution_profile=execution_profile,
            ),
        ),
    )
    config = replace(
        analysis_config,
        qc_decisions=(
            QcDecision(
                "golgi",
                "Cell1",
                3,
                "accept",
                ("target-blind",),
                "reviewer",
                "2026-08-27T00:00:00Z",
            ),
        ),
    )

    def validate_current_provenance(_cell):
        current_record = CompletionRecord.read(completion_record_path)
        current_native = Path(current_record.request.inputs[1].path)
        current_artifact = replace(
            artifact,
            completion_records=(current_record,),
            completion_record_identities=(
                task_module.ContentIdentity.from_path(completion_record_path),
            ),
            native_target=current_native,
            native_target_identity=task_module.ContentIdentity.from_path(
                current_native
            ),
        )
        return SimpleNamespace(rounds=(current_artifact,))

    monkeypatch.setattr(
        cyexm_artifacts, "validate_cell_inputs", validate_current_provenance
    )
    monkeypatch.setattr(provenance, "channel_tiff_paths", lambda _round: raw_paths)
    return (
        replace(base, cell=cell, config=config),
        completion_record_path,
        native_source,
    )


@pytest.mark.parametrize(
    ("cohort", "expected_native_rounds"),
    (("golgi", {1, 2}), ("lamp1", {1, 2}), ("lamin", set())),
)
def test_preflight_binds_only_native_targets_consumed_by_each_cohort(
    analysis_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cohort: str,
    expected_native_rounds: set[int],
) -> None:
    """Native targets are authoritative only where the selected analyzer consumes them."""
    from cyclic_stability import cyexm_artifacts, provenance
    from cyclic_stability.cyexm_artifacts import RoundArtifacts

    base = cell_task(tmp_path)
    rounds = []
    artifacts = []
    raw_paths: dict[int, tuple[tuple[str, Path], ...]] = {}
    native_paths: dict[int, Path] = {}
    registered_targets: dict[int, Path] = {}
    completion_paths: dict[int, Path] = {}
    execution_profiles: dict[int, Path] = {}
    for number in (1, 2, 3):
        round_root = tmp_path / "preflight" / cohort / f"round-{number}"
        nd2 = round_root / "source.nd2"
        nd2.parent.mkdir(parents=True, exist_ok=True)
        nd2.write_bytes(f"nd2-{number}".encode())
        raw_paths[number] = tuple(
            (name, round_root / "raw" / f"{name}.tif")
            for name in ("target", "tubulin", "dapi")
        )
        for _name, path in raw_paths[number]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"raw-{number}-{path.name}".encode())
        profile = round_root / "execution.yaml"
        profile.write_text(f"round: {number}\n", encoding="utf-8")
        execution_profiles[number] = profile
        dataset_manifest = round_root / "dataset.yaml"
        dataset_manifest.write_text(f"round: {number}\n", encoding="utf-8")
        round_spec = replace(
            base.cell.rounds[number - 1],
            cohort=cohort,
            acquisition_round=number,
            workflow_round=number,
            nd2_path=nd2,
            tiff_directory=round_root / "raw",
            dataset_manifest=dataset_manifest,
            execution_profile=profile,
        )
        rounds.append(round_spec)
        registered_dapi = round_root / "registered-dapi.tif"
        registered_tubulin = round_root / "registered-tubulin.tif"
        registered_target = round_root / "registered-target.tif"
        completion_record = round_root / "registered-target.tif.complete.json"
        native_target = round_root / "native-target.tif"
        for path in (
            registered_dapi,
            registered_tubulin,
            registered_target,
            completion_record,
            native_target,
        ):
            path.write_bytes(f"{number}-{path.name}".encode())
        native_paths[number] = native_target
        registered_targets[number] = registered_target
        completion_paths[number] = completion_record
        artifacts.append(
            RoundArtifacts(
                acquisition_round=number,
                workflow_round=number,
                workflow_output_root=round_root,
                registered_dapi=registered_dapi,
                registered_tubulin=registered_tubulin,
                registered_target=registered_target,
                affine=None,
                warp=None,
                pre_metrics=None,
                linear_metrics=None,
                nonlinear_metrics=None,
                chunk_manifest=None,
                completion_records=(SimpleNamespace(),),
                completion_record_paths=(completion_record,),
                completion_record_identities=(
                    task_module.ContentIdentity.from_path(completion_record),
                ),
                native_target=native_target,
                native_target_identity=task_module.ContentIdentity.from_path(
                    native_target
                ),
                moving_crop=None,
            )
        )
    cell = CellSpec(cohort, "Cell1", tuple(rounds))
    config = replace(
        analysis_config,
        cells=(cell,),
        qc_decisions=(
            QcDecision(
                cohort,
                "Cell1",
                1,
                "accept",
                ("target-blind",),
                "reviewer",
                "2026-08-27T00:00:00Z",
            ),
            QcDecision(
                cohort,
                "Cell1",
                3,
                "exclude",
                ("target-blind",),
                "reviewer",
                "2026-08-27T00:00:00Z",
            ),
        ),
    )

    def validate_current_native_sources(_cell):
        return SimpleNamespace(
            rounds=tuple(
                replace(
                    artifact,
                    native_target_identity=task_module.ContentIdentity.from_path(
                        artifact.native_target
                    ),
                )
                for artifact in artifacts
            )
        )

    monkeypatch.setattr(
        cyexm_artifacts, "validate_cell_inputs", validate_current_native_sources
    )
    monkeypatch.setattr(
        provenance,
        "channel_tiff_paths",
        lambda round_spec: raw_paths[round_spec.acquisition_round],
    )

    task = replace(base, cell=cell, config=config, input_identities=())
    preflighted = task_module._preflight_task(task)

    bound = {Path(identity.path) for identity in preflighted.input_identities}
    assert {native_paths[number].resolve() for number in expected_native_rounds} == {
        path.resolve() for path in bound if path.name == "native-target.tif"
    }
    for number in (1, 2):
        assert registered_targets[number].resolve() in bound
        assert completion_paths[number].resolve() in bound
    assert {path.resolve() for path in execution_profiles.values()}.issubset(bound)
    for number, path in native_paths.items():
        payload = path.read_bytes()
        original_mtime = path.stat().st_mtime_ns
        replacement = (b"X" if payload[:1] != b"X" else b"Y") + payload[1:]
        path.write_bytes(replacement)
        os.utime(path, ns=(path.stat().st_atime_ns, original_mtime))

        changed = task_module._preflight_task(task)

        assert (changed.input_identities != preflighted.input_identities) is (
            number in expected_native_rounds
        )
        path.write_bytes(payload)
        os.utime(path, ns=(path.stat().st_atime_ns, original_mtime))
    profile = execution_profiles[1]
    profile_payload = profile.read_bytes()
    profile_mtime = profile.stat().st_mtime_ns
    profile.write_bytes(profile_payload.replace(b"1", b"9", 1))
    os.utime(profile, ns=(profile.stat().st_atime_ns, profile_mtime))

    changed_profile = task_module._preflight_task(task)

    assert changed_profile.input_identities != preflighted.input_identities


@pytest.mark.parametrize("tampered_input", ("completion-record", "native-source"))
def test_production_task_rejects_same_metadata_consumed_provenance_tampering(
    analysis_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_input: str,
) -> None:
    """Every parsed record and record-derived source must invalidate stale reuse."""
    task, completion_record, native_source = _production_provenance_task(
        analysis_config, tmp_path, monkeypatch
    )
    first = run_cell_task(task, tmp_path)
    reopened = run_cell_task(task, tmp_path)
    assert reopened.attempt_id == first.attempt_id
    assert reopened.reused is True
    path = {
        "completion-record": completion_record,
        "native-source": native_source,
    }[tampered_input]
    original_mtime = path.stat().st_mtime_ns
    payload = path.read_bytes()
    if tampered_input == "completion-record":
        tampered = payload.replace(b'":', b'": ', 1)[:-1]
        assert json.loads(tampered) == json.loads(payload)
    else:
        tampered = b"source-v2"
    assert len(tampered) == len(payload)
    path.write_bytes(tampered)
    os.utime(path, ns=(path.stat().st_atime_ns, original_mtime))

    after_tamper = run_cell_task(task, tmp_path)

    assert after_tamper.status == "complete"
    assert after_tamper.reused is False
    assert after_tamper.attempt_id != first.attempt_id
    bound_paths = {identity.path for identity in after_tamper.input_identities}
    assert str(completion_record.resolve()) in bound_paths
    assert str(native_source.resolve()) in bound_paths


def test_artifact_task_invalidates_same_metadata_dapi_reuse(tmp_path: Path) -> None:
    """Metadata-only input checks would authorize stale DAPI evidence."""
    source = tmp_path / "registered-dapi.tif"
    source.write_bytes(b"dapi-v1")
    original_mtime = source.stat().st_mtime_ns

    def publish(output_dir: Path) -> tuple[Path, ...]:
        output = output_dir / "nuclear-inventory.json"
        output.write_text('{"value":"a"}\n', encoding="utf-8")
        return (output,)

    task = _artifact_task(source, publish)
    first = task_module.run_artifact_task(task, tmp_path / "run")
    source.write_bytes(b"dapi-v2")
    os.utime(source, ns=(source.stat().st_atime_ns, original_mtime))

    after_tamper = task_module.run_artifact_task(task, tmp_path / "run")

    assert after_tamper.status == "complete"
    assert after_tamper.reused is False
    assert after_tamper.attempt_id != first.attempt_id
    terminal = json.loads(
        (tmp_path / "run" / "tasks" / task.key / "terminal.json").read_text()
    )
    assert terminal["status"] == "complete"
    assert terminal["attempt_id"] == after_tamper.attempt_id
    assert (first.attempt_dir / "record.json").is_file()


@pytest.mark.parametrize("artifact", ("output", "record", "manifest"))
def test_artifact_task_rejects_same_metadata_published_byte_tampering(
    tmp_path: Path, artifact: str
) -> None:
    """Every file trusted during reopen must be bound to its exact content."""
    source = tmp_path / "registered-dapi.tif"
    source.write_bytes(b"dapi-v1")

    def publish(output_dir: Path) -> tuple[Path, ...]:
        output = output_dir / "nuclear-inventory.json"
        output.write_text('{"value":"a"}\n', encoding="utf-8")
        return (output,)

    task = _artifact_task(source, publish)
    first = task_module.run_artifact_task(task, tmp_path / "run")
    paths = {
        "output": first.attempt_dir / "nuclear-inventory.json",
        "record": first.attempt_dir / "record.json",
        "manifest": first.attempt_dir / "output-manifest.json",
    }
    path = paths[artifact]
    original_mtime = path.stat().st_mtime_ns
    payload = path.read_bytes()
    if artifact == "output":
        tampered = payload.replace(b'"a"', b'"b"', 1)
    elif artifact == "record":
        match = re.search(rb'"peak_rss_bytes":([0-9])', payload)
        assert match is not None
        digit = match.group(1)
        replacement = b"1" if digit != b"1" else b"2"
        tampered = payload[: match.start(1)] + replacement + payload[match.end(1) :]
    else:
        tampered = payload.replace(b'{"outputs":[', b'{"outputs": [', 1)[:-1]
    assert len(tampered) == len(payload)
    path.write_bytes(tampered)
    os.utime(path, ns=(path.stat().st_atime_ns, original_mtime))

    after_tamper = task_module.run_artifact_task(task, tmp_path / "run")

    assert after_tamper.status == "complete"
    assert after_tamper.reused is False
    assert after_tamper.attempt_id != first.attempt_id


def test_task_record_replacement_during_parse_invalidates_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One task record snapshot must remain the same directory entry after parsing."""
    source = tmp_path / "registered-dapi.tif"
    source.write_bytes(b"dapi-v1")

    def publish(output_dir: Path) -> tuple[Path, ...]:
        output = output_dir / "nuclear-inventory.json"
        output.write_text('{"value":"a"}\n', encoding="utf-8")
        return (output,)

    task = _artifact_task(source, publish)
    first = task_module.run_artifact_task(task, tmp_path / "run")
    record_path = first.attempt_dir / "record.json"
    replacement = tmp_path / "replacement-record.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["peak_rss_bytes"] += 1
    replacement.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    original_decoder = task_module._record_from_payload
    replaced = False

    def replace_after_read(value):
        nonlocal replaced
        if not replaced and value["attempt_id"] == first.attempt_id:
            replaced = True
            os.replace(replacement, record_path)
        return original_decoder(value)

    monkeypatch.setattr(task_module, "_record_from_payload", replace_after_read)

    reopened = task_module.run_artifact_task(task, tmp_path / "run")

    assert replaced is True
    assert reopened.status == "complete"
    assert reopened.reused is False
    assert reopened.attempt_id != first.attempt_id


def test_terminal_replacement_during_decode_invalidates_direct_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed terminal replacing decoded complete bytes must remain authoritative."""
    source = tmp_path / "registered-dapi.tif"
    source.write_bytes(b"dapi-v1")

    def publish(output_dir: Path) -> tuple[Path, ...]:
        output = output_dir / "nuclear-inventory.json"
        output.write_text('{"value":"a"}\n', encoding="utf-8")
        return (output,)

    task = _artifact_task(source, publish)
    first = task_module.run_artifact_task(task, tmp_path / "run")
    terminal = tmp_path / "run" / "tasks" / task.key / "terminal.json"
    replacement = tmp_path / "failed-terminal.json"
    replacement.write_text(
        json.dumps({"status": "failed", "attempt_id": "0" * 32}) + "\n",
        encoding="utf-8",
    )
    original_loads = task_module.json.loads
    replaced = False

    def replace_during_terminal_decode(payload, *args, **kwargs):
        nonlocal replaced
        text = payload.decode() if isinstance(payload, bytes) else str(payload)
        if (
            not replaced
            and first.attempt_id in text
            and '"status":"complete"' in text
            and '"record_identity"' in text
        ):
            replaced = True
            os.replace(replacement, terminal)
        return original_loads(payload, *args, **kwargs)

    monkeypatch.setattr(task_module.json, "loads", replace_during_terminal_decode)

    reopened = task_module.run_artifact_task(task, tmp_path / "run")

    assert replaced is True
    assert reopened.status == "complete"
    assert reopened.reused is False
    assert reopened.attempt_id != first.attempt_id


def test_terminal_replacement_during_decode_invalidates_production_reuse(
    analysis_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saved production terminals must be rechecked before running authority begins."""
    decisions = tuple(
        QcDecision(
            "golgi",
            "Cell1",
            number,
            "accept",
            ("target-blind",),
            "reviewer",
            "2026-08-27T00:00:00Z",
        )
        for number in (1, 3, 4, 5)
    )
    task = replace(
        cell_task(tmp_path),
        config=replace(analysis_config, qc_decisions=decisions),
    )
    monkeypatch.setattr(
        task_module,
        "_preflight_task",
        lambda value: replace(value, preflight=PreflightSnapshot((), ())),
    )
    first = run_cell_task(task, tmp_path / "run")
    terminal = tmp_path / "run" / "tasks" / task.key / "terminal.json"
    replacement = tmp_path / "failed-production-terminal.json"
    replacement.write_text(
        json.dumps({"status": "failed", "attempt_id": "0" * 32}) + "\n",
        encoding="utf-8",
    )
    original_loads = task_module.json.loads
    replaced = False

    def replace_during_terminal_decode(payload, *args, **kwargs):
        nonlocal replaced
        text = payload.decode() if isinstance(payload, bytes) else str(payload)
        if (
            not replaced
            and first.attempt_id in text
            and '"status":"complete"' in text
            and '"record_identity"' in text
        ):
            replaced = True
            os.replace(replacement, terminal)
        return original_loads(payload, *args, **kwargs)

    monkeypatch.setattr(task_module.json, "loads", replace_during_terminal_decode)

    reopened = run_cell_task(task, tmp_path / "run")

    assert replaced is True
    assert reopened.status == "complete"
    assert reopened.reused is False
    assert reopened.attempt_id != first.attempt_id


def test_production_reuse_rechecks_running_terminal_before_restoring_complete(
    analysis_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement during preflight must prevent restoration of old success."""
    decisions = tuple(
        QcDecision(
            "golgi",
            "Cell1",
            number,
            "accept",
            ("target-blind",),
            "reviewer",
            "2026-08-27T00:00:00Z",
        )
        for number in (1, 3, 4, 5)
    )
    task = replace(
        cell_task(tmp_path),
        config=replace(analysis_config, qc_decisions=decisions),
    )

    def preflight(value):
        return replace(value, preflight=PreflightSnapshot((), ()))

    monkeypatch.setattr(task_module, "_preflight_task", preflight)
    first = run_cell_task(task, tmp_path / "run")
    terminal = tmp_path / "run" / "tasks" / task.key / "terminal.json"
    replacement = tmp_path / "replacement-failed-terminal.json"
    replacement.write_text(
        json.dumps({"status": "failed", "attempt_id": "0" * 32}) + "\n",
        encoding="utf-8",
    )
    replaced = False

    def replace_running_during_preflight(value):
        nonlocal replaced
        pointer = json.loads(terminal.read_text(encoding="utf-8"))
        if pointer["status"] == "running":
            os.replace(replacement, terminal)
            replaced = True
        return preflight(value)

    monkeypatch.setattr(
        task_module, "_preflight_task", replace_running_during_preflight
    )

    reopened = run_cell_task(task, tmp_path / "run")

    assert replaced is True
    assert reopened.status == "failed"
    assert reopened.reused is False
    assert reopened.attempt_id != first.attempt_id
    pointer = json.loads(terminal.read_text(encoding="utf-8"))
    assert pointer["status"] == "failed"
    assert pointer["attempt_id"] == reopened.attempt_id
    assert (first.attempt_dir / "record.json").is_file()


def test_changed_input_failure_supersedes_prior_success_terminal(
    tmp_path: Path,
) -> None:
    """A stale success pointer must not survive a newer failed attempt."""
    source = tmp_path / "registered-dapi.tif"
    source.write_bytes(b"dapi-v1")

    def publish(output_dir: Path) -> tuple[Path, ...]:
        output = output_dir / "nuclear-inventory.json"
        output.write_text('{"value":"a"}\n', encoding="utf-8")
        return (output,)

    task = _artifact_task(source, publish)
    first = task_module.run_artifact_task(task, tmp_path / "run")
    source.write_bytes(b"dapi-v2")

    def fail(_output_dir: Path) -> tuple[Path, ...]:
        raise ValueError("synthetic changed-input failure")

    changed = _artifact_task(source, fail)
    failed = task_module.run_artifact_task(changed, tmp_path / "run")

    assert failed.status == "failed"
    assert failed.attempt_id != first.attempt_id
    terminal = json.loads(
        (tmp_path / "run" / "tasks" / task.key / "terminal.json").read_text()
    )
    assert terminal["status"] == "failed"
    assert terminal["attempt_id"] == failed.attempt_id
    assert (first.attempt_dir / "record.json").is_file()


def test_cleanup_failure_publishes_only_failed_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publishing success before fallible cleanup creates contradictory state."""
    source = tmp_path / "registered-dapi.tif"
    source.write_bytes(b"dapi-v1")

    def publish(output_dir: Path) -> tuple[Path, ...]:
        output = output_dir / "nuclear-inventory.json"
        output.write_text('{"value":"a"}\n', encoding="utf-8")
        return (output,)

    task = _artifact_task(source, publish)

    def fail_cleanup(_path: Path) -> None:
        raise OSError("synthetic scratch cleanup failure")

    monkeypatch.setattr(task_module.shutil, "rmtree", fail_cleanup)
    failed = task_module.run_artifact_task(task, tmp_path / "run")

    assert failed.status == "failed"
    terminal_path = tmp_path / "run" / "tasks" / task.key / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert terminal["attempt_id"] == failed.attempt_id
    assert not (failed.attempt_dir / "record.json").exists()
    assert not (tmp_path / "run" / "tasks" / task.key / "complete.json").exists()


@pytest.mark.parametrize("publication_failure", ("failure-json", "failed-terminal"))
def test_admitted_attempt_cannot_expose_prior_success_when_failure_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication_failure: str,
) -> None:
    """Failure-publication errors must leave running, never stale success, authoritative."""
    source = tmp_path / "registered-dapi.tif"
    source.write_bytes(b"dapi-v1")

    def publish(output_dir: Path) -> tuple[Path, ...]:
        output = output_dir / "nuclear-inventory.json"
        output.write_text('{"value":"a"}\n', encoding="utf-8")
        return (output,)

    task = _artifact_task(source, publish)
    first = task_module.run_artifact_task(task, tmp_path / "run")

    def fail(_output_dir: Path) -> tuple[Path, ...]:
        raise ValueError("synthetic scientific failure")

    original_atomic = task_module._atomic_bytes

    def fail_publication(path: Path, payload: bytes) -> None:
        if publication_failure == "failure-json" and path.name == "failure.json":
            raise OSError("synthetic failure-json publication failure")
        if path.name == "terminal.json":
            status = json.loads(payload)["status"]
            if publication_failure == "failed-terminal" and status == "failed":
                raise OSError("synthetic failed-terminal publication failure")
        original_atomic(path, payload)

    monkeypatch.setattr(task_module, "_atomic_bytes", fail_publication)
    changed = _artifact_task(source, fail, config_digest="config-v2")
    with pytest.raises(OSError, match="publication failure"):
        task_module.run_artifact_task(changed, tmp_path / "run")

    terminal = json.loads(
        (tmp_path / "run" / "tasks" / task.key / "terminal.json").read_text()
    )
    assert terminal["status"] == "running"
    assert terminal["attempt_id"] != first.attempt_id
    assert (first.attempt_dir / "record.json").is_file()


def test_terminal_authority_retires_coexisting_legacy_complete_pointer(
    tmp_path: Path,
) -> None:
    """A legacy completion pointer must never coexist as alternate authority."""
    source = tmp_path / "registered-dapi.tif"
    source.write_bytes(b"dapi-v1")

    def publish(output_dir: Path) -> tuple[Path, ...]:
        output = output_dir / "nuclear-inventory.json"
        output.write_text('{"value":"a"}\n', encoding="utf-8")
        return (output,)

    task = _artifact_task(source, publish)
    first = task_module.run_artifact_task(task, tmp_path / "run")
    task_dir = tmp_path / "run" / "tasks" / task.key
    legacy = task_dir / "complete.json"
    legacy.write_text('{"attempt_id":"stale-legacy"}\n', encoding="utf-8")

    reopened = task_module.run_artifact_task(task, tmp_path / "run")

    assert reopened.reused is True
    assert reopened.attempt_id == first.attempt_id
    assert not legacy.exists()


def test_cell_task_rejects_same_metadata_input_tampering(tmp_path: Path) -> None:
    """Normal task inputs need exact content identity, not metadata equality."""
    task = cell_task(tmp_path)
    first = run_cell_task(task, tmp_path)
    source = Path(task.input_identities[0].path)
    original_mtime = source.stat().st_mtime_ns
    payload = source.read_bytes()
    replacement = (b"X" if payload[:1] != b"X" else b"Y") + payload[1:]
    source.write_bytes(replacement)
    os.utime(source, ns=(source.stat().st_atime_ns, original_mtime))

    after_tamper = run_cell_task(task, tmp_path)

    assert after_tamper.status == "failed"
    assert after_tamper.attempt_id != first.attempt_id
    terminal = json.loads((tmp_path / "tasks" / task.key / "terminal.json").read_text())
    assert terminal["status"] == "failed"


def test_cell_task_rejects_same_metadata_output_tampering(tmp_path: Path) -> None:
    """Normal task output reopen must recompute exact content identity."""
    task = cell_task(tmp_path)
    first = run_cell_task(task, tmp_path)
    output = next(
        Path(identity.path)
        for identity in first.output_identities
        if Path(identity.path).name == "summary.csv"
    )
    original_mtime = output.stat().st_mtime_ns
    payload = output.read_bytes()
    tampered = payload.replace(b"synthetic", b"synthetix", 1)
    assert len(tampered) == len(payload)
    output.write_bytes(tampered)
    os.utime(output, ns=(output.stat().st_atime_ns, original_mtime))

    after_tamper = run_cell_task(task, tmp_path)

    assert after_tamper.status == "complete"
    assert after_tamper.reused is False
    assert after_tamper.attempt_id != first.attempt_id


def test_cell_task_rejects_metadata_only_legacy_record_schema(tmp_path: Path) -> None:
    """Old metadata-only records must fail closed even with an updated root hash."""
    task = cell_task(tmp_path)
    first = run_cell_task(task, tmp_path)
    record_path = first.attempt_dir / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    for field in ("input_identities", "output_identities"):
        for identity in record[field]:
            identity.pop("sha256", None)
    record_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    terminal_path = tmp_path / "tasks" / task.key / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["record_identity"] = task_module.ContentIdentity.from_path(
        record_path
    ).to_dict()
    terminal_path.write_text(
        json.dumps(terminal, sort_keys=True) + "\n", encoding="utf-8"
    )

    reopened = run_cell_task(task, tmp_path)

    assert reopened.status == "complete"
    assert reopened.reused is False
    assert reopened.attempt_id != first.attempt_id


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
