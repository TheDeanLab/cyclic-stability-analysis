"""Behavioral tests for the read-only Cy-ExM artifact boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from cyclic_stability.cyexm_artifacts import (
    StatusInspector,
    resolve_workflow_output_root,
    resolve_round_artifacts,
    validate_completed_artifact,
)
from cyclic_stability.models import RoundSpec
from cyexm.errors import InputValidationError
from cyexm.workflow.models import WorkflowStatus
from synthetic import write_completion_fixture


@pytest.fixture
def completed_artifact(tmp_path: Path) -> Path:
    """A complete artifact whose record initially matches its immutable bytes."""
    return write_completion_fixture(tmp_path / "artifact.tif", b"original")


@pytest.fixture
def complete_status_inspector() -> StatusInspector:
    """A fully complete status without contacting the real workflow runner."""
    return lambda dataset_root, dataset_id: WorkflowStatus(
        dataset_id=dataset_id,
        scientific_fingerprint="synthetic",
        rounds=(),
        complete=True,
    )


@pytest.fixture
def golgi_round_two(tmp_path: Path) -> RoundSpec:
    """The Golgi reference acquisition maps to workflow reference round one."""
    output_root = tmp_path / "workflow-root"
    profile = tmp_path / "execution.yaml"
    profile.write_text(
        """schema_version: 1
executor: sequential
paths:
  dataset_root: /synthetic/raw
  output_root: %s
  worker_local_directory: null
"""
        % output_root,
        encoding="utf-8",
    )
    dataset_root = output_root / "outputs" / "synthetic"
    (dataset_root / "scientific-plan.json").parent.mkdir(parents=True)
    (dataset_root / "scientific-plan.json").write_text(
        '{"dataset_id": "synthetic"}\n', encoding="utf-8"
    )
    channels = dataset_root / "rounds" / "round-001" / "channels"
    for filename in (
        "round_2_405_DAPI.tif",
        "round_2_488_tubulin.tif",
        "round_2_647_golgi.tif",
    ):
        write_completion_fixture(channels / filename, filename.encode())
    return RoundSpec(
        cohort="golgi",
        cell_id="Cell1",
        acquisition_round=2,
        workflow_round=1,
        nd2_path=tmp_path / "round2.nd2",
        tiff_directory=tmp_path / "raw",
        z_start_offset=1,
        target_exposure_ms=200.0,
        target_laser_percent=50.0,
        tubulin_exposure_ms=200.0,
        dapi_exposure_ms=200.0,
        dataset_manifest=tmp_path / "dataset.yaml",
        execution_profile=profile,
        workflow_dataset_id="synthetic",
        identity_method="exact-pixel-audit",
    )


def test_completion_identity_mismatch_is_rejected(completed_artifact: Path) -> None:
    """A completion record must not attest to bytes changed after commit."""
    completed_artifact.write_bytes(completed_artifact.read_bytes() + b"changed")
    with pytest.raises(InputValidationError, match="completion identity mismatch"):
        validate_completed_artifact(completed_artifact)


def test_golgi_reference_resolves_acquisition_round_two_from_workflow_round_one(
    golgi_round_two: RoundSpec, complete_status_inspector: StatusInspector
) -> None:
    """The acquisition/workflow permutation cannot silently select round two outputs."""
    artifacts = resolve_round_artifacts(golgi_round_two, complete_status_inspector)
    assert artifacts.acquisition_round == 2
    assert artifacts.workflow_round == 1
    assert artifacts.registered_target.name == "round_2_647_golgi.tif"
    assert artifacts.affine is None
    assert (
        artifacts.workflow_output_root
        == golgi_round_two.execution_profile.parent / "workflow-root" / "outputs"
    )


def test_legacy_outputs_child_is_rejected_when_two_dataset_plans_match(
    golgi_round_two: RoundSpec,
) -> None:
    """A profile and legacy child must not silently select between two datasets."""
    profile_root = golgi_round_two.execution_profile.parent / "workflow-root"
    plan = profile_root / "synthetic" / "scientific-plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text('{"dataset_id": "synthetic"}\n', encoding="utf-8")
    with pytest.raises(InputValidationError, match="ambiguous workflow output root"):
        resolve_workflow_output_root(golgi_round_two)
