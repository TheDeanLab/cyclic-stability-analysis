"""Behavioral tests for the read-only Cy-ExM artifact boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from cyclic_stability.cyexm_artifacts import (
    StatusInspector,
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
    dataset_root = output_root / "synthetic"
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
        == golgi_round_two.execution_profile.parent / "workflow-root"
    )


def _lamin_round(
    tmp_path: Path,
    *,
    acquisition_round: int,
    workflow_round: int,
    complete_moving: bool,
) -> RoundSpec:
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
    round_root = output_root / "synthetic" / "rounds" / f"round-{workflow_round:03d}"
    channels = round_root / "channels"
    for filename in (
        f"round_{acquisition_round}_405_DAPI.tif",
        f"round_{acquisition_round}_488_tubulin.tif",
        f"round_{acquisition_round}_647_lamin.tif",
    ):
        write_completion_fixture(channels / filename, filename.encode())
    if complete_moving:
        registration = round_root / "registration"
        for filename in (
            f"linear_transform_{workflow_round}.mat",
            f"nonlinear_warp_{workflow_round}.nii.gz",
            "image_metrics_pre-registration.txt",
            "image_metrics_linearly-registered.txt",
            "image_metrics_nonlinear-registered.txt",
            f"round_{workflow_round}_chunks/manifest.json",
        ):
            write_completion_fixture(registration / filename, filename.encode())
    return RoundSpec(
        cohort="lamin",
        cell_id="Cell1",
        acquisition_round=acquisition_round,
        workflow_round=workflow_round,
        nd2_path=tmp_path / "source.nd2",
        tiff_directory=tmp_path / "raw",
        z_start_offset=1,
        target_exposure_ms=200.0,
        target_laser_percent=40.0,
        tubulin_exposure_ms=200.0,
        dapi_exposure_ms=200.0,
        dataset_manifest=tmp_path / "dataset.yaml",
        execution_profile=profile,
        workflow_dataset_id="synthetic",
        identity_method="exact-pixel-audit",
    )


def test_lamin_acquisition_round_one_validates_the_moving_artifact_chain(
    tmp_path: Path, complete_status_inspector: StatusInspector
) -> None:
    """A moving biological round must validate transforms even at workflow round one."""
    artifacts = resolve_round_artifacts(
        _lamin_round(
            tmp_path, acquisition_round=1, workflow_round=1, complete_moving=True
        ),
        complete_status_inspector,
    )
    assert artifacts.affine is not None
    assert artifacts.warp is not None
    assert artifacts.pre_metrics is not None
    assert artifacts.linear_metrics is not None
    assert artifacts.nonlinear_metrics is not None
    assert artifacts.chunk_manifest is not None
    assert len(artifacts.completion_records) == 9


def test_lamin_acquisition_round_two_skips_the_moving_artifact_chain(
    tmp_path: Path, complete_status_inspector: StatusInspector
) -> None:
    """The biological reference stays reference when its workflow number is two."""
    artifacts = resolve_round_artifacts(
        _lamin_round(
            tmp_path, acquisition_round=2, workflow_round=2, complete_moving=False
        ),
        complete_status_inspector,
    )
    assert artifacts.affine is None
    assert artifacts.warp is None
    assert artifacts.pre_metrics is None
    assert artifacts.linear_metrics is None
    assert artifacts.nonlinear_metrics is None
    assert artifacts.chunk_manifest is None
    assert len(artifacts.completion_records) == 3


def test_consumed_moving_artifact_requires_matching_completion_identity(
    tmp_path: Path, complete_status_inspector: StatusInspector
) -> None:
    """Changing a moving-round transform after completion must reject the round."""
    round_spec = _lamin_round(
        tmp_path, acquisition_round=1, workflow_round=1, complete_moving=True
    )
    affine = (
        tmp_path
        / "workflow-root/synthetic/rounds/round-001/registration/linear_transform_1.mat"
    )
    affine.write_bytes(affine.read_bytes() + b"changed")
    with pytest.raises(InputValidationError, match="completion identity mismatch"):
        resolve_round_artifacts(round_spec, complete_status_inspector)
