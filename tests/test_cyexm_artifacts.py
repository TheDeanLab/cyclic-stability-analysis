"""Behavioral tests for the read-only Cy-ExM artifact boundary."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from cyclic_stability import cyexm_artifacts
from cyclic_stability.cyexm_artifacts import (
    StatusInspector,
    resolve_round_artifacts,
    snapshot_registered_dapi,
    validate_completed_artifact,
)
from cyclic_stability.models import RoundSpec
from cyexm.errors import InputValidationError
from cyexm.registration._request_policy import (
    channel_transform_parameters,
    reference_channel_parameters,
)
from cyexm.registration.artifacts import (
    ArtifactStore,
    CompletionRecord,
    FileIdentity,
    StageRequest,
)
from cyexm.spatial import ImageGeometry
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
        f"""schema_version: 1
executor: sequential
paths:
  dataset_root: /synthetic/raw
  output_root: {output_root}
  worker_local_directory: null
""",
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


def _write_registered_dapi(
    path: Path,
    *,
    moving: bool,
    cropped_reference: bool = False,
    shape: tuple[int, int, int] = (5, 7, 9),
) -> Path:
    """Write one realistic DAPI-only completion chain for inventory tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    source_root = path.parents[4] / "dapi-provenance"
    source_root.mkdir(parents=True, exist_ok=True)
    native = source_root / f"{path.stem}-native.tif"
    native.write_bytes(b"native-dapi")
    geometry = ImageGeometry(
        shape=shape,
        origin=(0.0, 0.0, 0.0),
        spacing=(0.2, 0.2, 0.5),
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    if moving:
        fixed = source_root / "fixed-registration-channel.tif"
        affine = source_root / "linear.mat"
        warp = source_root / "warp.nii.gz"
        fixed.write_bytes(b"fixed")
        affine.write_bytes(b"affine")
        warp.write_bytes(b"warp")
        request = StageRequest.for_files(
            stage="channel-transform-v1",
            inputs=(fixed, native, affine, warp),
            geometry=geometry,
            method="warp-affine",
            accuracy="not-applicable",
            native_threads={},
            upstream=FileIdentity.from_path(warp),
            parameters=channel_transform_parameters(
                pixel_size_xyz=geometry.spacing, interpolation="linear"
            ),
        )
    else:
        crop_bounds = source_root / "crop-bounds.json"
        if cropped_reference:
            crop_bounds.write_bytes(b"crop-bounds")
        request = StageRequest.for_files(
            stage="reference-channel",
            inputs=(native,),
            geometry=geometry,
            method="uint16-tiff",
            accuracy="exact",
            native_threads={},
            upstream=(
                FileIdentity.from_path(crop_bounds) if cropped_reference else None
            ),
            parameters=reference_channel_parameters(index=0, cropped=cropped_reference),
        )
    image = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    ArtifactStore().commit(
        path,
        request,
        lambda target: tifffile.imwrite(target, image),
        lambda target: tifffile.imread(target).shape == shape,
    )
    return path


def test_dapi_snapshot_accepts_cropped_reference_and_binds_crop_provenance(
    tmp_path: Path,
) -> None:
    """A valid cropped fixed DAPI must retain its authoritative crop upstream."""
    output_root = tmp_path / "workflow-root"
    profile = tmp_path / "execution.yaml"
    profile.write_text(
        f"""schema_version: 1
executor: sequential
paths:
  dataset_root: /synthetic/raw
  output_root: {output_root}
  worker_local_directory: null
""",
        encoding="utf-8",
    )
    round_spec = replace(
        _lamin_round(
            tmp_path, acquisition_round=2, workflow_round=1, complete_moving=False
        ),
        execution_profile=profile,
    )
    dapi = _write_registered_dapi(
        output_root / "synthetic/rounds/round-001/channels/round_2_405_DAPI.tif",
        moving=False,
        cropped_reference=True,
    )

    snapshot = snapshot_registered_dapi(round_spec)

    assert snapshot.path == dapi.resolve()
    assert snapshot.record.request.parameters["cropped"] is True
    assert snapshot.record.request.upstream is not None
    assert any(
        Path(item.path).name == "crop-bounds.json" for item in snapshot.identities
    )


@pytest.mark.parametrize("target_state", ("missing", "invalid"))
def test_dapi_snapshot_ignores_missing_or_invalid_target_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_state: str,
) -> None:
    """DAPI inventory must not inherit target-specific workflow completeness."""
    output_root = tmp_path / "workflow-root"
    profile = tmp_path / "execution.yaml"
    profile.write_text(
        f"""schema_version: 1
executor: sequential
paths:
  dataset_root: /synthetic/raw
  output_root: {output_root}
  worker_local_directory: null
""",
        encoding="utf-8",
    )
    round_root = output_root / "synthetic" / "rounds" / "round-003"
    dapi = _write_registered_dapi(
        round_root / "channels" / "round_3_405_DAPI.tif", moving=True
    )
    target = round_root / "channels" / "round_3_647_lamin.tif"
    if target_state == "invalid":
        target.write_bytes(b"invalid-target-without-completion")
    round_spec = RoundSpec(
        cohort="lamin",
        cell_id="Cell1",
        acquisition_round=3,
        workflow_round=3,
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
    monkeypatch.setattr(
        cyexm_artifacts,
        "inspect_workflow_status",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("DAPI-only inventory consulted whole-workflow status")
        ),
    )

    snapshot = snapshot_registered_dapi(round_spec)

    assert snapshot.path == dapi.resolve()
    assert snapshot.geometry.shape == (5, 7, 9)
    assert snapshot.record.request.stage == "channel-transform-v1"
    assert all("647_lamin" not in identity.path for identity in snapshot.identities)


def test_dapi_snapshot_rejects_non_authoritative_channel_stage(tmp_path: Path) -> None:
    """A generic artifact record cannot attest to registered DAPI provenance."""
    output_root = tmp_path / "workflow-root"
    profile = tmp_path / "execution.yaml"
    profile.write_text(
        f"""schema_version: 1
executor: sequential
paths:
  dataset_root: /synthetic/raw
  output_root: {output_root}
  worker_local_directory: null
""",
        encoding="utf-8",
    )
    dapi = output_root / "synthetic/rounds/round-003/channels/round_3_405_DAPI.tif"
    write_completion_fixture(dapi, b"not-a-channel-transform")
    round_spec = replace(
        _lamin_round(
            tmp_path, acquisition_round=3, workflow_round=3, complete_moving=False
        ),
        execution_profile=profile,
    )

    with pytest.raises(InputValidationError, match="DAPI completion stage"):
        snapshot_registered_dapi(round_spec)


def test_symlinked_completion_record_is_rejected(completed_artifact: Path) -> None:
    """A completion record path must not follow a retargetable symlink."""
    record_path = ArtifactStore().record_path(completed_artifact)
    real_record = record_path.with_name("saved-completion-record.json")
    record_path.rename(real_record)
    record_path.symlink_to(real_record)

    with pytest.raises(InputValidationError, match="completion record is unreadable"):
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
    assert artifacts.native_target is None
    assert artifacts.moving_crop is None
    assert artifacts.completion_record_paths == tuple(
        ArtifactStore().record_path(Path(record.artifact.path))
        for record in artifacts.completion_records
    )
    assert (
        artifacts.workflow_output_root
        == golgi_round_two.execution_profile.parent / "workflow-root"
    )


def test_execution_profile_rewrite_during_status_cannot_select_another_root(
    golgi_round_two: RoundSpec,
) -> None:
    """Status and artifact selection must consume one exact profile snapshot."""
    profile = golgi_round_two.execution_profile
    root_a = profile.parent / "workflow-root"
    root_b = profile.parent / "workflow-root-b"
    channels_b = root_b / "synthetic" / "rounds" / "round-001" / "channels"
    for filename in (
        "round_2_405_DAPI.tif",
        "round_2_488_tubulin.tif",
        "round_2_647_golgi.tif",
    ):
        write_completion_fixture(channels_b / filename, f"root-b-{filename}".encode())
    original_profile = profile.read_bytes()
    replacement_profile = original_profile.replace(
        str(root_a).encode(), str(root_b).encode()
    )
    assert replacement_profile != original_profile
    inspected_roots: list[Path] = []

    def rewrite_after_status_root_selected(output_root: str | Path, dataset_id: str):
        inspected_roots.append(Path(output_root))
        profile.write_bytes(replacement_profile)
        return WorkflowStatus(
            dataset_id=dataset_id,
            scientific_fingerprint="synthetic",
            rounds=(),
            complete=True,
        )

    artifacts = resolve_round_artifacts(
        golgi_round_two, rewrite_after_status_root_selected
    )

    assert inspected_roots == [root_a.resolve()]
    assert artifacts.workflow_output_root == root_a.resolve()
    assert (
        artifacts.registered_dapi.parent
        == (root_a / "synthetic" / "rounds" / "round-001" / "channels").resolve()
    )
    assert artifacts.execution_profile_identity is not None
    assert Path(artifacts.execution_profile_identity.path) == profile.resolve()
    assert artifacts.execution_profile_identity.matches_path() is False


def _golgi_moving_round_with_native_source(
    tmp_path: Path,
) -> tuple[RoundSpec, Path, Path, Path, Path]:
    """Create one complete moving round and an alternate valid target record."""
    output_root = tmp_path / "workflow-root"
    profile = tmp_path / "execution-moving.yaml"
    profile.write_text(
        f"""schema_version: 1
executor: sequential
paths:
  dataset_root: /synthetic/raw
  output_root: {output_root}
  worker_local_directory: null
""",
        encoding="utf-8",
    )
    round_root = output_root / "synthetic" / "rounds" / "round-003"
    channels = round_root / "channels"
    fixed = write_completion_fixture(
        channels / "round_3_405_DAPI.tif", b"registered-dapi"
    )
    write_completion_fixture(
        channels / "round_3_488_tubulin.tif", b"registered-tubulin"
    )
    registered_target = channels / "round_3_647_golgi.tif"
    registration = round_root / "registration"
    affine = write_completion_fixture(
        registration / "linear_transform_3.mat", b"affine"
    )
    warp = write_completion_fixture(registration / "nonlinear_warp_3.nii.gz", b"warp")
    for filename in (
        "image_metrics_pre-registration.txt",
        "image_metrics_linearly-registered.txt",
        "image_metrics_nonlinear-registered.txt",
        "round_3_chunks/manifest.json",
    ):
        write_completion_fixture(registration / filename, filename.encode())
    native_a = tmp_path / "native-a.tif"
    native_b = tmp_path / "native-b.tif"
    native_a.write_bytes(b"native-a")
    native_b.write_bytes(b"native-b")
    geometry = ImageGeometry(
        shape=(1, 1, 1),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )

    def request(source: Path) -> StageRequest:
        return StageRequest.for_files(
            stage="channel-transform-v1",
            inputs=(fixed, source, affine, warp),
            geometry=geometry,
            method="warp-affine",
            accuracy="not-applicable",
            native_threads={"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": 1},
            upstream=FileIdentity.from_path(warp),
            parameters=channel_transform_parameters(
                pixel_size_xyz=(1.0, 1.0, 1.0), interpolation="linear"
            ),
        )

    store = ArtifactStore()
    store.commit(
        registered_target,
        request(native_a),
        lambda path: path.write_bytes(b"registered-target"),
        lambda path: path.read_bytes() == b"registered-target",
    )
    record_path = store.record_path(registered_target)
    replacement = tmp_path / "replacement-completion-record.json"
    replacement.write_text(
        CompletionRecord(
            1, request(native_b), FileIdentity.from_path(registered_target)
        ).to_json(),
        encoding="utf-8",
    )
    round_spec = RoundSpec(
        cohort="golgi",
        cell_id="Cell1",
        acquisition_round=3,
        workflow_round=3,
        nd2_path=tmp_path / "round3.nd2",
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
    return round_spec, record_path, replacement, native_a, native_b


def test_completion_record_replacement_during_parse_is_rejected(
    tmp_path: Path,
    complete_status_inspector: StatusInspector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record parsing and hashing must describe one no-follow file snapshot."""
    round_spec, record_path, replacement, native_a, _native_b = (
        _golgi_moving_round_with_native_source(tmp_path)
    )
    original_loads = json.loads
    replaced = False

    def replace_after_read(payload, *args, **kwargs):
        nonlocal replaced
        text = payload.decode() if isinstance(payload, bytes) else str(payload)
        if not replaced and str(native_a.resolve()) in text:
            replaced = True
            os.replace(replacement, record_path)
        return original_loads(payload, *args, **kwargs)

    monkeypatch.setattr(json, "loads", replace_after_read)

    with pytest.raises(InputValidationError, match="completion record is unreadable"):
        resolve_round_artifacts(round_spec, complete_status_inspector)

    assert replaced is True


def test_native_source_retarget_during_snapshot_is_not_selected(
    tmp_path: Path,
    complete_status_inspector: StatusInspector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected native source must keep one directory entry for its exact hash."""
    round_spec, _record_path, _replacement, native_a, _native_b = (
        _golgi_moving_round_with_native_source(tmp_path)
    )
    retarget = tmp_path / "native-retarget.tif"
    retarget.write_bytes(b"retarget")
    original_read = os.read
    replaced = False

    def retarget_after_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        payload = original_read(descriptor, size)
        selected = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        if not replaced and selected == native_a.resolve():
            replaced = True
            os.replace(retarget, native_a)
        return payload

    monkeypatch.setattr(os, "read", retarget_after_read)

    artifacts = resolve_round_artifacts(round_spec, complete_status_inspector)

    assert replaced is True
    assert artifacts.native_target is None
    assert artifacts.native_target_identity is None


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
        f"""schema_version: 1
executor: sequential
paths:
  dataset_root: /synthetic/raw
  output_root: {output_root}
  worker_local_directory: null
""",
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
