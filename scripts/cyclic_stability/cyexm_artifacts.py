"""Read-only validation of the Cy-ExM artifacts consumed by this analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cyexm.config.execution import load_execution_profile
from cyexm.errors import InputValidationError
from cyexm.registration.artifacts import ArtifactStore, CompletionRecord, FileIdentity
from cyexm.registration.whole_image import RegistrationPaths
from cyexm.workflow.layout import WorkflowLayout
from cyexm.workflow.models import WorkflowStatus
from cyexm.workflow.status import inspect_workflow_status

from cyclic_stability.models import CellSpec, RoundSpec
from cyclic_stability.provenance import ChannelMatch, compare_nd2_export

type StatusInspector = Callable[[str | Path, str], WorkflowStatus]


@dataclass(frozen=True, slots=True)
class RoundArtifacts:
    """Validated source channels and registration-chain artifacts for one round."""

    acquisition_round: int
    workflow_round: int
    workflow_output_root: Path
    registered_dapi: Path
    registered_tubulin: Path
    registered_target: Path
    affine: Path | None
    warp: Path | None
    pre_metrics: Path | None
    linear_metrics: Path | None
    nonlinear_metrics: Path | None
    chunk_manifest: Path | None
    completion_records: tuple[CompletionRecord, ...]
    native_target: Path | None = None
    moving_crop: Path | None = None


@dataclass(frozen=True, slots=True)
class CellValidationReport:
    """Complete source and Cy-ExM validation evidence for one cell."""

    cohort: str
    cell_id: str
    source_matches: tuple[ChannelMatch, ...]
    rounds: tuple[RoundArtifacts, ...]


def validate_completed_artifact(path: Path) -> CompletionRecord:
    """Validate an artifact against the immutable identity in its completion record."""
    store = ArtifactStore()
    record_path = store.record_path(path)
    if not path.is_file() or not record_path.is_file():
        raise InputValidationError(f"artifact or completion record missing: {path}")
    try:
        record = CompletionRecord.read(record_path)
        identity = FileIdentity.from_path(path)
    except (OSError, RuntimeError, TypeError, UnicodeError, ValueError) as error:
        raise InputValidationError(
            f"completion record is unreadable: {path}"
        ) from error
    if identity != record.artifact:
        raise InputValidationError(f"completion identity mismatch: {path}")
    return record


def require_complete_workflow(
    round_spec: RoundSpec,
    status_inspector: StatusInspector = inspect_workflow_status,
) -> WorkflowStatus:
    """Require the declared Cy-ExM workflow to be authoritatively complete."""
    profile = load_execution_profile(round_spec.execution_profile)
    status = status_inspector(profile.paths.output_root, round_spec.workflow_dataset_id)
    if not status.complete:
        raise InputValidationError(
            f"Cy-ExM workflow is not complete: {round_spec.workflow_dataset_id}"
        )
    return status


def _target_name(round_spec: RoundSpec) -> str:
    return {"golgi": "647_golgi", "lamin": "647_lamin", "lamp1": "647_lamp1"}[
        round_spec.cohort
    ]


def resolve_round_artifacts(
    round_spec: RoundSpec,
    status_inspector: StatusInspector = inspect_workflow_status,
) -> RoundArtifacts:
    """Resolve and validate every Cy-ExM artifact consumed for a round."""
    require_complete_workflow(round_spec, status_inspector)
    output_root = load_execution_profile(round_spec.execution_profile).paths.output_root
    round_layout = WorkflowLayout.for_dataset(
        output_root, round_spec.workflow_dataset_id
    ).round(round_spec.workflow_round)
    acquisition = round_spec.acquisition_round
    channels = (
        round_layout.channel(f"round_{acquisition}_405_DAPI.tif"),
        round_layout.channel(f"round_{acquisition}_488_tubulin.tif"),
        round_layout.channel(f"round_{acquisition}_{_target_name(round_spec)}.tif"),
    )
    required: list[Path] = list(channels)
    affine: Path | None = None
    warp: Path | None = None
    pre_metrics: Path | None = None
    linear_metrics: Path | None = None
    nonlinear_metrics: Path | None = None
    chunk_manifest: Path | None = None
    native_target: Path | None = None
    if round_spec.acquisition_round != 2:
        registration = RegistrationPaths.for_round(
            round_layout.registration, round_spec.workflow_round
        )
        affine = registration.affine
        warp = registration.warp
        pre_metrics = registration.pre_metrics
        linear_metrics = registration.linear_metrics
        nonlinear_metrics = registration.nonlinear_metrics
        chunk_manifest = (
            round_layout.registration
            / f"round_{round_spec.workflow_round}_chunks"
            / "manifest.json"
        )
        required.extend(
            (
                affine,
                warp,
                pre_metrics,
                linear_metrics,
                nonlinear_metrics,
                chunk_manifest,
            )
        )
    records = tuple(validate_completed_artifact(path) for path in required)
    target_record = records[2]
    if (
        target_record.request.stage == "channel-transform-v1"
        and len(target_record.request.inputs) == 4
    ):
        candidate = Path(target_record.request.inputs[1].path)
        if (
            candidate.is_file()
            and FileIdentity.from_path(candidate) == target_record.request.inputs[1]
        ):
            native_target = candidate
    return RoundArtifacts(
        acquisition_round=acquisition,
        workflow_round=round_spec.workflow_round,
        workflow_output_root=output_root,
        registered_dapi=channels[0],
        registered_tubulin=channels[1],
        registered_target=channels[2],
        affine=affine,
        warp=warp,
        pre_metrics=pre_metrics,
        linear_metrics=linear_metrics,
        nonlinear_metrics=nonlinear_metrics,
        chunk_manifest=chunk_manifest,
        completion_records=records,
        native_target=native_target,
        # Channel-transform-v1 records name the native source directly.  A stale
        # registration crop is not a transform input and must never be inferred.
        moving_crop=None,
    )


def validate_cell_inputs(cell: CellSpec) -> CellValidationReport:
    """Validate exact source correspondence and Cy-ExM provenance for one cell."""
    source_matches = tuple(
        match for round_spec in cell.rounds for match in compare_nd2_export(round_spec)
    )
    mismatched = [match.channel for match in source_matches if not match.exact]
    if mismatched:
        raise InputValidationError(
            f"ND2/TIFF exact source correspondence failed for: {', '.join(mismatched)}"
        )
    return CellValidationReport(
        cohort=cell.cohort,
        cell_id=cell.cell_id,
        source_matches=source_matches,
        rounds=tuple(resolve_round_artifacts(round_spec) for round_spec in cell.rounds),
    )


__all__ = [
    "CellValidationReport",
    "RoundArtifacts",
    "StatusInspector",
    "require_complete_workflow",
    "resolve_round_artifacts",
    "validate_cell_inputs",
    "validate_completed_artifact",
]
