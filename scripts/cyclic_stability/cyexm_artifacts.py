"""Read-only validation of the Cy-ExM artifacts consumed by this analysis."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml
from cyexm.config import execution as cyexm_execution
from cyexm.config import load_dataset_manifest
from cyexm.config._validation import require_exact_keys
from cyexm.config._yaml import _StrictSafeLoader, _yaml_error_detail
from cyexm.config.models import (
    DatasetManifest,
    ExecutionProfile,
    ExecutorName,
    ParallelismSpec,
)
from cyexm.errors import (
    ConfigurationValidationError,
    InputValidationError,
    ManifestDecodeError,
)
from cyexm.registration.artifacts import (
    ArtifactStore,
    CompletionRecord,
    FileIdentity,
    StageRequest,
)
from cyexm.registration.whole_image import RegistrationPaths
from cyexm.spatial import ImageGeometry
from cyexm.workflow.layout import WorkflowLayout
from cyexm.workflow.models import WorkflowStatus
from cyexm.workflow.status import inspect_workflow_status

from cyclic_stability.models import AnalysisConfig, CellSpec, RoundSpec
from cyclic_stability.provenance import (
    ChannelMatch,
    channel_tiff_paths,
    compare_nd2_export,
)
from cyclic_stability.task import ContentIdentity, RetainedContent

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
    completion_record_paths: tuple[Path, ...] = ()
    completion_record_identities: tuple[ContentIdentity, ...] = ()
    native_target_identity: ContentIdentity | None = None
    execution_profile_identity: ContentIdentity | None = None
    artifact_identities: tuple[ContentIdentity, ...] = ()


@dataclass(frozen=True, slots=True)
class CellValidationReport:
    """Complete source and Cy-ExM validation evidence for one cell."""

    cohort: str
    cell_id: str
    source_matches: tuple[ChannelMatch, ...]
    rounds: tuple[RoundArtifacts, ...]


@dataclass(frozen=True, slots=True)
class DapiArtifactSnapshot:
    """Exact DAPI-only artifact, completion, geometry, and provenance snapshot."""

    path: Path
    record_path: Path
    record: CompletionRecord
    geometry: ImageGeometry
    identities: tuple[ContentIdentity, ...]
    content: RetainedContent

    @property
    def consumer_path(self) -> Path:
        """Return the descriptor path for decoding the exact attested DAPI bytes."""
        return self.content.consumer_path

    def close(self) -> None:
        """Close the retained DAPI descriptor when ownership was not transferred."""
        self.content.close()


@dataclass(frozen=True, slots=True)
class _ManifestDapiContract:
    """The exact DAPI channel position and producer source declared by Cy-ExM."""

    channel_index: int
    producer_source: Path


def _completion_record_from_bytes(payload: bytes) -> CompletionRecord:
    """Parse the exact completion-record bytes already hashed by the caller."""
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, Mapping) or set(decoded) != {
        "schema_version",
        "request",
        "artifact",
    }:
        raise ValueError("completion record has invalid fields")
    schema_version = decoded["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError("completion record schema_version must be an integer")
    request = decoded["request"]
    artifact = decoded["artifact"]
    if not isinstance(request, Mapping) or not isinstance(artifact, Mapping):
        raise TypeError("completion record request and artifact must be mappings")
    return CompletionRecord(
        schema_version=schema_version,
        request=StageRequest.from_dict(request),
        artifact=FileIdentity.from_dict(artifact, name="artifact identity"),
    )


def _execution_profile_from_bytes(
    payload: bytes, source_path: Path
) -> ExecutionProfile:
    """Parse one Cy-ExM profile from the exact bytes already hashed by the caller."""
    loader: _StrictSafeLoader | None = None
    try:
        loader = _StrictSafeLoader(payload.decode("utf-8"))
        value = loader.get_single_data()
        if not isinstance(value, Mapping):
            raise yaml.YAMLError("YAML document must contain a top-level mapping")
        raw_mapping = cast(Mapping[str, object], value)
    except UnicodeDecodeError as error:
        raise ManifestDecodeError(
            f"Could not decode YAML from {source_path}: {error}"
        ) from error
    except yaml.YAMLError as error:
        raise ManifestDecodeError(
            f"Could not decode YAML from {source_path}: {_yaml_error_detail(error)}"
        ) from error
    finally:
        if loader is not None:
            loader.dispose()

    raw_executor = raw_mapping.get("executor")
    executor = cast(
        ExecutorName,
        cyexm_execution._choice(  # noqa: SLF001 - pinned Cy-ExM parser adapter
            raw_executor,
            cyexm_execution._EXECUTORS,
            "executor",  # noqa: SLF001
        ),
    )
    expected_keys = cyexm_execution._BASE_EXECUTION_KEYS  # noqa: SLF001
    if "native_threads" in raw_mapping:
        expected_keys |= frozenset({"native_threads"})
    if "parallelism" in raw_mapping:
        expected_keys |= frozenset({"parallelism"})
    if executor != "sequential":
        expected_keys |= frozenset({executor})
    mapping = require_exact_keys(raw_mapping, expected_keys, "execution")

    schema_version = mapping["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise ConfigurationValidationError(
            f"schema_version must be the built-in integer 1; got {schema_version!r}"
        )
    native_threads = (
        cyexm_execution._native_threads(mapping["native_threads"])  # noqa: SLF001
        if "native_threads" in mapping
        else cyexm_execution._resolve_inherited_native_threads(  # noqa: SLF001
            os.environ
        )
    )
    profile = ExecutionProfile(
        schema_version=schema_version,
        executor=executor,
        paths=cyexm_execution._paths(  # noqa: SLF001
            mapping["paths"],
            source_path.parent,
            require_worker_scratch=executor != "sequential",
        ),
        native_threads=native_threads,
        local_dask=(
            cyexm_execution._local_dask(mapping["local_dask"])  # noqa: SLF001
            if executor == "local_dask"
            else None
        ),
        slurm=(
            cyexm_execution._slurm(mapping["slurm"])  # noqa: SLF001
            if executor == "slurm"
            else None
        ),
        source_path=source_path,
        parallelism=(
            cyexm_execution._parallelism(mapping["parallelism"])  # noqa: SLF001
            if "parallelism" in mapping
            else ParallelismSpec()
        ),
    )
    cyexm_execution._validate_execution_profile(profile)  # noqa: SLF001
    return profile


def _execution_profile_snapshot(
    path: Path,
) -> tuple[ExecutionProfile, ContentIdentity]:
    """Hash and parse one canonical execution-profile snapshot without reloading it."""
    identity, payload = ContentIdentity.read_bytes(path)
    profile = _execution_profile_from_bytes(payload, Path(identity.path))
    return profile, identity


def _dataset_manifest_snapshot(
    path: Path,
) -> tuple[DatasetManifest, ContentIdentity]:
    """Hash and parse the same no-follow dataset-manifest descriptor bytes."""
    retained: RetainedContent | None = None
    try:
        retained = RetainedContent.from_path(path)
        manifest = load_dataset_manifest(retained.consumer_path)
        retained.validate()
        if not retained.identity.matches_path():
            raise ValueError("dataset manifest changed during parsing")
        identity = retained.identity
        retained.close()
        return manifest, identity
    except (
        ConfigurationValidationError,
        ManifestDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as error:
        if retained is not None:
            retained.close()
        raise InputValidationError("dataset manifest is unreadable") from error
    except BaseException:
        if retained is not None:
            retained.close()
        raise


def _manifest_dapi_contract(
    round_spec: RoundSpec,
    config: AnalysisConfig,
    profile: ExecutionProfile,
    manifest: DatasetManifest,
) -> _ManifestDapiContract:
    """Resolve DAPI by the pinned manifest channel order and source semantics."""
    if manifest.dataset_id != round_spec.workflow_dataset_id:
        raise InputValidationError(
            "dataset manifest identity does not match the declared workflow"
        )
    if tuple(manifest.voxel_spacing_um) != tuple(config.voxel_spacing_xyz_um):
        raise InputValidationError(
            "dataset manifest pixel spacing does not match analysis configuration"
        )

    is_reference = round_spec.acquisition_round == config.reference_acquisition_round
    if is_reference:
        if round_spec.workflow_round != manifest.reference.round:
            raise InputValidationError(
                "dataset manifest reference round does not match the declared workflow"
            )
        channels = manifest.reference.channels
    else:
        round_matches = tuple(
            item for item in manifest.rounds if item.round == round_spec.workflow_round
        )
        if len(round_matches) != 1:
            raise InputValidationError(
                "dataset manifest moving round does not match the declared workflow"
            )
        channels = round_matches[0].channels

    correspondence_source = dict(channel_tiff_paths(round_spec))["dapi"]
    expected_file = correspondence_source.name
    matches = tuple(
        (index, channel)
        for index, channel in enumerate(channels)
        if channel.file.as_posix() == expected_file
    )
    if len(matches) != 1:
        raise InputValidationError(
            "dataset manifest does not declare exactly one expected DAPI channel"
        )
    channel_index, channel = matches[0]
    if not is_reference and channel.interpolation != "linear":
        raise InputValidationError(
            "dataset manifest DAPI channel interpolation is not authoritative"
        )
    producer_source = profile.paths.dataset_root.joinpath(*channel.file.parts)
    return _ManifestDapiContract(channel_index, producer_source)


def _retained_tiff_geometry(content: RetainedContent) -> tuple[tuple[int, ...], str]:
    """Read one scalar TIFF series geometry from the exact retained bytes."""
    import tifffile

    with content.open_binary() as source, tifffile.TiffFile(source) as handle:
        if len(handle.series) != 1:
            raise ValueError("DAPI source must contain exactly one TIFF series")
        series = handle.series[0]
        shape = tuple(int(value) for value in series.shape)
        if len(shape) != 3 or series.dtype.kind not in "uif":
            raise ValueError("DAPI source must be one scalar three-dimensional TIFF")
        return shape, str(series.dtype)


def _retained_tiff_planes_equal(
    producer: RetainedContent,
    correspondence: RetainedContent,
    z_start_offset: int,
) -> bool:
    """Compare exact decoded pixels one independently stored Z plane at a time."""
    import numpy as np
    import tifffile

    with (
        producer.open_binary() as producer_source,
        correspondence.open_binary() as correspondence_source,
        tifffile.TiffFile(producer_source) as producer_handle,
        tifffile.TiffFile(correspondence_source) as correspondence_handle,
    ):
        producer_series = producer_handle.series[0]
        correspondence_series = correspondence_handle.series[0]
        producer_shape = tuple(int(value) for value in producer_series.shape)
        correspondence_shape = tuple(
            int(value) for value in correspondence_series.shape
        )
        producer_pages = producer_series.pages
        correspondence_pages = correspondence_series.pages
        if (
            len(producer_pages) != producer_shape[0]
            or len(correspondence_pages) != correspondence_shape[0]
            or any(
                tuple(int(value) for value in page.shape) != producer_shape[1:]
                for page in producer_pages
            )
            or any(
                tuple(int(value) for value in page.shape) != correspondence_shape[1:]
                for page in correspondence_pages
            )
        ):
            raise ValueError(
                "DAPI source TIFF must expose one independently decodable page "
                "per Z plane"
            )
        for producer_index, producer_page in enumerate(producer_pages):
            correspondence_page = correspondence_pages[producer_index + z_start_offset]
            producer_pixels = producer_page.asarray()
            correspondence_pixels = correspondence_page.asarray()
            if not np.array_equal(producer_pixels, correspondence_pixels):
                return False
        return True


def _validated_dapi_source_copies(
    producer_path: Path,
    correspondence_path: Path,
    producer_source_z_start_offset: int | None,
) -> tuple[ContentIdentity, ContentIdentity]:
    """Bind producer pixels to the declared correspondence leading-Z transform."""
    producer: RetainedContent | None = None
    correspondence: RetainedContent | None = None
    try:
        producer = RetainedContent.from_path(producer_path)
        correspondence = RetainedContent.from_path(correspondence_path)
        producer_geometry = _retained_tiff_geometry(producer)
        correspondence_geometry = _retained_tiff_geometry(correspondence)
        correspondence_shape, correspondence_dtype = correspondence_geometry
        if producer_source_z_start_offset is None:
            if producer_geometry != correspondence_geometry:
                raise InputValidationError(
                    "producer_source_z_start_offset is unresolved for "
                    "non-equivalent DAPI source copies; attest this field "
                    "before analysis"
                )
            effective_offset = 0
        else:
            effective_offset = producer_source_z_start_offset
            if effective_offset < 0 or effective_offset >= correspondence_shape[0]:
                raise InputValidationError(
                    "producer_source_z_start_offset is outside correspondence TIFF Z"
                )
            expected_producer_geometry = (
                (
                    correspondence_shape[0] - effective_offset,
                    *correspondence_shape[1:],
                ),
                correspondence_dtype,
            )
            if producer_geometry != expected_producer_geometry:
                raise InputValidationError(
                    "DAPI source-copy geometry does not match declared "
                    "producer_source_z_start_offset"
                )
        pixels_equal = (
            effective_offset == 0
            and producer.identity.sha256 == correspondence.identity.sha256
        ) or _retained_tiff_planes_equal(
            producer,
            correspondence,
            effective_offset,
        )
        if not pixels_equal and producer_source_z_start_offset is None:
            raise InputValidationError(
                "producer_source_z_start_offset is unresolved for non-equivalent "
                "DAPI source copies; attest this field before analysis"
            )
        if not pixels_equal:
            raise InputValidationError(
                "DAPI source-copy pixels do not match declared "
                "producer_source_z_start_offset"
            )
        producer.validate()
        correspondence.validate()
        return producer.identity, correspondence.identity
    except InputValidationError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise InputValidationError("DAPI completion source is unreadable") from error
    finally:
        if producer is not None:
            producer.close()
        if correspondence is not None:
            correspondence.close()


def _validated_completion_snapshot(
    path: Path,
) -> tuple[CompletionRecord, Path, ContentIdentity, ContentIdentity]:
    """Return one parsed record and identity from the same no-follow snapshot."""
    store = ArtifactStore()
    record_path = store.record_path(path)
    if not path.is_file() or not record_path.is_file():
        raise InputValidationError(f"artifact or completion record missing: {path}")
    try:
        record_identity, record_bytes = ContentIdentity.read_bytes(record_path)
        record = _completion_record_from_bytes(record_bytes)
        if not record_identity.matches_path():
            raise ValueError("completion record changed during parsing")
        artifact_identity = ContentIdentity.from_path(path)
        identity = FileIdentity(
            artifact_identity.path,
            artifact_identity.st_size,
            artifact_identity.st_mtime_ns,
        )
    except (OSError, RuntimeError, TypeError, UnicodeError, ValueError) as error:
        raise InputValidationError(
            f"completion record is unreadable: {path}"
        ) from error
    if identity != record.artifact:
        raise InputValidationError(f"completion identity mismatch: {path}")
    return record, Path(record_identity.path), record_identity, artifact_identity


def validate_completed_artifact(path: Path) -> CompletionRecord:
    """Validate an artifact against the immutable identity in its completion record."""
    record, _, _, _ = _validated_completion_snapshot(path)
    return record


def _validated_retained_completion_snapshot(
    path: Path,
) -> tuple[CompletionRecord, Path, ContentIdentity, RetainedContent]:
    """Parse one completion record while retaining its exact artifact descriptor."""
    retained: RetainedContent | None = None
    try:
        retained = RetainedContent.from_path(path)
        canonical = Path(retained.identity.path)
        record_path = ArtifactStore().record_path(canonical)
        record_identity, record_bytes = ContentIdentity.read_bytes(record_path)
        record = _completion_record_from_bytes(record_bytes)
        retained.validate()
        if not record_identity.matches_path():
            raise ValueError("completion record changed during parsing")
        identity = FileIdentity(
            retained.identity.path,
            retained.identity.st_size,
            retained.identity.st_mtime_ns,
        )
        if identity != record.artifact:
            raise InputValidationError(f"completion identity mismatch: {canonical}")
        return record, Path(record_identity.path), record_identity, retained
    except InputValidationError:
        if retained is not None:
            retained.close()
        raise
    except (OSError, RuntimeError, TypeError, UnicodeError, ValueError) as error:
        if retained is not None:
            retained.close()
        raise InputValidationError(
            f"completion record is unreadable: {path}"
        ) from error
    except BaseException:
        if retained is not None:
            retained.close()
        raise


def snapshot_registered_dapi(
    round_spec: RoundSpec,
    config: AnalysisConfig,
    *,
    expected_fixed_geometry: ImageGeometry | None = None,
) -> DapiArtifactSnapshot:
    """Validate only one registered DAPI artifact and its necessary provenance.

    This boundary deliberately does not inspect whole-workflow status, tubulin
    channel completion, or the target-channel artifact. Moving DAPI must carry the
    exact channel-transform request; reference DAPI must carry the exact reference
    channel stage. The returned identities include the execution profile, dataset
    manifest, DAPI completion record, DAPI artifact, every input named by that DAPI
    request, and the independently declared correspondence source.

    Parameters
    ----------
    round_spec : RoundSpec
        One configured acquisition/workflow-round mapping.
    config : AnalysisConfig
        Authoritative reference-round and exact voxel-spacing contract.
    expected_fixed_geometry : cyexm.spatial.ImageGeometry or None, keyword-only
        Optional reference-round grid that this round must match exactly.

    Returns
    -------
    DapiArtifactSnapshot
        Canonical DAPI path, parsed authoritative record, fixed-grid geometry, and
        complete content identities needed by DAPI-only inventory.

    Raises
    ------
    InputValidationError
        If the DAPI artifact, completion stage, request inputs, or TIFF geometry is
        inconsistent. Target-channel state is never consulted.
    """
    import tifffile
    from cyexm.registration._request_policy import (
        channel_transform_parameters,
        reference_channel_parameters,
    )

    profile, profile_identity = _execution_profile_snapshot(
        round_spec.execution_profile
    )
    manifest, manifest_identity = _dataset_manifest_snapshot(
        round_spec.dataset_manifest
    )
    manifest_contract = _manifest_dapi_contract(round_spec, config, profile, manifest)
    correspondence_source = dict(channel_tiff_paths(round_spec))["dapi"]
    round_layout = WorkflowLayout.for_dataset(
        profile.paths.output_root, round_spec.workflow_dataset_id
    ).round(round_spec.workflow_round)
    expected = round_layout.channel(
        f"round_{round_spec.acquisition_round}_405_DAPI.tif"
    )
    record, record_path, record_identity, retained = (
        _validated_retained_completion_snapshot(expected)
    )
    try:
        canonical = Path(retained.identity.path)
        request = record.request
        expected_spacing = tuple(config.voxel_spacing_xyz_um)
        is_reference = (
            round_spec.acquisition_round == config.reference_acquisition_round
        )
        source_index = 0 if is_reference else 1
        if len(request.inputs) <= source_index:
            raise InputValidationError(
                "DAPI completion does not declare its authoritative source"
            )
        producer_source_identity, correspondence_source_identity = (
            _validated_dapi_source_copies(
                Path(request.inputs[source_index].path),
                correspondence_source,
                round_spec.producer_source_z_start_offset,
            )
        )
        try:
            manifest_source = manifest_contract.producer_source.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise InputValidationError(
                "dataset manifest DAPI source is unreadable"
            ) from error
        if Path(producer_source_identity.path) != manifest_source:
            raise InputValidationError(
                "DAPI completion source does not match the dataset manifest"
            )
        if tuple(request.geometry.spacing) != expected_spacing:
            raise InputValidationError(
                "DAPI request pixel spacing does not match analysis configuration"
            )
        if (
            expected_fixed_geometry is not None
            and request.geometry != expected_fixed_geometry
        ):
            raise InputValidationError(
                "DAPI request does not match the exact fixed output grid"
            )
        dapi_index = manifest_contract.channel_index
        expected_source = FileIdentity(
            producer_source_identity.path,
            producer_source_identity.st_size,
            producer_source_identity.st_mtime_ns,
        )
        if is_reference:
            cropped = request.parameters.get("cropped")
            reference_parameters_are_exact = isinstance(cropped, bool) and dict(
                request.parameters
            ) == reference_channel_parameters(index=dapi_index, cropped=cropped)
            if (
                request.stage != "reference-channel"
                or request.method != "uint16-tiff"
                or request.accuracy != "exact"
                or len(request.inputs) != 1
                or request.inputs[0] != expected_source
                or not reference_parameters_are_exact
                or (cropped is True) != (request.upstream is not None)
            ):
                raise InputValidationError(
                    "DAPI completion stage, channel, or source is not an "
                    "authoritative reference channel"
                )
        else:
            expected_parameters = channel_transform_parameters(
                pixel_size_xyz=config.voxel_spacing_xyz_um,
                interpolation="linear",
            )
            if (
                request.stage != "channel-transform-v1"
                or request.method != "warp-affine"
                or request.accuracy != "not-applicable"
                or len(request.inputs) != 4
                or request.inputs[1] != expected_source
                or request.upstream != request.inputs[3]
                or dict(request.parameters) != dict(expected_parameters)
            ):
                raise InputValidationError(
                    "DAPI completion stage, channel, or source is not an "
                    "authoritative channel transform"
                )
        input_identities: list[ContentIdentity] = []
        declared_provenance = list(request.inputs)
        if request.upstream is not None and request.upstream not in declared_provenance:
            declared_provenance.append(request.upstream)
        for declared in declared_provenance:
            if declared.path == producer_source_identity.path:
                identity = producer_source_identity
            else:
                try:
                    identity = ContentIdentity.from_path(Path(declared.path))
                except (OSError, ValueError) as error:
                    raise InputValidationError(
                        "DAPI completion input provenance is unreadable"
                    ) from error
            metadata = FileIdentity(
                identity.path, identity.st_size, identity.st_mtime_ns
            )
            if metadata != declared:
                raise InputValidationError(
                    "DAPI completion input provenance identity mismatch"
                )
            input_identities.append(identity)
        if producer_source_identity not in input_identities:
            raise InputValidationError(
                "DAPI completion source identity is not retained"
            )
        with retained.open_binary() as source, tifffile.TiffFile(source) as handle:
            if len(handle.series) != 1:
                raise ValueError("registered DAPI must contain exactly one TIFF series")
            actual_shape = tuple(int(value) for value in handle.series[0].shape)
            actual_dtype = handle.series[0].dtype
        if (
            actual_shape != tuple(request.geometry.shape)
            or actual_dtype.kind not in "uif"
        ):
            raise InputValidationError(
                "registered DAPI TIFF does not match authoritative fixed-grid geometry"
            )
        retained.validate()
        identities = (
            profile_identity,
            manifest_identity,
            record_identity,
            retained.identity,
            *input_identities,
            correspondence_source_identity,
        )
        if not all(identity.matches_path() for identity in identities):
            raise InputValidationError(
                "registered DAPI provenance changed during snapshot"
            )
        return DapiArtifactSnapshot(
            canonical,
            record_path,
            record,
            request.geometry,
            identities,
            retained,
        )
    except InputValidationError:
        retained.close()
        raise
    except (OSError, ValueError, tifffile.TiffFileError) as error:
        retained.close()
        raise InputValidationError("registered DAPI TIFF is unreadable") from error
    except BaseException:
        retained.close()
        raise


def require_complete_workflow(
    round_spec: RoundSpec,
    status_inspector: StatusInspector = inspect_workflow_status,
    *,
    profile: ExecutionProfile | None = None,
) -> WorkflowStatus:
    """Require the declared Cy-ExM workflow to be authoritatively complete."""
    if profile is None:
        profile, _ = _execution_profile_snapshot(round_spec.execution_profile)
    status = status_inspector(profile.paths.output_root, round_spec.workflow_dataset_id)
    if not status.complete:
        raise InputValidationError(
            f"Cy-ExM workflow is not complete: {round_spec.workflow_dataset_id}"
        )
    return status


def _target_name(round_spec: RoundSpec) -> str:
    return {"golgi": "647_golgi", "lamin": "647_lamin", "lamp1": "647_LAMP1"}[
        round_spec.cohort
    ]


def resolve_round_artifacts(
    round_spec: RoundSpec,
    status_inspector: StatusInspector = inspect_workflow_status,
) -> RoundArtifacts:
    """Resolve and validate every Cy-ExM artifact consumed for a round."""
    profile, profile_identity = _execution_profile_snapshot(
        round_spec.execution_profile
    )
    require_complete_workflow(round_spec, status_inspector, profile=profile)
    output_root = profile.paths.output_root
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
    validated = tuple(_validated_completion_snapshot(path) for path in required)
    records = tuple(item[0] for item in validated)
    record_paths = tuple(item[1] for item in validated)
    record_identities = tuple(item[2] for item in validated)
    artifact_identities = tuple(item[3] for item in validated)
    canonical_required = tuple(Path(identity.path) for identity in artifact_identities)
    channels = cast(tuple[Path, Path, Path], canonical_required[:3])
    if acquisition != 2:
        (
            affine,
            warp,
            pre_metrics,
            linear_metrics,
            nonlinear_metrics,
            chunk_manifest,
        ) = canonical_required[3:]
    target_record = records[2]
    native_target_identity: ContentIdentity | None = None
    if (
        target_record.request.stage == "channel-transform-v1"
        and len(target_record.request.inputs) == 4
    ):
        candidate = Path(target_record.request.inputs[1].path)
        try:
            candidate_identity = ContentIdentity.from_path(candidate)
        except (OSError, ValueError):
            pass
        else:
            metadata = FileIdentity(
                candidate_identity.path,
                candidate_identity.st_size,
                candidate_identity.st_mtime_ns,
            )
            if metadata == target_record.request.inputs[1]:
                native_target = Path(candidate_identity.path)
                native_target_identity = candidate_identity
    if not all(identity.matches_path() for identity in record_identities):
        raise InputValidationError(f"completion record is unreadable: {channels[2]}")
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
        completion_record_paths=record_paths,
        completion_record_identities=record_identities,
        native_target_identity=native_target_identity,
        execution_profile_identity=profile_identity,
        artifact_identities=artifact_identities,
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
    "DapiArtifactSnapshot",
    "RoundArtifacts",
    "StatusInspector",
    "require_complete_workflow",
    "resolve_round_artifacts",
    "snapshot_registered_dapi",
    "validate_cell_inputs",
    "validate_completed_artifact",
]
