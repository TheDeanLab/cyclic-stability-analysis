"""Target-blind physical registration and deformation quality-control summaries."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from cyexm.metrics import read_metrics
from cyexm.registration.backend import AntsRegistrationBackend
from numpy.typing import NDArray
from scipy.linalg import polar, rq
from scipy.spatial.transform import Rotation

from cyclic_stability.cyexm_artifacts import RoundArtifacts
from cyclic_stability.models import CellSpec, RoundSpec
from cyclic_stability.units import ResolutionReference, theoretical_resolution


@dataclass(frozen=True, slots=True)
class AffineSummary:
    """Physical XYZ affine decomposition; pose fields are not morphology endpoints."""

    determinant: float
    singular_values: tuple[float, float, float]
    scale_xyz: tuple[float, float, float]
    anisotropy: float
    rotation_xyz_deg: tuple[float, float, float]
    translation_xyz_um: tuple[float, float, float]
    shear_xy_deg: float
    shear_xz_deg: float
    shear_yz_deg: float
    polar_rotation_xyz_deg: tuple[float, float, float]
    polar_strain_principal_values: tuple[float, float, float]
    translation_role: str = "repositioning"
    rotation_role: str = "repositioning"


@dataclass(frozen=True, slots=True)
class WarpSummary:
    """Physical deformation summary for a selected set of fixed-grid voxels."""

    displacement_p50_um: float
    displacement_p95_um: float
    fraction_over_one_resolution: float
    jacobian_p05: float
    jacobian_p50: float
    jacobian_p95: float
    jacobian_nonpositive_count: int
    gradient_frobenius_p50: float
    gradient_frobenius_p95: float
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _WarpMeasurements:
    """Full-grid values shared by whole, seam, and interior summaries."""

    displacement: NDArray[np.float64]
    normalized: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    gradient_norm: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RegistrationQcResult:
    """Target-blind QC records, overlays, and explicitly undecided moving rounds."""

    registration_rows: tuple[dict[str, object], ...]
    deformation_rows: tuple[dict[str, object], ...]
    automated_validity: tuple[tuple[int, bool], ...]
    overlay_paths: tuple[Path, ...]
    seam_summaries: tuple[WarpSummary, ...]
    interior_summaries: tuple[WarpSummary, ...]
    decision_requests: tuple[dict[str, object], ...]


class _HardInvalid(ValueError):
    """A permitted automated exclusion based on registration evidence."""


def decompose_affine(matrix_xyz: NDArray[np.float64]) -> AffineSummary:
    """Return a sign-normalized, proper RQ decomposition of a physical affine.

    The reported ``scale_xyz``, shear angles, and ``rotation_xyz_deg`` obey
    ``linear = upper(scale, shear) @ rotation`` exactly (up to numerical
    precision).  The separately named polar fields are descriptive strain/pose
    quantities and are not substituted into that reconstruction.  Reflections
    are rejected because a valid proper rotation cannot represent them.
    """
    matrix = np.asarray(matrix_xyz, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("affine matrix must be finite with shape (4, 4)")
    linear = matrix[:3, :3]
    polar_rotation, polar_strain = polar(linear)
    upper, rotation_matrix = rq(linear)
    signs = np.where(np.diag(upper) < 0.0, -1.0, 1.0)
    sign_matrix = np.diag(signs)
    upper = upper @ sign_matrix
    rotation_matrix = sign_matrix @ rotation_matrix
    if np.linalg.det(rotation_matrix) <= 0.0:
        raise ValueError("affine matrix is improper (contains a reflection)")
    scales = np.diag(upper)
    if np.any(scales == 0.0):
        raise ValueError("affine matrix has a singular scale factor")
    rotation_deg = tuple(
        float(value)
        for value in Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)
    )
    shear_xy = math.degrees(math.atan(upper[0, 1] / upper[1, 1]))
    shear_xz = math.degrees(math.atan(upper[0, 2] / upper[2, 2]))
    shear_yz = math.degrees(math.atan(upper[1, 2] / upper[2, 2]))
    singular_values = tuple(
        float(value) for value in np.linalg.svd(linear, compute_uv=False)
    )
    polar_rotation_deg = tuple(
        float(value)
        for value in Rotation.from_matrix(polar_rotation).as_euler("xyz", degrees=True)
    )
    polar_principal_values = tuple(
        float(value) for value in np.linalg.eigvalsh(polar_strain)
    )
    return AffineSummary(
        determinant=float(np.linalg.det(linear)),
        singular_values=singular_values,
        scale_xyz=tuple(float(value) for value in scales),
        anisotropy=float(max(singular_values) / min(singular_values)),
        rotation_xyz_deg=rotation_deg,
        translation_xyz_um=tuple(float(value) for value in matrix[:3, 3]),
        shear_xy_deg=float(shear_xy),
        shear_xz_deg=float(shear_xz),
        shear_yz_deg=float(shear_yz),
        polar_rotation_xyz_deg=polar_rotation_deg,
        polar_strain_principal_values=polar_principal_values,
    )


def deformation_gradient(
    warp_zyx_xyz: NDArray[np.float64],
    spacing_xyz_um: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Return ``I + du/dx`` with physical XYZ derivatives on a ZYX volume."""
    sx, sy, sz = spacing_xyz_um
    gradients = [np.gradient(warp_zyx_xyz[..., c], sz, sy, sx) for c in range(3)]
    field = np.zeros((*warp_zyx_xyz.shape[:3], 3, 3), dtype=np.float64)
    field[..., 0, :] = np.stack(
        (gradients[0][2], gradients[0][1], gradients[0][0]), axis=-1
    )
    field[..., 1, :] = np.stack(
        (gradients[1][2], gradients[1][1], gradients[1][0]), axis=-1
    )
    field[..., 2, :] = np.stack(
        (gradients[2][2], gradients[2][1], gradients[2][0]), axis=-1
    )
    field += np.eye(3)
    return field


def _validate_warp(
    warp_zyx_xyz: NDArray[np.float64], spacing_xyz_um: tuple[float, float, float]
) -> NDArray[np.float64]:
    warp = np.asarray(warp_zyx_xyz, dtype=np.float64)
    if warp.ndim != 4 or warp.shape[-1] != 3 or not np.isfinite(warp).all():
        raise ValueError("warp must be a finite ZYX volume with three XYZ components")
    if any(not math.isfinite(value) or value <= 0.0 for value in spacing_xyz_um):
        raise ValueError("spacing_xyz_um must contain three finite positive values")
    return warp


def _measure_warp(
    warp_zyx_xyz: NDArray[np.float64],
    spacing_xyz_um: tuple[float, float, float],
    resolution: ResolutionReference,
) -> _WarpMeasurements:
    warp = _validate_warp(warp_zyx_xyz, spacing_xyz_um)
    gradient = deformation_gradient(warp, spacing_xyz_um)
    displacement = np.linalg.norm(warp, axis=-1)
    normalized = np.sqrt(
        (warp[..., 0] / resolution.lateral_expanded_um) ** 2
        + (warp[..., 1] / resolution.lateral_expanded_um) ** 2
        + (warp[..., 2] / resolution.axial_expanded_um) ** 2
    )
    jacobian = np.linalg.det(gradient)
    gradient_norm = np.linalg.norm(gradient - np.eye(3), ord="fro", axis=(-2, -1))
    return _WarpMeasurements(displacement, normalized, jacobian, gradient_norm)


def _summarize_measurements(
    measurements: _WarpMeasurements,
    mask: NDArray[np.bool_] | None = None,
) -> WarpSummary:
    selected = (
        np.ones(measurements.displacement.shape, dtype=bool) if mask is None else mask
    )
    if selected.shape != measurements.displacement.shape or not selected.any():
        raise ValueError("summary mask must select at least one ZYX voxel")

    def values(array: NDArray[np.float64]) -> NDArray[np.float64]:
        return array[selected]

    nonpositive = int(np.count_nonzero(values(measurements.jacobian) <= 0.0))
    flags = ("nonpositive-jacobian",) if nonpositive else ()
    return WarpSummary(
        displacement_p50_um=float(np.percentile(values(measurements.displacement), 50)),
        displacement_p95_um=float(np.percentile(values(measurements.displacement), 95)),
        fraction_over_one_resolution=float(
            np.mean(values(measurements.normalized) > 1.0)
        ),
        jacobian_p05=float(np.percentile(values(measurements.jacobian), 5)),
        jacobian_p50=float(np.percentile(values(measurements.jacobian), 50)),
        jacobian_p95=float(np.percentile(values(measurements.jacobian), 95)),
        jacobian_nonpositive_count=nonpositive,
        gradient_frobenius_p50=float(
            np.percentile(values(measurements.gradient_norm), 50)
        ),
        gradient_frobenius_p95=float(
            np.percentile(values(measurements.gradient_norm), 95)
        ),
        flags=flags,
    )


def _summarize_warp(
    warp_zyx_xyz: NDArray[np.float64],
    spacing_xyz_um: tuple[float, float, float],
    resolution: ResolutionReference,
    mask: NDArray[np.bool_] | None = None,
) -> WarpSummary:
    return _summarize_measurements(
        _measure_warp(warp_zyx_xyz, spacing_xyz_um, resolution), mask
    )


def summarize_warp(
    warp_zyx_xyz: NDArray[np.float64],
    spacing_xyz_um: tuple[float, float, float],
    resolution: ResolutionReference,
) -> WarpSummary:
    """Summarize a physical XYZ displacement vector field on its ZYX grid."""
    return _summarize_warp(warp_zyx_xyz, spacing_xyz_um, resolution)


def seam_adjacency_mask(
    manifest_path: Path, shape_zyx: tuple[int, int, int]
) -> NDArray[np.bool_]:
    """Mark voxels within 16 expanded-image pixels of an internal core boundary."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        raise TypeError("chunk manifest has no chunks list")
    boundaries: tuple[set[int], set[int], set[int]] = (set(), set(), set())
    for item in chunks:
        chunk = item.get("chunk", item) if isinstance(item, dict) else None
        if not isinstance(chunk, dict):
            raise TypeError("chunk manifest has an invalid chunk entry")
        for axis, name in enumerate(("z", "y", "x")):
            for suffix in ("start", "end"):
                value = chunk.get(f"{name}_{suffix}")
                if isinstance(value, int) and 0 < value < shape_zyx[axis]:
                    boundaries[axis].add(value)
    grid = np.indices(shape_zyx)
    mask = np.zeros(shape_zyx, dtype=bool)
    for axis, values in enumerate(boundaries):
        for boundary in values:
            mask |= np.abs(grid[axis] - boundary) <= 16
    return mask


def _physical_affine(transform: object) -> NDArray[np.float64]:
    parameters = np.asarray(transform.parameters, dtype=np.float64)
    center = np.asarray(transform.fixed_parameters, dtype=np.float64)
    if parameters.shape != (12,) or center.shape != (3,):
        raise ValueError("ANTs affine transform has unexpected parameters")
    linear = parameters[:9].reshape(3, 3)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = linear
    matrix[:3, 3] = parameters[9:] + center - linear @ center
    return matrix


def _load_overlay_volume(path: Path) -> NDArray[np.float64]:
    import tifffile

    array = np.asarray(tifffile.imread(path), dtype=np.float64)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise _HardInvalid(f"non-finite or invalid target-blind channel data: {path}")
    return array


def _projection_overlay(
    first: NDArray[np.float64], second: NDArray[np.float64], axis: int
) -> NDArray[np.uint8]:
    first_projection = np.max(first, axis=axis)
    second_projection = np.max(second, axis=axis)

    def scale(image: NDArray[np.float64]) -> NDArray[np.uint8]:
        high = float(np.percentile(image, 99.5))
        if not math.isfinite(high) or high <= 0.0:
            return np.zeros(image.shape, dtype=np.uint8)
        return np.clip(image * (255.0 / high), 0, 255).astype(np.uint8)

    red, green = scale(first_projection), scale(second_projection)
    return np.stack((red, green, np.zeros_like(red)), axis=-1)


def _padded_pair(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Center-pad native-space images solely for an unregistered visual overlay."""
    shape = tuple(
        max(left, right) for left, right in zip(first.shape, second.shape, strict=True)
    )

    def padded(image: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.zeros(shape, dtype=np.float64)
        starts = tuple(
            (target - actual) // 2
            for target, actual in zip(shape, image.shape, strict=True)
        )
        slices = tuple(
            slice(start, start + length)
            for start, length in zip(starts, image.shape, strict=True)
        )
        result[slices] = image
        return result

    return padded(first), padded(second)


def _source_channel(round_spec: RoundSpec, suffix: str) -> Path:
    return (
        round_spec.tiff_directory / f"round_{round_spec.acquisition_round}_{suffix}.tif"
    )


def _write_stage_overlays(
    stage: str,
    reference_dapi: NDArray[np.float64],
    moving_dapi: NDArray[np.float64],
    reference_tubulin: NDArray[np.float64],
    moving_tubulin: NDArray[np.float64],
    moving: RoundArtifacts,
    output_dir: Path,
) -> tuple[Path, ...]:
    import tifffile

    paths: list[Path] = []
    for marker, fixed, transformed in (
        ("dapi", reference_dapi, moving_dapi),
        ("tubulin", reference_tubulin, moving_tubulin),
    ):
        if fixed.shape != transformed.shape:
            if stage != "pre":
                raise ValueError(
                    f"{stage} target-blind overlay channels have mismatched shapes"
                )
            fixed, transformed = _padded_pair(fixed, transformed)
        for name, axis in (("xy", 0), ("xz", 1), ("yz", 2)):
            path = (
                output_dir
                / f"{moving.acquisition_round:02d}-{marker}-{stage}-{name}.tif"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(path, _projection_overlay(fixed, transformed, axis))
            paths.append(path)
    return tuple(paths)


def _write_target_blind_overlays(
    cell: CellSpec,
    reference: RoundArtifacts,
    moving: RoundArtifacts,
    transform: object,
    backend: AntsRegistrationBackend,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write pre-, affine-, and nonlinear DAPI/tubulin overlays without targets."""
    round_specs = {
        round_spec.acquisition_round: round_spec for round_spec in cell.rounds
    }
    reference_spec, moving_spec = round_specs[2], round_specs[moving.acquisition_round]
    reference_dapi_pre = _load_overlay_volume(
        _source_channel(reference_spec, "405_DAPI")
    )
    moving_dapi_pre = _load_overlay_volume(_source_channel(moving_spec, "405_DAPI"))
    reference_tubulin_pre = _load_overlay_volume(
        _source_channel(reference_spec, "488_tubulin")
    )
    moving_tubulin_pre = _load_overlay_volume(
        _source_channel(moving_spec, "488_tubulin")
    )
    reference_dapi_image = backend.load_image(
        _source_channel(reference_spec, "405_DAPI"), (0.1083, 0.1083, 0.3)
    )
    reference_tubulin_image = backend.load_image(
        _source_channel(reference_spec, "488_tubulin"), (0.1083, 0.1083, 0.3)
    )
    moving_dapi_image = backend.load_image(
        _source_channel(moving_spec, "405_DAPI"), (0.1083, 0.1083, 0.3)
    )
    moving_tubulin_image = backend.load_image(
        _source_channel(moving_spec, "488_tubulin"), (0.1083, 0.1083, 0.3)
    )
    moving_dapi_affine = np.asarray(
        backend.to_array(
            backend.apply_affine(
                fixed=reference_dapi_image,
                moving=moving_dapi_image,
                transform=transform,
                interpolation="linear",
            )
        ),
        dtype=np.float64,
    )
    moving_tubulin_affine = np.asarray(
        backend.to_array(
            backend.apply_affine(
                fixed=reference_tubulin_image,
                moving=moving_tubulin_image,
                transform=transform,
                interpolation="linear",
            )
        ),
        dtype=np.float64,
    )
    reference_dapi = _load_overlay_volume(reference.registered_dapi)
    moving_dapi = _load_overlay_volume(moving.registered_dapi)
    reference_tubulin = _load_overlay_volume(reference.registered_tubulin)
    moving_tubulin = _load_overlay_volume(moving.registered_tubulin)
    return (
        *_write_stage_overlays(
            "pre",
            reference_dapi_pre,
            moving_dapi_pre,
            reference_tubulin_pre,
            moving_tubulin_pre,
            moving,
            output_dir,
        ),
        *_write_stage_overlays(
            "affine",
            np.asarray(backend.to_array(reference_dapi_image), dtype=np.float64),
            moving_dapi_affine,
            np.asarray(backend.to_array(reference_tubulin_image), dtype=np.float64),
            moving_tubulin_affine,
            moving,
            output_dir,
        ),
        *_write_stage_overlays(
            "nonlinear",
            reference_dapi,
            moving_dapi,
            reference_tubulin,
            moving_tubulin,
            moving,
            output_dir,
        ),
    )


def _csv_rows(
    path: Path, fieldnames: tuple[str, ...], rows: tuple[dict[str, object], ...]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _summary_row(
    prefix: str, summary: AffineSummary | WarpSummary
) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in asdict(summary).items()}


def _atomic_replace_directory(
    destination: Path, populate: Callable[[Path], None]
) -> None:
    """Populate then atomically promote a same-filesystem directory tree."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    try:
        populate(staged)
        if destination.exists():
            os.replace(destination, backup)
        os.replace(staged, destination)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)


def publish_qc_outputs(
    destination: Path,
    registration_rows: tuple[dict[str, object], ...],
    deformation_rows: tuple[dict[str, object], ...],
    decision_rows: tuple[dict[str, object], ...],
    overlay_writer: Callable[[Path], tuple[Path, ...]] | None = None,
) -> tuple[Path, ...]:
    """Atomically replace one QC tree after all scientific evaluation succeeds."""
    staged_overlays: tuple[Path, ...] = ()
    staging_root: Path | None = None

    def populate(staged: Path) -> None:
        nonlocal staged_overlays, staging_root
        staging_root = staged
        registration_fields = tuple(
            sorted({key for row in registration_rows for key in row})
        ) or (
            "cohort",
            "cell_id",
            "acquisition_round",
            "workflow_round",
            "is_reference",
            "automated_valid",
            "invalid_reason",
        )
        deformation_fields = tuple(
            sorted({key for row in deformation_rows for key in row})
        ) or ("cohort", "cell_id", "acquisition_round", "region")
        _csv_rows(
            staged / "registration-qc.csv", registration_fields, registration_rows
        )
        _csv_rows(staged / "deformation.csv", deformation_fields, deformation_rows)
        _csv_rows(
            staged / "decision-requests.csv",
            (
                "cohort",
                "cell_id",
                "acquisition_round",
                "automated_valid",
                "automated_reason",
                "decision",
            ),
            decision_rows,
        )
        if overlay_writer is not None:
            staged_overlays = overlay_writer(staged / "overlays")

    _atomic_replace_directory(destination, populate)
    assert staging_root is not None
    return tuple(
        destination / path.relative_to(staging_root) for path in staged_overlays
    )


def analyze_registration_qc(
    cell: CellSpec, artifacts: tuple[RoundArtifacts, ...], output_dir: Path
) -> RegistrationQcResult:
    """Write target-blind QC evidence for validated artifacts from one cell.

    This function intentionally accesses only DAPI, tubulin, transforms, warp,
    chunk-manifest, and metric artifacts.  It never opens ``registered_target``.
    """
    by_acquisition = {artifact.acquisition_round: artifact for artifact in artifacts}
    reference = by_acquisition.get(2)
    if reference is None:
        raise ValueError("registration QC requires biological acquisition round 2")
    if tuple(round_spec.acquisition_round for round_spec in cell.rounds) != tuple(
        sorted(by_acquisition)
    ):
        raise ValueError("cell and artifact acquisition-round inventories disagree")
    resolution = theoretical_resolution(0.640, 1.4, 1.515, 4.5)
    spacing = (0.1083, 0.1083, 0.3)
    backend = AntsRegistrationBackend()
    registration_rows: list[dict[str, object]] = []
    deformation_rows: list[dict[str, object]] = []
    decision_requests: list[dict[str, object]] = []
    overlay_paths: list[Path] = []
    seam_summaries: list[WarpSummary] = []
    interior_summaries: list[WarpSummary] = []
    validity: list[tuple[int, bool]] = []
    overlay_jobs: list[tuple[RoundArtifacts, object]] = []
    for artifact in sorted(artifacts, key=lambda value: value.acquisition_round):
        row: dict[str, object] = {
            "cohort": cell.cohort,
            "cell_id": cell.cell_id,
            "acquisition_round": artifact.acquisition_round,
            "workflow_round": artifact.workflow_round,
            "is_reference": artifact.acquisition_round == 2,
            "automated_valid": True,
            "invalid_reason": "",
            "translation_role": "repositioning",
            "rotation_role": "repositioning",
        }
        if artifact.acquisition_round != 2:
            try:
                if None in (
                    artifact.affine,
                    artifact.warp,
                    artifact.pre_metrics,
                    artifact.linear_metrics,
                    artifact.nonlinear_metrics,
                    artifact.chunk_manifest,
                ):
                    raise _HardInvalid("validated moving artifact chain is incomplete")
                affine_path, warp_path = artifact.affine, artifact.warp
                assert affine_path is not None and warp_path is not None
                try:
                    transform = backend.read_affine(affine_path)
                    affine = decompose_affine(_physical_affine(transform))
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    raise _HardInvalid(
                        f"unreadable or invalid affine geometry: {error}"
                    ) from error
                try:
                    warp = np.asarray(
                        backend.to_array(backend.read_warp(warp_path)), dtype=np.float64
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    raise _HardInvalid(
                        f"unreadable or non-finite warp geometry: {error}"
                    ) from error
                dapi = _load_overlay_volume(artifact.registered_dapi)
                tubulin = _load_overlay_volume(artifact.registered_tubulin)
                if dapi.shape != tubulin.shape or warp.shape[:3] != dapi.shape:
                    raise _HardInvalid("registered DAPI/tubulin/warp shapes disagree")
                assert artifact.chunk_manifest is not None
                try:
                    seam_mask = seam_adjacency_mask(artifact.chunk_manifest, dapi.shape)
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    raise _HardInvalid(
                        f"invalid chunk-manifest evidence: {error}"
                    ) from error
                interior_mask = ~seam_mask
                try:
                    measurements = _measure_warp(warp, spacing, resolution)
                    whole = _summarize_measurements(measurements)
                    seam = (
                        _summarize_measurements(measurements, seam_mask)
                        if seam_mask.any()
                        else None
                    )
                    interior = (
                        _summarize_measurements(measurements, interior_mask)
                        if interior_mask.any()
                        else None
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    raise _HardInvalid(
                        f"unreadable or non-finite warp geometry: {error}"
                    ) from error
                metrics = {
                    "pre": read_metrics(artifact.pre_metrics),
                    "linear": read_metrics(artifact.linear_metrics),
                    "nonlinear": read_metrics(artifact.nonlinear_metrics),
                }
                row.update(_summary_row("affine", affine))
                row.update(
                    {
                        f"{stage}_{name}": value
                        for stage, values in metrics.items()
                        for name, value in values.items()
                    }
                )
                row["automated_valid"] = not whole.flags
                row["invalid_reason"] = ";".join(whole.flags)
                deformation_rows.append(
                    {
                        "cohort": cell.cohort,
                        "cell_id": cell.cell_id,
                        "acquisition_round": artifact.acquisition_round,
                        "region": "whole",
                        **_summary_row("warp", whole),
                    }
                )
                for region, summary in (("seam", seam), ("interior", interior)):
                    if summary is None:
                        continue
                    deformation_rows.append(
                        {
                            "cohort": cell.cohort,
                            "cell_id": cell.cell_id,
                            "acquisition_round": artifact.acquisition_round,
                            "region": region,
                            **_summary_row("warp", summary),
                        }
                    )
                if seam is not None:
                    seam_summaries.append(seam)
                if interior is not None:
                    interior_summaries.append(interior)
                overlay_jobs.append((artifact, transform))
            except _HardInvalid as error:
                row["automated_valid"] = False
                row["invalid_reason"] = str(error)
        registration_rows.append(row)
        valid = bool(row["automated_valid"])
        validity.append((artifact.acquisition_round, valid))
        if artifact.acquisition_round != 2:
            decision_requests.append(
                {
                    "cohort": cell.cohort,
                    "cell_id": cell.cell_id,
                    "acquisition_round": artifact.acquisition_round,
                    "automated_valid": valid,
                    "automated_reason": row["invalid_reason"],
                    "decision": "",
                }
            )
    registration = tuple(registration_rows)
    deformation = tuple(deformation_rows)
    decisions = tuple(decision_requests)

    def write_overlays(staged_overlays: Path) -> tuple[Path, ...]:
        return tuple(
            path
            for artifact, transform in overlay_jobs
            for path in _write_target_blind_overlays(
                cell, reference, artifact, transform, backend, staged_overlays
            )
        )

    overlay_paths = list(
        publish_qc_outputs(
            output_dir / "qc", registration, deformation, decisions, write_overlays
        )
    )
    return RegistrationQcResult(
        registration,
        deformation,
        tuple(validity),
        tuple(overlay_paths),
        tuple(seam_summaries),
        tuple(interior_summaries),
        decisions,
    )


__all__ = [
    "AffineSummary",
    "RegistrationQcResult",
    "WarpSummary",
    "analyze_registration_qc",
    "decompose_affine",
    "deformation_gradient",
    "seam_adjacency_mask",
    "summarize_warp",
]
