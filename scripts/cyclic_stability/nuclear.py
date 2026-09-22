"""Target-independent consensus nuclei and Lamin envelope measurements."""

from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    find_objects,
    gaussian_filter,
    label,
    map_coordinates,
)
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, marching_cubes, mesh_surface_area
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from cyclic_stability.cyexm_artifacts import RoundArtifacts
from cyclic_stability.images import BackgroundSummary, load_volume, robust_background
from cyclic_stability.models import (
    AnalysisConfig,
    CellSpec,
    EnsembleRow,
    LaminNuclearExclusion,
    MarkerResult,
    MetricRow,
    NuclearConfig,
)
from cyclic_stability.morphology import (
    gaussian_sigma_zyx,
    resolution_surface_metrics,
)
from cyclic_stability.units import ResolutionReference, theoretical_resolution

REFERENCE_ROUND = 2
_DAPI_LATERAL_RESOLUTION_UM = 0.61 * 0.405 / 1.4
_DIAGNOSTIC_MAX_POINTS_PER_PROJECTION = 2048


@dataclass(frozen=True, slots=True)
class NuclearComponentQc:
    """Target-blind physical adjudication of one consensus DAPI component.

    Face fractions are ordered ``z-min, z-max, y-min, y-max, x-min, x-max``.
    """

    nucleus_id: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    expanded_volume_um3: float
    nominal_volume_um3: float
    sphericity: float
    centroid_xyz_um: tuple[float, float, float]
    face_contact_fractions: tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class NuclearComponentProjection:
    """Bounded diagnostic-only orthogonal outlines for one DAPI component."""

    plane: str
    outlines_ab_um: tuple[NDArray[np.float64], ...]
    source_point_count: int
    retained_point_count: int
    decimated: bool


@dataclass(frozen=True, slots=True)
class NuclearComponentDiagnostic:
    """Target-blind accepted or rejected component outline evidence."""

    nucleus_id: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    projections: tuple[NuclearComponentProjection, ...]
    diagnostic_only: bool = True


@dataclass(frozen=True, slots=True)
class NuclearSurface:
    """A target-independent common nuclear surface in physical coordinates."""

    nucleus_id: str
    mask_zyx: NDArray[np.bool_]
    vertices_xyz_um: NDArray[np.float64]
    faces: NDArray[np.int_]
    normals_xyz: NDArray[np.float64]
    centroid_xyz_um: NDArray[np.float64]
    mean_curvature_um_inverse: NDArray[np.float64]
    spacing_xyz_um: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class NuclearConsensus:
    """Immutable accepted nuclei, their union, and all DAPI component QC."""

    nuclei: tuple[NuclearSurface, ...]
    mask_zyx: NDArray[np.bool_]
    labels_zyx: NDArray[np.uint16]
    component_qc: tuple[NuclearComponentQc, ...]
    component_diagnostics: tuple[NuclearComponentDiagnostic, ...]
    spacing_xyz_um: tuple[float, float, float]
    nominal_expansion_factor: float


@dataclass(frozen=True, slots=True)
class _BoundedNuclearRoi:
    """One tight, owning nuclear ROI with exact full-grid placement metadata."""

    nucleus_id: str
    full_shape_zyx: tuple[int, int, int]
    region_zyx: tuple[slice, slice, slice]
    origin_xyz_um: tuple[float, float, float]
    mask_zyx: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class _BoundedLaminMask:
    """One bounded Lamin member whose mask never retains a full-grid base array."""

    nucleus_id: str
    region_zyx: tuple[slice, slice, slice]
    origin_xyz_um: tuple[float, float, float]
    mask_zyx: NDArray[np.bool_]
    sigma_um: float
    threshold_sbr: float


@dataclass(frozen=True, slots=True)
class _LaminReferenceMasks:
    """The only Lamin masks retained across rounds: bounded reference members."""

    full_shape_zyx: tuple[int, int, int]
    members: Mapping[tuple[str, float, float], _BoundedLaminMask]

    def __post_init__(self) -> None:
        frozen = dict(self.members)
        for key, member in frozen.items():
            if key != (member.nucleus_id, member.sigma_um, member.threshold_sbr):
                raise ValueError("reference Lamin mask key does not match its member")
            expected_shape = tuple(
                int(region.stop) - int(region.start) for region in member.region_zyx
            )
            if member.mask_zyx.shape != expected_shape:
                raise ValueError("bounded Lamin mask does not match its global region")
            if member.mask_zyx.dtype != np.bool_ or member.mask_zyx.flags.writeable:
                raise ValueError("bounded Lamin masks must be immutable Boolean arrays")
            if member.mask_zyx.base is not None:
                raise ValueError("bounded Lamin masks must own their retained storage")
            if any(
                int(region.start) < 0
                or int(region.stop) > extent
                or int(region.stop) <= int(region.start)
                for region, extent in zip(
                    member.region_zyx, self.full_shape_zyx, strict=True
                )
            ):
                raise ValueError("bounded Lamin mask region is outside the full grid")
        object.__setattr__(self, "members", MappingProxyType(frozen))

    @property
    def retained_mask_voxels(self) -> int:
        """Return the exact number of Boolean voxels retained between rounds."""
        return sum(member.mask_zyx.size for member in self.members.values())


@dataclass(frozen=True, slots=True)
class _AcceptedCandidateGeometry:
    region_zyx: tuple[slice, slice, slice]
    mask_zyx: NDArray[np.bool_]
    vertices_xyz_um: NDArray[np.float64]
    faces: NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class _NuclearCandidate:
    accepted_geometry: _AcceptedCandidateGeometry | None
    diagnostic_projections: tuple[NuclearComponentProjection, ...]
    accepted: bool
    rejection_reasons: tuple[str, ...]
    expanded_volume_um3: float
    nominal_volume_um3: float
    sphericity: float
    centroid_xyz_um: tuple[float, float, float]
    face_contact_fractions: tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class NuclearComponentAdjudication:
    """Target-blind component evidence before strict surface realization.

    Parameters
    ----------
    component_qc : tuple[NuclearComponentQc, ...]
        Exact candidate rows ordered by physical XYZ centroid. Accepted rows use
        stable ``N###`` identifiers; rejected rows retain candidate ``C###``
        identifiers.
    component_diagnostics : tuple[NuclearComponentDiagnostic, ...]
        Deterministically bounded, diagnostic-only orthogonal component outlines.
        These outlines are target-blind and are never used for scientific metrics.
    spacing_xyz_um : tuple[float, float, float]
        Physical voxel spacing in public ``(x, y, z)`` micrometers.
    nominal_expansion_factor : float
        Linear expansion factor used for nominal-volume conversion.

    Notes
    -----
    This value deliberately does not claim that accepted candidates have usable
    normals or curvature. Pass it to :func:`realize_nuclear_surfaces` to enforce
    the normal scientific surface contract without repeating segmentation.
    """

    component_qc: tuple[NuclearComponentQc, ...]
    component_diagnostics: tuple[NuclearComponentDiagnostic, ...]
    spacing_xyz_um: tuple[float, float, float]
    nominal_expansion_factor: float
    _candidates: tuple[_NuclearCandidate, ...]
    _full_shape_zyx: tuple[int, int, int]
    _config: NuclearConfig


@dataclass(frozen=True, slots=True)
class LaminProfileRow:
    """One surface-vertex normal profile and its round-level support summaries."""

    acquisition_round: int
    vertex_index: int
    supported: bool
    peak_offset_um: float
    displacement_um: float
    absolute_displacement_um: float
    fwhm_um: float
    integrated_ridge_intensity: float
    surface_coverage: float
    continuity: float
    continuity_change_from_reference: float
    nucleus_id: str = "aggregate"


class _GeometryResult(TypedDict):
    """Exact-mask geometry and tolerant curvature completeness."""

    volume_um3: float
    area_to_volume_um_inverse: float
    sphericity: float
    mean_curvature_p25_um_inverse: float
    mean_curvature_p50_um_inverse: float
    mean_curvature_p75_um_inverse: float
    valid_curvature_vertex_fraction: float
    geometry_status: str


def _spacing(
    spacing_xyz_um: tuple[float, float, float],
) -> tuple[float, float, float]:
    if len(spacing_xyz_um) != 3:
        raise ValueError("spacing_xyz_um must contain exactly three values")
    result = tuple(float(value) for value in spacing_xyz_um)
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError("spacing_xyz_um must be finite and positive")
    return result


def _validate_config(config: NuclearConfig) -> None:
    values = (
        config.consensus_smoothing_um,
        config.normal_half_window_um,
        config.normal_step_um,
        config.ridge_min_sbr,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("nuclear configuration values must be finite and positive")
    h = config.watershed_seed_prominence_um
    if h is not None and (isinstance(h, bool) or not math.isfinite(h) or h <= 0):
        raise ValueError("watershed_seed_prominence_um must be finite and positive")
    quotient = config.normal_half_window_um / config.normal_step_um
    if not math.isclose(quotient, round(quotient), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("normal_half_window_um must be divisible by normal_step_um")


def _volume(value: NDArray[np.generic], name: str) -> NDArray[np.float64]:
    image = np.asarray(value)
    if (
        image.ndim != 3
        or not np.issubdtype(image.dtype, np.number)
        or np.issubdtype(image.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be one real numeric 3D ZYX volume")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{name} must contain only finite values")
    return image.astype(np.float64, copy=False)


def _bounded_nuclear_roi(
    nucleus_id: str,
    mask_zyx: NDArray[np.generic],
    spacing_xyz_um: tuple[float, float, float],
) -> _BoundedNuclearRoi:
    """Copy one complete nuclear mask into its tight global-grid bounding slice."""
    mask = np.asarray(mask_zyx)
    spacing = _spacing(spacing_xyz_um)
    if mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError("nuclear ROI must be one Boolean 3D ZYX mask")
    occupied = tuple(
        np.flatnonzero(np.any(mask, axis=axes)) for axes in ((1, 2), (0, 2), (0, 1))
    )
    if any(indices.size == 0 for indices in occupied):
        raise ValueError("nuclear ROI must contain foreground")
    region = tuple(slice(int(indices[0]), int(indices[-1]) + 1) for indices in occupied)
    region_zyx = (region[0], region[1], region[2])
    bounded = np.array(mask[region_zyx], dtype=bool, order="C", copy=True)
    bounded.setflags(write=False)
    sx, sy, sz = spacing
    z_region, y_region, x_region = region_zyx
    origin_xyz_um = (
        int(x_region.start) * sx,
        int(y_region.start) * sy,
        int(z_region.start) * sz,
    )
    return _BoundedNuclearRoi(
        nucleus_id,
        tuple(int(value) for value in mask.shape),
        region_zyx,
        origin_xyz_um,
        bounded,
    )


def _iter_bounded_lamin_masks(
    image: NDArray[np.generic],
    rois: tuple[_BoundedNuclearRoi, ...],
    sigma_values_um: tuple[float, ...],
    threshold_sbr: tuple[float, ...],
    spacing_xyz_um: tuple[float, float, float],
    background: BackgroundSummary,
) -> Iterator[_BoundedLaminMask]:
    """Smooth one full round once per sigma and yield disposable bounded masks."""
    source = _volume(image, "Lamin")
    spacing = _spacing(spacing_xyz_um)
    if not rois:
        raise ValueError("bounded Lamin ensemble requires at least one nuclear ROI")
    if not sigma_values_um or not threshold_sbr:
        raise ValueError("Lamin ensemble coordinates must be non-empty")
    if len({roi.nucleus_id for roi in rois}) != len(rois):
        raise ValueError("bounded nuclear ROI identifiers must be unique")
    if any(roi.full_shape_zyx != source.shape for roi in rois):
        raise ValueError("bounded nuclear ROIs must match the Lamin full grid")
    thresholds = tuple(float(value) for value in threshold_sbr)
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("threshold_sbr values must be finite and positive")
    if not math.isfinite(background.median) or not math.isfinite(background.scale):
        raise ValueError("Lamin background summary must be finite")
    for raw_sigma in sigma_values_um:
        sigma_um = float(raw_sigma)
        smoothed = gaussian_filter(source, sigma=gaussian_sigma_zyx(sigma_um, spacing))
        for roi in rois:
            bounded_signal = smoothed[roi.region_zyx]
            for sbr in thresholds:
                mask = np.logical_and(
                    bounded_signal > background.median + sbr * background.scale,
                    roi.mask_zyx,
                )
                mask.setflags(write=False)
                yield _BoundedLaminMask(
                    roi.nucleus_id,
                    roi.region_zyx,
                    roi.origin_xyz_um,
                    mask,
                    sigma_um,
                    sbr,
                )
        del smoothed


def _build_lamin_reference_masks(
    image: NDArray[np.generic],
    rois: tuple[_BoundedNuclearRoi, ...],
    sigma_values_um: tuple[float, ...],
    threshold_sbr: tuple[float, ...],
    spacing_xyz_um: tuple[float, float, float],
    background: BackgroundSummary,
) -> _LaminReferenceMasks:
    """Retain only bounded reference masks needed by later round comparisons."""
    source = _volume(image, "reference Lamin")
    members = {
        (member.nucleus_id, member.sigma_um, member.threshold_sbr): member
        for member in _iter_bounded_lamin_masks(
            source,
            rois,
            sigma_values_um,
            threshold_sbr,
            spacing_xyz_um,
            background,
        )
    }
    expected = len(rois) * len(sigma_values_um) * len(threshold_sbr)
    if len(members) != expected:
        raise ValueError("Lamin ensemble coordinates must be unique")
    return _LaminReferenceMasks(source.shape, members)


def _normalized_dapi(image: NDArray[np.float64]) -> NDArray[np.float64]:
    background = robust_background(image)
    corrected = np.clip(image - background.median, 0.0, None)
    positive = corrected[corrected > 0.0]
    if positive.size == 0:
        raise ValueError("DAPI has no positive background-corrected support")
    scale = float(np.percentile(positive, 99.0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("DAPI positive 99th percentile must be finite and positive")
    return corrected / scale


def _physical_ball(
    radius_um: float, spacing_xyz_um: tuple[float, float, float]
) -> NDArray[np.bool_]:
    sx, sy, sz = spacing_xyz_um
    radii = tuple(math.ceil(radius_um / step) for step in (sz, sy, sx))
    z, y, x = np.ogrid[
        -radii[0] : radii[0] + 1,
        -radii[1] : radii[1] + 1,
        -radii[2] : radii[2] + 1,
    ]
    return (z * sz) ** 2 + (y * sy) ** 2 + (x * sx) ** 2 <= radius_um**2


def _face_contact_fractions(
    mask: NDArray[np.bool_],
    region_zyx: tuple[slice, slice, slice],
    full_shape_zyx: tuple[int, int, int],
) -> tuple[float, float, float, float, float, float]:
    z_region, y_region, x_region = region_zyx
    full_z, full_y, full_x = full_shape_zyx
    return (
        float(mask[0].sum() / (full_y * full_x)) if z_region.start == 0 else 0.0,
        (float(mask[-1].sum() / (full_y * full_x)) if z_region.stop == full_z else 0.0),
        float(mask[:, 0, :].sum() / (full_z * full_x)) if y_region.start == 0 else 0.0,
        (
            float(mask[:, -1, :].sum() / (full_z * full_x))
            if y_region.stop == full_y
            else 0.0
        ),
        float(mask[:, :, 0].sum() / (full_z * full_y)) if x_region.start == 0 else 0.0,
        (
            float(mask[:, :, -1].sum() / (full_z * full_y))
            if x_region.stop == full_x
            else 0.0
        ),
    )


def _expanded_component_slice(
    region_zyx: tuple[slice, slice, slice],
    full_shape_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    """Include one real background voxel around a component where available."""
    return tuple(
        slice(max(int(region.start) - 1, 0), min(int(region.stop) + 1, extent))
        for region, extent in zip(region_zyx, full_shape_zyx, strict=True)
    )  # type: ignore[return-value]


def _decimate_diagnostic_outlines(
    outlines: tuple[NDArray[np.float64], ...],
) -> tuple[tuple[NDArray[np.float64], ...], int, int]:
    """Deterministically cap diagnostic outline points while retaining segments."""
    source_count = sum(outline.shape[0] for outline in outlines)
    if source_count == 0:
        return (), 0, 0
    selected = list(outlines)
    if len(selected) * 2 > _DIAGNOSTIC_MAX_POINTS_PER_PROJECTION:
        keep_count = _DIAGNOSTIC_MAX_POINTS_PER_PROJECTION // 2
        selected_indices = sorted(
            sorted(
                range(len(selected)),
                key=lambda index: (-selected[index].shape[0], index),
            )[:keep_count]
        )
        selected = [selected[index] for index in selected_indices]
    lengths = [outline.shape[0] for outline in selected]
    if sum(lengths) <= _DIAGNOSTIC_MAX_POINTS_PER_PROJECTION:
        budgets = lengths
    else:
        budgets = [min(length, 2) for length in lengths]
        while sum(budgets) < _DIAGNOSTIC_MAX_POINTS_PER_PROJECTION:
            candidates = [
                index
                for index, (budget, length) in enumerate(zip(budgets, lengths))
                if budget < length
            ]
            if not candidates:
                break
            index = max(
                candidates,
                key=lambda value: (lengths[value] - budgets[value], -value),
            )
            budgets[index] += 1
    retained: list[NDArray[np.float64]] = []
    for outline, budget in zip(selected, budgets, strict=True):
        if budget == outline.shape[0]:
            sampled = np.array(outline, dtype=np.float64, order="C", copy=True)
        else:
            indices = np.linspace(0, outline.shape[0] - 1, num=budget, dtype=np.int64)
            sampled = np.array(outline[indices], dtype=np.float64, order="C", copy=True)
        sampled.setflags(write=False)
        retained.append(sampled)
    retained_count = sum(outline.shape[0] for outline in retained)
    return tuple(retained), source_count, retained_count


def _component_diagnostic_projections(
    mask_zyx: NDArray[np.bool_],
    region_zyx: tuple[slice, slice, slice],
    spacing_xyz_um: tuple[float, float, float],
) -> tuple[NuclearComponentProjection, ...]:
    """Encode bounded global-physical orthogonal outlines without a volume mask."""
    sx, sy, sz = spacing_xyz_um
    z_region, y_region, x_region = region_zyx
    specifications = (
        (
            "XY",
            mask_zyx.max(axis=0),
            int(x_region.start) * sx,
            int(y_region.start) * sy,
            sx,
            sy,
        ),
        (
            "XZ",
            mask_zyx.max(axis=1),
            int(x_region.start) * sx,
            int(z_region.start) * sz,
            sx,
            sz,
        ),
        (
            "YZ",
            mask_zyx.max(axis=2),
            int(y_region.start) * sy,
            int(z_region.start) * sz,
            sy,
            sz,
        ),
    )
    projections: list[NuclearComponentProjection] = []
    for plane, projected, origin_a, origin_b, spacing_a, spacing_b in specifications:
        contours = find_contours(np.pad(projected.astype(float), 1), level=0.5)
        physical = tuple(
            np.column_stack(
                (
                    (contour[:, 1] - 1.0) * spacing_a + origin_a,
                    (contour[:, 0] - 1.0) * spacing_b + origin_b,
                )
            )
            for contour in contours
            if contour.shape[0] >= 2
        )
        outlines, source_count, retained_count = _decimate_diagnostic_outlines(physical)
        projections.append(
            NuclearComponentProjection(
                plane,
                outlines,
                source_count,
                retained_count,
                retained_count < source_count,
            )
        )
    return tuple(projections)


def _surface_from_mask(
    mask: NDArray[np.bool_], spacing_xyz_um: tuple[float, float, float]
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    sx, sy, sz = spacing_xyz_um
    padded = np.pad(mask.astype(np.float32), 1)
    try:
        vertices_zyx, faces, _, _ = marching_cubes(
            padded, level=0.5, spacing=(sz, sy, sx)
        )
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            "consensus nucleus cannot form a nondegenerate surface"
        ) from error
    vertices_zyx -= np.array((sz, sy, sx))
    return vertices_zyx[:, ::-1], faces.astype(int, copy=False)


def _surface_normals_and_curvature(
    mask: NDArray[np.bool_],
    vertices_xyz_um: NDArray[np.float64],
    spacing_xyz_um: tuple[float, float, float],
    *,
    allow_degenerate: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Sample outward physical normals and matching signed-distance curvature.

    The signed field is positive outside the foreground and negative inside:
    ``edt(~mask) - edt(mask)``. Its normalized physical gradient is outward and
    its divergence uses exactly the same orientation. One-voxel zero padding
    provides an outside field for an allowed face-contacting nucleus, with the
    interpolation coordinates shifted by exactly that padding.
    """
    sx, sy, sz = spacing_xyz_um
    padded = np.pad(mask, 1)
    signed = distance_transform_edt(
        ~padded, sampling=(sz, sy, sx)
    ) - distance_transform_edt(padded, sampling=(sz, sy, sx))
    gradients_zyx = np.gradient(signed, sz, sy, sx)
    gradient_xyz = np.stack(
        (gradients_zyx[2], gradients_zyx[1], gradients_zyx[0]), axis=-1
    )
    norm = np.linalg.norm(gradient_xyz, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norm)):
        raise ValueError("signed-distance gradient is non-finite")
    outward = np.divide(
        gradient_xyz,
        norm,
        out=np.zeros_like(gradient_xyz),
        where=norm > 0.0,
    )
    divergence = (
        np.gradient(outward[..., 0], sx, axis=2)
        + np.gradient(outward[..., 1], sy, axis=1)
        + np.gradient(outward[..., 2], sz, axis=0)
    )
    sample_zyx = np.vstack(
        (
            vertices_xyz_um[:, 2] / sz + 1.0,
            vertices_xyz_um[:, 1] / sy + 1.0,
            vertices_xyz_um[:, 0] / sx + 1.0,
        )
    )
    sampled_outward = np.vstack(
        [map_coordinates(outward[..., axis], sample_zyx, order=1) for axis in range(3)]
    ).T
    lengths = np.linalg.norm(sampled_outward, axis=1)
    valid = np.isfinite(lengths) & (lengths > 0.0)
    if not np.all(valid) and not allow_degenerate:
        raise ValueError("surface normal interpolation is degenerate")
    normals = np.full_like(sampled_outward, np.nan)
    normals[valid] = sampled_outward[valid] / lengths[valid, None]
    foreground_zyx = np.argwhere(mask)
    if foreground_zyx.size == 0:
        raise ValueError("surface mask has no foreground centroid")
    centroid = foreground_zyx.mean(axis=0)[::-1] * np.array((sx, sy, sz))
    curvature = map_coordinates(divergence, sample_zyx, order=1)
    curvature[~valid] = np.nan
    if not np.all(np.isfinite(curvature)) and not allow_degenerate:
        raise ValueError("surface curvature interpolation is non-finite")
    return normals, centroid, curvature


def _segmented_dapi_foreground_mask(
    dapi_rounds: Mapping[int, NDArray[np.generic]],
    spacing_xyz_um: tuple[float, float, float],
    config: NuclearConfig,
) -> NDArray[np.bool_]:
    """Segment the shared DAPI foreground without component adjudication.

    This internal primitive supplies the same target-independent DAPI
    segmentation to both strict nuclear morphology and permissive raw-signal
    ROI construction.  It deliberately makes no decision about component
    count, geometry, or edge contact.
    """
    spacing = _spacing(spacing_xyz_um)
    if REFERENCE_ROUND not in dapi_rounds:
        raise ValueError("consensus nuclear surface requires acquisition round 2")
    if not dapi_rounds:
        raise ValueError("consensus nuclear surface requires at least one DAPI round")
    volumes = {
        int(round_number): _volume(image, "DAPI")
        for round_number, image in dapi_rounds.items()
    }
    shapes = {volume.shape for volume in volumes.values()}
    if len(shapes) != 1:
        raise ValueError("QC-accepted DAPI rounds must share one registered geometry")
    median = np.median(
        np.stack([_normalized_dapi(volume) for volume in volumes.values()]), axis=0
    )
    smoothed = gaussian_filter(
        median, sigma=gaussian_sigma_zyx(config.consensus_smoothing_um, spacing)
    )
    if not np.any(np.isfinite(smoothed)) or float(smoothed.max()) <= float(
        smoothed.min()
    ):
        raise ValueError("consensus DAPI intensity is degenerate")
    threshold_mask = smoothed > threshold_otsu(smoothed)
    filled = binary_fill_holes(threshold_mask)
    mask = (
        binary_closing(
            filled, structure=_physical_ball(_DAPI_LATERAL_RESOLUTION_UM, spacing)
        )
        | filled
    )
    mask.setflags(write=False)
    return mask


def _watershed_nuclear_labels(
    foreground_mask: NDArray[np.bool_],
    spacing_xyz_um: tuple[float, float, float],
    seed_prominence_um: float,
) -> tuple[NDArray[np.uint16], int]:
    """Split touching DAPI foreground while preserving its exact outer union."""
    mask = np.asarray(foreground_mask, dtype=bool)
    source_labels, source_count = label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    source_regions = find_objects(source_labels, max_label=source_count)
    partition = np.zeros(mask.shape, dtype=np.uint16)
    marker_count = 0
    for component_index in range(1, source_count + 1):
        region = source_regions[component_index - 1]
        if region is None:
            raise RuntimeError("connected DAPI component has no bounding slice")
        component = source_labels[region] == component_index
        padded_component = np.pad(component, 1)
        distance = distance_transform_edt(
            padded_component, sampling=spacing_xyz_um[::-1]
        )[1:-1, 1:-1, 1:-1].astype(np.float32)
        prominent = h_maxima(distance, seed_prominence_um) & component
        local_markers, local_count = label(
            prominent, structure=np.ones((3, 3, 3), dtype=bool)
        )
        if local_count == 0:
            masked_distance = np.where(component, distance, -np.inf)
            peak_index = np.unravel_index(
                int(np.argmax(masked_distance)), masked_distance.shape
            )
            local_markers = np.zeros(component.shape, dtype=np.int32)
            local_markers[peak_index] = 1
            local_count = 1
        watershed_markers = np.zeros(local_markers.shape, dtype=np.int32)
        for local_index in range(1, local_count + 1):
            marker_count += 1
            if marker_count > np.iinfo(np.uint16).max:
                raise ValueError("nuclear watershed produced too many markers")
            watershed_markers[local_markers == local_index] = marker_count
        local_partition = np.asarray(
            watershed(
                -distance,
                markers=watershed_markers,
                mask=component,
                connectivity=np.ones((3, 3, 3), dtype=bool),
            ),
            dtype=np.uint16,
        )
        partition[region][component] = local_partition[component]
    if not np.array_equal(partition > 0, mask):
        raise RuntimeError("nuclear watershed did not preserve the DAPI foreground")
    return partition, marker_count


def adjudicate_nuclear_components(
    dapi_rounds: Mapping[int, NDArray[np.generic]],
    spacing_xyz_um: tuple[float, float, float],
    config: NuclearConfig,
    nominal_expansion_factor: float,
) -> NuclearComponentAdjudication:
    """Adjudicate target-independent median-DAPI components without normals.

    Parameters
    ----------
    dapi_rounds : Mapping[int, numpy.ndarray]
        Registered DAPI volumes keyed by acquisition round. The mapping must
        already contain only rounds admitted by recorded target-blind QC and
        must include biological reference round 2. Arrays use ``(z, y, x)``.
    spacing_xyz_um : tuple[float, float, float]
        Physical voxel spacing in public ``(x, y, z)`` micrometers.
    config : NuclearConfig
        Frozen segmentation and physical component-adjudication policy.
    nominal_expansion_factor : float
        Positive linear expansion factor for nominal-volume conversion.

    Returns
    -------
    NuclearComponentAdjudication
        Exact candidate QC plus bounded geometry that can subsequently undergo
        strict surface realization without repeating segmentation.

    Raises
    ------
    ValueError
        If inputs, spacing, configuration, or reference-round geometry are
        invalid. Component-count and surface-normal failures are intentionally
        deferred to :func:`realize_nuclear_surfaces`.
    """
    spacing = _spacing(spacing_xyz_um)
    _validate_config(config)
    expansion = float(nominal_expansion_factor)
    if not math.isfinite(expansion) or expansion <= 0.0:
        raise ValueError("nominal_expansion_factor must be finite and positive")
    mask = _segmented_dapi_foreground_mask(dapi_rounds, spacing, config)
    if config.watershed_seed_prominence_um is None:
        candidate_labels, candidate_count = label(
            mask, structure=np.ones((3, 3, 3), dtype=bool)
        )
    else:
        candidate_labels, candidate_count = _watershed_nuclear_labels(
            mask, spacing, config.watershed_seed_prominence_um
        )
    full_shape_zyx = mask.shape
    component_regions = find_objects(candidate_labels, max_label=candidate_count)
    del mask
    sx, sy, sz = spacing
    voxel_volume_um3 = sx * sy * sz
    minimum_volume, maximum_volume = config.nominal_volume_um3
    candidates: list[_NuclearCandidate] = []
    for component_index in range(1, candidate_count + 1):
        raw_region = component_regions[component_index - 1]
        if raw_region is None:
            raise RuntimeError("labeled DAPI component has no bounding slice")
        region_zyx = _expanded_component_slice(raw_region, full_shape_zyx)
        component_mask = candidate_labels[region_zyx] == component_index
        diagnostic_projections = _component_diagnostic_projections(
            component_mask, region_zyx, spacing
        )
        expanded_volume_um3 = float(component_mask.sum()) * voxel_volume_um3
        nominal_volume_um3 = expanded_volume_um3 / expansion**3
        vertices, faces = _surface_from_mask(component_mask, spacing)
        surface_area_um2 = float(mesh_surface_area(vertices, faces))
        sphericity = float(
            math.pi ** (1.0 / 3.0)
            * (6.0 * expanded_volume_um3) ** (2.0 / 3.0)
            / surface_area_um2
        )
        centroid_zyx = np.argwhere(component_mask).mean(axis=0)
        z_region, y_region, x_region = region_zyx
        centroid_xyz_um = (
            float((centroid_zyx[2] + int(x_region.start)) * sx),
            float((centroid_zyx[1] + int(y_region.start)) * sy),
            float((centroid_zyx[0] + int(z_region.start)) * sz),
        )
        face_contact_fractions = _face_contact_fractions(
            component_mask, region_zyx, full_shape_zyx
        )
        rejection_reasons: list[str] = []
        if nominal_volume_um3 < minimum_volume:
            rejection_reasons.append("volume-below-minimum")
        if nominal_volume_um3 > maximum_volume:
            rejection_reasons.append("volume-above-maximum")
        if sphericity < config.minimum_sphericity:
            rejection_reasons.append("sphericity-below-minimum")
        if any(
            fraction > config.maximum_face_contact_fraction
            for fraction in face_contact_fractions
        ):
            rejection_reasons.append("face-contact-above-maximum")
        if rejection_reasons:
            accepted_geometry = None
        else:
            for array in (component_mask, vertices, faces):
                array.setflags(write=False)
            accepted_geometry = _AcceptedCandidateGeometry(
                region_zyx, component_mask, vertices, faces
            )
        candidates.append(
            _NuclearCandidate(
                accepted_geometry,
                diagnostic_projections,
                not rejection_reasons,
                tuple(rejection_reasons),
                expanded_volume_um3,
                nominal_volume_um3,
                sphericity,
                centroid_xyz_um,
                face_contact_fractions,
            )
        )
        del component_mask, faces, vertices

    candidates.sort(key=lambda item: item.centroid_xyz_um)
    component_qc: list[NuclearComponentQc] = []
    component_diagnostics: list[NuclearComponentDiagnostic] = []
    accepted_index = 0
    for candidate_index, candidate in enumerate(candidates, start=1):
        if candidate.accepted:
            accepted_index += 1
            nucleus_id = f"N{accepted_index:03d}"
        else:
            nucleus_id = f"C{candidate_index:03d}"
        component_qc.append(
            NuclearComponentQc(
                nucleus_id,
                candidate.accepted,
                candidate.rejection_reasons,
                candidate.expanded_volume_um3,
                candidate.nominal_volume_um3,
                candidate.sphericity,
                candidate.centroid_xyz_um,
                candidate.face_contact_fractions,
            )
        )
        component_diagnostics.append(
            NuclearComponentDiagnostic(
                nucleus_id,
                candidate.accepted,
                candidate.rejection_reasons,
                candidate.diagnostic_projections,
            )
        )
    del candidate_labels, component_regions
    return NuclearComponentAdjudication(
        tuple(component_qc),
        tuple(component_diagnostics),
        spacing,
        expansion,
        tuple(candidates),
        full_shape_zyx,
        config,
    )


def _accepted_nuclear_components(
    adjudication: NuclearComponentAdjudication,
    config: NuclearConfig,
) -> tuple[tuple[_NuclearCandidate, NuclearComponentQc], ...]:
    """Validate and return accepted target-blind components in stable order."""
    if config != adjudication._config:
        raise ValueError("surface realization requires the adjudication configuration")
    candidates = adjudication._candidates
    component_qc = adjudication.component_qc
    accepted_count = sum(item.accepted for item in candidates)
    minimum_count, maximum_count = config.accepted_component_count
    if not minimum_count <= accepted_count <= maximum_count:
        rejected_reasons = sorted(
            {
                reason
                for candidate in candidates
                for reason in candidate.rejection_reasons
            }
        )
        reason_suffix = (
            f"; rejected component reasons: {', '.join(rejected_reasons)}"
            if rejected_reasons
            else ""
        )
        raise ValueError(
            "accepted nuclear component count "
            f"{accepted_count} is outside configured range "
            f"[{minimum_count}, {maximum_count}]{reason_suffix}"
        )
    return tuple(
        (candidate, qc)
        for candidate, qc in zip(candidates, component_qc, strict=True)
        if candidate.accepted
    )


def consensus_nuclear_mask(
    dapi_rounds: Mapping[int, NDArray[np.generic]],
    spacing_xyz_um: tuple[float, float, float],
    config: NuclearConfig,
    nominal_expansion_factor: float,
) -> NDArray[np.bool_]:
    """Return the accepted target-blind nuclear union mask without surface metrics.

    Parameters
    ----------
    dapi_rounds : Mapping[int, numpy.ndarray]
        Registered or native DAPI volumes keyed by acquisition round. Volumes use
        ``(z, y, x)`` order and are used only for target-independent component
        adjudication.
    spacing_xyz_um : tuple[float, float, float]
        Physical voxel spacing in public ``(x, y, z)`` micrometers.
    config : NuclearConfig
        Frozen segmentation and component-adjudication policy.
    nominal_expansion_factor : float
        Positive linear expansion factor for nominal-volume conversion.

    Returns
    -------
    numpy.ndarray
        Read-only boolean union mask of all accepted components in ``(z, y, x)``
        order.

    Raises
    ------
    ValueError
        If component adjudication fails or the accepted count is outside the
        configured range.

    Notes
    -----
    This function deliberately does not realize normals or curvature. Consumers
    that require surface metrics must use :func:`consensus_nuclear_surface`.
    """
    adjudication = adjudicate_nuclear_components(
        dapi_rounds, spacing_xyz_um, config, nominal_expansion_factor
    )
    union_mask = np.zeros(adjudication._full_shape_zyx, dtype=bool)
    for candidate, _qc in _accepted_nuclear_components(adjudication, config):
        geometry = candidate.accepted_geometry
        if geometry is None:
            raise RuntimeError("accepted DAPI component has no bounded geometry")
        union_mask[geometry.region_zyx] |= geometry.mask_zyx
    union_mask.setflags(write=False)
    return union_mask


def realize_nuclear_surfaces(
    adjudication: NuclearComponentAdjudication,
    config: NuclearConfig,
) -> NuclearConsensus:
    """Realize strict surfaces and normals from target-blind component evidence.

    Parameters
    ----------
    adjudication : NuclearComponentAdjudication
        Component evidence returned by :func:`adjudicate_nuclear_components`.
    config : NuclearConfig
        The exact configuration used for adjudication.

    Returns
    -------
    NuclearConsensus
        Immutable accepted surfaces, labels, union mask, and component QC.

    Raises
    ------
    ValueError
        If configuration changed, accepted count is outside its configured
        range, or any accepted component cannot satisfy strict surface-normal or
        curvature realization.
    """
    component_qc = adjudication.component_qc
    spacing = adjudication.spacing_xyz_um
    sx, sy, sz = spacing
    full_shape_zyx = adjudication._full_shape_zyx
    accepted_labels = np.zeros(full_shape_zyx, dtype=np.uint16)
    nuclei: list[NuclearSurface] = []
    for accepted_index, (candidate, qc) in enumerate(
        _accepted_nuclear_components(adjudication, config), start=1
    ):
        geometry = candidate.accepted_geometry
        if geometry is None:
            raise RuntimeError("accepted DAPI component has no bounded geometry")
        normals, local_centroid, curvature = _surface_normals_and_curvature(
            geometry.mask_zyx,
            geometry.vertices_xyz_um,
            spacing,
            allow_degenerate=True,
        )
        valid_vertices = np.all(np.isfinite(normals), axis=1) & np.isfinite(curvature)
        valid_faces = np.all(valid_vertices[geometry.faces], axis=1)
        if not np.any(valid_faces):
            raise ValueError("surface has no valid normal-supported faces")
        faces = geometry.faces[valid_faces]
        used = np.unique(faces)
        remap = np.full(normals.shape[0], -1, dtype=np.intp)
        remap[used] = np.arange(used.size, dtype=np.intp)
        faces = remap[faces]
        vertices_local = geometry.vertices_xyz_um[used]
        normals = normals[used]
        curvature = curvature[used]
        z_region, y_region, x_region = geometry.region_zyx
        origin_xyz_um = np.array(
            (
                int(x_region.start) * sx,
                int(y_region.start) * sy,
                int(z_region.start) * sz,
            )
        )
        vertices = vertices_local + origin_xyz_um
        centroid = local_centroid + origin_xyz_um
        component_mask = np.zeros(full_shape_zyx, dtype=bool)
        component_mask[geometry.region_zyx] = geometry.mask_zyx
        for array in (
            component_mask,
            vertices,
            faces,
            normals,
            centroid,
            curvature,
        ):
            array.setflags(write=False)
        accepted_labels[component_mask] = accepted_index
        nuclei.append(
            NuclearSurface(
                qc.nucleus_id,
                component_mask,
                vertices,
                faces,
                normals,
                centroid,
                curvature,
                spacing,
            )
        )
    union_mask = accepted_labels > 0
    union_mask.setflags(write=False)
    accepted_labels.setflags(write=False)
    return NuclearConsensus(
        tuple(nuclei),
        union_mask,
        accepted_labels,
        component_qc,
        adjudication.component_diagnostics,
        spacing,
        adjudication.nominal_expansion_factor,
    )


def consensus_nuclear_surface(
    dapi_rounds: Mapping[int, NDArray[np.generic]],
    spacing_xyz_um: tuple[float, float, float],
    config: NuclearConfig,
    nominal_expansion_factor: float,
) -> NuclearConsensus:
    """Construct a strict target-independent component-aware DAPI consensus."""
    adjudication = adjudicate_nuclear_components(
        dapi_rounds, spacing_xyz_um, config, nominal_expansion_factor
    )
    return realize_nuclear_surfaces(adjudication, config)


def _profile_offsets(config: NuclearConfig) -> NDArray[np.float64]:
    count = round(config.normal_half_window_um / config.normal_step_um)
    return np.arange(-count, count + 1, dtype=np.float64) * config.normal_step_um


def _sample_profiles(
    surface: NuclearSurface,
    image: NDArray[np.float64],
    offsets_um: NDArray[np.float64],
) -> NDArray[np.float64]:
    sx, sy, sz = surface.spacing_xyz_um
    xyz = (
        surface.vertices_xyz_um[:, None, :]
        + surface.normals_xyz[:, None, :] * offsets_um[None, :, None]
    )
    coordinates = np.stack(
        (xyz[..., 2] / sz, xyz[..., 1] / sy, xyz[..., 0] / sx), axis=0
    ).reshape(3, -1)
    sampled = map_coordinates(image, coordinates, order=1, mode="constant", cval=np.nan)
    return sampled.reshape(xyz.shape[:2])


def _crossing(
    positions: NDArray[np.float64],
    values: NDArray[np.float64],
    level: float,
    start: int,
    direction: int,
) -> float | None:
    index = start
    while 0 <= index + direction < values.size:
        next_index = index + direction
        first, second = values[index], values[next_index]
        if first >= level >= second or first <= level <= second:
            if first == second:
                return float(positions[index])
            fraction = float((level - first) / (second - first))
            return float(
                positions[index] + fraction * (positions[next_index] - positions[index])
            )
        index = next_index
    return None


def _ridge(
    profile: NDArray[np.float64],
    offsets_um: NDArray[np.float64],
    background: BackgroundSummary,
    config: NuclearConfig,
) -> tuple[bool, float, float, float]:
    if not np.all(np.isfinite(profile)):
        return False, math.nan, math.nan, math.nan
    corrected = profile - background.median
    peak_index = int(np.argmax(corrected))
    peak = float(corrected[peak_index])
    if (
        peak_index in (0, corrected.size - 1)
        or peak <= config.ridge_min_sbr * background.scale
    ):
        return False, math.nan, math.nan, math.nan
    level = peak / 2.0
    left = _crossing(offsets_um, corrected, level, peak_index, -1)
    right = _crossing(offsets_um, corrected, level, peak_index, 1)
    if left is None or right is None or right <= left:
        return False, math.nan, math.nan, math.nan
    return (
        True,
        float(offsets_um[peak_index]),
        float(right - left),
        float(np.trapezoid(np.clip(corrected, 0.0, None), offsets_um)),
    )


def _continuity(
    faces: NDArray[np.int_], supported: NDArray[np.bool_]
) -> tuple[float, float]:
    coverage = float(supported.mean())
    if not np.any(supported):
        return coverage, 0.0
    parent = np.arange(supported.size)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def join(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for triangle in faces:
        for first, second in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            if supported[first] and supported[second]:
                join(first, second)
    roots = np.array([find(int(index)) for index in np.flatnonzero(supported)])
    _, counts = np.unique(roots, return_counts=True)
    return coverage, float(counts.max() / supported.size)


def lamin_normal_profiles(
    surface: NuclearSurface,
    lamin_rounds: Mapping[int, NDArray[np.generic]],
    config: NuclearConfig,
    background_by_round: Mapping[int, BackgroundSummary],
) -> tuple[LaminProfileRow, ...]:
    """Measure signed Lamin ridge locations relative to acquisition round 2.

    The normal direction is physical XYZ, whereas volumes are sampled in ZYX.
    A missing or endpoint ridge remains unsupported with NaN geometric values;
    it is never assigned a zero displacement.
    """
    _spacing(surface.spacing_xyz_um)
    _validate_config(config)
    if REFERENCE_ROUND not in lamin_rounds:
        raise ValueError("Lamin profiles require acquisition round 2")
    if set(lamin_rounds) != set(background_by_round):
        raise ValueError(
            "Lamin volumes and background summaries must have matching rounds"
        )
    offsets = _profile_offsets(config)
    per_round: dict[int, list[tuple[bool, float, float, float]]] = {}
    for round_number, raw in lamin_rounds.items():
        image = _volume(raw, "Lamin")
        if image.shape != surface.mask_zyx.shape:
            raise ValueError("Lamin volume and consensus surface geometry must match")
        background = background_by_round[round_number]
        if (
            not math.isfinite(background.median)
            or not math.isfinite(background.scale)
            or background.scale <= 0.0
        ):
            raise ValueError(
                "Lamin background must be finite with positive robust scale"
            )
        per_round[int(round_number)] = [
            _ridge(profile, offsets, background, config)
            for profile in _sample_profiles(surface, image, offsets)
        ]
    reference = per_round[REFERENCE_ROUND]
    summaries: dict[int, tuple[float, float]] = {}
    for round_number, profiles in per_round.items():
        summaries[round_number] = _continuity(
            surface.faces, np.array([profile[0] for profile in profiles], dtype=bool)
        )
    reference_continuity = summaries[REFERENCE_ROUND][1]
    rows: list[LaminProfileRow] = []
    for round_number in sorted(per_round):
        coverage, continuity = summaries[round_number]
        for index, (supported, peak, fwhm, intensity) in enumerate(
            per_round[round_number]
        ):
            reference_supported, reference_peak, _, _ = reference[index]
            displacement = (
                0.0
                if round_number == REFERENCE_ROUND and supported
                else peak - reference_peak
                if supported and reference_supported
                else math.nan
            )
            rows.append(
                LaminProfileRow(
                    round_number,
                    index,
                    supported,
                    peak,
                    displacement,
                    abs(displacement) if math.isfinite(displacement) else math.nan,
                    fwhm,
                    intensity,
                    coverage,
                    continuity,
                    continuity - reference_continuity,
                    surface.nucleus_id,
                )
            )
    return tuple(rows)


def _accepted_rounds(cell: CellSpec, config: AnalysisConfig) -> tuple[int, ...]:
    decisions = {
        decision.acquisition_round: decision.decision
        for decision in config.qc_decisions
        if decision.cohort == cell.cohort and decision.cell_id == cell.cell_id
    }
    accepted = [REFERENCE_ROUND]
    for round_spec in cell.rounds:
        round_number = round_spec.acquisition_round
        if round_number == REFERENCE_ROUND:
            continue
        decision = decisions.get(round_number)
        if decision is None:
            raise ValueError(
                f"target-specific Lamin analysis requires QC decision for round {round_number}"
            )
        if decision == "accept":
            accepted.append(round_number)
    return tuple(sorted(accepted))


def _metric_rows(
    cell: CellSpec,
    profiles: tuple[LaminProfileRow, ...],
    config: AnalysisConfig,
) -> tuple[MetricRow, ...]:
    rows: list[MetricRow] = []
    for nucleus_id in sorted({row.nucleus_id for row in profiles}):
        nucleus_profiles = [row for row in profiles if row.nucleus_id == nucleus_id]
        for round_number in sorted({row.acquisition_round for row in nucleus_profiles}):
            subset = [
                row for row in nucleus_profiles if row.acquisition_round == round_number
            ]
            current_supported = [row for row in subset if row.supported]
            paired_supported = [
                row
                for row in subset
                if row.supported and math.isfinite(row.displacement_um)
            ]
            values = {
                "lamin_surface_coverage": (subset[0].surface_coverage, "fraction"),
                "lamin_continuity": (subset[0].continuity, "fraction"),
                "lamin_continuity_change_from_round_2": (
                    subset[0].continuity_change_from_reference,
                    "fraction",
                ),
            }
            if paired_supported:
                values.update(
                    {
                        "lamin_signed_displacement_median": (
                            float(
                                np.median(
                                    [row.displacement_um for row in paired_supported]
                                )
                            ),
                            "um",
                        ),
                        "lamin_absolute_displacement_median": (
                            float(
                                np.median(
                                    [
                                        row.absolute_displacement_um
                                        for row in paired_supported
                                    ]
                                )
                            ),
                            "um",
                        ),
                    }
                )
            if current_supported:
                values.update(
                    {
                        "lamin_fwhm_median": (
                            float(
                                np.median([row.fwhm_um for row in current_supported])
                            ),
                            "um",
                        ),
                        "lamin_integrated_ridge_intensity_median": (
                            float(
                                np.median(
                                    [
                                        row.integrated_ridge_intensity
                                        for row in current_supported
                                    ]
                                )
                            ),
                            "a.u. um",
                        ),
                    }
                )
            for endpoint, (value, units) in values.items():
                rows.append(
                    MetricRow(
                        cell.cohort,
                        cell.cell_id,
                        round_number,
                        endpoint,
                        value,
                        units,
                        "round-1-exposure-sensitivity"
                        if round_number == 1
                        else "matched-settings",
                        True,
                        config.config_digest,
                        nucleus_id,
                    )
                )
    return tuple(rows)


def _write_profile_diagnostics(
    output_dir: Path,
    cell_id: str,
    surface: NuclearSurface,
    profiles: tuple[LaminProfileRow, ...],
) -> tuple[Path, ...]:
    """Write portable surface coordinates plus XY/XZ/YZ profile projections."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_path = output_dir / "consensus-nuclear-surface.npz"
    np.savez_compressed(
        surface_path,
        vertices_xyz_um=surface.vertices_xyz_um,
        normals_xyz=surface.normals_xyz,
        faces=surface.faces,
    )
    profile_path = output_dir / "lamin-normal-profiles.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(LaminProfileRow.__dataclass_fields__)
        writer.writerows(
            [getattr(row, name) for name in LaminProfileRow.__dataclass_fields__]
            for row in profiles
        )
    latest_round = max(row.acquisition_round for row in profiles)
    displacements = np.full(surface.vertices_xyz_um.shape[0], np.nan)
    for row in profiles:
        if row.acquisition_round == latest_round:
            displacements[row.vertex_index] = row.displacement_um
    projection_paths: list[Path] = []
    for plane, axes in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
        figure, axis = plt.subplots(figsize=(5, 5), layout="constrained")
        plotted = np.isfinite(displacements)
        axis.scatter(
            surface.vertices_xyz_um[~plotted, axes[0]],
            surface.vertices_xyz_um[~plotted, axes[1]],
            color="lightgray",
            s=1.0,
            rasterized=True,
        )
        scatter = axis.scatter(
            surface.vertices_xyz_um[plotted, axes[0]],
            surface.vertices_xyz_um[plotted, axes[1]],
            c=displacements[plotted],
            cmap="coolwarm",
            s=1.0,
            rasterized=True,
        )
        figure.colorbar(scatter, ax=axis, label="Lamin displacement (um)")
        axis.set(
            xlabel=f"{plane[0].upper()} (um)",
            ylabel=f"{plane[1].upper()} (um)",
            aspect="equal",
            title=f"{cell_id} / {surface.nucleus_id} / round {latest_round}",
        )
        path = output_dir / f"lamin-surface-profile-{plane}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        projection_paths.append(path)
    return (surface_path, profile_path, *projection_paths)


def write_nuclear_component_diagnostic(
    output_dir: Path,
    cell_id: str,
    dapi_rounds: Mapping[int, NDArray[np.generic]],
    consensus: NuclearConsensus,
) -> Path:
    """Write an orthogonal overview followed by paginated component QC tables.

    Overview panels retain physical aspect ratios and fixed layout allocations.
    All accepted and rejected components remain visible, with short ID labels;
    full rejection reasons and measurements appear on subsequent table pages.
    Every page uses the same 12-by-8-inch canvas.
    """
    median_dapi = np.median(
        np.stack(
            [_normalized_dapi(_volume(image, "DAPI")) for image in dapi_rounds.values()]
        ),
        axis=0,
    )
    return _write_nuclear_component_inventory(
        output_dir,
        cell_id,
        {
            plane: median_dapi.max(axis=axis)
            for plane, axis in (("XY", 0), ("XZ", 1), ("YZ", 2))
        },
        consensus,
    )


def _write_nuclear_component_inventory(
    output_dir: Path,
    cell_id: str,
    images: Mapping[str, NDArray[np.generic]],
    consensus: NuclearConsensus,
) -> Path:
    """Render existing projections and component evidence without remeasurement."""
    import textwrap

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
    accepted = [item for item in consensus.component_diagnostics if item.accepted]
    color_by_id = {
        item.nucleus_id: colors[index % len(colors)]
        for index, item in enumerate(accepted)
    }
    diagnostic_by_id = {
        item.nucleus_id: item for item in consensus.component_diagnostics
    }
    if set(diagnostic_by_id) != {item.nucleus_id for item in consensus.component_qc}:
        raise ValueError("component diagnostics do not match DAPI component QC")
    sx, sy, sz = consensus.spacing_xyz_um
    projections = (
        ("XY", (0, 1), (sx, sy)),
        ("XZ", (0, 2), (sx, sz)),
        ("YZ", (1, 2), (sy, sz)),
    )
    footer = (
        "Diagnostic-only orthogonal outlines; deterministically capped at "
        f"{_DIAGNOSTIC_MAX_POINTS_PER_PROJECTION} points per component/projection."
    )
    path = output_dir / "nuclear-component-inventory.pdf"
    with PdfPages(path) as pdf:
        # Fixed allocations deliberately exclude annotations and tables from layout.
        figure = plt.figure(figsize=(12, 8))
        try:
            grid = figure.add_gridspec(
                1, 3, left=0.06, right=0.98, bottom=0.20, top=0.80, wspace=0.24
            )
            for column, (plane, centroid_axes, plot_spacing) in enumerate(projections):
                axis = figure.add_subplot(grid[0, column])
                projection = images[plane]
                extent = (
                    0.0,
                    (projection.shape[1] - 1) * plot_spacing[0],
                    0.0,
                    (projection.shape[0] - 1) * plot_spacing[1],
                )
                axis.imshow(projection, cmap="gray", origin="lower", extent=extent)
                axis.set(
                    xlim=extent[:2],
                    ylim=extent[2:],
                    xlabel=f"{plane[0]} (um)",
                    ylabel=f"{plane[1]} (um)",
                    aspect="equal",
                )
                figure.canvas.draw()
                height_points = axis.get_window_extent().height * 72 / figure.dpi
                ordered = sorted(
                    consensus.component_qc,
                    key=lambda item: (
                        item.centroid_xyz_um[centroid_axes[0]],
                        item.nucleus_id,
                    ),
                )
                slots = min(8, max(1, len(ordered)))
                for index, component in enumerate(ordered):
                    diagnostic = diagnostic_by_id[component.nucleus_id]
                    projected = next(
                        value
                        for value in diagnostic.projections
                        if value.plane == plane
                    )
                    color = color_by_id.get(component.nucleus_id, "#7F7F7F")
                    for outline in projected.outlines_ab_um:
                        axis.plot(
                            outline[:, 0],
                            outline[:, 1],
                            color=color,
                            linewidth=1.2 if component.accepted else 1.0,
                            linestyle="-" if component.accepted else "--",
                            zorder=2,
                        )
                    point = tuple(
                        component.centroid_xyz_um[dim] for dim in centroid_axes
                    )
                    axis.scatter(
                        *point,
                        color=color,
                        edgecolor="#D62728"
                        if max(component.face_contact_fractions) > 0.0
                        else "white",
                        linewidth=1.2,
                        marker="o" if component.accepted else "s",
                        s=22,
                        zorder=3,
                    )
                    label = axis.annotate(
                        component.nucleus_id,
                        point,
                        xytext=(
                            (index % slots + 0.5) / slots,
                            1 + (8 + 12 * (index // slots)) / height_points,
                        ),
                        textcoords="axes fraction",
                        ha="center",
                        va="bottom",
                        color=color,
                        fontsize=7,
                        arrowprops={
                            "arrowstyle": "-",
                            "color": color,
                            "linewidth": 0.5,
                        },
                        annotation_clip=False,
                    )
                    label.set_in_layout(False)
                position = axis.get_position(original=True)
                figure.text(
                    position.x0 + position.width / 2,
                    0.89,
                    f"DAPI {plane} maximum projection",
                    ha="center",
                    fontsize=11,
                )
            figure.suptitle(f"{cell_id}: target-blind DAPI component inventory", y=0.97)
            figure.text(
                0.5,
                0.13,
                f"{len(accepted)} accepted; {len(consensus.component_qc) - len(accepted)} rejected. Full component QC follows on subsequent pages.",
                ha="center",
                fontsize=10,
            )
            figure.text(
                0.5,
                0.09,
                "Accepted: colored solid outlines / circles. Rejected: gray dashed outlines / squares. Red marker edge: boundary contact.",
                ha="center",
                fontsize=8,
            )
            figure.text(0.5, 0.035, footer, ha="center", fontsize=7)
            pdf.savefig(figure)
        finally:
            plt.close(figure)

        columns = (
            "ID",
            "accepted",
            "expanded volume\n(um3)",
            "nominal volume\n(um3)",
            "sphericity",
            "centroid XYZ\n(um)",
            "max face\ncontact",
            "rejection code(s)",
            "outline points\nXY / XZ / YZ",
        )
        widths = (0.055, 0.065, 0.105, 0.095, 0.07, 0.10, 0.08, 0.31, 0.12)
        rows = []
        for component in consensus.component_qc:
            values = (
                component.nucleus_id,
                str(component.accepted),
                f"{component.expanded_volume_um3:.3f}",
                f"{component.nominal_volume_um3:.3f}",
                f"{component.sphericity:.4f}",
                "\n".join(f"{value:.3f}" for value in component.centroid_xyz_um),
                f"{max(component.face_contact_fractions):.5f}",
                "\n".join(
                    textwrap.fill(
                        reason, width=38, break_long_words=True, break_on_hyphens=True
                    )
                    for reason in component.rejection_reasons
                )
                or "accepted",
                " / ".join(
                    str(value.retained_point_count)
                    for value in diagnostic_by_id[component.nucleus_id].projections
                ),
            )
            # Heights are physical page fractions, independent of the row count.
            height = (max(value.count("\n") + 1 for value in values) * 11 + 10) / 576
            rows.append((values, height))
        pages = []
        current = []
        used = 0.0
        for row in rows:
            if current and used + row[1] > 0.72:
                pages.append(current)
                current, used = [], 0.0
            current.append(row)
            used += row[1]
        if current:
            pages.append(current)
        for number, page_rows in enumerate(pages, start=1):
            figure = plt.figure(figsize=(12, 8))
            try:
                axis = figure.add_axes((0.035, 0.0, 0.93, 1.0))
                axis.axis("off")
                table_height = 0.07 + sum(height for _, height in page_rows)
                table = axis.table(
                    cellText=[values for values, _ in page_rows],
                    colLabels=columns,
                    colWidths=widths,
                    cellLoc="center",
                    bbox=(0, 0.86 - table_height, 1, table_height),
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                for (row_index, _), cell in table.get_celld().items():
                    cell.set_height(
                        0.07 if row_index == 0 else page_rows[row_index - 1][1]
                    )
                    if row_index == 0:
                        cell.set_facecolor("#EAEAEA")
                figure.suptitle(
                    f"{cell_id}: DAPI component QC — table {number}/{len(pages)}",
                    y=0.97,
                )
                figure.text(0.5, 0.035, footer, ha="center", fontsize=7)
                pdf.savefig(figure)
            finally:
                plt.close(figure)
    return path


def _atomic_publish_directory(
    destination: Path, populate: Callable[[Path], None]
) -> None:
    """Atomically replace a complete Lamin output tree on its destination filesystem."""
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


def _geometry(
    mask: NDArray[np.bool_],
    spacing_xyz_um: tuple[float, float, float],
) -> _GeometryResult:
    """Return exact-mask geometry with tolerant ensemble-only curvature.

    Volume, surface area, and shape use the complete input mask. Curvature alone
    filters vertices whose signed-distance normal or sampled curvature is not
    finite and nondegenerate. The input mask is never relabeled, component-filtered,
    or otherwise changed.
    """
    volume = float(mask.sum() * np.prod(spacing_xyz_um))
    if volume == 0.0:
        return {
            "volume_um3": 0.0,
            "area_to_volume_um_inverse": math.nan,
            "sphericity": math.nan,
            "mean_curvature_p25_um_inverse": math.nan,
            "mean_curvature_p50_um_inverse": math.nan,
            "mean_curvature_p75_um_inverse": math.nan,
            "valid_curvature_vertex_fraction": math.nan,
            "geometry_status": "empty",
        }
    vertices, faces = _surface_from_mask(mask, spacing_xyz_um)
    area = float(mesh_surface_area(vertices, faces))
    curvature, valid_fraction = _tolerant_ensemble_curvature(
        mask, vertices, spacing_xyz_um
    )
    report_curvature = valid_fraction >= 0.99 and curvature.size > 0
    if valid_fraction == 1.0:
        status = "valid"
    elif report_curvature:
        status = "partial-curvature"
    else:
        status = "invalid-curvature"
    curvature_percentiles = (
        tuple(float(value) for value in np.percentile(curvature, (25.0, 50.0, 75.0)))
        if report_curvature
        else (math.nan, math.nan, math.nan)
    )
    return {
        "volume_um3": volume,
        "area_to_volume_um_inverse": area / volume,
        "sphericity": float(
            np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / area
        ),
        "mean_curvature_p25_um_inverse": curvature_percentiles[0],
        "mean_curvature_p50_um_inverse": curvature_percentiles[1],
        "mean_curvature_p75_um_inverse": curvature_percentiles[2],
        "valid_curvature_vertex_fraction": valid_fraction,
        "geometry_status": status,
    }


def _tolerant_ensemble_curvature(
    mask: NDArray[np.bool_],
    vertices_xyz_um: NDArray[np.float64],
    spacing_xyz_um: tuple[float, float, float],
) -> tuple[NDArray[np.float64], float]:
    """Sample curvature only at finite, nondegenerate ensemble-mask vertices.

    This intentionally does not replace the strict normal path used to accept a
    DAPI consensus surface. It exists only so one fragmented Lamin threshold
    member cannot abort the remaining ensemble or field.
    """
    sx, sy, sz = spacing_xyz_um
    padded = np.pad(mask, 1)
    signed = np.asarray(
        distance_transform_edt(~padded, sampling=(sz, sy, sx)), dtype=np.float64
    ) - np.asarray(
        distance_transform_edt(padded, sampling=(sz, sy, sx)), dtype=np.float64
    )
    gradients_zyx = np.gradient(signed, sz, sy, sx)
    gradient_xyz = np.stack(
        (gradients_zyx[2], gradients_zyx[1], gradients_zyx[0]), axis=-1
    )
    grid_lengths = np.linalg.norm(gradient_xyz, axis=-1, keepdims=True)
    outward = np.divide(
        gradient_xyz,
        grid_lengths,
        out=np.zeros_like(gradient_xyz),
        where=np.isfinite(grid_lengths) & (grid_lengths > 0.0),
    )
    divergence = (
        np.gradient(outward[..., 0], sx, axis=2)
        + np.gradient(outward[..., 1], sy, axis=1)
        + np.gradient(outward[..., 2], sz, axis=0)
    )
    sample_zyx = np.vstack(
        (
            vertices_xyz_um[:, 2] / sz + 1.0,
            vertices_xyz_um[:, 1] / sy + 1.0,
            vertices_xyz_um[:, 0] / sx + 1.0,
        )
    )
    sampled_outward = np.vstack(
        [map_coordinates(outward[..., axis], sample_zyx, order=1) for axis in range(3)]
    ).T
    sampled_lengths = np.linalg.norm(sampled_outward, axis=1)
    sampled_curvature = map_coordinates(divergence, sample_zyx, order=1)
    valid = (
        np.all(np.isfinite(sampled_outward), axis=1)
        & np.isfinite(sampled_lengths)
        & (sampled_lengths > 0.0)
        & np.isfinite(sampled_curvature)
    )
    return sampled_curvature[valid], float(np.count_nonzero(valid) / valid.size)


def _nuclear_surface_geometry(surface: NuclearSurface) -> dict[str, float]:
    """Summarize one already-strict, accepted DAPI surface without resampling it."""
    volume = float(surface.mask_zyx.sum() * np.prod(surface.spacing_xyz_um))
    area = float(mesh_surface_area(surface.vertices_xyz_um, surface.faces))
    curvature = surface.mean_curvature_um_inverse
    return {
        "volume_um3": volume,
        "area_to_volume_um_inverse": area / volume,
        "sphericity": float(
            np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / area
        ),
        "mean_curvature_p25_um_inverse": float(np.percentile(curvature, 25.0)),
        "mean_curvature_p50_um_inverse": float(np.percentile(curvature, 50.0)),
        "mean_curvature_p75_um_inverse": float(np.percentile(curvature, 75.0)),
    }


def _ensemble_row(
    cell: CellSpec,
    round_number: int,
    endpoint: str,
    name: str,
    ensemble_value: float,
    sigma_um: float,
    threshold_sbr: float,
    surface_tolerance_resolution_units: float | None,
    value: float | str | None,
    units: str,
    config: AnalysisConfig,
    nucleus_id: str,
) -> EnsembleRow:
    """Keep one declared ensemble coordinate beside every derived endpoint."""
    return EnsembleRow(
        cohort=cell.cohort,
        cell_id=cell.cell_id,
        acquisition_round=round_number,
        endpoint=endpoint,
        ensemble_name=name,
        ensemble_value=ensemble_value,
        sigma_um=sigma_um,
        threshold_sbr=threshold_sbr,
        surface_tolerance_resolution_units=surface_tolerance_resolution_units,
        value=math.nan if value is None else value,
        units=units,
        source_task_identity=config.config_digest,
        nucleus_id=nucleus_id,
    )


def _lamin_ensemble_rows_for_member(
    cell: CellSpec,
    round_number: int,
    reference: _BoundedLaminMask,
    current: _BoundedLaminMask,
    resolution: ResolutionReference,
    config: AnalysisConfig,
) -> tuple[EnsembleRow, ...]:
    """Measure one disposable bounded member against its matching reference."""
    if (
        reference.nucleus_id != current.nucleus_id
        or reference.region_zyx != current.region_zyx
        or reference.origin_xyz_um != current.origin_xyz_um
        or reference.sigma_um != current.sigma_um
        or reference.threshold_sbr != current.threshold_sbr
        or reference.mask_zyx.shape != current.mask_zyx.shape
    ):
        raise ValueError("bounded Lamin comparison metadata does not match")
    geometry = _geometry(current.mask_zyx, config.voxel_spacing_xyz_um)
    surface_metrics = resolution_surface_metrics(
        reference.mask_zyx,
        current.mask_zyx,
        config.voxel_spacing_xyz_um,
        resolution,
        config.ensembles.surface_tolerance_resolution_units,
    )
    name = f"sigma={current.sigma_um:g};sbr={current.threshold_sbr:g}"
    rows = [
        _ensemble_row(
            cell,
            round_number,
            f"lamin_mask_{endpoint}",
            name,
            current.threshold_sbr,
            current.sigma_um,
            current.threshold_sbr,
            None,
            value,
            units,
            config,
            current.nucleus_id,
        )
        for endpoint, value, units in (
            ("volume", geometry["volume_um3"], "um3"),
            ("area_to_volume", geometry["area_to_volume_um_inverse"], "1/um"),
            ("sphericity", geometry["sphericity"], "fraction"),
            (
                "mean_curvature_p25",
                geometry["mean_curvature_p25_um_inverse"],
                "1/um",
            ),
            (
                "mean_curvature_p50",
                geometry["mean_curvature_p50_um_inverse"],
                "1/um",
            ),
            (
                "mean_curvature_p75",
                geometry["mean_curvature_p75_um_inverse"],
                "1/um",
            ),
            (
                "valid_curvature_vertex_fraction",
                geometry["valid_curvature_vertex_fraction"],
                "fraction",
            ),
            ("geometry_status", geometry["geometry_status"], "status"),
        )
    ]
    rows.extend(
        (
            _ensemble_row(
                cell,
                round_number,
                "lamin_jaccard",
                name,
                current.threshold_sbr,
                current.sigma_um,
                current.threshold_sbr,
                None,
                surface_metrics.jaccard,
                "fraction",
                config,
                current.nucleus_id,
            ),
            _ensemble_row(
                cell,
                round_number,
                "lamin_fixed_to_moving_boundary_distance_p95",
                name,
                current.threshold_sbr,
                current.sigma_um,
                current.threshold_sbr,
                None,
                surface_metrics.fixed_to_moving_normalized_distance_p95,
                "resolution-units",
                config,
                current.nucleus_id,
            ),
            _ensemble_row(
                cell,
                round_number,
                "lamin_moving_to_fixed_boundary_distance_p95",
                name,
                current.threshold_sbr,
                current.sigma_um,
                current.threshold_sbr,
                None,
                surface_metrics.moving_to_fixed_normalized_distance_p95,
                "resolution-units",
                config,
                current.nucleus_id,
            ),
        )
    )
    rows.extend(
        _ensemble_row(
            cell,
            round_number,
            "lamin_surface_dice",
            name,
            tolerance,
            current.sigma_um,
            current.threshold_sbr,
            tolerance,
            dice,
            "fraction",
            config,
            current.nucleus_id,
        )
        for tolerance, dice in surface_metrics.surface_dice.items()
    )
    return tuple(rows)


def _aggregate_metric_rows(rows: tuple[MetricRow, ...]) -> tuple[MetricRow, ...]:
    """Take field medians over nucleus-level metrics without pooling vertices."""
    grouped: dict[tuple[object, ...], list[MetricRow]] = {}
    for row in rows:
        if row.nucleus_id == "aggregate":
            continue
        key = (
            row.cohort,
            row.cell_id,
            row.acquisition_round,
            row.endpoint,
            row.units,
            row.comparison_set,
            row.qc_included,
            row.source_task_identity,
        )
        grouped.setdefault(key, []).append(row)
    aggregated: list[MetricRow] = []
    for members in grouped.values():
        first = members[0]
        finite_values = [row.value for row in members if math.isfinite(row.value)]
        aggregated.append(
            MetricRow(
                first.cohort,
                first.cell_id,
                first.acquisition_round,
                first.endpoint,
                float(np.median(finite_values)) if finite_values else math.nan,
                first.units,
                first.comparison_set,
                first.qc_included,
                first.source_task_identity,
                "aggregate",
            )
        )
    return tuple(aggregated)


def _aggregate_ensemble_rows(
    rows: tuple[EnsembleRow, ...],
) -> tuple[EnsembleRow, ...]:
    """Take field medians over matching nucleus-level ensemble coordinates."""
    grouped: dict[tuple[object, ...], list[EnsembleRow]] = {}
    for row in rows:
        if row.nucleus_id == "aggregate":
            continue
        key = (
            row.cohort,
            row.cell_id,
            row.acquisition_round,
            row.endpoint,
            row.sigma_um,
            row.threshold_sbr,
            row.surface_tolerance_resolution_units,
            row.units,
            row.source_task_identity,
        )
        grouped.setdefault(key, []).append(row)
    aggregated: list[EnsembleRow] = []
    for members in grouped.values():
        first = members[0]
        values = [row.value for row in members]
        if all(isinstance(value, str) for value in values):
            statuses = {str(value) for value in values}
            aggregate_value: float | str = (
                str(values[0]) if len(statuses) == 1 else "mixed"
            )
        elif all(not isinstance(value, str) for value in values):
            finite_values = [
                float(value) for value in values if math.isfinite(float(value))
            ]
            aggregate_value = (
                float(np.median(finite_values)) if finite_values else math.nan
            )
        else:
            raise ValueError(
                "one ensemble endpoint cannot mix numeric and status values"
            )
        aggregated.append(replace(first, value=aggregate_value, nucleus_id="aggregate"))
    return tuple(aggregated)


def _apply_lamin_nuclear_exclusions(
    adjudication: NuclearComponentAdjudication,
    exclusions: tuple[LaminNuclearExclusion, ...],
) -> NuclearComponentAdjudication:
    """Retain explicit spatial QC rejections before strict surface realization.

    Exclusions must already be restricted to this field. Matching uses the
    nearest grid voxel and an exact foreground-count guard. Missing, repeated,
    or changed matches raise ValueError; the input adjudication is not mutated.
    """
    candidates = list(adjudication._candidates)
    rows = list(adjudication.component_qc)
    diagnostics = list(adjudication.component_diagnostics)
    spacing = np.asarray(adjudication.spacing_xyz_um)
    for rule in exclusions:
        point = np.asarray(rule.point_xyz_um, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)) or np.any(point < 0):
            raise ValueError("exclusion point must be finite nonnegative XYZ")
        index = np.rint(point / spacing).astype(np.int64)[::-1]
        matches = []
        for i, candidate in enumerate(candidates):
            geometry = candidate.accepted_geometry
            if not candidate.accepted or geometry is None:
                continue
            local = index - np.array([s.start for s in geometry.region_zyx])
            if (
                np.all(local >= 0)
                and np.all(local < geometry.mask_zyx.shape)
                and geometry.mask_zyx[tuple(local)]
            ):
                matches.append(i)
        if len(matches) != 1:
            raise ValueError(
                "Lamin nuclear exclusion must match one accepted component"
            )
        i = matches[0]
        geometry = candidates[i].accepted_geometry
        assert geometry is not None
        if int(geometry.mask_zyx.sum()) != rule.expected_voxel_count:
            raise ValueError("Lamin nuclear exclusion voxel count changed")
        reasons = (*candidates[i].rejection_reasons, f"explicit-qc:{rule.reason}")
        candidates[i] = replace(
            candidates[i],
            accepted=False,
            rejection_reasons=reasons,
            accepted_geometry=None,
        )
        rows[i] = replace(rows[i], accepted=False, rejection_reasons=reasons)
        diagnostics[i] = replace(
            diagnostics[i], accepted=False, rejection_reasons=reasons
        )
    return replace(
        adjudication,
        _candidates=tuple(candidates),
        component_qc=tuple(rows),
        component_diagnostics=tuple(diagnostics),
    )


def analyze_lamin_cell(
    cell: CellSpec,
    artifacts: tuple[RoundArtifacts, ...],
    config: AnalysisConfig,
    output_dir: Path,
) -> MarkerResult:
    """Analyze one QC-eligible Lamin cell without any group-level inference."""
    if cell.cohort != "lamin":
        raise ValueError("Lamin analysis requires a lamin CellSpec")
    by_round = {artifact.acquisition_round: artifact for artifact in artifacts}
    if set(by_round) != {round_spec.acquisition_round for round_spec in cell.rounds}:
        raise ValueError(
            "cell and validated artifact acquisition-round inventories disagree"
        )
    accepted = _accepted_rounds(cell, config)
    dapi = {
        round_number: load_volume(by_round[round_number].registered_dapi)
        for round_number in accepted
    }
    lamin = {
        round_number: load_volume(by_round[round_number].registered_target)
        for round_number in accepted
    }
    adjudication = adjudicate_nuclear_components(
        dapi,
        config.voxel_spacing_xyz_um,
        config.nuclear,
        config.nominal_expansion_factor,
    )
    adjudication = _apply_lamin_nuclear_exclusions(
        adjudication,
        tuple(
            rule
            for rule in config.lamin_nuclear_exclusions
            if rule.cell_id == cell.cell_id
        ),
    )
    consensus = realize_nuclear_surfaces(adjudication, config.nuclear)
    del adjudication
    backgrounds = {
        round_number: robust_background(image) for round_number, image in lamin.items()
    }
    resolution = theoretical_resolution(
        config.optics.target_wavelength_um,
        config.optics.numerical_aperture,
        config.optics.immersion_refractive_index,
        config.nominal_expansion_factor,
    )
    profiles_by_nucleus: dict[str, tuple[LaminProfileRow, ...]] = {}
    profile_rows: list[LaminProfileRow] = []
    summary_rows: list[MetricRow] = []
    for surface in consensus.nuclei:
        profiles = lamin_normal_profiles(surface, lamin, config.nuclear, backgrounds)
        profiles_by_nucleus[surface.nucleus_id] = profiles
        profile_rows.extend(profiles)
        nuclear_geometry = _nuclear_surface_geometry(surface)
        summary_rows.extend(
            MetricRow(
                cell.cohort,
                cell.cell_id,
                REFERENCE_ROUND,
                f"consensus_nuclear_{endpoint}",
                value,
                units,
                "QC-accepted-DAPI-rounds",
                True,
                config.config_digest,
                surface.nucleus_id,
            )
            for endpoint, value, units in (
                ("volume", nuclear_geometry["volume_um3"], "um3"),
                (
                    "area_to_volume",
                    nuclear_geometry["area_to_volume_um_inverse"],
                    "1/um",
                ),
                ("sphericity", nuclear_geometry["sphericity"], "fraction"),
                (
                    "mean_curvature_p25",
                    nuclear_geometry["mean_curvature_p25_um_inverse"],
                    "1/um",
                ),
                (
                    "mean_curvature_p50",
                    nuclear_geometry["mean_curvature_p50_um_inverse"],
                    "1/um",
                ),
                (
                    "mean_curvature_p75",
                    nuclear_geometry["mean_curvature_p75_um_inverse"],
                    "1/um",
                ),
            )
        )
    rois = tuple(
        _bounded_nuclear_roi(
            surface.nucleus_id,
            surface.mask_zyx,
            config.voxel_spacing_xyz_um,
        )
        for surface in consensus.nuclei
    )
    reference_masks = _build_lamin_reference_masks(
        lamin[REFERENCE_ROUND],
        rois,
        config.ensembles.gaussian_sigma_um,
        config.ensembles.threshold_sbr,
        config.voxel_spacing_xyz_um,
        backgrounds[REFERENCE_ROUND],
    )
    ensemble_blocks: dict[tuple[str, int, float, float], tuple[EnsembleRow, ...]] = {}
    for round_number in accepted:
        if round_number == REFERENCE_ROUND:
            current_members: Iterator[_BoundedLaminMask] = (
                reference_masks.members[(roi.nucleus_id, sigma_um, threshold)]
                for sigma_um in config.ensembles.gaussian_sigma_um
                for roi in rois
                for threshold in config.ensembles.threshold_sbr
            )
        else:
            current_members = _iter_bounded_lamin_masks(
                lamin[round_number],
                rois,
                config.ensembles.gaussian_sigma_um,
                config.ensembles.threshold_sbr,
                config.voxel_spacing_xyz_um,
                backgrounds[round_number],
            )
        for current in current_members:
            key = (current.nucleus_id, current.sigma_um, current.threshold_sbr)
            reference = reference_masks.members[key]
            block_key = (
                current.nucleus_id,
                round_number,
                current.sigma_um,
                current.threshold_sbr,
            )
            ensemble_blocks[block_key] = _lamin_ensemble_rows_for_member(
                cell, round_number, reference, current, resolution, config
            )
            del current
    ensemble_rows = [
        row
        for surface in consensus.nuclei
        for round_number in accepted
        for sigma_um in config.ensembles.gaussian_sigma_um
        for threshold in config.ensembles.threshold_sbr
        for row in ensemble_blocks[
            (surface.nucleus_id, round_number, sigma_um, threshold)
        ]
    ]
    del reference_masks
    summary_rows[:0] = _metric_rows(cell, tuple(profile_rows), config)
    per_nucleus_summary = tuple(summary_rows)
    per_nucleus_ensemble = tuple(ensemble_rows)
    all_summary_rows = (
        *per_nucleus_summary,
        *_aggregate_metric_rows(per_nucleus_summary),
    )
    all_ensemble_rows = (
        *per_nucleus_ensemble,
        *_aggregate_ensemble_rows(per_nucleus_ensemble),
    )
    staged_diagnostics: tuple[Path, ...] = ()
    staging_root: Path | None = None

    def populate_diagnostics(staged: Path) -> None:
        nonlocal staged_diagnostics, staging_root
        staging_root = staged
        paths: list[Path] = [
            write_nuclear_component_diagnostic(staged, cell.cell_id, dapi, consensus)
        ]
        label_path = staged / "consensus-nuclear-labels.npz"
        np.savez_compressed(
            label_path,
            labels_zyx=consensus.labels_zyx,
            spacing_xyz_um=consensus.spacing_xyz_um,
            nucleus_ids=np.asarray(
                [surface.nucleus_id for surface in consensus.nuclei]
            ),
        )
        paths.append(label_path)
        for surface in consensus.nuclei:
            paths.extend(
                _write_profile_diagnostics(
                    staged / surface.nucleus_id,
                    cell.cell_id,
                    surface,
                    profiles_by_nucleus[surface.nucleus_id],
                )
            )
        staged_diagnostics = tuple(paths)

    _atomic_publish_directory(output_dir, populate_diagnostics)
    assert staging_root is not None
    diagnostics = tuple(
        output_dir / path.relative_to(staging_root) for path in staged_diagnostics
    )
    identities = tuple(
        str(path)
        for artifact in artifacts
        for path in (artifact.registered_dapi, artifact.registered_target)
    )
    return MarkerResult(
        cell.cohort,
        cell.cell_id,
        all_summary_rows,
        all_ensemble_rows,
        diagnostics,
        identities,
        tuple(MappingProxyType(asdict(row)) for row in consensus.component_qc),
    )


__all__ = [
    "LaminProfileRow",
    "NuclearComponentAdjudication",
    "NuclearComponentDiagnostic",
    "NuclearComponentProjection",
    "NuclearComponentQc",
    "NuclearConsensus",
    "NuclearSurface",
    "adjudicate_nuclear_components",
    "analyze_lamin_cell",
    "consensus_nuclear_mask",
    "consensus_nuclear_surface",
    "lamin_normal_profiles",
    "realize_nuclear_surfaces",
    "write_nuclear_component_diagnostic",
]
