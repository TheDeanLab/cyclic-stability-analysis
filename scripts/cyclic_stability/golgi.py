"""Per-cell Golgi continuous-image and threshold-ensemble stability measures."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
from cyexm.registration._request_policy import channel_transform_parameters
from cyexm.registration.artifacts import FileIdentity
from cyexm.registration.backend import AntsRegistrationBackend
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, label
from scipy.spatial import cKDTree

from cyclic_stability.cyexm_artifacts import RoundArtifacts
from cyclic_stability.images import BackgroundSummary, load_volume, robust_background
from cyclic_stability.models import (
    AnalysisConfig,
    CellSpec,
    EnsembleRow,
    MarkerResult,
    MetricRow,
)
from cyclic_stability.morphology import (
    MaskMember,
    gaussian_sigma_zyx,
    resolution_surface_metrics,
    validated_binary_mask,
)
from cyclic_stability.nuclear import (
    REFERENCE_ROUND,
    NuclearConsensus,
    NuclearSurface,
    _accepted_rounds,
    _atomic_publish_directory,
    consensus_nuclear_surface,
    write_nuclear_component_diagnostic,
)
from cyclic_stability.units import theoretical_resolution


@dataclass(frozen=True, slots=True)
class GradientSimilarityRow:
    """One physical smoothing-scale gradient correlation in the fixed perinuclear ROI."""

    sigma_um: float
    correlation: float


@dataclass(frozen=True, slots=True)
class _SharedCropBounds:
    """One conservative fixed-grid ZYX crop shared by every accepted round."""

    start_zyx: tuple[int, int, int]
    stop_zyx: tuple[int, int, int]
    full_shape_zyx: tuple[int, int, int]

    @property
    def retained_fraction(self) -> float:
        """Return the fraction of the fixed grid retained by this crop."""
        retained_voxels = math.prod(
            stop - start
            for start, stop in zip(self.start_zyx, self.stop_zyx, strict=True)
        )
        return retained_voxels / math.prod(self.full_shape_zyx)

    @property
    def slices_zyx(self) -> tuple[slice, slice, slice]:
        """Return the crop as ZYX slices."""
        return tuple(
            slice(start, stop)
            for start, stop in zip(self.start_zyx, self.stop_zyx, strict=True)
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class _GolgiArrayAnalysis:
    """Numerical Golgi result together with reusable diagnostic intermediates."""

    result: MarkerResult
    consensus: NuclearConsensus
    roi: NDArray[np.bool_]
    threshold_masks: Mapping[int, tuple[MaskMember, ...]]


@dataclass(frozen=True, slots=True)
class _GolgiCropPlan:
    """Small target-specific crop evidence produced after target-blind QC."""

    crop: _SharedCropBounds
    dapi_bounds_by_round: Mapping[
        int, tuple[tuple[int, int, int], tuple[int, int, int]]
    ]
    golgi_bounds_by_round: Mapping[
        int, tuple[tuple[int, int, int], tuple[int, int, int]] | None
    ]
    dapi_support_mips: Mapping[int, NDArray[np.bool_]]
    golgi_support_mips: Mapping[int, NDArray[np.bool_]]
    sample_step_zyx: tuple[int, int, int]


_MAX_RETAINED_GOLGI_CROP_FRACTION = 0.75


ENSEMBLE_ENDPOINTS = (
    "absolute_background_corrected_signal",
    "volume_um3",
    "component_count",
    "component_volume_gini",
    "fragmentation_per_um3",
    "compactness",
    "area_to_volume_um_inverse",
    "nuclear_distance_p25_um",
    "nuclear_distance_p50_um",
    "nuclear_distance_p75_um",
    "jaccard",
    "fixed_to_moving_boundary_distance_p95",
    "moving_to_fixed_boundary_distance_p95",
)


def _mask_bounds(
    mask: NDArray[np.bool_],
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """Return exclusive ZYX bounds for one non-empty binary support mask."""
    positive = validated_binary_mask(mask, "coarse support mask")
    if positive.ndim != 3:
        raise ValueError("coarse support mask must be 3D ZYX")
    occupied = tuple(
        np.flatnonzero(np.any(positive, axis=axes)) for axes in ((1, 2), (0, 2), (0, 1))
    )
    if any(axis.size == 0 for axis in occupied):
        return None
    return (
        tuple(int(axis[0]) for axis in occupied),
        tuple(int(axis[-1]) + 1 for axis in occupied),
    )


def _margin_zyx(
    distance_um: float, spacing_xyz_um: tuple[float, float, float]
) -> tuple[int, int, int]:
    """Convert a physical conservative distance into ceiling ZYX pixel margins."""
    sx, sy, sz = spacing_xyz_um
    return tuple(math.ceil(distance_um / value) for value in (sz, sy, sx))


def _shared_crop_bounds_from_support_masks(
    supports: Mapping[int, tuple[NDArray[np.bool_], NDArray[np.bool_]]],
    config: AnalysisConfig,
) -> _SharedCropBounds:
    """Bound all loose DAPI and Golgi supports with one conservative crop.

    DAPI receives the full exterior-ROI distance plus three standard deviations of
    the widest frozen Gaussian. Golgi receives the smoothing margin alone. Any
    missing support retains the full fixed grid rather than guessing a crop.
    """
    if not supports:
        raise ValueError("shared Golgi crop requires at least one accepted round")
    shape: tuple[int, int, int] | None = None
    starts: list[tuple[int, int, int]] = []
    stops: list[tuple[int, int, int]] = []
    smoothing_margin_um = 3.0 * max(config.ensembles.gaussian_sigma_um)
    dapi_margin = _margin_zyx(
        config.golgi.perinuclear_outer_distance_um + smoothing_margin_um,
        config.voxel_spacing_xyz_um,
    )
    golgi_margin = _margin_zyx(smoothing_margin_um, config.voxel_spacing_xyz_um)
    for dapi_support, golgi_support in supports.values():
        for support_name, support, margin in (
            ("DAPI", dapi_support, dapi_margin),
            ("Golgi", golgi_support, golgi_margin),
        ):
            binary = validated_binary_mask(support, f"coarse {support_name} support")
            if binary.ndim != 3:
                raise ValueError(f"coarse {support_name} support must be 3D ZYX")
            if shape is None:
                shape = tuple(int(value) for value in binary.shape)
            elif binary.shape != shape:
                raise ValueError(
                    "coarse DAPI and Golgi supports must share fixed-grid geometry"
                )
            bounds = _mask_bounds(binary)
            if bounds is None:
                assert shape is not None
                return _SharedCropBounds((0, 0, 0), shape, shape)
            start, stop = bounds
            starts.append(
                tuple(
                    max(0, value - pad)
                    for value, pad in zip(start, margin, strict=True)
                )
            )
            stops.append(
                tuple(
                    min(limit, value + pad)
                    for value, pad, limit in zip(stop, margin, shape, strict=True)
                )
            )
    assert shape is not None
    if not starts:
        return _SharedCropBounds((0, 0, 0), shape, shape)
    return _SharedCropBounds(
        tuple(min(start[axis] for start in starts) for axis in range(3)),
        tuple(max(stop[axis] for stop in stops) for axis in range(3)),
        shape,
    )


def _sampled_support(
    image: NDArray[np.generic],
    *,
    threshold_sbr: float,
    sample_step_zyx: tuple[int, int, int],
    maximum_components: int,
    minimum_quantile: float | None = None,
) -> tuple[tuple[tuple[int, int, int], tuple[int, int, int]], NDArray[np.bool_]] | None:
    """Return full-resolution bounds and sampled mask of plausible support."""
    source = np.asarray(image)
    if source.ndim != 3 or not np.all(np.isfinite(source)):
        raise ValueError("coarse support source must be one finite 3D ZYX image")
    if (
        any(step <= 0 for step in sample_step_zyx)
        or maximum_components <= 0
        or not math.isfinite(threshold_sbr)
        or threshold_sbr <= 0.0
        or (
            minimum_quantile is not None
            and (
                not math.isfinite(minimum_quantile) or not 0.0 < minimum_quantile < 1.0
            )
        )
    ):
        raise ValueError("coarse support settings must be finite and positive")
    sample = source[:: sample_step_zyx[0], :: sample_step_zyx[1], :: sample_step_zyx[2]]
    try:
        background = robust_background(sample)
    except ValueError:
        return None
    threshold = background.median + threshold_sbr * background.scale
    if minimum_quantile is not None:
        threshold = max(threshold, float(np.quantile(sample, minimum_quantile)))
    support = sample >= threshold
    labels, count = label(support, structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        return None
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.max())
    minimum_size = max(2, math.ceil(largest * 0.02))
    selected = np.flatnonzero(sizes >= minimum_size)
    if selected.size == 0:
        return None
    selected = selected[np.argsort(sizes[selected])[::-1][:maximum_components]] + 1
    retained = np.isin(labels, selected)
    bounds = _mask_bounds(retained)
    if bounds is None:
        return None
    start, stop = bounds
    full_start = tuple(
        index * step for index, step in zip(start, sample_step_zyx, strict=True)
    )
    full_stop = tuple(
        min(limit, index * step)
        for index, step, limit in zip(stop, sample_step_zyx, source.shape, strict=True)
    )
    retained.setflags(write=False)
    return (full_start, full_stop), retained


def _sampled_support_bounds(
    image: NDArray[np.generic],
    *,
    threshold_sbr: float,
    sample_step_zyx: tuple[int, int, int],
    maximum_components: int,
    minimum_quantile: float | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """Return full-resolution bounds of plausible supports from a small 3D sample."""
    support = _sampled_support(
        image,
        threshold_sbr=threshold_sbr,
        sample_step_zyx=sample_step_zyx,
        maximum_components=maximum_components,
        minimum_quantile=minimum_quantile,
    )
    return None if support is None else support[0]


def _shared_crop_bounds_from_artifacts(
    artifacts: Mapping[int, RoundArtifacts],
    accepted: tuple[int, ...],
    config: AnalysisConfig,
) -> _SharedCropBounds:
    """Plan one crop sequentially, retaining support bounds rather than image stacks."""
    return _golgi_crop_plan_from_artifacts(artifacts, accepted, config).crop


def _golgi_crop_plan_from_artifacts(
    artifacts: Mapping[int, RoundArtifacts],
    accepted: tuple[int, ...],
    config: AnalysisConfig,
) -> _GolgiCropPlan:
    """Build bounded target-specific crop evidence after target-blind QC passes."""
    shape: tuple[int, int, int] | None = None
    starts: list[tuple[int, int, int]] = []
    stops: list[tuple[int, int, int]] = []
    dapi_bounds_by_round: dict[
        int, tuple[tuple[int, int, int], tuple[int, int, int]]
    ] = {}
    golgi_bounds_by_round: dict[
        int, tuple[tuple[int, int, int], tuple[int, int, int]] | None
    ] = {}
    dapi_support_mips: dict[int, NDArray[np.bool_]] = {}
    golgi_support_mips: dict[int, NDArray[np.bool_]] = {}
    threshold_sbr = max(8.0, min(config.ensembles.threshold_sbr))
    sample_step_zyx = (2, 8, 8)
    smoothing_margin_um = 3.0 * max(config.ensembles.gaussian_sigma_um)
    dapi_margin = _margin_zyx(
        config.golgi.perinuclear_outer_distance_um + smoothing_margin_um,
        config.voxel_spacing_xyz_um,
    )
    for number in accepted:
        artifact = artifacts[number]
        dapi_image = load_volume(artifact.registered_dapi)
        golgi_image = load_volume(artifact.registered_target)
        if shape is None:
            shape = tuple(int(value) for value in dapi_image.shape)
        if dapi_image.shape != shape or golgi_image.shape != shape:
            raise ValueError("registered Golgi rounds must share fixed-grid geometry")
        dapi_support = _sampled_support(
            dapi_image,
            threshold_sbr=threshold_sbr,
            sample_step_zyx=sample_step_zyx,
            maximum_components=6,
            minimum_quantile=0.995,
        )
        golgi_support = _sampled_support(
            golgi_image,
            threshold_sbr=threshold_sbr,
            sample_step_zyx=sample_step_zyx,
            maximum_components=6,
        )
        if dapi_support is None:
            assert shape is not None
            return _GolgiCropPlan(
                _SharedCropBounds((0, 0, 0), shape, shape),
                MappingProxyType(dapi_bounds_by_round),
                MappingProxyType(golgi_bounds_by_round),
                MappingProxyType(dapi_support_mips),
                MappingProxyType(golgi_support_mips),
                sample_step_zyx,
            )
        dapi_bounds, dapi_mask = dapi_support
        dapi_bounds_by_round[number] = dapi_bounds
        dapi_support_mips[number] = dapi_mask.max(axis=0)
        if golgi_support is None:
            golgi_bounds = None
            golgi_support_mips[number] = np.zeros(dapi_mask.shape[1:], dtype=bool)
        else:
            golgi_bounds, golgi_mask = golgi_support
            golgi_support_mips[number] = golgi_mask.max(axis=0)
        golgi_bounds_by_round[number] = golgi_bounds
        # The fixed round defines the common measurement domain.  Unioning every
        # registered moving support allows a residual edge/background artifact to
        # re-expand a bounded analysis back to the full image.  Moving-round
        # support remains published as QC evidence, while transform-valid masks
        # exclude any unsupported moving pixels within this fixed anchored crop.
        if number != REFERENCE_ROUND:
            continue
        for bounds, margin in ((dapi_bounds, dapi_margin),):
            start, stop = bounds
            starts.append(
                tuple(
                    max(0, value - pad)
                    for value, pad in zip(start, margin, strict=True)
                )
            )
            stops.append(
                tuple(
                    min(limit, value + pad)
                    for value, pad, limit in zip(stop, margin, shape, strict=True)
                )
            )
        if golgi_bounds is None:
            continue
        # Golgi morphology is explicitly measured only inside the DAPI-defined
        # perinuclear domain.  Keep the independent target support in the
        # preflight evidence, but never let distal target signal enlarge the
        # fixed measurement volume.  The DAPI envelope already includes the
        # full declared perinuclear distance and three Gaussian sigmas.
    assert shape is not None
    crop = _SharedCropBounds(
        tuple(min(start[axis] for start in starts) for axis in range(3)),
        tuple(max(stop[axis] for stop in stops) for axis in range(3)),
        shape,
    )
    return _GolgiCropPlan(
        crop,
        MappingProxyType(dapi_bounds_by_round),
        MappingProxyType(golgi_bounds_by_round),
        MappingProxyType(dapi_support_mips),
        MappingProxyType(golgi_support_mips),
        sample_step_zyx,
    )


def _ensemble_units(config: AnalysisConfig) -> dict[str, str]:
    """Return the fixed units schema for every non-image ensemble endpoint."""
    return {
        "absolute_background_corrected_signal": "a.u.",
        "volume_um3": "um3",
        "component_count": "count",
        "component_volume_gini": "fraction",
        "fragmentation_per_um3": "1/um3",
        "compactness": "fraction",
        "area_to_volume_um_inverse": "1/um",
        "nuclear_distance_p25_um": "um",
        "nuclear_distance_p50_um": "um",
        "nuclear_distance_p75_um": "um",
        "jaccard": "fraction",
        "fixed_to_moving_boundary_distance_p95": "resolution-units",
        "moving_to_fixed_boundary_distance_p95": "resolution-units",
        **{
            f"surface_dice_{tolerance:g}": "fraction"
            for tolerance in config.ensembles.surface_tolerance_resolution_units
        },
    }


def _volume(image: NDArray[np.generic], name: str) -> NDArray[np.float64]:
    """Validate a finite real ZYX volume at this public numerical boundary."""
    value = np.asarray(image)
    if (
        value.ndim != 3
        or not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be one real numeric 3D ZYX volume")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value.astype(np.float64, copy=False)


def _spacing(config: AnalysisConfig) -> tuple[float, float, float]:
    """Validate public XYZ spacing through the shared physical-sigma primitive."""
    # ``gaussian_sigma_zyx`` owns the canonical public-XYZ validation.
    gaussian_sigma_zyx(
        config.ensembles.gaussian_sigma_um[0], config.voxel_spacing_xyz_um
    )
    return config.voxel_spacing_xyz_um


def _gradient_magnitude(
    image: NDArray[np.float64],
    validity_mask: NDArray[np.bool_],
    spacing_xyz_um: tuple[float, float, float],
    sigma_um: float,
) -> NDArray[np.float64]:
    """Background-correct, physically smooth, and differentiate one target volume."""
    background = _support_background(image, validity_mask)
    # Gradients are invariant to the robust offset; retaining negative corrected
    # values avoids fabricating a zero-gradient plateau at the support boundary.
    corrected = image - background.median
    smoothed = _masked_gaussian(
        corrected, validity_mask, gaussian_sigma_zyx(sigma_um, spacing_xyz_um)
    )
    sx, sy, sz = spacing_xyz_um
    gradients = np.gradient(smoothed, sz, sy, sx)
    return np.sqrt(sum(gradient * gradient for gradient in gradients))


def _support_background(
    image: NDArray[np.float64], validity_mask: NDArray[np.bool_]
) -> BackgroundSummary:
    """Estimate robust background solely from the common transform-valid support."""
    return robust_background(np.where(validity_mask, image, 0.0))


def _masked_gaussian(
    image: NDArray[np.float64],
    validity_mask: NDArray[np.bool_],
    sigma_zyx: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Smooth valid samples with normalized convolution, never padding intensities."""
    support = validity_mask.astype(np.float64, copy=False)
    denominator = gaussian_filter(support, sigma_zyx)
    numerator = gaussian_filter(image * support, sigma_zyx)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(np.float64).eps,
    )


def _effective_gradient_roi(
    roi: NDArray[np.bool_], validity_mask: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """Exclude all samples whose three-dimensional gradient stencil touches padding.

    A one-voxel 26-neighbour erosion makes the central-difference policy used by
    :func:`numpy.gradient` explicit.  Thin support fails rather than acquiring a
    favourable derivative from a padded neighbour.
    """
    effective = roi & binary_erosion(
        validity_mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0
    )
    if not np.any(effective):
        raise ValueError(
            "gradient similarity ROI is empty after support-boundary exclusion"
        )
    return effective


def multiscale_gradient_similarity(
    reference: NDArray[np.generic],
    moving: NDArray[np.generic],
    roi: NDArray[np.bool_],
    spacing_xyz_um: tuple[float, float, float],
    sigma_values_um: tuple[float, ...],
    *,
    validity_mask: NDArray[np.bool_] | None = None,
) -> tuple[GradientSimilarityRow, ...]:
    """Correlate 3D gradient magnitudes at each frozen physical smoothing scale.

    The target-independent ``roi`` is fixed before any target thresholding.  Each
    image receives its own robust background correction before physical smoothing,
    so a global positive intensity scale and offset do not change this similarity.
    """
    fixed = _volume(reference, "reference")
    moved = _volume(moving, "moving")
    support = validated_binary_mask(roi, "ROI")
    if fixed.shape != moved.shape or fixed.shape != support.shape:
        raise ValueError("reference, moving, and ROI must be matching 3D ZYX volumes")
    if not np.any(support):
        raise ValueError("gradient similarity ROI is empty")
    if not sigma_values_um:
        raise ValueError("sigma_values_um must be non-empty")
    valid = (
        np.ones(fixed.shape, dtype=bool)
        if validity_mask is None
        else validated_binary_mask(validity_mask, "validity mask")
    )
    if valid.shape != fixed.shape:
        raise ValueError("validity mask must match reference geometry")
    if np.any(support & ~valid):
        raise ValueError("ROI must be contained in the validity mask")
    effective_roi = _effective_gradient_roi(support, valid)
    rows: list[GradientSimilarityRow] = []
    for raw_sigma in sigma_values_um:
        sigma_um = float(raw_sigma)
        fixed_gradient = _gradient_magnitude(fixed, valid, spacing_xyz_um, sigma_um)[
            effective_roi
        ]
        moving_gradient = _gradient_magnitude(moved, valid, spacing_xyz_um, sigma_um)[
            effective_roi
        ]
        if np.ptp(fixed_gradient) == 0.0 or np.ptp(moving_gradient) == 0.0:
            raise ValueError("gradient similarity has zero variance in the fixed ROI")
        correlation = float(np.corrcoef(fixed_gradient, moving_gradient)[0, 1])
        if not math.isfinite(correlation):
            raise ValueError("gradient similarity correlation is non-finite")
        rows.append(GradientSimilarityRow(sigma_um, correlation))
    return tuple(rows)


def golgi_nucleus_distances(
    mask: NDArray[np.bool_],
    nuclear_surfaces: NuclearSurface | tuple[NuclearSurface, ...],
    spacing_xyz_um: tuple[float, float, float],
    *,
    distance_field_zyx: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Return physical distances from Golgi voxel centers to accepted nuclei.

    Without a cached field, this retains the legacy nearest-vertex approximation to
    a triangulated surface. A supplied ``distance_field_zyx`` instead samples a
    physical fixed-grid exterior distance field for the same consensus nuclei and
    avoids constructing one nearest-neighbor query per threshold member.
    """
    positive = validated_binary_mask(mask, "Golgi mask")
    surfaces = (
        (nuclear_surfaces,)
        if isinstance(nuclear_surfaces, NuclearSurface)
        else nuclear_surfaces
    )
    if positive.ndim != 3 or not surfaces:
        raise ValueError("Golgi distance requires a 3D mask and accepted nuclei")
    if any(positive.shape != surface.mask_zyx.shape for surface in surfaces):
        raise ValueError("Golgi mask and nuclear surfaces must share 3D ZYX geometry")
    if any(
        tuple(spacing_xyz_um) != tuple(surface.spacing_xyz_um) for surface in surfaces
    ):
        raise ValueError("Golgi and nuclear-surface spacing must match")
    if distance_field_zyx is not None:
        field = np.asarray(distance_field_zyx)
        if (
            field.shape != positive.shape
            or not np.issubdtype(field.dtype, np.floating)
            or not np.all(np.isfinite(field))
            or np.any(field < 0.0)
        ):
            raise ValueError(
                "cached nuclear distance field must be finite, non-negative, and share mask geometry"
            )
        return np.asarray(field[positive], dtype=np.float64)
    indices_zyx = np.argwhere(positive)
    if indices_zyx.size == 0:
        return np.empty(0, dtype=np.float64)
    sx, sy, sz = spacing_xyz_um
    points_xyz = indices_zyx[:, ::-1] * np.array((sx, sy, sz))
    vertices_xyz_um = np.concatenate(
        tuple(surface.vertices_xyz_um for surface in surfaces), axis=0
    )
    return cKDTree(vertices_xyz_um).query(points_xyz, k=1)[0]


def _nuclear_distance_field(
    nuclear_surfaces: tuple[NuclearSurface, ...],
) -> NDArray[np.float32]:
    """Return one physical exterior distance field for all accepted nuclei."""
    if not nuclear_surfaces:
        raise ValueError(
            "nuclear distance field requires at least one accepted nucleus"
        )
    shape = nuclear_surfaces[0].mask_zyx.shape
    spacing = nuclear_surfaces[0].spacing_xyz_um
    if any(
        surface.mask_zyx.shape != shape or surface.spacing_xyz_um != spacing
        for surface in nuclear_surfaces
    ):
        raise ValueError("accepted nuclear surfaces must share fixed-grid geometry")
    sx, sy, sz = spacing
    union = np.logical_or.reduce(
        tuple(surface.mask_zyx for surface in nuclear_surfaces)
    )
    field = distance_transform_edt(~union, sampling=(sz, sy, sx)).astype(
        np.float32, copy=False
    )
    field.setflags(write=False)
    return field


def _component_volume_gini(volumes_um3: NDArray[np.float64]) -> float:
    """Return the finite-sample Gini coefficient of positive component volumes."""
    values = np.sort(np.asarray(volumes_um3, dtype=np.float64))
    if (
        values.ndim != 1
        or values.size == 0
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("component volumes must be a non-empty finite positive vector")
    if values.size == 1:
        return 0.0
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    population_gini = (
        2.0 * float(np.dot(ranks, values)) / (values.size * float(values.sum()))
        - (values.size + 1.0) / values.size
    )
    return float(population_gini * values.size / (values.size - 1.0))


def _mask_geometry(
    mask: NDArray[np.bool_], spacing_xyz_um: tuple[float, float, float]
) -> dict[str, float]:
    """Measure all positive components, preserving explicit empty-mask semantics."""
    positive = validated_binary_mask(mask, "Golgi mask")
    if positive.ndim != 3:
        raise ValueError("Golgi mask must be 3D ZYX")
    voxel_volume = float(np.prod(spacing_xyz_um))
    volume = float(positive.sum() * voxel_volume)
    if volume == 0.0:
        return {
            "volume_um3": 0.0,
            "component_count": 0.0,
            "component_volume_gini": math.nan,
            "fragmentation_per_um3": math.nan,
            "compactness": math.nan,
            "area_to_volume_um_inverse": math.nan,
        }
    labels, count = label(positive, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())[1:]
    component_volumes = sizes.astype(np.float64) * voxel_volume
    sx, sy, sz = spacing_xyz_um
    # Fixed-grid voxel-interface area is deliberately used instead of a dense
    # marching-cubes mesh.  The study compares the same acquisition grid and
    # threshold ensemble across rounds, so this stable discretization is the
    # relevant relative morphology measure and has bounded memory.
    area = float(
        np.count_nonzero(positive[0]) * sx * sy
        + np.count_nonzero(positive[-1]) * sx * sy
        + np.count_nonzero(positive[:, 0]) * sx * sz
        + np.count_nonzero(positive[:, -1]) * sx * sz
        + np.count_nonzero(positive[:, :, 0]) * sy * sz
        + np.count_nonzero(positive[:, :, -1]) * sy * sz
        + np.count_nonzero(positive[1:] != positive[:-1]) * sx * sy
        + np.count_nonzero(positive[:, 1:] != positive[:, :-1]) * sx * sz
        + np.count_nonzero(positive[:, :, 1:] != positive[:, :, :-1]) * sy * sz
    )
    return {
        "volume_um3": volume,
        "component_count": float(count),
        "component_volume_gini": _component_volume_gini(component_volumes),
        "fragmentation_per_um3": float(count / volume),
        "compactness": float(36.0 * math.pi * volume * volume / area**3),
        "area_to_volume_um_inverse": area / volume,
    }


def _perinuclear_roi(
    nuclear_mask: NDArray[np.bool_],
    validity_masks: Mapping[int, NDArray[np.bool_]],
    config: AnalysisConfig,
) -> NDArray[np.bool_]:
    """Build one fixed exterior ROI from explicit transform-valid support masks.

    A registered target can contain finite padding outside its transformed source
    support.  Therefore validity is an input derived from the transform chain, not
    from target intensities.
    """
    nucleus = validated_binary_mask(nuclear_mask, "consensus nuclear mask")
    if nucleus.ndim != 3:
        raise ValueError("consensus nuclear mask must be 3D ZYX")
    if not validity_masks:
        raise ValueError("Golgi ROI requires at least one registered round")
    valid = np.ones(nucleus.shape, dtype=bool)
    for round_number, mask in validity_masks.items():
        transform_valid = validated_binary_mask(
            mask, f"validity mask for round {round_number}"
        )
        if transform_valid.ndim != 3 or transform_valid.shape != nucleus.shape:
            raise ValueError(
                "transform validity masks and consensus nuclear mask must share geometry"
            )
        valid &= transform_valid
    sx, sy, sz = _spacing(config)
    exterior_distance = distance_transform_edt(~nucleus, sampling=(sz, sy, sx))
    roi = (
        (~nucleus)
        & (exterior_distance <= config.golgi.perinuclear_outer_distance_um)
        & valid
    )
    if not np.any(roi):
        raise ValueError("fixed perinuclear ROI is empty")
    roi.setflags(write=False)
    return roi


def _common_validity_masks(
    rounds: Mapping[int, NDArray[np.float64]],
    validity_masks: Mapping[int, NDArray[np.bool_]],
) -> dict[int, NDArray[np.bool_]]:
    """Validate explicit fixed-grid transform support for every registered round."""
    if set(validity_masks) != set(rounds):
        raise ValueError("transform validity masks must match registered Golgi rounds")
    result: dict[int, NDArray[np.bool_]] = {}
    for round_number, raw_mask in validity_masks.items():
        mask = validated_binary_mask(
            raw_mask, f"validity mask for round {round_number}"
        )
        if mask.ndim != 3 or mask.shape != rounds[round_number].shape:
            raise ValueError(
                "transform validity mask geometry must match its Golgi round"
            )
        result[round_number] = mask
    return result


def _support_threshold_ensemble(
    image: NDArray[np.float64],
    roi: NDArray[np.bool_],
    validity_mask: NDArray[np.bool_],
    config: AnalysisConfig,
) -> tuple[MaskMember, ...]:
    """Build the frozen Golgi mask grid using only common-valid source support."""
    background = _support_background(image, validity_mask)
    members: list[MaskMember] = []
    for sigma_um in config.ensembles.gaussian_sigma_um:
        smoothed = _masked_gaussian(
            image,
            validity_mask,
            gaussian_sigma_zyx(sigma_um, config.voxel_spacing_xyz_um),
        )
        for threshold_sbr in config.ensembles.threshold_sbr:
            mask = (
                smoothed > background.median + threshold_sbr * background.scale
            ) & roi
            mask.setflags(write=False)
            members.append(
                MaskMember(
                    mask,
                    sigma_um,
                    threshold_sbr,
                    background,
                    config.voxel_spacing_xyz_um,
                    "nonempty" if np.any(mask) else "empty",
                )
            )
    return tuple(members)


def _artifact_validity_masks(
    artifacts: Mapping[int, RoundArtifacts],
    config: AnalysisConfig,
    *,
    _backend: AntsRegistrationBackend | None = None,
    fixed_shape_zyx: tuple[int, int, int] | None = None,
) -> tuple[dict[int, NDArray[np.bool_]], str, tuple[str, ...]]:
    """Recreate support from the registered-target producer record alone.

    The target-transform completion record is authoritative for native source,
    fixed grid, recorded XYZ pixel size, and affine-before-warp chain.  Analysis
    spacing is deliberately not substituted for the producer geometry.
    """
    reference_artifact = artifacts[REFERENCE_ROUND]
    backend = _backend or AntsRegistrationBackend()
    fixed_shape = fixed_shape_zyx or tuple(
        load_volume(reference_artifact.registered_target).shape
    )
    reference_mask = np.ones(fixed_shape, dtype=bool)
    masks: dict[int, NDArray[np.bool_]] = {REFERENCE_ROUND: reference_mask}
    provenance: dict[str, object] = {
        "reference": {
            "registered_target": FileIdentity.from_path(
                reference_artifact.registered_target
            ).to_dict(),
            "semantics": "untransformed-fixed-reference-grid-all-valid-v1",
            "mask": _validity_mask_provenance(reference_mask),
        },
        "moving": [],
    }
    for round_number, artifact in artifacts.items():
        if round_number == REFERENCE_ROUND:
            continue
        if (
            artifact.native_target is None
            or artifact.affine is None
            or artifact.warp is None
        ):
            raise ValueError(
                "validated moving artifact lacks exact native target or transform chain"
            )
        target_record = next(
            (
                record
                for record in artifact.completion_records
                if record.artifact.path == str(artifact.registered_target.resolve())
            ),
            None,
        )
        if target_record is None or target_record.artifact != FileIdentity.from_path(
            artifact.registered_target
        ):
            raise ValueError("registered target lacks matching completion provenance")
        request = target_record.request
        if (
            request.stage != "channel-transform-v1"
            or request.method != "warp-affine"
            or request.accuracy != "not-applicable"
            or len(request.inputs) != 4
            or request.parameters.get("interpolation") != "linear"
        ):
            raise ValueError(
                "registered target is not an exact linear scalar channel transform"
            )
        try:
            spacing = tuple(
                float(value) for value in request.parameters["pixel_size_xyz_um"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "registered target lacks recorded producer pixel size"
            ) from error
        if len(spacing) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in spacing
        ):
            raise ValueError(
                "registered target has invalid recorded producer pixel size"
            )
        if request.parameters != channel_transform_parameters(
            pixel_size_xyz=spacing, interpolation="linear"
        ):
            raise ValueError(
                "registered target transform parameters do not match producer policy"
            )
        fixed_identity, source_identity, affine_identity, warp_identity = request.inputs
        for identity in request.inputs:
            try:
                current = FileIdentity.from_path(Path(identity.path))
            except (OSError, RuntimeError, ValueError) as error:
                raise ValueError(
                    "registered target input identity is unavailable"
                ) from error
            if current != identity:
                raise ValueError(
                    "registered target input identity changed after completion"
                )
        source = artifact.native_target
        selected_fixed_identity = FileIdentity.from_path(
            reference_artifact.registered_tubulin
        )
        fixed_reference_match = _fixed_reference_match(
            fixed_identity, selected_fixed_identity
        )
        if (
            source_identity != FileIdentity.from_path(source)
            or fixed_reference_match is None
            or affine_identity != FileIdentity.from_path(artifact.affine)
            or warp_identity != FileIdentity.from_path(artifact.warp)
            or request.upstream != warp_identity
        ):
            raise ValueError(
                "registered target inputs do not match the selected artifact chain"
            )
        fixed = backend.load_image(Path(fixed_identity.path), spacing)
        moving = backend.load_image(source, spacing)
        if backend.geometry(fixed) != request.geometry:
            raise ValueError(
                "registered target fixed geometry differs from its completion request"
            )
        ones = backend.from_array(
            np.ones(
                tuple(np.asarray(backend.to_array(moving)).shape), dtype=np.float32
            ),
            reference=moving,
        )
        transformed = backend.apply_transform_chain(
            fixed=fixed,
            moving=ones,
            affine_path=artifact.affine,
            warp_path=artifact.warp,
            interpolation="nearestNeighbor",
        )
        mask = np.asarray(backend.to_array(transformed)) >= 0.5
        if mask.shape != fixed_shape:
            raise ValueError(
                "transformed validity mask does not match reference geometry"
            )
        masks[round_number] = mask
        moving_provenance = provenance["moving"]
        assert isinstance(moving_provenance, list)
        moving_provenance.append(
            {
                "acquisition_round": round_number,
                "registered_target": target_record.artifact.to_dict(),
                "request": request.to_dict(),
                "fixed": fixed_identity.to_dict(),
                "selected_fixed": selected_fixed_identity.to_dict(),
                "fixed_reference_match": fixed_reference_match,
                "source": source_identity.to_dict(),
                "affine": affine_identity.to_dict(),
                "warp": warp_identity.to_dict(),
                "semantics": "native-all-ones-affine-before-warp-nearest-neighbor-v1",
                "mask": {
                    **_validity_mask_provenance(mask),
                },
            }
        )
    serialized = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    return (
        masks,
        digest,
        (
            f"validity-provenance:{serialized}",
            f"validity-provenance-sha256:{digest}",
        ),
    )


def _fixed_reference_match(
    recorded: FileIdentity, selected: FileIdentity
) -> str | None:
    """Return the sole admissible match mode for a producer fixed reference.

    A normal chain uses the exact recorded path, size, and modification time.
    Cy-ExM may also materialize an independently copied, byte-identical fixed
    reference while preserving the same fixed grid.  That copy is admissible
    only after a full streaming SHA-256 comparison; all moving inputs and both
    transform identities remain exact-file checks.
    """
    if recorded == selected:
        return "exact-file-identity"
    try:
        recorded_digest = _file_sha256(Path(recorded.path))
        selected_digest = _file_sha256(Path(selected.path))
    except OSError:
        return None
    return "exact-byte-equivalent" if recorded_digest == selected_digest else None


def _file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading a volume into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validity_mask_provenance(mask: NDArray[np.bool_]) -> dict[str, object]:
    """Return canonical bytes-and-layout evidence for one returned validity mask."""
    contiguous = np.ascontiguousarray(mask)
    return {
        "shape_zyx": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "order": "C",
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _ensemble_row(
    round_number: int,
    endpoint: str,
    sigma_um: float,
    threshold_sbr: float,
    value: float,
    units: str,
    config: AnalysisConfig,
    *,
    surface_tolerance_resolution_units: float | None = None,
) -> EnsembleRow:
    return EnsembleRow(
        cohort="golgi",
        cell_id="array",
        acquisition_round=round_number,
        endpoint=endpoint,
        ensemble_name=f"sigma={sigma_um:g};sbr={threshold_sbr:g}",
        ensemble_value=(
            surface_tolerance_resolution_units
            if surface_tolerance_resolution_units is not None
            else threshold_sbr
        ),
        sigma_um=sigma_um,
        threshold_sbr=threshold_sbr,
        surface_tolerance_resolution_units=surface_tolerance_resolution_units,
        value=value,
        units=units,
        source_task_identity=config.config_digest,
    )


def _comparison_values(
    reference_mask: NDArray[np.bool_],
    moving_mask: NDArray[np.bool_],
    config: AnalysisConfig,
) -> tuple[tuple[str, float, float | None], ...]:
    """Return reference-round surface measures without favorable empty agreement."""
    if not np.any(reference_mask) or not np.any(moving_mask):
        return (
            ("jaccard", math.nan, None),
            ("fixed_to_moving_boundary_distance_p95", math.nan, None),
            ("moving_to_fixed_boundary_distance_p95", math.nan, None),
            *(
                (f"surface_dice_{tolerance:g}", math.nan, tolerance)
                for tolerance in config.ensembles.surface_tolerance_resolution_units
            ),
        )
    resolution = theoretical_resolution(
        config.optics.target_wavelength_um,
        config.optics.numerical_aperture,
        config.optics.immersion_refractive_index,
        config.nominal_expansion_factor,
    )
    metrics = resolution_surface_metrics(
        reference_mask,
        moving_mask,
        config.voxel_spacing_xyz_um,
        resolution,
        config.ensembles.surface_tolerance_resolution_units,
    )
    return (
        ("jaccard", metrics.jaccard, None),
        (
            "fixed_to_moving_boundary_distance_p95",
            float(metrics.fixed_to_moving_normalized_distance_p95),
            None,
        ),
        (
            "moving_to_fixed_boundary_distance_p95",
            float(metrics.moving_to_fixed_normalized_distance_p95),
            None,
        ),
        *(
            (f"surface_dice_{tolerance:g}", value, tolerance)
            for tolerance, value in metrics.surface_dice.items()
        ),
    )


def _analyze_golgi_arrays(
    golgi_rounds: Mapping[int, NDArray],
    dapi_rounds: Mapping[int, NDArray],
    config: AnalysisConfig,
    *,
    validity_masks: Mapping[int, NDArray[np.bool_]],
) -> _GolgiArrayAnalysis:
    """Compute Golgi measurements and retain intermediates for diagnostics."""
    if REFERENCE_ROUND not in golgi_rounds or REFERENCE_ROUND not in dapi_rounds:
        raise ValueError("Golgi analysis requires acquisition round 2")
    if set(golgi_rounds) != set(dapi_rounds):
        raise ValueError(
            "registered Golgi and DAPI rounds must have identical inventories"
        )
    rounds = {
        int(number): _volume(image, f"Golgi round {number}")
        for number, image in golgi_rounds.items()
    }
    if len({image.shape for image in rounds.values()}) != 1:
        raise ValueError("registered Golgi rounds must share fixed-grid geometry")
    common_validity = _common_validity_masks(rounds, validity_masks)
    common_support = np.logical_and.reduce(tuple(common_validity.values()))
    consensus = consensus_nuclear_surface(
        dapi_rounds,
        config.voxel_spacing_xyz_um,
        config.nuclear,
        config.nominal_expansion_factor,
    )
    roi = _perinuclear_roi(consensus.mask_zyx, common_validity, config)
    reference = rounds[REFERENCE_ROUND]
    ensemble_by_round = {
        round_number: _support_threshold_ensemble(
            image,
            roi,
            common_support,
            config,
        )
        for round_number, image in rounds.items()
    }
    reference_members = ensemble_by_round[REFERENCE_ROUND]
    nuclear_distance_field = _nuclear_distance_field(consensus.nuclei)
    ensemble_rows: list[EnsembleRow] = []
    summary_rows: list[MetricRow] = []
    units = _ensemble_units(config)
    endpoints = (
        *ENSEMBLE_ENDPOINTS,
        *(
            f"surface_dice_{tolerance:g}"
            for tolerance in config.ensembles.surface_tolerance_resolution_units
        ),
    )
    for round_number in sorted(rounds):
        similarities = multiscale_gradient_similarity(
            reference,
            rounds[round_number],
            roi,
            config.voxel_spacing_xyz_um,
            config.ensembles.gaussian_sigma_um,
            validity_mask=common_support,
        )
        comparison_set = (
            "round-1-exposure-sensitivity" if round_number == 1 else "matched-settings"
        )
        for row in similarities:
            summary_rows.append(
                MetricRow(
                    "golgi",
                    "array",
                    round_number,
                    f"golgi_gradient_similarity_sigma_{row.sigma_um:g}",
                    row.correlation,
                    "correlation",
                    comparison_set,
                    True,
                    config.config_digest,
                )
            )
        summary_rows.append(
            MetricRow(
                "golgi",
                "array",
                round_number,
                "golgi_gradient_similarity_median",
                float(np.median([row.correlation for row in similarities])),
                "correlation",
                comparison_set,
                True,
                config.config_digest,
            )
        )
        current_values: dict[str, list[float]] = {}
        for reference_member, member in zip(
            reference_members, ensemble_by_round[round_number], strict=True
        ):
            geometry = _mask_geometry(member.mask_zyx, config.voxel_spacing_xyz_um)
            distances = golgi_nucleus_distances(
                member.mask_zyx,
                consensus.nuclei,
                config.voxel_spacing_xyz_um,
                distance_field_zyx=nuclear_distance_field,
            )
            background = member.background
            corrected_signal = float(
                np.clip(rounds[round_number] - background.median, 0.0, None)[
                    member.mask_zyx
                ].sum()
            )
            values = {
                "absolute_background_corrected_signal": corrected_signal,
                **geometry,
                "nuclear_distance_p25_um": float(np.percentile(distances, 25.0))
                if distances.size
                else math.nan,
                "nuclear_distance_p50_um": float(np.percentile(distances, 50.0))
                if distances.size
                else math.nan,
                "nuclear_distance_p75_um": float(np.percentile(distances, 75.0))
                if distances.size
                else math.nan,
            }
            exact_values = [
                (endpoint, value, None) for endpoint, value in values.items()
            ]
            exact_values.extend(
                _comparison_values(reference_member.mask_zyx, member.mask_zyx, config)
            )
            for endpoint, value, tolerance in exact_values:
                ensemble_rows.append(
                    _ensemble_row(
                        round_number,
                        endpoint,
                        member.sigma_um,
                        member.threshold_sbr,
                        value,
                        units[endpoint],
                        config,
                        surface_tolerance_resolution_units=tolerance,
                    )
                )
                if math.isfinite(value):
                    current_values.setdefault(endpoint, []).append(value)
        for endpoint in endpoints:
            values = current_values.get(endpoint, [])
            summary_rows.append(
                MetricRow(
                    "golgi",
                    "array",
                    round_number,
                    f"golgi_{endpoint}_median",
                    float(np.median(values)) if values else math.nan,
                    units[endpoint],
                    comparison_set,
                    True,
                    config.config_digest,
                )
            )
    return _GolgiArrayAnalysis(
        MarkerResult(
            "golgi",
            "array",
            tuple(summary_rows),
            tuple(ensemble_rows),
            (),
            (),
            tuple(MappingProxyType(asdict(row)) for row in consensus.component_qc),
        ),
        consensus,
        roi,
        MappingProxyType(ensemble_by_round),
    )


def analyze_golgi_arrays(
    golgi_rounds: Mapping[int, NDArray],
    dapi_rounds: Mapping[int, NDArray],
    config: AnalysisConfig,
    *,
    validity_masks: Mapping[int, NDArray[np.bool_]],
) -> MarkerResult:
    """Analyze one cell's registered Golgi arrays with round 2 fixed as reference.

    This numerical kernel makes no filesystem writes and has no cohort aggregation.
    Its ``array`` identity is rebound to the validated cell only by
    :func:`analyze_golgi_cell`, the I/O-owning entry point.
    """
    return _analyze_golgi_arrays(
        golgi_rounds, dapi_rounds, config, validity_masks=validity_masks
    ).result


def _rebind_cell(result: MarkerResult, cell: CellSpec) -> MarkerResult:
    """Replace array-kernel placeholder identity without changing numerical results."""
    return MarkerResult(
        cell.cohort,
        cell.cell_id,
        tuple(
            replace(row, cohort=cell.cohort, cell_id=cell.cell_id)
            for row in result.summary_rows
        ),
        tuple(
            replace(row, cohort=cell.cohort, cell_id=cell.cell_id)
            for row in result.ensemble_rows
        ),
        result.diagnostics,
        result.input_identities,
        result.component_qc_rows,
    )


def _write_diagnostics(
    output_dir: Path,
    rounds: Mapping[int, NDArray[np.float64]],
    dapi_rounds: Mapping[int, NDArray[np.float64]],
    result: MarkerResult,
    consensus: NuclearConsensus,
    roi: NDArray[np.bool_],
    threshold_masks: Mapping[int, tuple[MaskMember, ...]],
    crop: _SharedCropBounds,
) -> tuple[Path, ...]:
    """Atomically publish fixed-grid composite overlays and threshold-response curves."""
    staged_paths: tuple[Path, ...] = ()
    staging_root: Path | None = None

    def populate(staged: Path) -> None:
        nonlocal staged_paths, staging_root
        import matplotlib.pyplot as plt

        staging_root = staged
        component_diagnostic = write_nuclear_component_diagnostic(
            staged, result.cell_id, dapi_rounds, consensus
        )
        overlay = staged / "golgi-registered-overlays.npz"
        np.savez_compressed(
            overlay,
            consensus_nuclear_mask_zyx=consensus.mask_zyx,
            fixed_perinuclear_roi_zyx=roi,
            crop_start_zyx=np.asarray(crop.start_zyx, dtype=np.int64),
            crop_stop_zyx=np.asarray(crop.stop_zyx, dtype=np.int64),
            full_shape_zyx=np.asarray(crop.full_shape_zyx, dtype=np.int64),
            **{f"round_{number}_target_zyx": image for number, image in rounds.items()},
            **{
                f"round_{number}_sigma_{member.sigma_um:g}_sbr_{member.threshold_sbr:g}_threshold_mask_zyx": member.mask_zyx
                for number, members in threshold_masks.items()
                for member in members
            },
        )
        curves = staged / "golgi-threshold-robustness.csv"
        with curves.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(EnsembleRow.__dataclass_fields__)
            writer.writerows(
                [getattr(row, name) for name in EnsembleRow.__dataclass_fields__]
                for row in result.ensemble_rows
            )
        plt.rcParams.update(
            {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
        )
        reference = rounds[REFERENCE_ROUND].max(axis=0)
        roi_mip = roi.max(axis=0)
        nucleus_mip = consensus.mask_zyx.max(axis=0)
        figure_paths: list[Path] = []
        union_masks = {
            number: np.logical_or.reduce([member.mask_zyx for member in members])
            for number, members in threshold_masks.items()
        }
        for round_number, image in sorted(rounds.items()):
            figure, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
            for axis, title, mip, mask in (
                (
                    axes[0],
                    "round 2 reference",
                    reference,
                    union_masks[REFERENCE_ROUND],
                ),
                (
                    axes[1],
                    f"round {round_number}",
                    image.max(axis=0),
                    union_masks[round_number],
                ),
            ):
                finite = mip[np.isfinite(mip)]
                low, high = np.percentile(finite, (1.0, 99.5))
                axis.imshow(mip, cmap="magma", vmin=low, vmax=high)
                axis.contour(roi_mip, levels=(0.5,), colors="cyan", linewidths=0.7)
                axis.contour(nucleus_mip, levels=(0.5,), colors="white", linewidths=0.7)
                axis.contour(
                    mask.max(axis=0), levels=(0.5,), colors="lime", linewidths=0.7
                )
                axis.set_title(title)
                axis.set_axis_off()
            path = staged / f"golgi-overlay-round-{round_number}.png"
            figure.savefig(path, dpi=200)
            plt.close(figure)
            figure_paths.append(path)
        staged_paths = (component_diagnostic, overlay, curves, *figure_paths)

    _atomic_publish_directory(output_dir, populate)
    assert staging_root is not None
    return tuple(output_dir / path.relative_to(staging_root) for path in staged_paths)


def _write_crop_preflight(
    output_dir: Path, cell: CellSpec, accepted: tuple[int, ...], plan: _GolgiCropPlan
) -> tuple[Path, ...]:
    """Atomically publish a small, inspectable Golgi crop plan before analysis."""
    staged_paths: tuple[Path, ...] = ()
    staging_root: Path | None = None

    def bounds_json(
        bounds: tuple[tuple[int, int, int], tuple[int, int, int]] | None,
    ) -> dict[str, list[int]] | None:
        if bounds is None:
            return None
        return {"start_zyx": list(bounds[0]), "stop_zyx": list(bounds[1])}

    def populate(staged: Path) -> None:
        nonlocal staged_paths, staging_root
        import matplotlib.pyplot as plt

        staging_root = staged
        plan_path = staged / "crop-plan.json"
        plan_path.write_text(
            json.dumps(
                {
                    "cohort": cell.cohort,
                    "cell_id": cell.cell_id,
                    "accepted_rounds": list(accepted),
                    "sample_step_zyx": list(plan.sample_step_zyx),
                    "full_shape_zyx": list(plan.crop.full_shape_zyx),
                    "crop_start_zyx": list(plan.crop.start_zyx),
                    "crop_stop_zyx": list(plan.crop.stop_zyx),
                    "retained_fraction": plan.crop.retained_fraction,
                    "maximum_allowed_retained_fraction": _MAX_RETAINED_GOLGI_CROP_FRACTION,
                    "dapi_support_bounds": {
                        str(number): bounds_json(bounds)
                        for number, bounds in sorted(plan.dapi_bounds_by_round.items())
                    },
                    "golgi_support_bounds": {
                        str(number): bounds_json(bounds)
                        for number, bounds in sorted(plan.golgi_bounds_by_round.items())
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        plt.rcParams.update(
            {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
        )
        figure_paths: list[Path] = []
        for number in accepted:
            figure, axes = plt.subplots(1, 2, figsize=(7, 3.5), layout="constrained")
            for axis, title, support, color in (
                (
                    axes[0],
                    "DAPI coarse support",
                    plan.dapi_support_mips.get(number),
                    "#C44EAD",
                ),
                (
                    axes[1],
                    "Golgi coarse support",
                    plan.golgi_support_mips.get(number),
                    "#009E73",
                ),
            ):
                if support is None or not np.any(support):
                    axis.text(
                        0.5, 0.5, "no plausible support", ha="center", va="center"
                    )
                else:
                    axis.imshow(support, cmap="gray", interpolation="nearest")
                    axis.contour(support, levels=(0.5,), colors=color, linewidths=0.8)
                axis.set_title(title)
                axis.set_axis_off()
            figure.suptitle(f"{cell.cell_id}: round {number} (coarse sampled MIP)")
            path = staged / f"crop-support-round-{number}.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            figure_paths.append(path)
        staged_paths = (plan_path, *figure_paths)

    _atomic_publish_directory(output_dir, populate)
    assert staging_root is not None
    return tuple(output_dir / path.relative_to(staging_root) for path in staged_paths)


def _require_compact_crop(plan: _GolgiCropPlan, preflight_dir: Path) -> None:
    """Fail before large allocations when the candidate crop remains near-full field."""
    fraction = plan.crop.retained_fraction
    if fraction > _MAX_RETAINED_GOLGI_CROP_FRACTION:
        raise ValueError(
            "Golgi crop preflight retained "
            f"{fraction:.1%} of the fixed grid (limit "
            f"{_MAX_RETAINED_GOLGI_CROP_FRACTION:.0%}); inspect "
            f"{preflight_dir / 'crop-plan.json'} before target-specific analysis"
        )


def analyze_golgi_cell(
    cell: CellSpec,
    artifacts: tuple[RoundArtifacts, ...],
    config: AnalysisConfig,
    output_dir: Path,
) -> MarkerResult:
    """Load validated registered Golgi inputs and publish one cell's diagnostics.

    The entry point reconstructs target-independent support from each audited native
    target input and its persisted affine-before-warp chain. The scalar Golgi
    channel is required to have the producer's linear interpolation request;
    nearest-neighbor is used only when replaying binary validity support. It
    never accepts caller-supplied target-support masks.
    """
    if cell.cohort != "golgi":
        raise ValueError("Golgi analysis requires a golgi CellSpec")
    by_round = {artifact.acquisition_round: artifact for artifact in artifacts}
    if set(by_round) != {round_spec.acquisition_round for round_spec in cell.rounds}:
        raise ValueError(
            "cell and validated artifact acquisition-round inventories disagree"
        )
    accepted = _accepted_rounds(cell, config)
    crop_plan = _golgi_crop_plan_from_artifacts(by_round, accepted, config)
    preflight_dir = output_dir.parent / f"{output_dir.name}-crop-preflight"
    preflight_paths = _write_crop_preflight(preflight_dir, cell, accepted, crop_plan)
    _require_compact_crop(crop_plan, preflight_dir)
    crop = crop_plan.crop
    dapi = {
        number: load_volume(by_round[number].registered_dapi)[crop.slices_zyx]
        for number in accepted
    }
    golgi = {
        number: load_volume(by_round[number].registered_target)[crop.slices_zyx]
        for number in accepted
    }
    full_validity, validity_digest, validity_identities = _artifact_validity_masks(
        {number: by_round[number] for number in accepted},
        config,
        fixed_shape_zyx=crop.full_shape_zyx,
    )
    common_validity = {
        number: mask[crop.slices_zyx] for number, mask in full_validity.items()
    }
    analysis = _analyze_golgi_arrays(
        golgi, dapi, config, validity_masks=common_validity
    )
    result = _rebind_cell(analysis.result, cell)
    diagnostics = _write_diagnostics(
        output_dir,
        golgi,
        dapi,
        result,
        analysis.consensus,
        analysis.roi,
        analysis.threshold_masks,
        crop,
    )
    return replace(
        result,
        diagnostics=(*preflight_paths, *diagnostics),
        input_identities=tuple(
            str(path)
            for number in accepted
            for path in (
                by_round[number].registered_dapi,
                by_round[number].registered_target,
            )
        )
        + validity_identities
        + (f"validity-mask-sha256:{validity_digest}",),
    )


__all__ = [
    "ENSEMBLE_ENDPOINTS",
    "GradientSimilarityRow",
    "analyze_golgi_arrays",
    "analyze_golgi_cell",
    "golgi_nucleus_distances",
    "multiscale_gradient_similarity",
]
