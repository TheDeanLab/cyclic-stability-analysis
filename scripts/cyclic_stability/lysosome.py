"""Gated LAMP1 object detection and one-to-one resolution-aware matching."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    find_objects,
    gaussian_filter,
    label,
    maximum_filter,
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.feature import blob_dog
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.segmentation import watershed

from cyclic_stability.cyexm_artifacts import RoundArtifacts
from cyclic_stability.golgi import _artifact_validity_masks
from cyclic_stability.images import BackgroundSummary, load_volume
from cyclic_stability.models import (
    AnalysisConfig,
    CellSpec,
    EnsembleRow,
    MarkerResult,
    MetricRow,
)
from cyclic_stability.morphology import gaussian_sigma_zyx, validated_binary_mask
from cyclic_stability.nuclear import (
    REFERENCE_ROUND,
    NuclearSurface,
    _accepted_rounds,
    _atomic_publish_directory,
    consensus_nuclear_surface,
    write_nuclear_component_diagnostic,
)
from cyclic_stability.units import ResolutionReference, theoretical_resolution

_MIP_MIN_LATERAL_DIAMETER_PX = 5.0
_MIP_MAX_LATERAL_DIAMETER_PX = 100.0
_MIP_BACKGROUND_SIGMA_PX = 70.0
_MIP_DOG_SIGMA_RATIO = 1.5


@dataclass(frozen=True, slots=True)
class LysosomeObject:
    """One watershed-separated LAMP1 object in physical XYZ coordinates."""

    centroid_xyz_um: tuple[float, float, float]
    volume_um3: float
    equivalent_diameter_um: float
    sphericity: float
    nearest_neighbor_distance_um: float
    nucleus_relative_distance_um: float

    def __post_init__(self) -> None:
        if len(self.centroid_xyz_um) != 3 or not all(
            math.isfinite(value) for value in self.centroid_xyz_um
        ):
            raise ValueError(
                "lysosome object coordinates must be three finite XYZ values"
            )


@dataclass(frozen=True, slots=True)
class DetectionSet:
    """All detected LAMP1 objects for one MIP-smoothing/SBR ensemble member."""

    sigma_um: float
    detection_sbr: float
    objects: tuple[LysosomeObject, ...]

    @property
    def object_count(self) -> int:
        """Return the number of watershed objects in this ensemble member."""
        return len(self.objects)

    @property
    def coordinates_xyz_um(self) -> tuple[tuple[float, float, float], ...]:
        """Return stable object centroids in physical XYZ micrometers."""
        return tuple(object_.centroid_xyz_um for object_ in self.objects)


@dataclass(frozen=True, slots=True)
class ObjectMatchResult:
    """One finite, one-to-one LAMP1 assignment with explicit lost/new outcomes."""

    matched_pairs: tuple[tuple[int, int], ...]
    normalized_displacements: tuple[float, ...]
    lateral_displacements_um: tuple[float, ...]
    axial_displacements_um: tuple[float, ...]
    matched_count: int
    lost_count: int
    new_count: int
    matched_fraction: float
    lost_fraction: float
    new_fraction: float


def _spacing(
    spacing_xyz_um: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Validate public XYZ spacing through the shared physical conversion helper."""
    gaussian_sigma_zyx(1.0, spacing_xyz_um)
    return tuple(float(value) for value in spacing_xyz_um)


def _volume(image: NDArray[np.generic], name: str) -> NDArray[np.float64]:
    """Return exactly one finite, real 3-D ZYX volume."""
    values = np.asarray(image)
    if (
        values.ndim != 3
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be one real numeric 3D ZYX volume")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values.astype(np.float64, copy=False)


def _positive_values(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    """Validate one non-empty ordered physical parameter sequence."""
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    return result


def _maxima(
    response: NDArray[np.float64],
    roi: NDArray[np.bool_],
    threshold: float,
    spacing_xyz_um: tuple[float, float, float],
    resolution: ResolutionReference,
) -> tuple[tuple[int, int, int], ...]:
    """Select 26-neighbour maxima with lateral/lateral/axial duplicate units."""
    maxima = (
        response == maximum_filter(response, size=(3, 3, 3), mode="nearest")
    ) & roi
    candidates = np.argwhere(maxima & (response > threshold))
    ordered = sorted(
        candidates.tolist(), key=lambda item: (-response[tuple(item)], *item)
    )
    sx, sy, sz = spacing_xyz_um
    resolution_xyz = np.array(
        (
            resolution.lateral_expanded_um,
            resolution.lateral_expanded_um,
            resolution.axial_expanded_um,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(resolution_xyz)) or np.any(resolution_xyz <= 0.0):
        raise ValueError(
            "resolution must contain finite positive expanded-space values"
        )
    retained: list[tuple[int, int, int]] = []
    for zyx in ordered:
        xyz = np.array((zyx[2] * sx, zyx[1] * sy, zyx[0] * sz), dtype=np.float64)
        if all(
            np.linalg.norm(
                (xyz - np.array((point[2] * sx, point[1] * sy, point[0] * sz)))
                / resolution_xyz
            )
            > 1.0
            for point in retained
        ):
            retained.append(tuple(zyx))
    return tuple(retained)


def _physical_log_response(
    image: NDArray[np.float64],
    sigma_um: float,
    spacing_xyz_um: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Return scale-normalized negative physical Laplacian in source-intensity units.

    SciPy derivative filters are expressed per voxel coordinate.  Multiplying each
    second derivative by that axis's squared voxel sigma yields ``sigma_um**2``
    times the corresponding physical-coordinate derivative, preserving response
    units across anisotropic samplings.
    """
    sigma_zyx = gaussian_sigma_zyx(sigma_um, spacing_xyz_um)
    response = np.zeros_like(image, dtype=np.float64)
    for axis, sigma_vox in enumerate(sigma_zyx):
        order = [0, 0, 0]
        order[axis] = 2
        response -= sigma_vox**2 * gaussian_filter(
            image, sigma=sigma_zyx, order=tuple(order), mode="nearest"
        )
    return response


def _support_background(
    image: NDArray[np.float64], support: NDArray[np.bool_]
) -> BackgroundSummary:
    """Estimate robust background only from finite positive transform-valid samples."""
    positive = image[support & (image > 0.0)]
    if positive.size == 0:
        raise ValueError("support background has no positive voxels")
    upper = float(np.percentile(positive, 30.0))
    background = positive[positive <= upper]
    if background.size == 0:
        raise ValueError("support background selection is empty")
    median = float(np.median(background))
    scale = float(1.4826 * np.median(np.abs(background - median)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("support background robust scale must be finite and positive")
    return BackgroundSummary(median, scale, int(positive.size), int(background.size))


def _support_safe_physical_log(
    image: NDArray[np.float64],
    support: NDArray[np.bool_],
    sigma_um: float,
    spacing_xyz_um: tuple[float, float, float],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return masked-normalized physical LoG only where its 3-point stencil is safe."""
    sigma_zyx = gaussian_sigma_zyx(sigma_um, spacing_xyz_um)
    weights = support.astype(np.float64, copy=False)
    denominator = gaussian_filter(weights, sigma_zyx, mode="nearest")
    smoothed = np.divide(
        gaussian_filter(image * weights, sigma_zyx, mode="nearest"),
        denominator,
        out=np.zeros_like(image),
        where=denominator > np.finfo(np.float64).eps,
    )
    safe = binary_erosion(support, structure=np.ones((3, 3, 3), dtype=bool))
    if not np.any(safe):
        raise ValueError(
            "LAMP1 support-boundary exclusion leaves no derivative stencil"
        )
    response = np.zeros_like(smoothed)
    for axis, sigma_vox in enumerate(sigma_zyx):
        ahead = [slice(None)] * 3
        center = [slice(None)] * 3
        behind = [slice(None)] * 3
        ahead[axis] = slice(2, None)
        center[axis] = slice(1, -1)
        behind[axis] = slice(None, -2)
        destination = [slice(None)] * 3
        destination[axis] = slice(1, -1)
        response[tuple(destination)] -= sigma_vox**2 * (
            smoothed[tuple(ahead)]
            - 2.0 * smoothed[tuple(center)]
            + smoothed[tuple(behind)]
        )
    return response, safe


def _surface_area(
    mask: NDArray[np.bool_], spacing_xyz_um: tuple[float, float, float]
) -> float:
    """Return a padded marching-cubes physical surface area for one non-empty object."""
    sx, sy, sz = spacing_xyz_um
    vertices, faces, _, _ = marching_cubes(
        np.pad(mask.astype(np.float32), 1), level=0.5, spacing=(sz, sy, sx)
    )
    return float(mesh_surface_area(vertices, faces))


def _object_from_local_label(
    labels: NDArray[np.int32],
    index: int,
    bounds: tuple[slice, slice, slice],
    spacing_xyz_um: tuple[float, float, float],
) -> LysosomeObject | None:
    """Summarize one watershed label from its minimal fixed-grid bounding box."""
    mask = labels[bounds] == index
    voxels = np.argwhere(mask)
    if not len(voxels):
        return None
    origin = np.array([axis.start for axis in bounds], dtype=np.float64)
    centroid_zyx = voxels.mean(axis=0) + origin
    sx, sy, sz = spacing_xyz_um
    centroid = (
        float(centroid_zyx[2] * sx),
        float(centroid_zyx[1] * sy),
        float(centroid_zyx[0] * sz),
    )
    volume = float(mask.sum() * sx * sy * sz)
    area = _surface_area(mask, spacing_xyz_um)
    return LysosomeObject(
        centroid,
        volume,
        float((6.0 * volume / np.pi) ** (1.0 / 3.0)),
        float(np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / area),
        math.nan,
        math.nan,
    )


def _objects(
    response: NDArray[np.float64],
    roi: NDArray[np.bool_],
    peaks: tuple[tuple[int, int, int], ...],
    spacing_xyz_um: tuple[float, float, float],
    *,
    local_workers: int,
) -> tuple[LysosomeObject, ...]:
    """Watershed positive LoG support and summarize every retained physical object."""
    if not peaks:
        return ()
    markers = np.zeros(response.shape, dtype=np.int32)
    for index, peak in enumerate(peaks, start=1):
        markers[peak] = index
    labels = watershed(-response, markers, mask=(response > 0.0) & roi)
    label_bounds = find_objects(labels, max_label=len(peaks))
    work = tuple(
        (index, bounds)
        for index, bounds in enumerate(label_bounds, start=1)
        if bounds is not None
    )
    if local_workers == 1:
        measured = tuple(
            _object_from_local_label(labels, index, bounds, spacing_xyz_um)
            for index, bounds in work
        )
    else:
        with ThreadPoolExecutor(max_workers=local_workers) as executor:
            measured = tuple(
                executor.map(
                    lambda item: _object_from_local_label(
                        labels, item[0], item[1], spacing_xyz_um
                    ),
                    work,
                )
            )
    objects = [item for item in measured if item is not None]
    if len(objects) < 2:
        return tuple(objects)
    points = np.array([object_.centroid_xyz_um for object_ in objects])
    distances, _ = cKDTree(points).query(points, k=2)
    return tuple(
        replace(object_, nearest_neighbor_distance_um=float(distances[index, 1]))
        for index, object_ in enumerate(objects)
    )


def detect_lysosomes(
    image: NDArray[np.generic],
    roi: NDArray[np.generic],
    spacing_xyz_um: tuple[float, float, float],
    sigma_values_um: tuple[float, ...],
    detection_sbr: tuple[float, ...],
    *,
    resolution: ResolutionReference,
    local_workers: int = 1,
) -> tuple[DetectionSet, ...]:
    """Detect physical 3-D LoG LAMP1 objects for every declared ensemble member."""
    source = _volume(image, "LAMP1 image")
    structure_roi = validated_binary_mask(roi, "LAMP1 ROI")
    if source.shape != structure_roi.shape:
        raise ValueError("LAMP1 image and ROI must share a 3D ZYX geometry")
    spacing = _spacing(spacing_xyz_um)
    sigmas = _positive_values(sigma_values_um, "sigma_values_um")
    thresholds = _positive_values(detection_sbr, "detection_sbr")
    if (
        isinstance(local_workers, bool)
        or not isinstance(local_workers, int)
        or local_workers < 1
    ):
        raise ValueError("local_workers must be a positive integer")
    background = _support_background(source, structure_roi)
    corrected = source - background.median
    result: list[DetectionSet] = []
    for sigma_um in sigmas:
        response, derivative_support = _support_safe_physical_log(
            corrected, structure_roi, sigma_um, spacing
        )
        for threshold_sbr in thresholds:
            peaks = _maxima(
                response,
                derivative_support,
                threshold_sbr * background.scale,
                spacing,
                resolution,
            )
            result.append(
                DetectionSet(
                    sigma_um,
                    threshold_sbr,
                    _objects(
                        response,
                        derivative_support,
                        peaks,
                        spacing,
                        local_workers=local_workers,
                    ),
                )
            )
    return tuple(result)


def _mip_dog_candidates(
    image: NDArray[np.float64],
    roi: NDArray[np.bool_],
    *,
    detection_sbr: float,
    spacing_xyz_um: tuple[float, float, float],
    min_lateral_diameter_px: float = _MIP_MIN_LATERAL_DIAMETER_PX,
    max_lateral_diameter_px: float = _MIP_MAX_LATERAL_DIAMETER_PX,
    min_sigma_px: float | None = None,
    max_sigma_px: float | None = None,
) -> tuple[tuple[tuple[slice, slice], NDArray[np.bool_]], ...]:
    """Return filled MIP candidate disks centered on multiscale DoG extrema.

    A broad background subtraction removes diffuse LAMP1 texture before
    scale-space center detection.  ``detection_sbr`` maps monotonically to a
    fixed normalized DoG objectness threshold (3, 4, 5 map to 0.15, 0.20,
    0.25), avoiding an image-wide foreground-percentile mask.
    """
    sx, sy, _sz = _spacing(spacing_xyz_um)
    if not math.isclose(sx, sy, rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError("MIP LAMP1 detection requires equal XY pixel spacing")
    if not math.isfinite(min_lateral_diameter_px) or min_lateral_diameter_px <= 0.0:
        raise ValueError("min_lateral_diameter_px must be finite and positive")
    if (
        not math.isfinite(max_lateral_diameter_px)
        or max_lateral_diameter_px <= min_lateral_diameter_px
    ):
        raise ValueError(
            "max_lateral_diameter_px must be finite and larger than the minimum"
        )
    if min_sigma_px is None:
        min_sigma_px = min_lateral_diameter_px / (
            2.0 * math.sqrt(2.0) * _MIP_DOG_SIGMA_RATIO
        )
    if max_sigma_px is None:
        max_sigma_px = (
            max_lateral_diameter_px * _MIP_DOG_SIGMA_RATIO / (2.0 * math.sqrt(2.0))
        )
    if not math.isfinite(min_sigma_px) or min_sigma_px <= 0.0:
        raise ValueError("min_sigma_px must be finite and positive")
    if not math.isfinite(max_sigma_px) or max_sigma_px < min_sigma_px:
        raise ValueError("max_sigma_px must be finite and no smaller than min_sigma_px")
    support = roi.any(axis=0)
    projection = image.max(axis=0).astype(np.float64, copy=False)
    weights = support.astype(np.float64, copy=False)
    denominator = gaussian_filter(weights, _MIP_BACKGROUND_SIGMA_PX, mode="nearest")
    local_background = np.divide(
        gaussian_filter(projection * weights, _MIP_BACKGROUND_SIGMA_PX, mode="nearest"),
        denominator,
        out=np.zeros_like(projection),
        where=denominator > np.finfo(np.float64).eps,
    )
    foreground = np.maximum(projection - local_background, 0.0)
    normalization = float(np.quantile(foreground[support], 0.999))
    if not math.isfinite(normalization) or normalization <= 0.0:
        return ()
    objectness_threshold = 0.05 * detection_sbr
    if not math.isfinite(objectness_threshold) or objectness_threshold <= 0.0:
        raise ValueError("detection_sbr must yield a positive objectness threshold")
    blobs = blob_dog(
        foreground / normalization,
        min_sigma=min_sigma_px,
        max_sigma=max_sigma_px,
        sigma_ratio=_MIP_DOG_SIGMA_RATIO,
        threshold=objectness_threshold,
        overlap=0.5,
        exclude_border=False,
    )
    result: list[tuple[tuple[slice, slice], NDArray[np.bool_]]] = []
    for center_y, center_x, sigma_px in blobs:
        radius_px = math.sqrt(2.0) * float(sigma_px)
        diameter_px = 2.0 * radius_px
        above_minimum = diameter_px >= min_lateral_diameter_px or math.isclose(
            diameter_px, min_lateral_diameter_px, rel_tol=1e-9, abs_tol=1e-12
        )
        below_maximum = diameter_px <= max_lateral_diameter_px or math.isclose(
            diameter_px, max_lateral_diameter_px, rel_tol=1e-9, abs_tol=1e-12
        )
        if not (above_minimum and below_maximum):
            continue
        y0 = max(0, math.floor(float(center_y) - radius_px))
        y1 = min(projection.shape[0], math.ceil(float(center_y) + radius_px) + 1)
        x0 = max(0, math.floor(float(center_x) - radius_px))
        x1 = min(projection.shape[1], math.ceil(float(center_x) + radius_px) + 1)
        y, x = np.ogrid[y0:y1, x0:x1]
        footprint = (y - center_y) ** 2 + (x - center_x) ** 2 <= radius_px**2
        footprint &= support[y0:y1, x0:x1]
        if np.any(footprint):
            result.append(((slice(y0, y1), slice(x0, x1)), footprint))
    return tuple(result)


def _merge_mip_seed_footprints(
    image: NDArray[np.float64],
    roi: NDArray[np.bool_],
    seeds: tuple[tuple[tuple[slice, slice], NDArray[np.bool_]], ...],
) -> tuple[tuple[tuple[slice, slice], NDArray[np.bool_]], ...]:
    """Merge DoG seeds that share one connected, filled MIP foreground object.

    Each incoming seed remains the required evidence for a candidate, while the
    measured lateral support is a background-corrected MIP component.  This joins
    several scale-space maxima on one hollow or extended lysosome without joining
    components that lack a seed.  The returned footprints retain the original
    fixed-grid XY coordinates and are suitable for the existing local 3-D stage.
    """
    support = roi.any(axis=0)
    projection = image.max(axis=0).astype(np.float64, copy=False)
    seed_mask = np.zeros(projection.shape, dtype=bool)
    for (y_bounds, x_bounds), footprint in seeds:
        seed_mask[y_bounds, x_bounds] |= footprint
    if not np.any(seed_mask):
        return ()
    weights = support.astype(np.float64, copy=False)
    denominator = gaussian_filter(weights, _MIP_BACKGROUND_SIGMA_PX, mode="nearest")
    local_background = np.divide(
        gaussian_filter(projection * weights, _MIP_BACKGROUND_SIGMA_PX, mode="nearest"),
        denominator,
        out=np.zeros_like(projection),
        where=denominator > np.finfo(np.float64).eps,
    )
    foreground = np.maximum(projection - local_background, 0.0)
    positive = foreground[(foreground > 0.0) & support]
    if positive.size < 2 or float(np.max(positive)) <= float(np.min(positive)):
        return ()
    try:
        threshold = float(threshold_otsu(positive))
    except ValueError:
        return ()
    signal = binary_closing(
        (foreground > threshold) & support,
        structure=np.ones((3, 3), dtype=bool),
    )
    labels, count = label(signal, structure=np.ones((3, 3), dtype=bool))
    seed_support = binary_dilation(seed_mask, structure=np.ones((5, 5), dtype=bool))
    result: list[tuple[tuple[slice, slice], NDArray[np.bool_]]] = []
    for index in range(1, count + 1):
        component = labels == index
        if not np.any(component & seed_support):
            continue
        filled = binary_fill_holes(component) & support
        y_coordinates, x_coordinates = np.nonzero(filled)
        if not len(y_coordinates):
            continue
        y_bounds = slice(int(y_coordinates.min()), int(y_coordinates.max()) + 1)
        x_bounds = slice(int(x_coordinates.min()), int(x_coordinates.max()) + 1)
        result.append(
            ((y_bounds, x_bounds), filled[y_bounds, x_bounds])
        )
    return tuple(result)


def _mip_candidates(
    image: NDArray[np.float64],
    roi: NDArray[np.bool_],
    *,
    sigma_um: float,
    detection_sbr: float,
    spacing_xyz_um: tuple[float, float, float],
) -> tuple[tuple[tuple[slice, slice], NDArray[np.bool_]], ...]:
    """Return MIP DoG candidates in the physical scale band of one ensemble member."""
    sx, _sy, _sz = _spacing(spacing_xyz_um)
    center_sigma_px = sigma_um / sx
    minimum = max(
        _MIP_MIN_LATERAL_DIAMETER_PX / (2.0 * math.sqrt(2.0)),
        center_sigma_px / 2.0,
    )
    maximum = min(
        _MIP_MAX_LATERAL_DIAMETER_PX / (2.0 * math.sqrt(2.0)),
        center_sigma_px * 3.0,
    )
    if maximum < minimum:
        return ()
    seeds = _mip_dog_candidates(
        image,
        roi,
        detection_sbr=detection_sbr,
        spacing_xyz_um=spacing_xyz_um,
        min_sigma_px=minimum,
        max_sigma_px=maximum,
    )
    return _merge_mip_seed_footprints(image, roi, seeds)


def _object_from_mip_candidate(
    image: NDArray[np.float64],
    roi: NDArray[np.bool_],
    candidate: tuple[tuple[slice, slice], NDArray[np.bool_]],
    *,
    spacing_xyz_um: tuple[float, float, float],
) -> LysosomeObject | None:
    """Measure one MIP-admitted object from a padded local 3-D crop."""
    (y_bounds, x_bounds), footprint = candidate
    y0 = max(0, y_bounds.start - 2)
    y1 = min(image.shape[1], y_bounds.stop + 2)
    x0 = max(0, x_bounds.start - 2)
    x1 = min(image.shape[2], x_bounds.stop + 2)
    local_footprint = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    local_footprint[
        y_bounds.start - y0 : y_bounds.stop - y0,
        x_bounds.start - x0 : x_bounds.stop - x0,
    ] = footprint
    allowed_xy = binary_dilation(local_footprint, structure=np.ones((5, 5), dtype=bool))
    local_image = image[:, y0:y1, x0:x1]
    local_roi = roi[:, y0:y1, x0:x1]
    local_projection = local_image.max(axis=0)
    values = local_projection[allowed_xy & local_roi.any(axis=0)]
    if values.size < 2 or float(np.max(values)) <= float(np.min(values)):
        return None
    try:
        threshold = float(threshold_otsu(values))
    except ValueError:
        return None
    mip_signal = (local_projection > threshold) & allowed_xy & local_roi.any(axis=0)
    mip_labels, mip_count = label(mip_signal, structure=np.ones((3, 3), dtype=bool))
    if mip_count == 0:
        return None
    areas = [int(np.sum(mip_labels == index)) for index in range(1, mip_count + 1)]
    selected_xy = binary_fill_holes(mip_labels == (int(np.argmax(areas)) + 1))
    local_signal = (local_image > threshold) & local_roi & selected_xy[None]
    labels, count = label(local_signal, structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        return None
    selected = np.zeros(local_signal.shape, dtype=bool)
    for index in range(1, count + 1):
        component = labels == index
        if np.any(component):
            selected |= component
    if not np.any(selected):
        return None
    voxels = np.argwhere(selected)
    sx, sy, sz = spacing_xyz_um
    centroid_zyx = voxels.mean(axis=0) + (0.0, float(y0), float(x0))
    volume = float(selected.sum() * sx * sy * sz)
    try:
        area = _surface_area(selected, spacing_xyz_um)
    except (RuntimeError, ValueError):
        return None
    if not math.isfinite(area) or area <= 0.0:
        return None
    return LysosomeObject(
        (
            float(centroid_zyx[2] * sx),
            float(centroid_zyx[1] * sy),
            float(centroid_zyx[0] * sz),
        ),
        volume,
        float((6.0 * volume / np.pi) ** (1.0 / 3.0)),
        float(np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / area),
        math.nan,
        math.nan,
    )


def detect_lysosomes_mip(
    image: NDArray[np.generic],
    roi: NDArray[np.generic],
    spacing_xyz_um: tuple[float, float, float],
    sigma_values_um: tuple[float, ...],
    detection_sbr: tuple[float, ...],
    *,
    local_workers: int = 1,
) -> tuple[DetectionSet, ...]:
    """Detect 5--100 pixel lateral LAMP1 objects on MIPs, then measure locally in 3-D.

    The named sigma values smooth the MIP before thresholding; they are no longer
    3-D LoG scales.  Filled 2-D candidates admit hollow structures as one object.
    The returned physical measurements always arise from the original 3-D volume.
    """
    source = _volume(image, "LAMP1 image")
    structure_roi = validated_binary_mask(roi, "LAMP1 ROI")
    if source.shape != structure_roi.shape:
        raise ValueError("LAMP1 image and ROI must share a 3D ZYX geometry")
    spacing = _spacing(spacing_xyz_um)
    sigmas = _positive_values(sigma_values_um, "sigma_values_um")
    thresholds = _positive_values(detection_sbr, "detection_sbr")
    if (
        isinstance(local_workers, bool)
        or not isinstance(local_workers, int)
        or local_workers < 1
    ):
        raise ValueError("local_workers must be a positive integer")
    result: list[DetectionSet] = []
    for sigma_um in sigmas:
        for threshold_sbr in thresholds:
            candidates = _mip_candidates(
                source,
                structure_roi,
                sigma_um=sigma_um,
                detection_sbr=threshold_sbr,
                spacing_xyz_um=spacing,
            )
            measure = lambda candidate: _object_from_mip_candidate(
                source,
                structure_roi,
                candidate,
                spacing_xyz_um=spacing,
            )
            if local_workers == 1:
                measured = tuple(measure(candidate) for candidate in candidates)
            else:
                with ThreadPoolExecutor(max_workers=local_workers) as executor:
                    measured = tuple(executor.map(measure, candidates))
            objects = tuple(item for item in measured if item is not None)
            if len(objects) >= 2:
                points = np.asarray([item.centroid_xyz_um for item in objects])
                distances, _ = cKDTree(points).query(points, k=2)
                objects = tuple(
                    replace(
                        object_, nearest_neighbor_distance_um=float(distances[index, 1])
                    )
                    for index, object_ in enumerate(objects)
                )
            result.append(DetectionSet(sigma_um, threshold_sbr, objects))
    return tuple(result)


def _coordinates(detections: DetectionSet) -> NDArray[np.float64]:
    """Return a finite N-by-3 XYZ coordinate matrix for a detection set."""
    result = np.asarray(detections.coordinates_xyz_um, dtype=np.float64)
    if result.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if result.shape != (detections.object_count, 3) or not np.all(np.isfinite(result)):
        raise ValueError("detection coordinates must be finite XYZ values")
    return result


def match_lysosomes(
    reference: DetectionSet,
    moving: DetectionSet,
    resolution: ResolutionReference,
    max_units: float,
) -> ObjectMatchResult:
    """Assign only finite one-to-one LAMP1 pairs within the stated resolution limit."""
    if not math.isfinite(max_units) or max_units <= 0.0:
        raise ValueError("max_units must be finite and positive")
    reference_xyz = _coordinates(reference)
    moving_xyz = _coordinates(moving)
    reference_count, moving_count = len(reference_xyz), len(moving_xyz)
    if not reference_count or not moving_count:
        return ObjectMatchResult(
            (),
            (),
            (),
            (),
            0,
            reference_count,
            moving_count,
            0.0,
            0.0 if not reference_count else 1.0,
            0.0 if not moving_count else 1.0,
        )
    resolution_xyz = np.array(
        (
            resolution.lateral_expanded_um,
            resolution.lateral_expanded_um,
            resolution.axial_expanded_um,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(resolution_xyz)) or np.any(resolution_xyz <= 0.0):
        raise ValueError(
            "resolution must contain finite positive expanded-space values"
        )
    delta = moving_xyz[:, None, :] - reference_xyz[None, :, :]
    cost = np.sqrt(np.sum((delta / resolution_xyz) ** 2, axis=-1))
    cost[cost > max_units] = np.inf
    possible_rows = np.any(np.isfinite(cost), axis=1)
    possible_cols = np.any(np.isfinite(cost), axis=0)
    pairs: list[tuple[int, int]] = []
    if np.any(possible_rows) and np.any(possible_cols):
        row_indices = np.flatnonzero(possible_rows)
        col_indices = np.flatnonzero(possible_cols)
        finite_cost = cost[np.ix_(row_indices, col_indices)]
        sentinel = (
            max_units
            + max(1.0, float(np.max(finite_cost[np.isfinite(finite_cost)])))
            * 1_000_000.0
        )
        assignment_rows, assignment_cols = linear_sum_assignment(
            np.where(np.isfinite(finite_cost), finite_cost, sentinel)
        )
        pairs = sorted(
            (
                (int(col_indices[col]), int(row_indices[row]))
                for row, col in zip(assignment_rows, assignment_cols, strict=True)
                if math.isfinite(float(finite_cost[row, col]))
            )
        )
    normalized = tuple(
        float(cost[moving_index, reference_index])
        for reference_index, moving_index in pairs
    )
    lateral = tuple(
        float(
            np.hypot(
                *(moving_xyz[moving_index, :2] - reference_xyz[reference_index, :2])
            )
        )
        for reference_index, moving_index in pairs
    )
    axial = tuple(
        float(abs(moving_xyz[moving_index, 2] - reference_xyz[reference_index, 2]))
        for reference_index, moving_index in pairs
    )
    matched = len(pairs)
    return ObjectMatchResult(
        tuple(pairs),
        normalized,
        lateral,
        axial,
        matched,
        reference_count - matched,
        moving_count - matched,
        matched / reference_count,
        (reference_count - matched) / reference_count,
        (moving_count - matched) / moving_count,
    )


def eligible_lamp_cells(config: AnalysisConfig) -> tuple[CellSpec, ...]:
    """Return only complete, image-verified LAMP1 cells in stable biological-ID order."""
    return tuple(
        cell
        for cell in config.cells
        if cell.cohort == "lamp1"
        and tuple(round_spec.acquisition_round for round_spec in cell.rounds)
        == (1, 2, 3, 4, 5)
        and all(
            round_spec.identity_method == "image-verified" for round_spec in cell.rounds
        )
    )


def _with_nuclear_distances(
    detections: DetectionSet,
    surfaces: tuple[NuclearSurface, ...],
) -> DetectionSet:
    """Annotate objects with minimum distance to any accepted nuclear surface."""
    if not surfaces:
        raise ValueError("LAMP1 nuclear distance requires accepted nuclei")
    spacing = surfaces[0].spacing_xyz_um
    if any(surface.spacing_xyz_um != spacing for surface in surfaces):
        raise ValueError("accepted nuclear surfaces must share physical spacing")
    vertices_xyz_um = np.concatenate(
        tuple(surface.vertices_xyz_um for surface in surfaces), axis=0
    )
    if not detections.objects:
        return detections
    coordinates_xyz_um = np.asarray(detections.coordinates_xyz_um, dtype=np.float64)
    distances = cKDTree(vertices_xyz_um).query(coordinates_xyz_um, k=1)[0]
    return replace(
        detections,
        objects=tuple(
            replace(object_, nucleus_relative_distance_um=float(distance))
            for object_, distance in zip(detections.objects, distances, strict=True)
        ),
    )


def _fixed_lamp_roi(
    nuclear_mask: NDArray[np.bool_],
    validity_masks: Mapping[int, NDArray[np.bool_]],
) -> NDArray[np.bool_]:
    """Return one target-independent exterior ROI on common producer-valid support."""
    if not validity_masks:
        raise ValueError("LAMP1 ROI requires producer-attested transform support")
    nucleus = validated_binary_mask(nuclear_mask, "consensus nuclear mask")
    if nucleus.ndim != 3:
        raise ValueError("consensus nuclear mask must be 3D ZYX")
    common = np.ones(nucleus.shape, dtype=bool)
    for number, raw_mask in validity_masks.items():
        mask = validated_binary_mask(raw_mask, f"validity mask for round {number}")
        if mask.ndim != 3 or mask.shape != common.shape:
            raise ValueError(
                "transform support and consensus nucleus must share geometry"
            )
        common &= mask
    roi = common & ~nucleus
    if not np.any(roi):
        raise ValueError("fixed LAMP1 ROI is empty")
    roi.setflags(write=False)
    return roi


def _median_or_nan(values: list[float]) -> float:
    """Return a median only when a finite measurement exists, without warnings."""
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else math.nan


def _comparison_specs(rounds: tuple[int, ...]) -> tuple[tuple[str, int, int], ...]:
    """Return explicit round-2-relative and adjacent comparison pairs in order."""
    ordered = tuple(sorted(rounds))
    return tuple(
        ("round2-relative", REFERENCE_ROUND, number) for number in ordered
    ) + tuple(
        ("adjacent", previous, current)
        for previous, current in pairwise(ordered)
        if current == previous + 1
    )


def _rows_for_detections(
    cell: CellSpec, round_number: int, detections: DetectionSet, config: AnalysisConfig
) -> tuple[EnsembleRow, ...]:
    """Emit all object summaries for one declared MIP-smoothing/SBR coordinate."""
    name = (
        f"mip_smoothing_sigma_um={detections.sigma_um:g};"
        f"detection_sbr={detections.detection_sbr:g}"
    )
    objects = detections.objects
    values = {
        "lysosome_object_count": (float(detections.object_count), "count"),
        "lysosome_volume_median": (
            float(np.median([item.volume_um3 for item in objects]))
            if objects
            else math.nan,
            "um3",
        ),
        "lysosome_equivalent_diameter_median": (
            float(np.median([item.equivalent_diameter_um for item in objects]))
            if objects
            else math.nan,
            "um",
        ),
        "lysosome_sphericity_median": (
            float(np.median([item.sphericity for item in objects]))
            if objects
            else math.nan,
            "fraction",
        ),
        "lysosome_nearest_neighbor_distance_median": (
            _median_or_nan([item.nearest_neighbor_distance_um for item in objects]),
            "um",
        ),
        "lysosome_nucleus_relative_distance_median": (
            _median_or_nan([item.nucleus_relative_distance_um for item in objects]),
            "um",
        ),
    }
    return tuple(
        EnsembleRow(
            cohort=cell.cohort,
            cell_id=cell.cell_id,
            acquisition_round=round_number,
            endpoint=endpoint,
            ensemble_name=name,
            ensemble_value=detections.detection_sbr,
            sigma_um=detections.sigma_um,
            threshold_sbr=detections.detection_sbr,
            surface_tolerance_resolution_units=None,
            value=value,
            units=units,
            source_task_identity=config.config_digest,
        )
        for endpoint, (value, units) in values.items()
    )


def _mip_unit_interval(image: NDArray[np.generic]) -> NDArray[np.float64]:
    """Return a robustly normalized XY maximum-intensity projection."""
    projection = np.asarray(image, dtype=np.float64).max(axis=0)
    low, high = np.quantile(projection, (0.01, 0.999))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return np.zeros(projection.shape, dtype=np.float64)
    return np.clip((projection - low) / (high - low), 0.0, 1.0)


def _draw_mip_match_status(
    axis: object,
    reference: DetectionSet,
    moving: DetectionSet,
    match: ObjectMatchResult,
    spacing_xy_um: tuple[float, float],
) -> None:
    """Draw colorblind object-status ellipses and matched-object links on one MIP."""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    sx, sy = spacing_xy_um
    matched_reference = {reference_index for reference_index, _ in match.matched_pairs}
    matched_moving = {moving_index for _, moving_index in match.matched_pairs}

    def add_object(object_: LysosomeObject, color: str, linestyle: str) -> None:
        x_um, y_um, _z_um = object_.centroid_xyz_um
        diameter_um = object_.equivalent_diameter_um
        axis.add_patch(
            Ellipse(
                (x_um / sx, y_um / sy),
                width=diameter_um / sx,
                height=diameter_um / sy,
                fill=False,
                edgecolor=color,
                linestyle=linestyle,
                linewidth=0.8,
                alpha=0.9,
                rasterized=True,
            )
        )

    for index, object_ in enumerate(reference.objects):
        add_object(
            object_,
            "#666666" if index in matched_reference else "#009E73",
            "-" if index in matched_reference else "--",
        )
    for index, object_ in enumerate(moving.objects):
        add_object(
            object_,
            "#666666" if index in matched_moving else "#0072B2",
            "-" if index in matched_moving else "--",
        )
    segments = [
        (
            np.asarray(reference.coordinates_xyz_um[reference_index][:2])
            / np.asarray((sx, sy)),
            np.asarray(moving.coordinates_xyz_um[moving_index][:2])
            / np.asarray((sx, sy)),
        )
        for reference_index, moving_index in match.matched_pairs
    ]
    if segments:
        axis.add_collection(
            LineCollection(
                segments,
                colors="#666666",
                linewidths=0.5,
                alpha=0.65,
                rasterized=True,
            )
        )
    axis.legend(
        handles=(
            Line2D([], [], color="#009E73", linestyle="--", label="round 2 only"),
            Line2D([], [], color="#0072B2", linestyle="--", label="round N only"),
            Line2D([], [], color="#666666", linestyle="-", label="matched 2 ↔ N"),
        ),
        frameon=True,
        framealpha=0.9,
        fontsize=8,
        loc="upper right",
    )


def _write_mip_match_overlays(
    staged: Path,
    lamp_mips: Mapping[int, NDArray[np.floating]],
    detections: Mapping[int, tuple[DetectionSet, ...]],
    matches: list[tuple[str, int, int, DetectionSet, ObjectMatchResult]],
    config: AnalysisConfig,
) -> tuple[Path, ...]:
    """Render one representative round-2 MIP comparison for each moving round.

    A neutral grayscale signal MIP is overlaid with colorblind object status at the
    median configured MIP-smoothing/SBR setting. The complete nine-member table
    remains the quantitative result; these figures are deliberately visual QC only.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    sigma_um = config.lysosome.log_sigma_um[len(config.lysosome.log_sigma_um) // 2]
    detection_sbr = config.lysosome.detection_sbr[
        len(config.lysosome.detection_sbr) // 2
    ]
    reference_round = REFERENCE_ROUND
    reference = next(
        item
        for item in detections[reference_round]
        if (item.sigma_um, item.detection_sbr) == (sigma_um, detection_sbr)
    )
    sx, sy, _sz = config.voxel_spacing_xyz_um
    paths: list[Path] = []
    for comparison, ref_round, moving_round, reference_member, match in matches:
        if (
            comparison != "round2-relative"
            or ref_round != reference_round
            or (reference_member.sigma_um, reference_member.detection_sbr)
            != (sigma_um, detection_sbr)
        ):
            continue
        moving = next(
            item
            for item in detections[moving_round]
            if (item.sigma_um, item.detection_sbr) == (sigma_um, detection_sbr)
        )
        signal = np.maximum(lamp_mips[reference_round], lamp_mips[moving_round])
        figure, axis = plt.subplots(figsize=(7.2, 7.0), layout="constrained")
        axis.imshow(signal, cmap="gray", origin="upper", interpolation="nearest")
        _draw_mip_match_status(axis, reference, moving, match, (sx, sy))
        axis.set_axis_off()
        axis.set_title(
            f"LAMP1 MIP object status: round 2 vs round {moving_round}\n"
            f"MIP smoothing σ={sigma_um:g} µm; SBR={detection_sbr:g}; "
            f"matched {match.matched_count}, lost {match.lost_count}, new {match.new_count}"
        )
        path = staged / f"lamp1-mip-overlay-round-{moving_round}.png"
        figure.savefig(path, dpi=220, facecolor="white")
        plt.close(figure)
        paths.append(path)
    return tuple(paths)


def analyze_lamp1_cell(
    cell: CellSpec,
    artifacts: tuple[RoundArtifacts, ...],
    config: AnalysisConfig,
    output_dir: Path,
    *,
    local_workers: int = 1,
) -> MarkerResult:
    """Analyze one eligible LAMP1 cell without cohort-level inference or implicit IDs."""
    if cell not in eligible_lamp_cells(config):
        raise ValueError("LAMP1 analysis requires a complete image-verified CellSpec")
    by_round: Mapping[int, RoundArtifacts] = {
        item.acquisition_round: item for item in artifacts
    }
    if set(by_round) != {item.acquisition_round for item in cell.rounds}:
        raise ValueError(
            "cell and validated artifact acquisition-round inventories disagree"
        )
    # The target-blind gate must run before any LAMP target is opened.
    try:
        accepted = _accepted_rounds(cell, config)
    except ValueError as error:
        raise ValueError(
            "target-blind QC decisions are required for every moving LAMP1 round"
        ) from error
    if REFERENCE_ROUND not in accepted:
        raise ValueError("LAMP1 analysis requires acquisition round 2")
    dapi = {
        number: load_volume(by_round[number].registered_dapi) for number in accepted
    }
    consensus = consensus_nuclear_surface(
        dapi,
        config.voxel_spacing_xyz_um,
        config.nuclear,
        config.nominal_expansion_factor,
    )
    validity_masks, validity_digest, validity_identities = _artifact_validity_masks(
        {number: by_round[number] for number in accepted}, config
    )
    roi = _fixed_lamp_roi(consensus.mask_zyx, validity_masks)
    lamp = {
        number: load_volume(by_round[number].registered_target) for number in accepted
    }
    if len({image.shape for image in (*lamp.values(), *dapi.values())}) != 1:
        raise ValueError(
            "registered LAMP1 and DAPI volumes must share fixed-grid geometry"
        )
    resolution = theoretical_resolution(
        config.optics.target_wavelength_um,
        config.optics.numerical_aperture,
        config.optics.immersion_refractive_index,
        config.nominal_expansion_factor,
    )
    detections = {
        number: tuple(
            _with_nuclear_distances(item, consensus.nuclei)
            for item in detect_lysosomes_mip(
                lamp[number],
                roi,
                config.voxel_spacing_xyz_um,
                config.lysosome.log_sigma_um,
                config.lysosome.detection_sbr,
                local_workers=local_workers,
            )
        )
        for number in accepted
    }
    ensemble_rows = tuple(
        row
        for number in sorted(detections)
        for item in detections[number]
        for row in _rows_for_detections(cell, number, item, config)
    )
    summary_rows: list[MetricRow] = []
    matches: list[tuple[str, int, int, DetectionSet, ObjectMatchResult]] = []
    for comparison, reference_round, moving_round in _comparison_specs(accepted):
        reference_members = {
            (item.sigma_um, item.detection_sbr): item
            for item in detections[reference_round]
        }
        moving_members = {
            (item.sigma_um, item.detection_sbr): item
            for item in detections[moving_round]
        }
        for key, reference_member in reference_members.items():
            match = match_lysosomes(
                reference_member,
                moving_members[key],
                resolution,
                config.lysosome.match_resolution_units,
            )
            matches.append(
                (comparison, reference_round, moving_round, reference_member, match)
            )
            for endpoint, value, units in (
                ("matched_fraction", match.matched_fraction, "fraction"),
                ("lost_fraction", match.lost_fraction, "fraction"),
                ("new_fraction", match.new_fraction, "fraction"),
                (
                    "lateral_displacement_median",
                    _median_or_nan(list(match.lateral_displacements_um)),
                    "um",
                ),
                (
                    "axial_displacement_median",
                    _median_or_nan(list(match.axial_displacements_um)),
                    "um",
                ),
                (
                    "normalized_displacement_median",
                    _median_or_nan(list(match.normalized_displacements)),
                    "resolution-units",
                ),
            ):
                summary_rows.append(
                    MetricRow(
                        cell.cohort,
                        cell.cell_id,
                        moving_round,
                        f"lysosome_{endpoint}_sigma_{key[0]:g}_sbr_{key[1]:g}",
                        value,
                        units,
                        comparison,
                        True,
                        config.config_digest,
                    )
                )
    staged_paths: tuple[Path, ...] = ()
    staging_root: Path | None = None

    def populate(staged: Path) -> None:
        nonlocal staged_paths, staging_root
        staging_root = staged
        component_diagnostic = write_nuclear_component_diagnostic(
            staged, cell.cell_id, dapi, consensus
        )
        diagnostic = staged / "lamp1-detections.npz"
        np.savez_compressed(
            diagnostic,
            **{
                f"round_{number}_sigma_{item.sigma_um:g}_sbr_{item.detection_sbr:g}_xyz_um": np.asarray(
                    item.coordinates_xyz_um
                )
                for number, items in detections.items()
                for item in items
            },
        )
        table = staged / "lamp1-ensemble.csv"
        with table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(EnsembleRow.__dataclass_fields__)
            writer.writerows(
                [getattr(row, field) for field in EnsembleRow.__dataclass_fields__]
                for row in ensemble_rows
            )
        objects_table = staged / "objects.csv"
        with objects_table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "acquisition_round",
                    "mip_smoothing_sigma_um",
                    "detection_sbr",
                    "object_index",
                    "x_um",
                    "y_um",
                    "z_um",
                    "volume_um3",
                    "equivalent_diameter_um",
                    "sphericity",
                    "nearest_neighbor_distance_um",
                    "nucleus_relative_distance_um",
                    "nucleus_id",
                )
            )
            for number, members in sorted(detections.items()):
                for member in members:
                    for index, object_ in enumerate(member.objects):
                        writer.writerow(
                            (
                                number,
                                member.sigma_um,
                                member.detection_sbr,
                                index,
                                *object_.centroid_xyz_um,
                                object_.volume_um3,
                                object_.equivalent_diameter_um,
                                object_.sphericity,
                                object_.nearest_neighbor_distance_um,
                                object_.nucleus_relative_distance_um,
                                "aggregate",
                            )
                        )
        matches_table = staged / "matches.csv"
        with matches_table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "comparison_type",
                    "reference_round",
                    "moving_round",
                    "mip_smoothing_sigma_um",
                    "detection_sbr",
                    "reference_object_index",
                    "moving_object_index",
                    "reference_x_um",
                    "reference_y_um",
                    "reference_z_um",
                    "moving_x_um",
                    "moving_y_um",
                    "moving_z_um",
                    "lateral_displacement_um",
                    "axial_displacement_um",
                    "normalized_displacement",
                    "nucleus_id",
                )
            )
            for (
                comparison,
                reference_round,
                moving_round,
                reference_member,
                match,
            ) in matches:
                moving_member = next(
                    item
                    for item in detections[moving_round]
                    if (item.sigma_um, item.detection_sbr)
                    == (reference_member.sigma_um, reference_member.detection_sbr)
                )
                for pair_index, (reference_index, moving_index) in enumerate(
                    match.matched_pairs
                ):
                    writer.writerow(
                        (
                            comparison,
                            reference_round,
                            moving_round,
                            reference_member.sigma_um,
                            reference_member.detection_sbr,
                            reference_index,
                            moving_index,
                            *reference_member.objects[reference_index].centroid_xyz_um,
                            *moving_member.objects[moving_index].centroid_xyz_um,
                            match.lateral_displacements_um[pair_index],
                            match.axial_displacements_um[pair_index],
                            match.normalized_displacements[pair_index],
                            "aggregate",
                        )
                    )
        lamp_mips = {
            number: _mip_unit_interval(image) for number, image in lamp.items()
        }
        mip_overlays = _write_mip_match_overlays(
            staged, lamp_mips, detections, matches, config
        )
        staged_paths = (
            component_diagnostic,
            diagnostic,
            table,
            objects_table,
            matches_table,
            *mip_overlays,
        )

    _atomic_publish_directory(output_dir, populate)
    assert staging_root is not None
    diagnostics = tuple(
        output_dir / item.relative_to(staging_root) for item in staged_paths
    )
    identities = (
        tuple(
            str(path)
            for number in accepted
            for artifact in (by_round[number],)
            for path in (artifact.registered_dapi, artifact.registered_target)
        )
        + validity_identities
        + (f"validity-mask-sha256:{validity_digest}",)
    )
    return MarkerResult(
        cell.cohort,
        cell.cell_id,
        tuple(summary_rows),
        ensemble_rows,
        diagnostics,
        identities,
        tuple(MappingProxyType(asdict(row)) for row in consensus.component_qc),
    )


__all__ = [
    "DetectionSet",
    "LysosomeObject",
    "ObjectMatchResult",
    "analyze_lamp1_cell",
    "detect_lysosomes",
    "eligible_lamp_cells",
    "match_lysosomes",
]
