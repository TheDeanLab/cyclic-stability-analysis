"""Physical threshold ensembles and resolution-normalized surface measures."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from cyclic_stability.images import BackgroundSummary, robust_background
from cyclic_stability.units import ResolutionReference


@dataclass(frozen=True, slots=True)
class MaskMember:
    """One reproducible physical-smoothing and SBR-threshold mask choice."""

    mask_zyx: NDArray[np.bool_]
    sigma_um: float
    threshold_sbr: float
    background: BackgroundSummary
    spacing_xyz_um: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SurfaceMetrics:
    """Bidirectional resolution-normalized boundary agreement measures."""

    fixed_to_moving_normalized_distance_p95: float
    moving_to_fixed_normalized_distance_p95: float
    normalized_distance_p95: float
    surface_dice: Mapping[float, float]
    jaccard: float


def _spacing_xyz(
    spacing_xyz_um: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Validate public XYZ spacing without changing its documented order."""
    if len(spacing_xyz_um) != 3:
        raise ValueError("spacing_xyz_um must contain exactly three values")
    spacing = tuple(float(value) for value in spacing_xyz_um)
    if any(not math.isfinite(value) or value <= 0.0 for value in spacing):
        raise ValueError("spacing_xyz_um must be finite and positive")
    return spacing


def gaussian_sigma_zyx(
    sigma_um: float, spacing_xyz_um: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Convert an isotropic physical Gaussian sigma to NumPy ZYX voxel units."""
    if not math.isfinite(sigma_um) or sigma_um <= 0.0:
        raise ValueError("sigma_um must be finite and positive")
    sx, sy, sz = _spacing_xyz(spacing_xyz_um)
    return (sigma_um / sz, sigma_um / sy, sigma_um / sx)


def _validated_image_and_roi(
    image: NDArray[np.generic], roi: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Validate matching finite ZYX signal and independent structure ROI volumes."""
    source = np.asarray(image)
    mask = np.asarray(roi, dtype=bool)
    if source.ndim != 3 or mask.ndim != 3 or source.shape != mask.shape:
        raise ValueError("image and ROI must be matching 3D ZYX volumes")
    if not np.issubdtype(source.dtype, np.number) or np.issubdtype(
        source.dtype, np.complexfloating
    ):
        raise ValueError("image must have a real numeric dtype")
    if not np.all(np.isfinite(source)):
        raise ValueError("image must contain only finite values")
    return source.astype(np.float64, copy=False), mask


def threshold_ensemble(
    image: NDArray[np.generic],
    roi: NDArray[np.bool_],
    sigma_values_um: tuple[float, ...],
    threshold_sbr: tuple[float, ...],
    spacing_xyz_um: tuple[float, float, float],
) -> tuple[MaskMember, ...]:
    """Return every physical-smoothing/SBR threshold mask in the sensitivity grid.

    Gaussian smoothing is used solely to construct the masks.  Source image
    intensities are deliberately not replaced, so downstream signal summaries
    can integrate the unregistered original target values.
    """
    source, structure_roi = _validated_image_and_roi(image, roi)
    spacing = _spacing_xyz(spacing_xyz_um)
    if not sigma_values_um or not threshold_sbr:
        raise ValueError("sigma_values_um and threshold_sbr must be non-empty")
    background = robust_background(source)
    members: list[MaskMember] = []
    for sigma_um in sigma_values_um:
        sigma_zyx = gaussian_sigma_zyx(float(sigma_um), spacing)
        smoothed = gaussian_filter(source, sigma=sigma_zyx)
        for sbr in threshold_sbr:
            threshold = float(sbr)
            if not math.isfinite(threshold) or threshold <= 0.0:
                raise ValueError("threshold_sbr values must be finite and positive")
            mask = (
                smoothed > background.median + threshold * background.scale
            ) & structure_roi
            mask.setflags(write=False)
            members.append(
                MaskMember(
                    mask_zyx=mask,
                    sigma_um=float(sigma_um),
                    threshold_sbr=threshold,
                    background=background,
                    spacing_xyz_um=spacing,
                )
            )
    return tuple(members)


def _surface_vertices_xyz(
    mask_zyx: NDArray[np.bool_], spacing_xyz_um: tuple[float, float, float]
) -> NDArray[np.float64]:
    """Return padded binary-mask marching-cubes vertices in physical XYZ order."""
    mask = np.asarray(mask_zyx, dtype=bool)
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError("surface masks must be non-empty 3D ZYX volumes")
    sx, sy, sz = spacing_xyz_um
    padded = np.pad(mask.astype(np.float32), 1)
    vertices_zyx, _, _, _ = marching_cubes(padded, level=0.5, spacing=(sz, sy, sx))
    vertices_zyx -= np.array((sz, sy, sx))
    return vertices_zyx[:, ::-1]


def _tolerances(tolerance_units: float | tuple[float, ...]) -> tuple[float, ...]:
    """Normalize one or more strictly positive resolution-unit tolerances."""
    raw = (
        (tolerance_units,)
        if isinstance(tolerance_units, (float, int))
        else tolerance_units
    )
    result = tuple(float(value) for value in raw)
    if not result or any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError("tolerance_units must be finite and positive")
    return result


def resolution_surface_metrics(
    fixed_mask: NDArray[np.bool_],
    moving_mask: NDArray[np.bool_],
    spacing_xyz_um: tuple[float, float, float],
    resolution: ResolutionReference,
    tolerance_units: float | tuple[float, ...],
) -> SurfaceMetrics:
    """Measure bidirectional surface agreement in theoretical-resolution units."""
    fixed = np.asarray(fixed_mask, dtype=bool)
    moving = np.asarray(moving_mask, dtype=bool)
    if fixed.ndim != 3 or moving.ndim != 3 or fixed.shape != moving.shape:
        raise ValueError("fixed and moving masks must be matching 3D ZYX volumes")
    spacing = _spacing_xyz(spacing_xyz_um)
    fixed_xyz = _surface_vertices_xyz(fixed, spacing)
    moving_xyz = _surface_vertices_xyz(moving, spacing)
    reference_xyz = np.array(
        (
            resolution.lateral_expanded_um,
            resolution.lateral_expanded_um,
            resolution.axial_expanded_um,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(reference_xyz)) or np.any(reference_xyz <= 0.0):
        raise ValueError("resolution reference must be finite and positive")
    fixed_normalized = fixed_xyz / reference_xyz
    moving_normalized = moving_xyz / reference_xyz
    fixed_to_moving = cKDTree(moving_normalized).query(fixed_normalized, k=1)[0]
    moving_to_fixed = cKDTree(fixed_normalized).query(moving_normalized, k=1)[0]
    distances = np.concatenate((fixed_to_moving, moving_to_fixed))
    dice = {
        tolerance: float(
            0.5
            * (
                np.mean(fixed_to_moving <= tolerance)
                + np.mean(moving_to_fixed <= tolerance)
            )
        )
        for tolerance in _tolerances(tolerance_units)
    }
    union = np.logical_or(fixed, moving).sum()
    return SurfaceMetrics(
        fixed_to_moving_normalized_distance_p95=float(
            np.percentile(fixed_to_moving, 95.0)
        ),
        moving_to_fixed_normalized_distance_p95=float(
            np.percentile(moving_to_fixed, 95.0)
        ),
        normalized_distance_p95=float(np.percentile(distances, 95.0)),
        surface_dice=MappingProxyType(dice),
        jaccard=float(np.logical_and(fixed, moving).sum() / union),
    )


__all__ = [
    "MaskMember",
    "SurfaceMetrics",
    "gaussian_sigma_zyx",
    "resolution_surface_metrics",
    "threshold_ensemble",
]
