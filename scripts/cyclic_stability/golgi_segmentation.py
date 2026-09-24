"""Plane-background-corrected 3D Otsu with an explicit physical size cutoff."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, label
from skimage.filters import threshold_otsu

from cyclic_stability.images import BackgroundSummary
from cyclic_stability.morphology import MaskMember, validated_binary_mask


def corrected_mask_values(
    image: NDArray[np.generic], member: MaskMember, mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Background-correct original selected voxels without smoothing intensities."""
    raw = np.asarray(image)[mask].astype(np.float64)
    if member.plane_background is None:
        background = member.background.median
    else:
        if member.plane_background.shape != (image.shape[0],):
            raise ValueError("plane background must match target Z geometry")
        background = member.plane_background[np.nonzero(mask)[0]]
    return np.maximum(raw - background, 0.0)


def otsu_component_member(
    image: NDArray[np.generic],
    roi: NDArray[np.bool_],
    spacing_xyz_um: tuple[float, float, float],
    minimum_volume_um3: float,
    *,
    smoothing_um: float | None = None,
) -> MaskMember:
    """Return one prespecified mask plus every removed small component.

    Background is the median of the eligible exterior ROI in each Z plane.
    Otsu uses the clipped background-corrected ROI histogram at 1 intensity-unit
    bin width, including zeros. Thresholding, exclusion and 26-connected size
    filtering all operate in 3D. No hole filling or component ranking is applied.
    Optional normalized Gaussian smoothing uses only eligible ROI voxels after
    plane-offset subtraction. Its residual plane medians are added to the saved
    background before original-intensity integration.
    Volume units match the supplied physical voxel spacing.
    """
    source = np.asarray(image)
    domain = validated_binary_mask(roi, "Golgi ROI")
    if source.ndim != 3 or domain.shape != source.shape or not np.any(domain):
        raise ValueError("Golgi target and nonempty ROI must share 3D ZYX geometry")
    if (
        not np.issubdtype(source.dtype, np.number)
        or np.iscomplexobj(source)
        or not np.all(np.isfinite(source))
    ):
        raise ValueError("Golgi target must contain finite real intensities")
    if len(spacing_xyz_um) != 3 or any(
        not math.isfinite(x) or x <= 0 for x in spacing_xyz_um
    ):
        raise ValueError("Golgi spacing must contain three finite positive values")
    if not math.isfinite(minimum_volume_um3) or minimum_volume_um3 <= 0:
        raise ValueError("minimum component volume must be finite and positive")
    if smoothing_um is not None and (
        not math.isfinite(smoothing_um) or smoothing_um <= 0
    ):
        raise ValueError("Golgi smoothing must be finite and positive or None")
    full_shape = source.shape
    # Crop only to the ROI's exact bounding box; normalized convolution uses
    # constant zero weights outside it, so this preserves full-grid results.
    bounds = []
    for axis in range(3):
        indices = np.flatnonzero(
            domain.any(axis=tuple(a for a in range(3) if a != axis))
        )
        bounds.append(slice(int(indices[0]), int(indices[-1]) + 1))
    region = tuple(bounds)
    source = source[region]
    domain = domain[region]
    backgrounds = np.zeros(source.shape[0], dtype=np.float64)
    corrected = np.zeros(source.shape, dtype=np.float32)
    active_planes = []
    for z in range(source.shape[0]):
        values = source[z][domain[z]].astype(np.float64)
        if values.size:
            active_planes.append(z)
            backgrounds[z] = np.median(values)
            corrected[z] = source[z].astype(np.float64) - backgrounds[z]
    if smoothing_um is not None:
        sigmas = tuple(smoothing_um / spacing for spacing in spacing_xyz_um[::-1])
        corrected[~domain] = 0
        weights = gaussian_filter(domain.astype(np.float32), sigmas, mode="constant")
        corrected = gaussian_filter(corrected, sigmas, mode="constant")
        np.divide(corrected, weights, out=corrected, where=weights > 0)
        del weights
    deviations = []
    threshold_deviations = []
    for z in active_planes:
        if smoothing_um is not None:
            # Remove any residual median shift introduced by smoothing a sparse
            # asymmetric signal. Record the exact resulting background for raw
            # intensity integration; never integrate the smoothed image.
            offset = float(np.median(corrected[z][domain[z]]))
            corrected[z] -= offset
            backgrounds[z] += offset
        deviations.append(
            np.abs(source[z][domain[z]].astype(np.float64) - backgrounds[z])
        )
        threshold_deviations.append(np.abs(corrected[z][domain[z]]))
    scale = float(1.4826 * np.median(np.concatenate(deviations)))
    threshold_scale = float(1.4826 * np.median(np.concatenate(threshold_deviations)))
    del deviations, threshold_deviations
    np.maximum(corrected, 0, out=corrected)
    values = corrected[domain]
    maximum = float(values.max())
    if maximum > 1_000_000:
        raise ValueError("1-unit Otsu histogram requires intensities below 1,000,000")
    # Degenerate input is an explicit unmeasured mask, never perfect agreement.
    threshold = None
    foreground = np.zeros(source.shape, dtype=bool)
    if maximum > 0:
        edges = np.arange(-0.5, maximum + 1.5, 1.0)
        histogram, edges = np.histogram(values, bins=edges)
        threshold = float(
            threshold_otsu(hist=(histogram, (edges[:-1] + edges[1:]) / 2))
        )
        if (
            not math.isfinite(threshold_scale)
            or threshold_scale <= 0
            or threshold < 3.0 * threshold_scale
        ):
            raise ValueError(
                "Otsu foreground is indistinguishable from background: "
                f"threshold={threshold:g}, robust noise scale={threshold_scale:g}; "
                "requires threshold >= 3 noise scales"
            )
        foreground = (corrected > threshold) & domain
    labels, count = cast(
        tuple[NDArray[np.int32], int],
        label(foreground, structure=np.ones((3, 3, 3), dtype=bool)),
    )
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    min_voxels = math.ceil(minimum_volume_um3 / math.prod(spacing_xyz_um))
    keep = sizes >= min_voxels
    keep[0] = False
    mask = keep[labels]
    rejected = foreground & ~mask
    background = BackgroundSummary(
        float(np.median(backgrounds[active_planes])),
        scale,
        int(np.count_nonzero(values)),
        int(domain.sum()),
    )
    if source.shape != full_shape:
        full_mask = np.zeros(full_shape, dtype=bool)
        full_mask[region] = mask
        mask = full_mask
        full_rejected = np.zeros(full_shape, dtype=bool)
        full_rejected[region] = rejected
        rejected = full_rejected
        full_backgrounds = np.zeros(full_shape[0], dtype=np.float64)
        full_backgrounds[region[0]] = backgrounds
        backgrounds = full_backgrounds
    for array in (mask, rejected, backgrounds):
        array.setflags(write=False)
    return MaskMember(
        mask,
        smoothing_um or 0.0,
        None,
        background,
        spacing_xyz_um,
        "nonempty" if np.any(mask) else "empty",
        threshold_method="plane-median-otsu",
        threshold_noise_scale=threshold_scale,
        threshold_value=threshold,
        minimum_component_volume_um3=minimum_volume_um3,
        plane_background=backgrounds,
        rejected_mask_zyx=rejected,
        component_sizes_voxels=tuple(int(x) for x in sizes[1:]),
    )


def write_native_otsu_diagnostics(
    output_dir: Path,
    round_number: int,
    member: MaskMember,
    roi: NDArray[np.bool_],
    *,
    nuclear_exclusion: NDArray[np.bool_],
    outer_distance_um: float,
) -> tuple[Path, Path]:
    """Save the exact native masks and backgrounds used for signal integration."""
    if (
        member.threshold_method != "plane-median-otsu"
        or member.minimum_component_volume_um3 is None
        or member.plane_background is None
        or member.rejected_mask_zyx is None
    ):
        raise ValueError("native Otsu diagnostics require a complete Otsu member")
    roi = validated_binary_mask(roi, "native structure ROI")
    nuclear_exclusion = validated_binary_mask(
        nuclear_exclusion, "native DAPI exclusion"
    )
    if (
        roi.shape != member.mask_zyx.shape
        or nuclear_exclusion.shape != roi.shape
        or np.any(roi & nuclear_exclusion)
        or np.any(member.mask_zyx & ~roi)
    ):
        raise ValueError("native Otsu masks must share geometry and exclude DAPI")
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = output_dir / f"round-{round_number}-segmentation.npz"
    metadata = output_dir / f"round-{round_number}-segmentation.json"
    np.savez_compressed(
        masks,
        retained_mask_zyx=member.mask_zyx,
        rejected_small_components_zyx=member.rejected_mask_zyx,
        nuclear_exclusion_mask_zyx=nuclear_exclusion,
        structure_roi_zyx=roi,
        background_by_plane=member.plane_background,
        spacing_xyz_um=member.spacing_xyz_um,
    )
    minimum_voxels = math.ceil(
        member.minimum_component_volume_um3 / math.prod(member.spacing_xyz_um)
    )
    metadata.write_text(
        json.dumps(
            {
                "acquisition_round": round_number,
                "coordinate_space": "native acquisition; DAPI-defined perinuclear ROI",
                "perinuclear_outer_distance_um": outer_distance_um,
                "method": member.threshold_method,
                "threshold_above_plane_background": member.threshold_value,
                "background_by_plane": member.plane_background.tolist(),
                "robust_noise_scale": member.threshold_noise_scale,
                "raw_background_noise_scale": member.background.scale,
                "minimum_threshold_noise_scales": 3.0,
                "minimum_component_volume_expanded_um3": member.minimum_component_volume_um3,
                "minimum_component_voxels": minimum_voxels,
                "component_sizes_voxels": member.component_sizes_voxels,
                "retained_voxels": int(member.mask_zyx.sum()),
                "removed_voxels": int(member.rejected_mask_zyx.sum()),
                "nuclear_overlap_voxels": int(
                    np.count_nonzero(member.mask_zyx & nuclear_exclusion)
                ),
                "connectivity": 26,
                "smoothing": "normalized Gaussian in eligible ROI"
                if member.sigma_um
                else "none",
                "smoothing_sigma_um": member.sigma_um,
                "intensity_policy": "sum max(original intensity - plane background, 0) / exposure_ms",
            },
            indent=2,
        )
        + "\n"
    )
    return masks, metadata
