"""Plane-background-corrected 3D Otsu with an explicit physical size cutoff."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label
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
) -> MaskMember:
    """Return one prespecified mask plus every removed small component.

    Background is the median of the eligible exterior ROI in each Z plane.
    Otsu uses the clipped background-corrected ROI histogram at 1 intensity-unit
    bin width, including zeros. Thresholding, exclusion and 26-connected size
    filtering all operate in 3D. No smoothing, hole filling or component ranking
    changes the topology. Volume units match the supplied physical voxel spacing.
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
    backgrounds = np.zeros(source.shape[0], dtype=np.float64)
    corrected = np.zeros(source.shape, dtype=np.float32)
    deviations = []
    active_planes = []
    for z in range(source.shape[0]):
        values = source[z][domain[z]].astype(np.float64)
        if values.size:
            active_planes.append(z)
            backgrounds[z] = np.median(values)
            deviations.append(np.abs(values - backgrounds[z]))
            corrected[z] = np.maximum(source[z].astype(np.float64) - backgrounds[z], 0)
    values = corrected[domain]
    maximum = float(values.max())
    scale = float(1.4826 * np.median(np.concatenate(deviations)))
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
        if not math.isfinite(scale) or scale <= 0 or threshold < 3.0 * scale:
            raise ValueError(
                "Otsu foreground is indistinguishable from background: "
                f"threshold={threshold:g}, robust noise scale={scale:g}; "
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
    for array in (mask, rejected, backgrounds):
        array.setflags(write=False)
    return MaskMember(
        mask,
        0.0,
        None,
        background,
        spacing_xyz_um,
        "nonempty" if np.any(mask) else "empty",
        threshold_method="plane-median-otsu",
        threshold_value=threshold,
        minimum_component_volume_um3=minimum_volume_um3,
        plane_background=backgrounds,
        rejected_mask_zyx=rejected,
        component_sizes_voxels=tuple(int(x) for x in sizes[1:]),
    )


def write_native_otsu_diagnostics(
    output_dir: Path, round_number: int, member: MaskMember, roi: NDArray[np.bool_]
) -> tuple[Path, Path]:
    """Save the exact native masks and backgrounds used for signal integration."""
    if (
        member.threshold_method != "plane-median-otsu"
        or member.minimum_component_volume_um3 is None
        or member.plane_background is None
        or member.rejected_mask_zyx is None
    ):
        raise ValueError("native Otsu diagnostics require a complete Otsu member")
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = output_dir / f"round-{round_number}-segmentation.npz"
    metadata = output_dir / f"round-{round_number}-segmentation.json"
    np.savez_compressed(
        masks,
        retained_mask_zyx=member.mask_zyx,
        rejected_small_components_zyx=member.rejected_mask_zyx,
        nuclear_exclusion_mask_zyx=~roi,
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
                "coordinate_space": "native acquisition; full extranuclear field",
                "method": member.threshold_method,
                "threshold_above_plane_background": member.threshold_value,
                "background_by_plane": member.plane_background.tolist(),
                "robust_noise_scale": member.background.scale,
                "minimum_threshold_noise_scales": 3.0,
                "minimum_component_volume_expanded_um3": member.minimum_component_volume_um3,
                "minimum_component_voxels": minimum_voxels,
                "component_sizes_voxels": member.component_sizes_voxels,
                "retained_voxels": int(member.mask_zyx.sum()),
                "removed_voxels": int(member.rejected_mask_zyx.sum()),
                "nuclear_overlap_voxels": int(np.count_nonzero(member.mask_zyx & ~roi)),
                "connectivity": 26,
                "smoothing": "none",
                "intensity_policy": "sum max(original intensity - plane background, 0) / exposure_ms",
            },
            indent=2,
        )
        + "\n"
    )
    return masks, metadata
