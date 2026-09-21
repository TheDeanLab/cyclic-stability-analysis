# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "numpy>=1.26,<3",
#   "scikit-image>=0.22,<1",
#   "scipy>=1.12,<2",
#   "tifffile>=2025.1.10",
# ]
# ///
"""Compare fast Golgi segmentation candidates outside the production pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, label
from skimage.filters import apply_hysteresis_threshold


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One fixed local-background and hysteresis segmentation policy."""

    name: str
    feature_sigma_um: float
    background_sigma_um: float
    low_sbr: float
    high_sbr: float
    minimum_size_voxels: int

    def __post_init__(self) -> None:
        """Reject ambiguous or non-physical candidate definitions."""
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("candidate name must be non-empty and contain no spaces")
        for value, name in (
            (self.feature_sigma_um, "feature_sigma_um"),
            (self.background_sigma_um, "background_sigma_um"),
            (self.low_sbr, "low_sbr"),
            (self.high_sbr, "high_sbr"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.background_sigma_um <= self.feature_sigma_um:
            raise ValueError("background sigma must exceed feature sigma")
        if self.high_sbr <= self.low_sbr:
            raise ValueError("high_sbr must exceed low_sbr")
        if isinstance(self.minimum_size_voxels, bool) or self.minimum_size_voxels < 1:
            raise ValueError("minimum_size_voxels must be a positive integer")


DEFAULT_CANDIDATES = (
    CandidateSpec("sensitive", 0.25, 2.0, 2.0, 6.0, 32),
    CandidateSpec("balanced", 0.25, 4.0, 3.0, 8.0, 32),
    CandidateSpec("conservative", 0.50, 4.0, 4.0, 10.0, 32),
)


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """One candidate mask plus its local-background threshold evidence."""

    mask: NDArray[np.bool_]
    feature: NDArray[np.float32]
    center: float
    noise_scale: float
    low_threshold: float
    high_threshold: float


def _validated_image_and_support(
    image: NDArray[np.generic], support: NDArray[np.generic]
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    source = np.asarray(image)
    domain = np.asarray(support)
    if source.ndim != 2 or domain.shape != source.shape or domain.dtype != np.bool_:
        raise ValueError("image and Boolean support must be matching 2-D arrays")
    if not np.issubdtype(source.dtype, np.number) or not np.all(np.isfinite(source)):
        raise ValueError("image must contain finite numeric values")
    if not np.any(domain):
        raise ValueError("support must contain at least one pixel")
    return source.astype(np.float32, copy=False), domain


def _masked_gaussian(
    image: NDArray[np.float32],
    support: NDArray[np.bool_],
    sigma_yx: tuple[float, float],
) -> NDArray[np.float32]:
    weights = support.astype(np.float32, copy=False)
    numerator = gaussian_filter(image * weights, sigma_yx, mode="nearest")
    denominator = gaussian_filter(weights, sigma_yx, mode="nearest")
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > np.finfo(np.float32).eps,
    )


def _remove_small_components(
    mask: NDArray[np.bool_], minimum_size: int
) -> NDArray[np.bool_]:
    labels, count = label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_size
    keep[0] = False
    return keep[labels]


def dog_hysteresis_mask(
    image: NDArray[np.generic],
    support: NDArray[np.generic],
    pixel_size_xy_um: tuple[float, float],
    spec: CandidateSpec,
) -> SegmentationResult:
    """Segment bright Golgi structure after local-background subtraction."""
    source, domain = _validated_image_and_support(image, support)
    if len(pixel_size_xy_um) != 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in pixel_size_xy_um
    ):
        raise ValueError("pixel_size_xy_um must contain positive finite X,Y values")
    pixel_x_um, pixel_y_um = pixel_size_xy_um
    feature_sigma_yx = (
        spec.feature_sigma_um / pixel_y_um,
        spec.feature_sigma_um / pixel_x_um,
    )
    background_sigma_yx = (
        spec.background_sigma_um / pixel_y_um,
        spec.background_sigma_um / pixel_x_um,
    )
    feature = _masked_gaussian(source, domain, feature_sigma_yx) - _masked_gaussian(
        source, domain, background_sigma_yx
    )
    samples = feature[domain].astype(np.float64, copy=False)
    center = float(np.median(samples))
    noise_scale = float(1.4826 * np.median(np.abs(samples - center)))
    if not math.isfinite(noise_scale) or noise_scale <= 0.0:
        raise ValueError("local-background residual has no finite positive noise scale")
    low_threshold = center + spec.low_sbr * noise_scale
    high_threshold = center + spec.high_sbr * noise_scale
    mask = np.asarray(
        apply_hysteresis_threshold(feature, low_threshold, high_threshold), dtype=bool
    )
    mask &= domain
    mask = _remove_small_components(mask, spec.minimum_size_voxels)
    mask.setflags(write=False)
    feature.setflags(write=False)
    return SegmentationResult(
        mask,
        feature,
        center,
        noise_scale,
        low_threshold,
        high_threshold,
    )


def mask_statistics(
    image: NDArray[np.generic],
    mask: NDArray[np.generic],
    support: NDArray[np.generic],
    nuclear_mask: NDArray[np.generic],
) -> dict[str, Any]:
    """Summarize a mask without treating nuclear overlap as exclusion."""
    source, domain = _validated_image_and_support(image, support)
    segmented = np.asarray(mask)
    nucleus = np.asarray(nuclear_mask)
    if (
        segmented.shape != source.shape
        or segmented.dtype != np.bool_
        or nucleus.shape != source.shape
        or nucleus.dtype != np.bool_
    ):
        raise ValueError("mask and nuclear_mask must be matching Boolean arrays")
    if np.any(segmented & ~domain):
        raise ValueError("mask must be contained in support")
    mask_pixels = int(np.count_nonzero(segmented))
    background = domain & ~segmented
    masked_values = source[segmented]
    background_values = source[background]
    return {
        "mask_pixels": mask_pixels,
        "support_fraction": mask_pixels / int(np.count_nonzero(domain)),
        "nuclear_overlap_fraction": (
            float(np.count_nonzero(segmented & nucleus) / mask_pixels)
            if mask_pixels
            else math.nan
        ),
        "masked_intensity_median": (
            float(np.median(masked_values)) if masked_values.size else math.nan
        ),
        "background_intensity_median": (
            float(np.median(background_values)) if background_values.size else math.nan
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_registered_volume(
    path: Path,
    volume: NDArray[np.generic],
    voxel_spacing_xyz_um: tuple[float, float, float],
) -> None:
    import tifffile

    pixel_x_um, pixel_y_um, pixel_z_um = voxel_spacing_xyz_um
    tifffile.imwrite(
        path,
        np.asarray(volume),
        bigtiff=True,
        ome=True,
        compression="zlib",
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": pixel_x_um,
            "PhysicalSizeXUnit": "um",
            "PhysicalSizeY": pixel_y_um,
            "PhysicalSizeYUnit": "um",
            "PhysicalSizeZ": pixel_z_um,
            "PhysicalSizeZUnit": "um",
        },
    )


def _write_image(path: Path, image: NDArray[np.generic]) -> None:
    import tifffile

    tifffile.imwrite(path, np.asarray(image), compression="zlib")


def _render_candidate_report(
    path: Path,
    candidate: CandidateSpec,
    mips: Sequence[NDArray[np.float32]],
    results: Sequence[SegmentationResult],
    nuclear_mip: NDArray[np.bool_],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    raw_samples = np.concatenate([image.ravel() for image in mips])
    raw_min, raw_max = np.percentile(raw_samples, (0.5, 99.8))
    positive_feature = np.concatenate(
        [result.feature[result.feature > 0.0] for result in results]
    )
    feature_max = (
        float(np.percentile(positive_feature, 99.8)) if positive_feature.size else 1.0
    )
    figure, axes = plt.subplots(
        len(mips),
        4,
        figsize=(15, 3.2 * len(mips)),
        constrained_layout=True,
        squeeze=False,
    )
    for index, (image, result) in enumerate(zip(mips, results, strict=True)):
        row = axes[index]
        row[0].imshow(image, cmap="gray", vmin=raw_min, vmax=raw_max)
        row[0].set_title(f"Round {index + 1}: raw Golgi MIP")
        row[1].imshow(result.feature, cmap="magma", vmin=0.0, vmax=feature_max)
        row[1].set_title("local-background residual")
        row[2].imshow(result.mask, cmap="gray", vmin=0, vmax=1)
        row[2].set_title("candidate mask")
        row[3].imshow(image, cmap="gray", vmin=raw_min, vmax=raw_max)
        if np.any(nuclear_mip):
            row[3].contour(nuclear_mip, levels=(0.5,), colors=("cyan",), linewidths=0.6)
        if np.any(result.mask):
            row[3].contour(result.mask, levels=(0.5,), colors=("lime",), linewidths=0.8)
        row[3].set_title("cyan nucleus; green Golgi")
        for axis in row:
            axis.set_axis_off()
    figure.suptitle(
        f"{candidate.name}: DoG {candidate.feature_sigma_um:g}/{candidate.background_sigma_um:g} um; "
        f"hysteresis {candidate.low_sbr:g}/{candidate.high_sbr:g} MAD",
        fontsize=13,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _validated_spacing(
    voxel_spacing_xyz_um: tuple[float, float, float],
) -> tuple[float, float, float]:
    if len(voxel_spacing_xyz_um) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in voxel_spacing_xyz_um
    ):
        raise ValueError("voxel spacing must contain positive finite X,Y,Z values")
    return tuple(float(value) for value in voxel_spacing_xyz_um)


def run_pilot(
    source_npz: Path,
    output_dir: Path,
    candidates: tuple[CandidateSpec, ...],
    voxel_spacing_xyz_um: tuple[float, float, float],
) -> None:
    """Export inputs and compare independent 2-D Golgi segmentations."""
    source_npz = Path(source_npz).resolve()
    output_dir = Path(output_dir).resolve()
    spacing = _validated_spacing(voxel_spacing_xyz_um)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not source_npz.is_file():
        raise FileNotFoundError(source_npz)
    if not candidates:
        raise ValueError("at least one segmentation candidate is required")
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        inputs_dir = staging / "inputs"
        candidates_dir = staging / "candidates"
        reports_dir = staging / "reports"
        inputs_dir.mkdir()
        candidates_dir.mkdir()
        reports_dir.mkdir()

        with np.load(source_npz, allow_pickle=False) as source:
            required = {
                "consensus_nuclear_mask_zyx",
                "fixed_perinuclear_roi_zyx",
                "crop_start_zyx",
                "crop_stop_zyx",
                "full_shape_zyx",
                *(f"round_{number}_target_zyx" for number in range(1, 6)),
            }
            missing = sorted(required - set(source.files))
            if missing:
                raise ValueError(f"source NPZ is missing arrays: {', '.join(missing)}")

            nuclear = np.asarray(source["consensus_nuclear_mask_zyx"], dtype=bool)
            perinuclear = np.asarray(source["fixed_perinuclear_roi_zyx"], dtype=bool)
            if nuclear.ndim != 3 or perinuclear.shape != nuclear.shape:
                raise ValueError(
                    "nuclear and perinuclear masks must be matching 3-D arrays"
                )
            cropped_shape = tuple(int(value) for value in nuclear.shape)
            nuclear_mip = np.max(nuclear, axis=0)
            analysis_mip = np.max(nuclear | perinuclear, axis=0)
            del nuclear, perinuclear

            mips: list[NDArray[np.float32]] = []
            common_positive = np.ones(cropped_shape[1:], dtype=bool)
            for round_number in range(1, 6):
                volume = np.asarray(source[f"round_{round_number}_target_zyx"])
                if volume.shape != cropped_shape or not np.issubdtype(
                    volume.dtype, np.number
                ):
                    raise ValueError(
                        f"round {round_number} target must be numeric with shape {cropped_shape}"
                    )
                if not np.all(np.isfinite(volume)):
                    raise ValueError(
                        f"round {round_number} target contains non-finite values"
                    )
                _write_registered_volume(
                    inputs_dir / f"round-{round_number}-golgi-registered.ome.tif",
                    volume,
                    spacing,
                )
                mip = np.max(volume, axis=0).astype(np.float32, copy=False)
                mips.append(mip)
                common_positive &= mip > 0.0
            analysis_mip &= common_positive
            if not np.any(analysis_mip):
                raise ValueError(
                    "analysis support is empty after intersecting valid data"
                )
            _write_image(
                inputs_dir / "nuclear-mask-mip.tif", nuclear_mip.astype(np.uint8)
            )
            _write_image(
                inputs_dir / "analysis-support-mip.tif", analysis_mip.astype(np.uint8)
            )

            geometry = {
                "crop_start_zyx": [
                    int(value) for value in source["crop_start_zyx"].tolist()
                ],
                "crop_stop_zyx": [
                    int(value) for value in source["crop_stop_zyx"].tolist()
                ],
                "cropped_shape_zyx": list(cropped_shape),
                "full_shape_zyx": [
                    int(value) for value in source["full_shape_zyx"].tolist()
                ],
                "voxel_spacing_xyz_um": list(spacing),
            }

        metrics: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_dir = candidates_dir / candidate.name
            candidate_dir.mkdir()
            results: list[SegmentationResult] = []
            for round_number, mip in enumerate(mips, start=1):
                result = dog_hysteresis_mask(mip, analysis_mip, spacing[:2], candidate)
                results.append(result)
                _write_image(
                    candidate_dir / f"round-{round_number}-mip-mask.tif",
                    result.mask.astype(np.uint8),
                )
                _write_image(
                    candidate_dir / f"round-{round_number}-feature.tif",
                    result.feature,
                )
                row = {
                    "candidate": candidate.name,
                    "acquisition_round": round_number,
                    **asdict(candidate),
                    "center": result.center,
                    "noise_scale": result.noise_scale,
                    "low_threshold": result.low_threshold,
                    "high_threshold": result.high_threshold,
                    **mask_statistics(mip, result.mask, analysis_mip, nuclear_mip),
                }
                row.pop("name")
                metrics.append(row)
            _render_candidate_report(
                reports_dir / f"{candidate.name}-all-rounds.png",
                candidate,
                mips,
                results,
                nuclear_mip,
            )

        fieldnames = list(metrics[0])
        with (staging / "candidate-metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)

        manifest = {
            "schema_version": 1,
            "pilot_scope": "xy-maximum-projection",
            "source": {"path": str(source_npz), "sha256": _sha256(source_npz)},
            "geometry": geometry,
            "segmentation_input": "per-round XY maximum projection",
            "analysis_support": "union of nuclear and fixed perinuclear ROI, intersected with positive data in every round",
            "nuclear_mask_policy": "context-only-not-hard-exclusion",
            "ground_truth_available": False,
            "candidates": [asdict(candidate) for candidate in candidates],
            "outputs": {
                "registered_volumes": "inputs/round-N-golgi-registered.ome.tif",
                "candidate_masks": "candidates/NAME/round-N-mip-mask.tif",
                "candidate_features": "candidates/NAME/round-N-feature.tif",
                "reports": "reports/NAME-all-rounds.png",
                "metrics": "candidate-metrics.csv",
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--voxel-spacing-xyz-um",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.1083, 0.1083, 0.3),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed three-candidate Cell1 pilot from the command line."""
    args = _parse_args(argv)
    run_pilot(
        args.source_npz,
        args.output_dir,
        DEFAULT_CANDIDATES,
        tuple(args.voxel_spacing_xyz_um),
    )
    print(args.output_dir.resolve())
    return 0


__all__ = [
    "DEFAULT_CANDIDATES",
    "CandidateSpec",
    "SegmentationResult",
    "dog_hysteresis_mask",
    "mask_statistics",
    "run_pilot",
]


if __name__ == "__main__":
    raise SystemExit(main())
