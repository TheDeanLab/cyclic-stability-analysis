# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cy-exm[workflow] @ git+https://github.com/TheDeanLab/Cy-ExM.git@f6e0de36cd7f6b5e85f093884c2cf224225f4ec2",
#   "matplotlib>=3.8,<4",
#   "nd2==0.11.3",
#   "PyYAML>=6,<7",
#   "psutil>=5.9,<8",
#   "scikit-image>=0.22,<1",
#   "scipy>=1.12,<2",
#   "tifffile>=2025.1.10",
# ]
# ///
"""Render rapid MIP-first LAMP1 candidate previews without running full analysis."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    gaussian_filter,
    label,
)
from skimage.filters import threshold_otsu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cyclic_stability.images import load_volume
from cyclic_stability.lysosome import _mip_dog_candidates, _mip_unit_interval


def parse_args() -> argparse.Namespace:
    """Parse one or more registered LAMP1 volumes and MIP preview settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lamp1", type=Path, nargs="+", required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--output-dir", type=Path)
    parser.add_argument("--pixel-size-um", type=float, default=0.1083)
    parser.add_argument("--z-step-um", type=float, default=0.3)
    parser.add_argument("--detection-sbr", type=float, default=4.0)
    parser.add_argument("--min-diameter-px", type=float, default=5.0)
    parser.add_argument("--max-diameter-px", type=float, default=100.0)
    return parser.parse_args()


def _merge_candidate_footprints(mip: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Return filled MIP foreground components supported by one or more DoG seeds.

    This visual-only alternative deliberately retains the scale-space candidates as
    evidence of a real LAMP1 feature, but defines its displayed footprint from the
    connected background-corrected MIP signal.  Several local maxima on the same
    hollow object consequently produce one filled outline.
    """
    if mip.ndim != 2 or candidates.shape != mip.shape or candidates.dtype != bool:
        raise ValueError("MIP and candidate mask must be same-shape 2-D boolean data")
    if not np.any(candidates):
        return np.zeros_like(candidates)
    background = gaussian_filter(
        mip.astype(np.float64, copy=False), 70.0, mode="nearest"
    )
    foreground = np.maximum(mip - background, 0.0)
    positive = foreground[foreground > 0.0]
    if positive.size < 2 or float(np.max(positive)) <= float(np.min(positive)):
        return np.zeros_like(candidates)
    threshold = float(threshold_otsu(positive))
    signal = binary_closing(
        foreground > threshold, structure=np.ones((3, 3), dtype=bool)
    )
    labels, count = label(signal, structure=np.ones((3, 3), dtype=bool))
    candidate_support = binary_dilation(
        candidates, structure=np.ones((5, 5), dtype=bool)
    )
    merged = np.zeros_like(candidates)
    for index in range(1, count + 1):
        component = labels == index
        if np.any(component & candidate_support):
            merged |= binary_fill_holes(component)
    return merged


def _preview(
    lamp1: Path,
    *,
    pixel_size_um: float,
    z_step_um: float,
    detection_sbr: float,
    min_diameter_px: float,
    max_diameter_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return MIP, current candidates, merged footprints, and candidate count."""
    volume = load_volume(lamp1)
    candidates = _mip_dog_candidates(
        volume,
        np.ones(volume.shape, dtype=bool),
        detection_sbr=detection_sbr,
        spacing_xyz_um=(pixel_size_um, pixel_size_um, z_step_um),
        min_lateral_diameter_px=min_diameter_px,
        max_lateral_diameter_px=max_diameter_px,
    )
    admitted = np.zeros(volume.shape[1:], dtype=bool)
    for (y_bounds, x_bounds), footprint in candidates:
        admitted[y_bounds, x_bounds] |= footprint
    mip = _mip_unit_interval(volume)
    return mip, admitted, _merge_candidate_footprints(mip, admitted), len(candidates)


def _render(
    axis: object,
    mip: np.ndarray,
    admitted: np.ndarray,
    title: str,
) -> None:
    """Draw one grayscale MIP with thin colorblind candidate outlines."""
    axis.imshow(mip, cmap="gray", origin="upper")
    if np.any(admitted):
        axis.contour(
            admitted,
            levels=(0.5,),
            colors="#0072B2",
            linewidths=0.55,
        )
    axis.set_axis_off()
    axis.set_title(title, fontsize=9)


def _render_merged(
    axis: object,
    mip: np.ndarray,
    merged: np.ndarray,
    title: str,
) -> None:
    """Draw one grayscale MIP with magenta merged-object outlines."""
    axis.imshow(mip, cmap="gray", origin="upper")
    if np.any(merged):
        axis.contour(merged, levels=(0.5,), colors="#CC79A7", linewidths=0.75)
    axis.set_axis_off()
    axis.set_title(title, fontsize=9)


def _output_name(lamp1: Path) -> str:
    """Return a dataset-qualified preview filename for a channel TIFF."""
    try:
        dataset_id = lamp1.parents[3].name
    except IndexError as error:
        raise ValueError(
            f"LAMP1 path is too shallow for a dataset-qualified name: {lamp1}"
        ) from error
    if not dataset_id:
        raise ValueError(f"LAMP1 path has no dataset directory: {lamp1}")
    return f"{dataset_id}-{lamp1.stem}-mip-detection.png"


def main() -> int:
    """Write a rapid preview, avoiding QC, nuclei, and local 3-D measurement."""
    args = parse_args()
    if args.output is not None and len(args.lamp1) != 1:
        raise ValueError(
            "--output accepts exactly one --lamp1 volume; use --output-dir"
        )
    for value, name in (
        (args.pixel_size_um, "--pixel-size-um"),
        (args.z_step_um, "--z-step-um"),
        (args.detection_sbr, "--detection-sbr"),
        (args.min_diameter_px, "--min-diameter-px"),
        (args.max_diameter_px, "--max-diameter-px"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if args.max_diameter_px <= args.min_diameter_px:
        raise ValueError("--max-diameter-px must be larger than --min-diameter-px")

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    previews: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray, int]] = []
    for lamp1 in args.lamp1:
        mip, admitted, merged, count = _preview(
            lamp1,
            pixel_size_um=args.pixel_size_um,
            z_step_um=args.z_step_um,
            detection_sbr=args.detection_sbr,
            min_diameter_px=args.min_diameter_px,
            max_diameter_px=args.max_diameter_px,
        )
        previews.append((lamp1, mip, admitted, merged, count))

    output_dir = args.output_dir if args.output_dir is not None else args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (lamp1, mip, admitted, merged, count) in enumerate(previews):
        output = (
            args.output if args.output is not None else output_dir / _output_name(lamp1)
        )
        figure, axes = plt.subplots(1, 2, figsize=(13, 6.5), layout="constrained")
        _render(
            axes[0],
            mip,
            admitted,
            f"Current DoG candidate footprints\n{count} candidates",
        )
        _render_merged(
            axes[1],
            mip,
            merged,
            "Merged connected-signal footprints\n"
            "one outline per seeded foreground component",
        )
        figure.suptitle(
            f"{lamp1.stem}: MIP segmentation preview\n"
            f"SBR={args.detection_sbr:g}; diameter={args.min_diameter_px:g}–{args.max_diameter_px:g} px; "
            "preview only (no QC or 3-D measurement)",
            fontsize=12,
        )
        figure.savefig(output, dpi=220, facecolor="white")
        plt.close(figure)
        merged_count = label(merged, structure=np.ones((3, 3), dtype=bool))[1]
        print(
            f"wrote {count} DoG candidates and {merged_count} merged footprints to {output}"
        )

    if len(previews) > 1:
        figure, axes = plt.subplots(
            2, len(previews), figsize=(4.5 * len(previews), 9.5), layout="constrained"
        )
        for column, (lamp1, mip, admitted, merged, count) in enumerate(previews):
            _render(axes[0, column], mip, admitted, f"{lamp1.stem}\n{count} candidates")
            _render_merged(axes[1, column], mip, merged, "merged footprints")
        figure.suptitle(
            "LAMP1 MIP-first candidate and merged-footprint overview: "
            f"{args.min_diameter_px:g}–{args.max_diameter_px:g} px lateral diameter\n"
            f"MIP DoG centers; SBR={args.detection_sbr:g}; preview only",
            fontsize=12,
        )
        overview = output_dir / "lamp1-mip-detection-overview.png"
        figure.savefig(overview, dpi=220, facecolor="white")
        plt.close(figure)
        print(f"wrote multi-round overview to {overview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
