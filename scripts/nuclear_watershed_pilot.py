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
"""Compare physical 3-D watershed splits of a saved DAPI consensus mask."""

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
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


@dataclass(frozen=True, slots=True)
class WatershedSpec:
    """One physical seed-prominence policy for splitting touching nuclei."""

    name: str
    seed_prominence_um: float

    def __post_init__(self) -> None:
        """Reject ambiguous names and non-physical prominence values."""
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("candidate name must be non-empty and contain no spaces")
        if not math.isfinite(self.seed_prominence_um) or self.seed_prominence_um <= 0.0:
            raise ValueError("seed_prominence_um must be finite and positive")


@dataclass(frozen=True, slots=True)
class WatershedResult:
    """A labeled foreground partition and its marker count."""

    labels: NDArray[np.uint16]
    seed_count: int


DEFAULT_SPECS = (
    WatershedSpec("prominence-2um", 2.0),
    WatershedSpec("prominence-6um", 6.0),
    WatershedSpec("prominence-10um", 10.0),
)


def split_touching_nuclei(
    foreground_mask: NDArray[np.generic],
    spacing_xyz_um: tuple[float, float, float],
    spec: WatershedSpec,
) -> WatershedResult:
    """Partition a 3-D DAPI foreground without changing its outer boundary."""
    mask = np.asarray(foreground_mask)
    if mask.ndim != 3 or mask.dtype != np.bool_ or not np.any(mask):
        raise ValueError("foreground_mask must be a non-empty 3-D Boolean array")
    if len(spacing_xyz_um) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in spacing_xyz_um
    ):
        raise ValueError("spacing_xyz_um must contain positive finite X,Y,Z values")
    distance = distance_transform_edt(mask, sampling=spacing_xyz_um[::-1]).astype(
        np.float32
    )
    maxima = h_maxima(distance, spec.seed_prominence_um) & mask
    markers, seed_count = label(maxima, structure=np.ones((3, 3, 3), dtype=bool))
    if seed_count < 1:
        raise ValueError("seed prominence produced no watershed markers")
    if seed_count > np.iinfo(np.uint16).max:
        raise ValueError("seed prominence produced too many watershed markers")
    partition = np.asarray(
        watershed(-distance, markers=markers, mask=mask), dtype=np.uint16
    )
    if not np.array_equal(partition > 0, mask):
        raise RuntimeError("watershed partition did not preserve the foreground union")
    partition.setflags(write=False)
    return WatershedResult(partition, int(seed_count))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_ome_volume(
    path: Path,
    volume: NDArray[np.generic],
    spacing_xyz_um: tuple[float, float, float],
) -> None:
    import tifffile

    pixel_x_um, pixel_y_um, pixel_z_um = spacing_xyz_um
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


def _component_metrics(
    candidate: WatershedSpec,
    labels: NDArray[np.uint16],
    spacing_xyz_um: tuple[float, float, float],
    crop_start_zyx: tuple[int, int, int],
    nominal_expansion_factor: float,
) -> list[dict[str, Any]]:
    pixel_x_um, pixel_y_um, pixel_z_um = spacing_xyz_um
    voxel_volume_um3 = pixel_x_um * pixel_y_um * pixel_z_um
    rows: list[dict[str, Any]] = []
    component_count = int(labels.max())
    for component_label in range(1, component_count + 1):
        coordinates = np.argwhere(labels == component_label)
        if coordinates.size == 0:
            raise RuntimeError("watershed labels must be consecutive and non-empty")
        voxel_count = int(coordinates.shape[0])
        expanded_volume_um3 = voxel_count * voxel_volume_um3
        nominal_volume_um3 = expanded_volume_um3 / nominal_expansion_factor**3
        centroid_zyx = coordinates.mean(axis=0) + np.asarray(crop_start_zyx)
        centroid_xyz_um = (
            float(centroid_zyx[2] * pixel_x_um),
            float(centroid_zyx[1] * pixel_y_um),
            float(centroid_zyx[0] * pixel_z_um),
        )
        rows.append(
            {
                "candidate": candidate.name,
                "seed_prominence_um": candidate.seed_prominence_um,
                "seed_count": component_count,
                "nucleus_label": component_label,
                "voxel_count": voxel_count,
                "expanded_volume_um3": expanded_volume_um3,
                "nominal_volume_um3": nominal_volume_um3,
                "nominal_equivalent_sphere_diameter_um": (
                    6.0 * nominal_volume_um3 / math.pi
                )
                ** (1.0 / 3.0),
                "centroid_xyz_expanded_um": repr(centroid_xyz_um),
            }
        )
    return rows


def dapi_review_projections(
    dapi_volume: NDArray[np.generic],
    *,
    xy_plane_index: int,
    plane_background_percentile: float,
    orthogonal_projection_percentile: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Build a plane-matched XY view and background-corrected orthogonal views."""
    volume = np.asarray(dapi_volume)
    if (
        volume.ndim != 3
        or not np.issubdtype(volume.dtype, np.number)
        or not np.all(np.isfinite(volume))
    ):
        raise ValueError("dapi_volume must be a finite numeric 3-D array")
    if not 0 <= xy_plane_index < volume.shape[0]:
        raise ValueError("xy_plane_index is outside the DAPI volume")
    if not 0.0 <= plane_background_percentile < 100.0:
        raise ValueError("plane_background_percentile must be in [0, 100)")
    if not 0.0 < orthogonal_projection_percentile <= 100.0:
        raise ValueError("orthogonal_projection_percentile must be in (0, 100]")
    source = volume.astype(np.float32, copy=False)
    plane_background = np.percentile(
        source, plane_background_percentile, axis=(1, 2)
    ).astype(np.float32)
    corrected = np.maximum(source - plane_background[:, None, None], 0.0)
    xy = source[xy_plane_index].astype(np.float32, copy=True)
    xz = np.percentile(corrected, orthogonal_projection_percentile, axis=1).astype(
        np.float32
    )
    yz = np.percentile(corrected, orthogonal_projection_percentile, axis=2).astype(
        np.float32
    )
    return xy, xz, yz


def label_review_projections(
    labels: NDArray[np.uint16],
    *,
    xy_plane_index: int,
) -> tuple[list[NDArray[np.bool_]], list[NDArray[np.bool_]], list[NDArray[np.bool_]]]:
    """Project labels while keeping XY on the exact reviewed DAPI plane."""
    if labels.ndim != 3 or not 0 <= xy_plane_index < labels.shape[0]:
        raise ValueError("labels must be 3-D and contain the requested XY plane")
    xy: list[NDArray[np.bool_]] = []
    xz: list[NDArray[np.bool_]] = []
    yz: list[NDArray[np.bool_]] = []
    for component_label in range(1, int(labels.max()) + 1):
        component = labels == component_label
        xy.append(component[xy_plane_index])
        xz.append(np.max(component, axis=1))
        yz.append(np.max(component, axis=2))
    return xy, xz, yz


def physical_image_geometry(
    image_shape_yx: tuple[int, int],
    *,
    horizontal_spacing_um: float,
    vertical_spacing_um: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[float, float, float, float]]:
    """Return shared raster edges and contour centers with row zero at the top."""
    height, width = image_shape_yx
    if height < 1 or width < 1:
        raise ValueError("image_shape_yx must be positive")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (horizontal_spacing_um, vertical_spacing_um)
    ):
        raise ValueError("image spacing must be finite and positive")
    x_centers = (np.arange(width, dtype=np.float64) + 0.5) * horizontal_spacing_um
    y_centers = (np.arange(height, dtype=np.float64) + 0.5) * vertical_spacing_um
    extent = (
        0.0,
        width * horizontal_spacing_um,
        height * vertical_spacing_um,
        0.0,
    )
    return x_centers, y_centers, extent


def _render_report(
    path: Path,
    dapi_crop: NDArray[np.generic],
    golgi_mask: NDArray[np.bool_],
    baseline_labels: NDArray[np.uint16],
    candidates: Sequence[
        tuple[
            WatershedSpec,
            int,
            tuple[
                list[NDArray[np.bool_]],
                list[NDArray[np.bool_]],
                list[NDArray[np.bool_]],
            ],
        ]
    ],
    spacing_xyz_um: tuple[float, float, float],
    *,
    xy_plane_index: int,
    plane_background_percentile: float,
    orthogonal_projection_percentile: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    pixel_x_um, pixel_y_um, pixel_z_um = spacing_xyz_um
    dapi_xy, dapi_xz, dapi_yz = dapi_review_projections(
        dapi_crop,
        xy_plane_index=xy_plane_index,
        plane_background_percentile=plane_background_percentile,
        orthogonal_projection_percentile=orthogonal_projection_percentile,
    )
    rows = [
        (
            "baseline: connected components",
            int(baseline_labels.max()),
            label_review_projections(baseline_labels, xy_plane_index=xy_plane_index),
        ),
        *(
            (
                f"{spec.name}: h={spec.seed_prominence_um:g} um",
                seed_count,
                projections,
            )
            for spec, seed_count, projections in candidates
        ),
    ]
    figure, axes = plt.subplots(
        len(rows),
        3,
        figsize=(14, 4.2 * len(rows)),
        constrained_layout=True,
        squeeze=False,
    )
    colors = plt.get_cmap("tab10").colors
    panels = (
        (
            dapi_xy,
            pixel_x_um,
            pixel_y_um,
            f"XY plane {xy_plane_index + 1}; green Golgi MIP context",
        ),
        (
            dapi_xz,
            pixel_x_um,
            pixel_z_um,
            f"XZ corrected p{orthogonal_projection_percentile:g} projection",
        ),
        (
            dapi_yz,
            pixel_y_um,
            pixel_z_um,
            f"YZ corrected p{orthogonal_projection_percentile:g} projection",
        ),
    )
    for row_index, (row_name, component_count, projections) in enumerate(rows):
        for column_index, (
            image,
            horizontal_spacing,
            vertical_spacing,
            title,
        ) in enumerate(panels):
            axis = axes[row_index, column_index]
            x_centers, y_centers, extent = physical_image_geometry(
                image.shape,
                horizontal_spacing_um=horizontal_spacing,
                vertical_spacing_um=vertical_spacing,
            )
            lower, upper = np.percentile(image[image > 0], (1.0, 99.8))
            axis.imshow(
                image,
                cmap="gray",
                vmin=lower,
                vmax=upper,
                extent=extent,
                interpolation="nearest",
            )
            for label_index, component_projection in enumerate(
                projections[column_index], start=1
            ):
                if np.any(component_projection):
                    axis.contour(
                        x_centers,
                        y_centers,
                        component_projection,
                        levels=(0.5,),
                        colors=(colors[(label_index - 1) % len(colors)],),
                        linewidths=1.0,
                    )
            if column_index == 0 and np.any(golgi_mask):
                axis.contour(
                    x_centers,
                    y_centers,
                    golgi_mask,
                    levels=(0.5,),
                    colors=("lime",),
                    linewidths=0.8,
                )
            axis.set_title(f"{row_name}; n={component_count}\n{title}")
            axis.set_xlabel("expanded micrometers")
            axis.set_ylabel("expanded micrometers")
            axis.set_aspect("equal")
            axis.set_xlim(extent[0], extent[1])
            axis.set_ylim(extent[2], extent[3])
    figure.suptitle(
        "Cell1 DAPI watershed pilot; colored nuclear labels; green round-5 conservative Golgi",
        fontsize=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _validated_spacing(
    spacing_xyz_um: tuple[float, float, float],
) -> tuple[float, float, float]:
    if len(spacing_xyz_um) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in spacing_xyz_um
    ):
        raise ValueError("spacing_xyz_um must contain positive finite X,Y,Z values")
    return tuple(float(value) for value in spacing_xyz_um)


def run_nuclear_pilot(
    source_consensus_npz: Path,
    display_reference_dapi: Path,
    round_5_conservative_golgi_mask: Path,
    output_dir: Path,
    specs: tuple[WatershedSpec, ...],
    spacing_xyz_um: tuple[float, float, float],
    *,
    nominal_expansion_factor: float,
    xy_plane_1based: int,
) -> None:
    """Run an atomic watershed comparison without altering production inputs."""
    import tifffile

    source_consensus_npz = Path(source_consensus_npz).resolve()
    display_reference_dapi = Path(display_reference_dapi).resolve()
    round_5_conservative_golgi_mask = Path(round_5_conservative_golgi_mask).resolve()
    output_dir = Path(output_dir).resolve()
    spacing = _validated_spacing(spacing_xyz_um)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    for path in (
        source_consensus_npz,
        display_reference_dapi,
        round_5_conservative_golgi_mask,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("watershed candidate names must be present and unique")
    if not math.isfinite(nominal_expansion_factor) or nominal_expansion_factor <= 0.0:
        raise ValueError("nominal_expansion_factor must be finite and positive")

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

        with np.load(source_consensus_npz, allow_pickle=False) as source:
            required = {
                "consensus_nuclear_mask_zyx",
                "crop_start_zyx",
                "crop_stop_zyx",
                "full_shape_zyx",
            }
            missing = sorted(required - set(source.files))
            if missing:
                raise ValueError(
                    f"source consensus NPZ is missing arrays: {', '.join(missing)}"
                )
            foreground = np.asarray(source["consensus_nuclear_mask_zyx"], dtype=bool)
            if foreground.ndim != 3 or not np.any(foreground):
                raise ValueError("saved consensus mask must be a non-empty 3-D array")
            crop_start = tuple(int(value) for value in source["crop_start_zyx"])
            crop_stop = tuple(int(value) for value in source["crop_stop_zyx"])
            full_shape = tuple(int(value) for value in source["full_shape_zyx"])
        expected_shape = tuple(
            stop - start for start, stop in zip(crop_start, crop_stop)
        )
        if foreground.shape != expected_shape:
            raise ValueError("saved consensus mask does not match its crop geometry")

        dapi_memmap = tifffile.memmap(display_reference_dapi)
        if dapi_memmap.shape != full_shape:
            raise ValueError("display DAPI does not match the saved full-grid geometry")
        crop_slices = tuple(
            slice(start, stop) for start, stop in zip(crop_start, crop_stop)
        )
        dapi_crop = np.asarray(dapi_memmap[crop_slices])
        if dapi_crop.shape != foreground.shape:
            raise RuntimeError("cropped display DAPI geometry is inconsistent")
        xy_plane_index = xy_plane_1based - 1
        if not 0 <= xy_plane_index < dapi_crop.shape[0]:
            raise ValueError("xy_plane_1based is outside the cropped DAPI volume")
        golgi_mask = np.asarray(
            tifffile.imread(round_5_conservative_golgi_mask), dtype=bool
        )
        if golgi_mask.shape != foreground.shape[1:]:
            raise ValueError("round-5 Golgi mask must match the consensus XY crop")

        _write_ome_volume(
            inputs_dir / "original-consensus-mask.ome.tif",
            foreground.astype(np.uint8),
            spacing,
        )
        _write_ome_volume(
            inputs_dir / "reference-dapi-crop.ome.tif", dapi_crop, spacing
        )
        baseline_labels, _baseline_count = label(
            foreground, structure=np.ones((3, 3, 3), dtype=bool)
        )
        baseline_labels = baseline_labels.astype(np.uint16, copy=False)

        metrics: list[dict[str, Any]] = []
        report_candidates: list[
            tuple[
                WatershedSpec,
                int,
                tuple[
                    list[NDArray[np.bool_]],
                    list[NDArray[np.bool_]],
                    list[NDArray[np.bool_]],
                ],
            ]
        ] = []
        for spec in specs:
            result = split_touching_nuclei(foreground, spacing, spec)
            candidate_dir = candidates_dir / spec.name
            candidate_dir.mkdir()
            _write_ome_volume(candidate_dir / "labels.ome.tif", result.labels, spacing)
            metrics.extend(
                _component_metrics(
                    spec,
                    result.labels,
                    spacing,
                    crop_start,
                    nominal_expansion_factor,
                )
            )
            report_candidates.append(
                (
                    spec,
                    result.seed_count,
                    label_review_projections(
                        result.labels, xy_plane_index=xy_plane_index
                    ),
                )
            )

        with (staging / "component-metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        _render_report(
            reports_dir / "nuclear-watershed-comparison.png",
            dapi_crop,
            golgi_mask,
            baseline_labels,
            report_candidates,
            spacing,
            xy_plane_index=xy_plane_index,
            plane_background_percentile=10.0,
            orthogonal_projection_percentile=95.0,
        )

        manifest = {
            "schema_version": 1,
            "pilot_scope": "physical-3d-distance-watershed",
            "foreground_policy": "preserve-saved-consensus-union-exactly",
            "source_consensus_npz": {
                "path": str(source_consensus_npz),
                "sha256": _sha256(source_consensus_npz),
            },
            "display_reference_dapi": {
                "path": str(display_reference_dapi),
                "sha256": _sha256(display_reference_dapi),
                "role": "display-only-acquisition-round-2",
            },
            "round_5_conservative_golgi_mask": {
                "path": str(round_5_conservative_golgi_mask),
                "sha256": _sha256(round_5_conservative_golgi_mask),
                "role": "display-only-accepted-pilot-context",
            },
            "geometry": {
                "crop_start_zyx": list(crop_start),
                "crop_stop_zyx": list(crop_stop),
                "cropped_shape_zyx": list(foreground.shape),
                "full_shape_zyx": list(full_shape),
                "voxel_spacing_xyz_um": list(spacing),
                "nominal_expansion_factor": nominal_expansion_factor,
            },
            "baseline_connected_component_count": int(baseline_labels.max()),
            "review_display": {
                "xy_plane_1based": xy_plane_1based,
                "xy_method": "raw-plane",
                "orthogonal_plane_background_percentile": 10.0,
                "orthogonal_projection_percentile": 95.0,
                "coordinate_policy": "shared-physical-pixel-centers-row-zero-at-top",
            },
            "candidates": [asdict(spec) for spec in specs],
            "ground_truth_available": False,
            "outputs": {
                "original_consensus_mask": "inputs/original-consensus-mask.ome.tif",
                "reference_dapi_crop": "inputs/reference-dapi-crop.ome.tif",
                "candidate_labels": "candidates/NAME/labels.ome.tif",
                "component_metrics": "component-metrics.csv",
                "review_report": "reports/nuclear-watershed-comparison.png",
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
    parser.add_argument("--source-consensus-npz", required=True, type=Path)
    parser.add_argument("--display-reference-dapi", required=True, type=Path)
    parser.add_argument("--round-5-conservative-golgi-mask", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--voxel-spacing-xyz-um",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.1083, 0.1083, 0.3),
    )
    parser.add_argument("--nominal-expansion-factor", type=float, default=4.5)
    parser.add_argument("--xy-plane-1based", type=int, default=36)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed Cell1 watershed comparison from the command line."""
    args = _parse_args(argv)
    run_nuclear_pilot(
        args.source_consensus_npz,
        args.display_reference_dapi,
        args.round_5_conservative_golgi_mask,
        args.output_dir,
        DEFAULT_SPECS,
        tuple(args.voxel_spacing_xyz_um),
        nominal_expansion_factor=args.nominal_expansion_factor,
        xy_plane_1based=args.xy_plane_1based,
    )
    print(args.output_dir.resolve())
    return 0


__all__ = [
    "DEFAULT_SPECS",
    "WatershedResult",
    "WatershedSpec",
    "dapi_review_projections",
    "label_review_projections",
    "physical_image_geometry",
    "run_nuclear_pilot",
    "split_touching_nuclei",
]


if __name__ == "__main__":
    raise SystemExit(main())
