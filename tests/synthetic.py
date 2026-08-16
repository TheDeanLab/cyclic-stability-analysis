"""Deterministic configuration fixtures for cyclic-stability tests."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from cyclic_stability.images import BackgroundSummary
from cyclic_stability.lysosome import DetectionSet, LysosomeObject
from cyclic_stability.models import NuclearConfig, RoundSpec
from cyclic_stability.units import theoretical_resolution
from cyexm.registration.artifacts import ArtifactStore, StageRequest
from cyexm.spatial import ImageGeometry
from numpy.typing import NDArray

CORRESPONDENCE_COLUMNS = (
    "cohort",
    "cell_id",
    "acquisition_round",
    "workflow_round",
    "nd2_relative",
    "tiff_directory_relative",
    "z_start_offset",
    "target_channel",
    "target_exposure_ms",
    "target_laser_percent",
    "tubulin_exposure_ms",
    "tubulin_laser_percent",
    "dapi_exposure_ms",
    "dapi_laser_percent",
    "dataset_manifest",
    "execution_profile",
    "workflow_dataset_id",
    "identity_method",
)


ANALYSIS_YAML = """\
schema_version: 1
reference_acquisition_round: 2
voxel_spacing_xyz_um: [0.108333, 0.108333, 0.300000]
nominal_expansion_factor: 4.5
optics:
  numerical_aperture: 1.4
  immersion_refractive_index: 1.515
  wavelength_um:
    dapi: 0.405
    tubulin: 0.488
    target: 0.640
ensembles:
  gaussian_sigma_um: [0.279, 0.558, 1.116]
  threshold_sbr: [3.0, 4.0, 5.0]
  surface_tolerance_resolution_units: [0.5, 1.0, 1.5]
nuclear:
  consensus_smoothing_um: 0.558
  normal_half_window_um: 1.5
  normal_step_um: 0.05
  ridge_min_sbr: 3.0
golgi:
  perinuclear_outer_distance_um: 18.0
lysosome:
  log_sigma_um: [0.35, 0.70, 1.40]
  detection_sbr: [3.0, 4.0, 5.0]
  match_resolution_units: 1.0
paths:
  nd2_root: /archive/bioinformatics/Danuser_lab/Dean/Seweryn/resubmission/U2OS_repeat_60x
  resubmission_root: /archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission
  correspondence_markdown: /archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/nd2-tiff-file-correspondence.md
"""


def write_synthetic_correspondence(path: Path) -> None:
    """Write five complete synthetic Golgi cells in production column order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORRESPONDENCE_COLUMNS)
        writer.writeheader()
        for cell_index in range(1, 6):
            for acquisition_round in range(1, 6):
                writer.writerow(
                    {
                        "cohort": "golgi",
                        "cell_id": f"Cell{cell_index}",
                        "acquisition_round": acquisition_round,
                        "workflow_round": 2
                        if acquisition_round == 1
                        else 1
                        if acquisition_round == 2
                        else acquisition_round,
                        "nd2_relative": f"round{acquisition_round}/DAPI_tubulin_golgi_{cell_index:03d}.nd2",
                        "tiff_directory_relative": f"Golgi/Cell{cell_index}",
                        "z_start_offset": 1,
                        "target_channel": "647_golgi",
                        "target_exposure_ms": 150 if acquisition_round == 1 else 200,
                        "target_laser_percent": 50,
                        "tubulin_exposure_ms": 200,
                        "tubulin_laser_percent": 50,
                        "dapi_exposure_ms": 200,
                        "dapi_laser_percent": 75,
                        "dataset_manifest": f"golgi-cell{cell_index}.yaml",
                        "execution_profile": "execution-local.yaml",
                        "workflow_dataset_id": f"golgi-cell{cell_index}",
                        "identity_method": "exact-pixel-audit",
                    }
                )


def write_completion_fixture(path: Path, payload: bytes) -> Path:
    """Write one synthetic Cy-ExM artifact with its canonical completion record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    request = StageRequest.for_files(
        stage="synthetic",
        inputs=(),
        geometry=ImageGeometry(
            shape=(1, 1, 1),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        method="synthetic",
        accuracy="synthetic",
        native_threads={},
        upstream=None,
        parameters={},
    )
    ArtifactStore().commit(
        path,
        request,
        lambda target: target.write_bytes(payload),
        lambda target: target.read_bytes() == payload,
    )
    return path


def synthetic_folded_warp(shape: tuple[int, int, int]) -> NDArray[np.float64]:
    """Return a ZYX/XYZ displacement with a folded central X slab."""
    warp = np.zeros((*shape, 3), dtype=np.float64)
    x0 = shape[2] // 3
    x1 = shape[2] - x0
    warp[:, :, x0:x1, 0] = -2.0 * np.arange(x1 - x0, dtype=np.float64)
    return warp


def single_voxel(
    shape_zyx: tuple[int, int, int], index_zyx: tuple[int, int, int]
) -> NDArray[np.bool_]:
    """Return a Boolean ZYX volume with exactly one foreground voxel."""
    result = np.zeros(shape_zyx, dtype=bool)
    result[index_zyx] = True
    return result


def synthetic_target() -> NDArray[np.float64]:
    """Return a deterministic target image with a bright central structure."""
    image = np.arange(1.0, 730.0).reshape(9, 9, 9)
    image[3:6, 3:6, 3:6] += 5_000.0
    return image


def synthetic_roi() -> NDArray[np.bool_]:
    """Return a target-independent central structure ROI in ZYX order."""
    roi = np.zeros((9, 9, 9), dtype=bool)
    roi[2:7, 2:7, 2:7] = True
    return roi


def synthetic_lamp_spots() -> NDArray[np.float64]:
    """Return three separated physical LAMP1-like spots on an anisotropic grid."""
    z, y, x = np.indices((31, 81, 81), dtype=np.float64)
    image = 1.0 + 0.04 * np.sin(x * SPACING[0]) * np.cos(y * SPACING[1])
    for center_xyz in ((2.0, 2.0, 2.4), (5.6, 2.0, 4.2), (2.0, 5.6, 5.4)):
        cx, cy, cz = center_xyz
        radius_squared = (
            ((x * SPACING[0] - cx) / 0.24) ** 2
            + ((y * SPACING[1] - cy) / 0.24) ** 2
            + ((z * SPACING[2] - cz) / 0.24) ** 2
        )
        image += 400.0 * np.exp(-0.5 * radius_squared)
    return image


def _detection(
    coordinates_xyz_um: tuple[tuple[float, float, float], ...],
) -> DetectionSet:
    """Construct hand-checked detection values used to exercise real matching."""
    return DetectionSet(
        sigma_um=0.70,
        detection_sbr=4.0,
        objects=tuple(
            LysosomeObject(
                centroid_xyz_um=coordinate,
                volume_um3=0.12,
                equivalent_diameter_um=(6.0 * 0.12 / np.pi) ** (1.0 / 3.0),
                sphericity=0.9,
                nearest_neighbor_distance_um=1.0,
                nucleus_relative_distance_um=2.0,
            )
            for coordinate in coordinates_xyz_um
        ),
    )


def reference_detections() -> DetectionSet:
    """Return three reference objects with distinct lateral and axial positions."""
    return _detection(((1.0, 1.0, 1.0), (3.0, 1.0, 1.0), (1.0, 3.0, 2.0)))


def moved_lost_new_detections() -> DetectionSet:
    """Return two retained objects, one lost object, and one unmatched new object."""
    return _detection(((1.1, 1.0, 1.0), (3.0, 1.1, 1.1), (8.0, 8.0, 5.0)))


def synthetic_round_specs() -> tuple[RoundSpec, ...]:
    """Return five immutable rounds with the round-one exposure sensitivity case."""
    return tuple(
        RoundSpec(
            cohort="golgi",
            cell_id="Cell1",
            acquisition_round=acquisition_round,
            workflow_round=2 if acquisition_round == 1 else acquisition_round,
            nd2_path=Path(f"round{acquisition_round}.nd2"),
            tiff_directory=Path("tiffs"),
            z_start_offset=0,
            target_exposure_ms=100.0 if acquisition_round == 1 else 200.0,
            target_laser_percent=50.0,
            tubulin_exposure_ms=200.0,
            dapi_exposure_ms=200.0,
            dataset_manifest=Path("dataset.yaml"),
            execution_profile=Path("execution.yaml"),
            workflow_dataset_id=f"golgi-cell1-round{acquisition_round}",
            identity_method="exact-pixel-audit",
        )
        for acquisition_round in range(1, 6)
    )


def synthetic_nucleus(
    radius_um: float, shell_width_um: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a centered DAPI ball and co-located smooth Lamin shell in ZYX."""
    if radius_um <= 0.0 or shell_width_um <= 0.0:
        raise ValueError("synthetic nucleus dimensions must be positive")
    z, y, x = np.indices((48, 128, 128), dtype=np.float64)
    physical_z = (z - 23.5) * SPACING[2]
    physical_y = (y - 63.5) * SPACING[1]
    physical_x = (x - 63.5) * SPACING[0]
    radius = np.sqrt(physical_x**2 + physical_y**2 + physical_z**2)
    background = 1.0 + 0.05 * np.sin(physical_x) * np.cos(physical_y)
    dapi = background + 40.0 * (radius <= radius_um)
    lamin = background + 20.0 * np.exp(
        -0.5 * ((radius - radius_um) / (shell_width_um / 2.355)) ** 2
    )
    return dapi, lamin


def synthetic_shell(radius_um: float, shell_width_um: float) -> NDArray[np.float64]:
    """Return a smooth centered Lamin shell in the shared synthetic geometry."""
    _, lamin = synthetic_nucleus(radius_um, shell_width_um)
    return lamin


def synthetic_golgi(components: int = 1) -> NDArray[np.float64]:
    """Return deterministic, separated perinuclear Golgi objects in physical ZYX."""
    if components < 1 or components > 4:
        raise ValueError("components must be between one and four")
    z, y, x = np.indices((48, 128, 128), dtype=np.float64)
    physical_z = (z - 23.5) * SPACING[2]
    physical_y = (y - 63.5) * SPACING[1]
    physical_x = (x - 63.5) * SPACING[0]
    image = 1.0 + 0.05 * np.sin(physical_x) * np.cos(physical_y)
    centers = ((2.4, 0.0, 0.0), (-2.4, 0.0, 0.0), (0.0, 2.4, 0.0), (0.0, -2.4, 0.0))
    for center in centers[:components]:
        radius = np.sqrt(
            (physical_x - center[0]) ** 2
            + (physical_y - center[1]) ** 2
            + (physical_z - center[2]) ** 2
        )
        image += 80.0 * np.exp(-0.5 * (radius / 0.32) ** 2)
    return image


def synthetic_perinuclear_roi() -> NDArray[np.bool_]:
    """Return a fixed target-independent exterior ring in the Golgi geometry."""
    z, y, x = np.indices((48, 128, 128), dtype=np.float64)
    radius = np.sqrt(
        ((x - 63.5) * SPACING[0]) ** 2
        + ((y - 63.5) * SPACING[1]) ** 2
        + ((z - 23.5) * SPACING[2]) ** 2
    )
    return (radius > 1.2) & (radius <= 4.5)


def synthetic_dapi_rounds() -> dict[int, NDArray[np.float64]]:
    """Return deterministic registered DAPI rounds sharing the Golgi geometry."""
    dapi, _ = synthetic_nucleus(radius_um=1.2, shell_width_um=0.4)
    return {2: dapi, 3: dapi.copy()}


def synthetic_backgrounds(rounds: tuple[int, ...]) -> dict[int, BackgroundSummary]:
    """Return the exact robust-background fixture for the synthetic volumes."""
    return {
        acquisition_round: BackgroundSummary(
            median=1.0,
            scale=0.05,
            positive_count=48 * 128 * 128,
            background_count=48 * 128 * 128,
        )
        for acquisition_round in rounds
    }


SPACING = (0.1, 0.1, 0.3)
RESOLUTION = theoretical_resolution(0.640, 1.4, 1.515, 4.5)
NUCLEAR_CONFIG = NuclearConfig(
    consensus_smoothing_um=0.558,
    normal_half_window_um=1.5,
    normal_step_um=0.05,
    ridge_min_sbr=3.0,
)
