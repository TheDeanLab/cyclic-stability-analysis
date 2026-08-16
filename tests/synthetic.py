"""Deterministic configuration fixtures for cyclic-stability tests."""

from __future__ import annotations

import csv
from pathlib import Path

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
