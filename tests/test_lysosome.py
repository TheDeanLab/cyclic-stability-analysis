"""Tests for LAMP1 eligibility, physical spot detection, and exact object matching."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from cyclic_stability.config import load_analysis_config
from cyclic_stability.cyexm_artifacts import RoundArtifacts
from cyclic_stability.lysosome import (
    DetectionSet,
    LysosomeObject,
    detect_lysosomes,
    eligible_lamp_cells,
    match_lysosomes,
)
from cyclic_stability.models import CellSpec, RoundSpec
from cyclic_stability.morphology import gaussian_sigma_zyx
from synthetic import (
    RESOLUTION,
    SPACING,
    moved_lost_new_detections,
    reference_detections,
    synthetic_lamp_spots,
)


def _detections(*coordinates: tuple[float, float, float]) -> DetectionSet:
    """Create exact, independent object coordinates for matching boundary tests."""
    return DetectionSet(
        0.7,
        4.0,
        tuple(
            LysosomeObject(point, 0.1, 0.5, 0.9, math.nan, math.nan)
            for point in coordinates
        ),
    )


def test_unverified_lamp_inventory_is_not_schedulable(
    config_files: tuple[Path, ...],
) -> None:
    """Adding no reviewed mappings must schedule no LAMP1 work."""
    assert eligible_lamp_cells(load_analysis_config(*config_files)) == ()


def test_log_detector_recovers_anisotropic_physical_spots() -> None:
    """Using voxel-isotropic LoG scales would miss the three physical spots."""
    image = synthetic_lamp_spots()
    roi = np.ones(image.shape, dtype=bool)
    detected = detect_lysosomes(
        image, roi, SPACING, (0.35, 0.70, 1.40), (3.0, 4.0, 5.0)
    )
    assert len(detected) == 9
    assert [item.object_count for item in detected] == [3] * 9


def test_scale_normalization_uses_physical_zyx_sigma() -> None:
    """A lateral scale must map to fewer Z voxels on the anisotropic acquisition."""
    assert gaussian_sigma_zyx(0.7, SPACING) == pytest.approx((0.7 / 0.3, 7.0, 7.0))


def test_duplicate_maxima_across_scales_are_suppressed() -> None:
    """One spot responding at several LoG scales must remain one detected object."""
    image = synthetic_lamp_spots()
    roi = np.ones(image.shape, dtype=bool)
    detected = detect_lysosomes(image, roi, SPACING, (0.35, 0.70), (3.0,))
    assert [item.object_count for item in detected] == [3, 3]
    assert all(
        len({tuple(point) for point in item.coordinates_xyz_um}) == 3
        for item in detected
    )


def test_watershed_objects_have_physical_geometry() -> None:
    """Replacing physical voxel volume or mesh area changes object geometry."""
    image = synthetic_lamp_spots()
    result = detect_lysosomes(
        image, np.ones(image.shape, dtype=bool), SPACING, (0.70,), (3.0,)
    )[0]
    assert all(object_.volume_um3 > 0.0 for object_ in result.objects)
    assert all(object_.equivalent_diameter_um > 0.0 for object_ in result.objects)
    assert all(0.0 < object_.sphericity <= 1.1 for object_ in result.objects)


def test_matching_allows_lost_and_new_objects_without_forcing_pairs() -> None:
    """A dense assignment that forces a distant pair hides a lost/new event."""
    result = match_lysosomes(
        reference_detections(), moved_lost_new_detections(), RESOLUTION, 1.0
    )
    assert result.matched_count == 2
    assert result.lost_count == 1
    assert result.new_count == 1
    assert result.matched_fraction == pytest.approx(2.0 / 3.0)
    assert result.lost_fraction == pytest.approx(1.0 / 3.0)
    assert result.new_fraction == pytest.approx(1.0 / 3.0)


def test_empty_sides_use_explicit_zero_denominators() -> None:
    """Empty detections must not create an optimistic matching fraction or NaN count."""
    empty = _detections()
    result = match_lysosomes(empty, _detections((1.0, 1.0, 1.0)), RESOLUTION, 1.0)
    assert (result.matched_count, result.lost_count, result.new_count) == (0, 0, 1)
    assert result.matched_fraction == 0.0
    assert result.lost_fraction == 0.0
    assert result.new_fraction == 1.0


def test_match_threshold_boundary_is_inclusive_and_max_units_is_respected() -> None:
    """A finite cost exactly at max_units is eligible, while any farther pair is not."""
    lateral = RESOLUTION.lateral_expanded_um
    reference = _detections((0.0, 0.0, 0.0))
    boundary = _detections((lateral, 0.0, 0.0))
    beyond = _detections((np.nextafter(lateral, math.inf), 0.0, 0.0))
    assert match_lysosomes(reference, boundary, RESOLUTION, 1.0).matched_count == 1
    assert match_lysosomes(reference, beyond, RESOLUTION, 1.0).matched_count == 0
    assert match_lysosomes(reference, boundary, RESOLUTION, 0.5).matched_count == 0


def test_matching_uses_axial_resolution_separately_from_lateral() -> None:
    """Treating Z like XY would accept an axial displacement beyond its limit."""
    reference = _detections((0.0, 0.0, 0.0))
    axial = _detections((0.0, 0.0, RESOLUTION.axial_expanded_um * 1.01))
    assert match_lysosomes(reference, axial, RESOLUTION, 1.0).matched_count == 0


@pytest.mark.parametrize(
    ("image", "roi", "message"),
    (
        (np.full((3, 3, 3), np.nan), np.ones((3, 3, 3), dtype=bool), "finite"),
        (np.ones((3, 3, 3)), np.ones((3, 3, 2), dtype=bool), "share"),
        (np.ones((3, 3, 3)), np.full((3, 3, 3), 2), "binary"),
        (np.ones((3, 3, 3)), np.ones((3, 3, 3), dtype=bool), "robust scale"),
    ),
)
def test_detector_hard_gates_invalid_signal_roi_and_background(
    image: np.ndarray, roi: np.ndarray, message: str
) -> None:
    """Invalid finite/binary geometry or a zero robust background cannot invent spots."""
    with pytest.raises(ValueError, match=message):
        detect_lysosomes(image, roi, SPACING, (0.7,), (3.0,))


def _lamp_cell_and_artifacts(
    tmp_path: Path,
) -> tuple[CellSpec, tuple[RoundArtifacts, ...]]:
    """Build five actual fixed-grid image artifacts for a single reviewed LAMP1 cell."""
    rounds: list[RoundSpec] = []
    artifacts: list[RoundArtifacts] = []
    for number in range(1, 6):
        dapi_path = tmp_path / f"dapi-{number}.tif"
        lamp_path = tmp_path / f"lamp-{number}.tif"
        image = synthetic_lamp_spots()
        tifffile.imwrite(dapi_path, image)
        tifffile.imwrite(lamp_path, image)
        rounds.append(
            RoundSpec(
                "lamp1",
                "verified-alpha",
                number,
                number,
                tmp_path / f"round-{number}.nd2",
                tmp_path,
                0,
                200.0,
                50.0,
                200.0,
                200.0,
                tmp_path / "dataset.yaml",
                tmp_path / "execution.yaml",
                f"lamp-{number}",
                "image-verified",
            )
        )
        artifacts.append(
            RoundArtifacts(
                number,
                number,
                tmp_path,
                dapi_path,
                dapi_path,
                lamp_path,
                None,
                None,
                None,
                None,
                None,
                None,
                (),
            )
        )
    return CellSpec("lamp1", "verified-alpha", tuple(rounds)), tuple(artifacts)


def test_lamp_analysis_emits_round_two_and_adjacent_comparisons_for_every_ensemble(
    analysis_config, tmp_path: Path
) -> None:
    """Dropping a five-round comparison would hide an acquisition-specific instability."""
    from cyclic_stability.lysosome import analyze_lamp1_cell

    cell, artifacts = _lamp_cell_and_artifacts(tmp_path)
    config = replace(analysis_config, cells=(cell,))
    result = analyze_lamp1_cell(cell, artifacts, config, tmp_path / "output")
    assert len(result.ensemble_rows) == 5 * 3 * 3 * 6
    assert {row.comparison_set for row in result.summary_rows} == {
        "acquisition-round-2-reference",
        "adjacent-acquisition-rounds",
    }
    assert (tmp_path / "output" / "lamp1-detections.npz").is_file()
    assert (tmp_path / "output" / "lamp1-ensemble.csv").is_file()


def test_real_lamp_gate_returns_machine_readable_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No image-verified mappings is an unavailable gate, not a failed biological result."""
    import sys

    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    from validate_inputs import main

    run_dir = tmp_path / "lamp-gate"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_inputs.py",
            "--analysis-config",
            str(Path(__file__).parents[1] / "config" / "analysis.yaml"),
            "--correspondence",
            str(Path(__file__).parents[1] / "config" / "correspondence.csv"),
            "--qc-decisions",
            str(Path(__file__).parents[1] / "config" / "qc-decisions.csv"),
            "--cohort",
            "lamp1",
            "--run-dir",
            str(run_dir),
        ],
    )
    assert main() == 0
    payload = json.loads((run_dir / "validation" / "inputs.json").read_text())
    assert payload["status"] == "unavailable"
    assert payload["reason"] == "no image-verified LAMP1 cell mappings"
    assert payload["scheduled_cell_count"] == 0
