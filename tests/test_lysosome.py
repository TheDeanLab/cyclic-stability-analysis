"""Tests for LAMP1 eligibility, physical spot detection, and exact object matching."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from cyclic_stability.config import load_analysis_config
from cyclic_stability.cyexm_artifacts import RoundArtifacts
from cyclic_stability.lysosome import (
    DetectionSet,
    LysosomeObject,
    _comparison_specs,
    _fixed_lamp_roi,
    _maxima,
    _physical_log_response,
    _rows_for_detections,
    _with_nuclear_distances,
    analyze_lamp1_cell,
    detect_lysosomes,
    eligible_lamp_cells,
    match_lysosomes,
)
from cyclic_stability.models import CellSpec, QcDecision, RoundSpec
from cyclic_stability.morphology import gaussian_sigma_zyx
from cyclic_stability.nuclear import consensus_nuclear_surface
from cyclic_stability.units import ResolutionReference
from synthetic import (
    RESOLUTION,
    SPACING,
    ellipsoid_dapi,
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
        image,
        roi,
        SPACING,
        (0.35, 0.70, 1.40),
        (3.0, 4.0, 5.0),
        resolution=RESOLUTION,
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
    detected = detect_lysosomes(
        image, roi, SPACING, (0.35, 0.70), (3.0,), resolution=RESOLUTION
    )
    assert [item.object_count for item in detected] == [3, 3]
    assert all(
        len({tuple(point) for point in item.coordinates_xyz_um}) == 3
        for item in detected
    )


def test_watershed_objects_have_physical_geometry() -> None:
    """Replacing physical voxel volume or mesh area changes object geometry."""
    image = synthetic_lamp_spots()
    result = detect_lysosomes(
        image,
        np.ones(image.shape, dtype=bool),
        SPACING,
        (0.70,),
        (3.0,),
        resolution=RESOLUTION,
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
    "spacing", ((0.1, 0.1, 0.3), (0.12, 0.12, 0.24), (0.08, 0.1, 0.36))
)
def test_physical_log_response_is_sampling_invariant(
    spacing: tuple[float, float, float],
) -> None:
    """Voxel-coordinate LoG normalization must not change a physical peak response."""
    sx, sy, sz = spacing
    z, y, x = np.indices((41, 81, 81), dtype=float)
    radius_squared = (
        ((x - 40.0) * sx / 0.70) ** 2
        + ((y - 40.0) * sy / 0.70) ** 2
        + ((z - 20.0) * sz / 0.70) ** 2
    )
    response = _physical_log_response(
        100.0 * np.exp(-0.5 * radius_squared), 0.70, spacing
    )
    assert response[20, 40, 40] == pytest.approx(53.0, rel=0.12)


def test_duplicate_suppression_uses_lateral_and_axial_resolution_units() -> None:
    """Replacing axial resolution with a scalar sigma misclassifies clustered Z peaks."""
    resolution = ResolutionReference(0.3, 0.9, 0.3, 0.9, "synthetic")
    response = np.zeros((13, 13, 13), dtype=float)
    response[6, 6, 6] = 10.0
    response[6, 6, 9] = 9.0  # 1.0 lateral resolution unit: suppress.
    response[9, 6, 6] = 8.0  # 1.0 axial resolution unit: suppress.
    roi = np.ones(response.shape, dtype=bool)
    assert _maxima(response, roi, 1.0, SPACING, resolution) == ((6, 6, 6),)
    response[9, 6, 6] = 0.0
    response[6, 6, 9] = 0.0
    response[6, 6, 10] = 8.0  # 4 lateral voxels: outside the 0.3 um limit.
    assert _maxima(response, roi, 1.0, SPACING, resolution) == (
        (6, 6, 6),
        (6, 6, 10),
    )


def test_fixed_lamp_roi_intersects_producer_support_outside_one_consensus_nucleus() -> (
    None
):
    """Round-specific padding must not create an intensity-dependent ROI edge."""
    nucleus = np.zeros((7, 7, 7), dtype=bool)
    nucleus[3, 3, 3] = True
    padded = np.ones_like(nucleus)
    padded[:, :, 0] = False
    roi = _fixed_lamp_roi(nucleus, {1: np.ones_like(nucleus), 3: padded})
    assert not roi[:, :, 0].any()
    assert not roi[3, 3, 3]
    with pytest.raises(ValueError, match="empty"):
        _fixed_lamp_roi(nucleus, {2: nucleus})


def test_lamp_union_roi_and_surface_distance_consume_every_accepted_nucleus(
    analysis_config,
) -> None:
    """Using one selected nucleus would admit or mis-distance the second nucleus."""
    spacing = (0.5, 0.5, 0.5)
    dapi = ellipsoid_dapi(
        ((6.0, 6.0, 6.0), (20.0, 6.0, 6.0)),
        ((4.0, 4.0, 4.0), (4.0, 4.0, 4.0)),
        spacing,
        (28, 28, 56),
    )
    consensus = consensus_nuclear_surface(
        {2: dapi},
        spacing,
        analysis_config.nuclear,
        nominal_expansion_factor=1.0,
    )
    roi = _fixed_lamp_roi(
        consensus.mask_zyx, {2: np.ones(consensus.mask_zyx.shape, dtype=bool)}
    )
    assert not np.any(roi & consensus.nuclei[0].mask_zyx)
    assert not np.any(roi & consensus.nuclei[1].mask_zyx)

    centroid = (24.5, 6.0, 6.0)
    annotated = _with_nuclear_distances(_detections(centroid), consensus.nuclei)
    expected = min(
        float(
            np.linalg.norm(surface.vertices_xyz_um - np.asarray(centroid), axis=1).min()
        )
        for surface in consensus.nuclei
    )
    assert annotated.objects[0].nucleus_relative_distance_um == pytest.approx(expected)
    assert expected < float(
        np.linalg.norm(
            consensus.nuclei[0].vertices_xyz_um - np.asarray(centroid), axis=1
        ).min()
    )


def test_lamp_rows_serialize_explicit_aggregate_nucleus_id(analysis_config) -> None:
    """Field-level LAMP1 rows must not infer ownership by a particular nucleus."""
    cell = analysis_config.cell("golgi", "Cell1")
    lamp_cell = replace(cell, cohort="lamp1")
    rows = _rows_for_detections(lamp_cell, 2, reference_detections(), analysis_config)
    assert rows
    assert all(asdict(row)["nucleus_id"] == "aggregate" for row in rows)


def test_detector_is_invariant_to_arbitrary_values_outside_common_support() -> None:
    """Transform-invalid padding must not influence background, LoG, or watershed objects."""
    image = synthetic_lamp_spots()
    support = np.zeros(image.shape, dtype=bool)
    support[4:-4, 12:-12, 12:-12] = True
    baseline = detect_lysosomes(
        image, support, SPACING, (0.7,), (3.0,), resolution=RESOLUTION
    )
    padded = image.copy()
    padded[~support] = 1e12
    changed = detect_lysosomes(
        padded, support, SPACING, (0.7,), (3.0,), resolution=RESOLUTION
    )
    assert changed == baseline


def test_detector_rejects_support_too_thin_for_physical_derivatives() -> None:
    """A thin support must fail rather than produce favorable empty detections."""
    image = synthetic_lamp_spots()
    support = np.zeros(image.shape, dtype=bool)
    support[14:16, 20:60, 20:60] = True
    with pytest.raises(ValueError, match="support-boundary"):
        detect_lysosomes(image, support, SPACING, (0.7,), (3.0,), resolution=RESOLUTION)


def test_adjacency_skips_qc_gaps_but_round_two_relative_does_not() -> None:
    """A 2-to-4 transition is not adjacent after an excluded acquisition round."""
    comparisons = _comparison_specs((1, 2, 4, 5))
    assert ("adjacent", 2, 4) not in comparisons
    assert ("adjacent", 4, 5) in comparisons
    assert ("round2-relative", 2, 4) in comparisons


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
        detect_lysosomes(image, roi, SPACING, (0.7,), (3.0,), resolution=RESOLUTION)


def _lamp_cell_and_artifacts(
    tmp_path: Path,
) -> tuple[CellSpec, tuple[RoundArtifacts, ...]]:
    """Build five actual fixed-grid image artifacts for a single reviewed LAMP1 cell."""
    rounds: list[RoundSpec] = []
    artifacts: list[RoundArtifacts] = []
    for number in range(1, 6):
        dapi_path = tmp_path / f"dapi-{number}.tif"
        lamp_path = tmp_path / f"lamp-{number}.tif"
        lamp = synthetic_lamp_spots()
        z, y, x = np.indices(lamp.shape, dtype=float)
        radius = np.sqrt(
            (x * SPACING[0] - 4.0) ** 2
            + (y * SPACING[1] - 4.0) ** 2
            + (z * SPACING[2] - 4.5) ** 2
        )
        dapi = 1.0 + 0.03 * np.sin(x * SPACING[0]) + 40.0 * (radius <= 1.1)
        tifffile.imwrite(dapi_path, dapi)
        tifffile.imwrite(lamp_path, lamp)
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
    analysis_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping a five-round comparison would hide an acquisition-specific instability."""
    cell, artifacts = _lamp_cell_and_artifacts(tmp_path)
    decisions = tuple(
        QcDecision(
            "lamp1",
            cell.cell_id,
            number,
            "accept",
            ("target-blind",),
            "reviewer",
            "2026-08-16T00:00:00Z",
        )
        for number in (1, 3, 4, 5)
    )
    config = replace(
        analysis_config,
        cells=(cell,),
        qc_decisions=decisions,
        nominal_expansion_factor=1.0,
        nuclear=replace(analysis_config.nuclear, nominal_volume_um3=(1.0, 5000.0)),
    )
    from cyclic_stability import lysosome

    shape = synthetic_lamp_spots().shape
    monkeypatch.setattr(
        lysosome,
        "_artifact_validity_masks",
        lambda _artifacts, _config: (
            {number: np.ones(shape, dtype=bool) for number in (1, 2, 3, 4, 5)},
            "synthetic-digest",
            ("synthetic-validity",),
        ),
    )
    result = analyze_lamp1_cell(cell, artifacts, config, tmp_path / "output")
    assert len(result.ensemble_rows) == 5 * 3 * 3 * 6
    assert {row.comparison_set for row in result.summary_rows} == {
        "round2-relative",
        "adjacent",
    }
    assert all(
        asdict(row)["nucleus_id"] == "aggregate"
        for row in (*result.summary_rows, *result.ensemble_rows)
    )
    assert all(
        row.sigma_um in config.lysosome.log_sigma_um
        and row.threshold_sbr in config.lysosome.detection_sbr
        and row.surface_tolerance_resolution_units is None
        for row in result.ensemble_rows
    )
    assert (tmp_path / "output" / "nuclear-component-inventory.pdf").is_file()
    assert (tmp_path / "output" / "lamp1-detections.npz").is_file()
    assert (tmp_path / "output" / "lamp1-ensemble.csv").is_file()
    assert (tmp_path / "output" / "objects.csv").is_file()
    assert (tmp_path / "output" / "matches.csv").is_file()
    with (tmp_path / "output" / "lamp1-ensemble.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        ensemble_csv = list(csv.DictReader(handle))
    assert ensemble_csv
    assert all(row["surface_tolerance_resolution_units"] == "" for row in ensemble_csv)
    with (tmp_path / "output" / "objects.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        objects = list(csv.DictReader(handle))
    with (tmp_path / "output" / "matches.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        matches = list(csv.DictReader(handle))
    assert objects and set(objects[0]) >= {
        "object_index",
        "x_um",
        "volume_um3",
        "nucleus_relative_distance_um",
    }
    assert matches and set(matches[0]) >= {
        "comparison_type",
        "reference_object_index",
        "moving_object_index",
        "normalized_displacement",
    }
    assert {row["nucleus_id"] for row in objects} == {"aggregate"}
    assert {row["nucleus_id"] for row in matches} == {"aggregate"}
    stale = tmp_path / "output" / "stale.txt"
    stale.write_text("obsolete", encoding="utf-8")
    analyze_lamp1_cell(cell, artifacts, config, tmp_path / "output")
    assert not stale.exists()


def test_missing_lamp_qc_rejects_before_any_target_volume_is_opened(
    analysis_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening a LAMP target before target-blind QC would leak target information."""
    from cyclic_stability import lysosome

    cell, artifacts = _lamp_cell_and_artifacts(tmp_path)
    config = replace(analysis_config, cells=(cell,))
    monkeypatch.setattr(
        lysosome,
        "load_volume",
        lambda _path: (_ for _ in ()).throw(AssertionError("target volume opened")),
    )
    with pytest.raises(ValueError, match="target-blind QC"):
        analyze_lamp1_cell(cell, artifacts, config, tmp_path / "output")


def test_excluded_lamp_round_never_opens_its_target(
    analysis_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An excluded moving round must be absent from both target reads and comparisons."""
    from cyclic_stability import lysosome

    cell, artifacts = _lamp_cell_and_artifacts(tmp_path)
    decisions = tuple(
        QcDecision(
            "lamp1",
            cell.cell_id,
            number,
            "exclude" if number == 3 else "accept",
            ("target-blind",),
            "reviewer",
            "2026-08-16T00:00:00Z",
        )
        for number in (1, 3, 4, 5)
    )
    config = replace(
        analysis_config,
        cells=(cell,),
        qc_decisions=decisions,
        nominal_expansion_factor=1.0,
        nuclear=replace(analysis_config.nuclear, nominal_volume_um3=(1.0, 5000.0)),
    )
    shape = synthetic_lamp_spots().shape
    monkeypatch.setattr(
        lysosome,
        "_artifact_validity_masks",
        lambda _artifacts, _config: (
            {number: np.ones(shape, dtype=bool) for number in (1, 2, 4, 5)},
            "synthetic-digest",
            ("synthetic-validity",),
        ),
    )
    original = lysosome.load_volume
    opened: list[Path] = []
    monkeypatch.setattr(
        lysosome,
        "load_volume",
        lambda path: (opened.append(path), original(path))[1],
    )
    result = analyze_lamp1_cell(cell, artifacts, config, tmp_path / "output")
    assert artifacts[2].registered_target not in opened
    assert str(artifacts[2].registered_target) not in result.input_identities
    assert str(artifacts[2].registered_dapi) not in result.input_identities
    assert "synthetic-validity" in result.input_identities


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
