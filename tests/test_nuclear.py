"""Synthetic tests for consensus DAPI surfaces and Lamin normal profiles."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from cyclic_stability import nuclear
from cyclic_stability.cyexm_artifacts import RoundArtifacts
from cyclic_stability.models import (
    AnalysisConfig,
    CellSpec,
    NuclearConfig,
    QcDecision,
)
from cyclic_stability.nuclear import (
    LaminProfileRow,
    _metric_rows,
    _ridge,
    _surface_from_mask,
    _surface_normals_and_curvature,
    analyze_lamin_cell,
    consensus_nuclear_surface,
    lamin_normal_profiles,
)
from synthetic import (
    NUCLEAR_CONFIG,
    SPACING,
    ellipsoid_dapi,
    separated_nuclei,
    synthetic_backgrounds,
    synthetic_nucleus,
    synthetic_shell,
)


@pytest.mark.parametrize("count", range(1, 7))
def test_consensus_accepts_one_through_six_nuclei(count: int) -> None:
    """Discarding all but the largest component loses valid nuclei in one field."""
    dapi = separated_nuclei(count, SPACING)
    result = consensus_nuclear_surface(
        {2: dapi}, SPACING, NUCLEAR_CONFIG, nominal_expansion_factor=4.5
    )
    assert [item.nucleus_id for item in result.nuclei] == [
        f"N{index:03d}" for index in range(1, count + 1)
    ]
    assert [item.nucleus_id for item in result.component_qc if item.accepted] == [
        f"N{index:03d}" for index in range(1, count + 1)
    ]
    assert set(np.unique(result.labels_zyx)) == set(range(count + 1))
    assert np.array_equal(result.mask_zyx, result.labels_zyx > 0)
    centroids = [tuple(item.centroid_xyz_um) for item in result.nuclei]
    assert centroids == sorted(centroids)


def test_consensus_rejects_zero_valid_nuclear_components() -> None:
    """A field containing only subminimum debris must fail closed as count zero."""
    dapi = ellipsoid_dapi(
        ((5.0, 5.0, 5.0),), ((2.0, 2.0, 2.0),), (0.5, 0.5, 0.5), (24, 24, 24)
    )
    with pytest.raises(ValueError, match="accepted nuclear component count 0"):
        consensus_nuclear_surface(
            {2: dapi}, (0.5, 0.5, 0.5), NUCLEAR_CONFIG, nominal_expansion_factor=1.0
        )


def test_consensus_rejects_seven_valid_nuclear_components() -> None:
    """A field with seven plausible nuclei must not silently select six."""
    dapi = separated_nuclei(7, SPACING)
    with pytest.raises(ValueError, match="accepted nuclear component count 7"):
        consensus_nuclear_surface(
            {2: dapi}, SPACING, NUCLEAR_CONFIG, nominal_expansion_factor=4.5
        )


@pytest.mark.parametrize(
    ("centers", "radii", "shape", "config", "reason"),
    (
        (
            ((6.0, 6.0, 6.0), (15.0, 6.0, 6.0)),
            ((4.0, 4.0, 4.0), (2.0, 2.0, 2.0)),
            (28, 48, 48),
            NUCLEAR_CONFIG,
            "volume-below-minimum",
        ),
        (
            ((7.0, 7.0, 7.0), (24.0, 7.0, 7.0)),
            ((4.0, 4.0, 4.0), (6.0, 6.0, 6.0)),
            (32, 48, 64),
            replace(NUCLEAR_CONFIG, nominal_volume_um3=(100.0, 500.0)),
            "volume-above-maximum",
        ),
        (
            ((7.0, 7.0, 25.0), (22.0, 7.0, 7.0)),
            ((1.5, 1.5, 20.0), (4.0, 4.0, 4.0)),
            (100, 48, 56),
            replace(
                NUCLEAR_CONFIG,
                nominal_volume_um3=(100.0, 500.0),
                minimum_sphericity=0.75,
            ),
            "sphericity-below-minimum",
        ),
        (
            ((1.0, 7.0, 7.0), (20.0, 15.0, 10.0)),
            ((4.0, 4.0, 4.0), (4.0, 4.0, 4.0)),
            (50, 80, 80),
            replace(
                NUCLEAR_CONFIG,
                nominal_volume_um3=(10.0, 1000.0),
                minimum_sphericity=0.10,
                maximum_face_contact_fraction=0.02,
            ),
            "face-contact-above-maximum",
        ),
    ),
)
def test_component_qc_records_exact_rejection_reason(
    centers: tuple[tuple[float, float, float], ...],
    radii: tuple[tuple[float, float, float], ...],
    shape: tuple[int, int, int],
    config: NuclearConfig,
    reason: str,
) -> None:
    """Rejected DAPI candidates remain auditable beside a valid nucleus."""
    spacing = (0.5, 0.5, 0.5)
    dapi = ellipsoid_dapi(centers, radii, spacing, shape)
    result = consensus_nuclear_surface(
        {2: dapi}, spacing, config, nominal_expansion_factor=1.0
    )
    rejected = [item for item in result.component_qc if not item.accepted]
    assert len(result.nuclei) == 1
    assert len(rejected) == 1
    assert rejected[0].rejection_reasons == (reason,)


def test_component_qc_uses_exact_expansion_volume_conversion() -> None:
    """Using a linear or area expansion correction misclassifies nuclear volume."""
    spacing = (0.5, 0.5, 0.5)
    dapi = ellipsoid_dapi(((6.0, 6.0, 6.0),), ((4.0, 4.0, 4.0),), spacing, (28, 28, 28))
    config = replace(NUCLEAR_CONFIG, nominal_volume_um3=(1.0, 10.0))
    result = consensus_nuclear_surface(
        {2: dapi}, spacing, config, nominal_expansion_factor=4.5
    )
    qc = result.component_qc[0]
    assert qc.nominal_volume_um3 == pytest.approx(qc.expanded_volume_um3 / 4.5**3)


def test_multinuclear_consensus_arrays_are_read_only() -> None:
    """Mutating accepted labels or one surface would invalidate component QC."""
    result = consensus_nuclear_surface(
        {2: separated_nuclei(1, SPACING)},
        SPACING,
        NUCLEAR_CONFIG,
        nominal_expansion_factor=4.5,
    )
    surface = result.nuclei[0]
    assert result.mask_zyx.flags.writeable is False
    assert result.labels_zyx.dtype == np.uint16
    assert result.labels_zyx.flags.writeable is False
    assert all(
        array.flags.writeable is False
        for array in (
            surface.mask_zyx,
            surface.vertices_xyz_um,
            surface.faces,
            surface.normals_xyz,
            surface.centroid_xyz_um,
            surface.mean_curvature_um_inverse,
        )
    )


def test_lamin_profiles_recover_known_outward_displacement() -> None:
    """Flipping the outward normal would turn a known positive shell shift negative."""
    dapi, reference_lamin = synthetic_nucleus(radius_um=5.0, shell_width_um=0.4)
    moved_lamin = synthetic_shell(radius_um=5.6, shell_width_um=0.4)
    surface = consensus_nuclear_surface(
        {2: dapi, 3: dapi}, SPACING, NUCLEAR_CONFIG, nominal_expansion_factor=1.0
    ).nuclei[0]
    rows = lamin_normal_profiles(
        surface,
        {2: reference_lamin, 3: moved_lamin},
        NUCLEAR_CONFIG,
        synthetic_backgrounds((2, 3)),
    )
    round_three = [row for row in rows if row.acquisition_round == 3 and row.supported]
    assert np.median([row.displacement_um for row in round_three]) == pytest.approx(
        0.6, abs=0.12
    )
    assert np.median([row.fwhm_um for row in round_three]) == pytest.approx(
        0.4, abs=0.12
    )


def test_missing_lamin_peak_is_missing_not_zero_displacement() -> None:
    """Replacing missing ridges with zero would invent stability evidence."""
    dapi, reference_lamin = synthetic_nucleus(radius_um=5.0, shell_width_um=0.4)
    surface = consensus_nuclear_surface(
        {2: dapi, 4: dapi}, SPACING, NUCLEAR_CONFIG, nominal_expansion_factor=1.0
    ).nuclei[0]
    rows = lamin_normal_profiles(
        surface,
        {2: reference_lamin, 4: np.zeros_like(reference_lamin)},
        NUCLEAR_CONFIG,
        synthetic_backgrounds((2, 4)),
    )
    round_four = [row for row in rows if row.acquisition_round == 4]
    assert all(row.supported is False for row in round_four)
    assert all(math.isnan(row.displacement_um) for row in round_four)
    assert round_four[0].surface_coverage == 0.0


def test_acquisition_round_two_is_reference_not_mapping_order() -> None:
    """Using insertion order as reference changes the scientific comparison silently."""
    dapi, reference_lamin = synthetic_nucleus(radius_um=5.0, shell_width_um=0.4)
    moved_lamin = synthetic_shell(radius_um=5.4, shell_width_um=0.4)
    surface = consensus_nuclear_surface(
        {3: dapi, 2: dapi}, SPACING, NUCLEAR_CONFIG, nominal_expansion_factor=1.0
    ).nuclei[0]
    rows = lamin_normal_profiles(
        surface,
        {3: moved_lamin, 2: reference_lamin},
        NUCLEAR_CONFIG,
        synthetic_backgrounds((2, 3)),
    )
    reference = [row for row in rows if row.acquisition_round == 2 and row.supported]
    moved = [row for row in rows if row.acquisition_round == 3 and row.supported]
    assert np.median([row.displacement_um for row in reference]) == pytest.approx(0.0)
    assert np.median([row.displacement_um for row in moved]) == pytest.approx(
        0.4, abs=0.12
    )


def test_normals_are_physical_xyz_and_point_away_from_centroid() -> None:
    """Treating anisotropic ZYX indices as XYZ produces inward or distorted normals."""
    dapi, _ = synthetic_nucleus(radius_um=5.0, shell_width_um=0.4)
    surface = consensus_nuclear_surface(
        {2: dapi}, SPACING, NUCLEAR_CONFIG, nominal_expansion_factor=1.0
    ).nuclei[0]
    outward = surface.vertices_xyz_um - surface.centroid_xyz_um
    assert np.all(np.sum(outward * surface.normals_xyz, axis=1) > 0.0)
    assert np.allclose(np.linalg.norm(surface.normals_xyz, axis=1), 1.0)


def test_outward_signed_distance_handles_asymmetric_concavity_consistently() -> None:
    """Radially flipping each vertex reverses the normal or curvature of a cavity."""
    z, y, x = np.indices((41, 41, 41), dtype=np.float64)
    xyz = np.stack((x - 20.0, y - 20.0, z - 20.0), axis=-1)
    outer = np.linalg.norm(xyz, axis=-1) <= 14.0
    cavity_center = np.array((5.0, 0.0, 0.0))
    cavity = np.linalg.norm(xyz - cavity_center, axis=-1) < 4.5
    mask = outer & ~cavity
    vertices, _ = _surface_from_mask(mask, (1.0, 1.0, 1.0))
    normals, centroid, curvature = _surface_normals_and_curvature(
        mask, vertices, (1.0, 1.0, 1.0)
    )
    inner = np.linalg.norm(vertices - (cavity_center + 20.0), axis=1) < 5.0
    outer_vertices = np.linalg.norm(vertices - centroid, axis=1) > 11.0
    assert np.all(
        np.sum(normals[inner] * (vertices[inner] - (cavity_center + 20.0)), axis=1)
        < 0.0
    )
    assert np.median(curvature[inner]) < 0.0
    assert np.median(curvature[outer_vertices]) > 0.0


def test_consensus_keeps_surface_for_permitted_small_face_contact() -> None:
    """Padding after accepting <=10% contact must not erase the allowed boundary."""
    z, y, x = np.indices((48, 128, 128), dtype=np.float64)
    radius = np.sqrt(
        ((x * SPACING[0] - 4.8) ** 2)
        + ((y * SPACING[1] - 6.35) ** 2)
        + ((z * SPACING[2] - 7.05) ** 2)
    )
    dapi = 1.0 + 0.05 * np.sin(x * SPACING[0]) * np.cos(y * SPACING[1])
    dapi += 40.0 * (radius <= 5.0)
    surface = consensus_nuclear_surface(
        {2: dapi}, SPACING, NUCLEAR_CONFIG, nominal_expansion_factor=1.0
    ).nuclei[0]
    assert surface.vertices_xyz_um[:, 0].min() < 0.05


def test_consensus_rejects_face_touching_nucleus() -> None:
    """A truncated nucleus must not be summarized as a complete envelope."""
    dapi, _ = synthetic_nucleus(radius_um=5.0, shell_width_um=0.4)
    _, y, x = np.indices(dapi.shape, dtype=np.float64)
    shifted = 1.0 + 0.05 * np.sin(x * SPACING[0]) * np.cos(y * SPACING[1])
    shifted[:, :, :78] = dapi[:, :, 50:]
    with pytest.raises(ValueError, match="face"):
        consensus_nuclear_surface(
            {2: shifted}, SPACING, NUCLEAR_CONFIG, nominal_expansion_factor=1.0
        )


def test_profiles_reject_missing_acquisition_round_two() -> None:
    """A displacement has no scientific origin when round 2 is unavailable."""
    dapi, lamin = synthetic_nucleus(radius_um=5.0, shell_width_um=0.4)
    surface = consensus_nuclear_surface(
        {2: dapi}, SPACING, NUCLEAR_CONFIG, nominal_expansion_factor=1.0
    ).nuclei[0]
    with pytest.raises(ValueError, match="round 2"):
        lamin_normal_profiles(
            surface,
            {3: lamin},
            NUCLEAR_CONFIG,
            synthetic_backgrounds((3,)),
        )


def test_continuity_is_edge_connected_and_coverage_is_not_imputed() -> None:
    """Disconnected supported regions cannot be reported as a continuous envelope."""
    dapi, lamin = synthetic_nucleus(radius_um=5.0, shell_width_um=0.4)
    surface = consensus_nuclear_surface(
        {2: dapi}, SPACING, NUCLEAR_CONFIG, nominal_expansion_factor=1.0
    ).nuclei[0]
    half_shell = lamin.copy()
    half_shell[:, :, :64] = 1.0
    rows = lamin_normal_profiles(
        surface,
        {2: lamin, 3: half_shell},
        NUCLEAR_CONFIG,
        synthetic_backgrounds((2, 3)),
    )
    round_three = [row for row in rows if row.acquisition_round == 3]
    assert 0.35 < round_three[0].surface_coverage < 0.65
    assert round_three[0].continuity < round_three[0].surface_coverage
    assert round_three[0].continuity_change_from_reference < 0.0


def test_current_support_retains_width_and_intensity_when_reference_is_missing(
    analysis_config: AnalysisConfig,
) -> None:
    """Reference-only pairing must not suppress measured current-round ridge shape."""
    cell = replace(analysis_config.cells[0], cohort="lamin", cell_id="SyntheticCell")
    rows = _metric_rows(
        cell,
        (
            LaminProfileRow(
                2,
                0,
                False,
                math.nan,
                math.nan,
                math.nan,
                math.nan,
                math.nan,
                0.0,
                0.0,
                0.0,
            ),
            LaminProfileRow(
                3, 0, True, 0.2, math.nan, math.nan, 0.4, 9.0, 1.0, 1.0, 1.0
            ),
        ),
        analysis_config,
    )
    current = {row.endpoint: row for row in rows if row.acquisition_round == 3}
    assert current["lamin_fwhm_median"].value == pytest.approx(0.4)
    assert current["lamin_integrated_ridge_intensity_median"].value == pytest.approx(
        9.0
    )
    assert "lamin_signed_displacement_median" not in current


def test_metric_rows_mark_round_one_as_exposure_sensitivity(
    analysis_config: AnalysisConfig,
) -> None:
    """Calling round 1 matched settings would overstate comparability without normalization."""
    cell = replace(analysis_config.cells[0], cohort="lamin", cell_id="SyntheticCell")
    rows = _metric_rows(
        cell,
        (
            LaminProfileRow(
                1, 0, True, 0.0, math.nan, math.nan, 0.4, 9.0, 1.0, 1.0, 0.0
            ),
            LaminProfileRow(2, 0, True, 0.0, 0.0, 0.0, 0.4, 9.0, 1.0, 1.0, 0.0),
        ),
        analysis_config,
    )
    assert {row.comparison_set for row in rows if row.acquisition_round == 1} == {
        "round-1-exposure-sensitivity"
    }
    assert {row.comparison_set for row in rows if row.acquisition_round == 2} == {
        "matched-settings"
    }


def test_ridge_requires_strict_sbr_and_rejects_search_endpoints() -> None:
    """An equality threshold or endpoint peak is not a localized supported ridge."""
    offsets = np.array((-0.1, 0.0, 0.1))
    background = synthetic_backgrounds((2,))[2]
    assert (
        _ridge(np.array((1.0, 1.15, 1.0)), offsets, background, NUCLEAR_CONFIG)[0]
        is False
    )
    assert (
        _ridge(np.array((1.2, 1.1, 1.0)), offsets, background, NUCLEAR_CONFIG)[0]
        is False
    )


def test_lamin_output_tree_replaces_stale_tree_and_rolls_back_on_failure(
    tmp_path: Path,
) -> None:
    """A diagnostic error must neither mix generations nor destroy a prior tree."""
    destination = tmp_path / "lamin"
    destination.mkdir()
    (destination / "stale.txt").write_text("old", encoding="utf-8")

    def fail(staged: Path) -> None:
        (staged / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected diagnostic failure")

    with pytest.raises(RuntimeError, match="injected"):
        nuclear._atomic_publish_directory(destination, fail)
    assert (destination / "stale.txt").read_text(encoding="utf-8") == "old"
    assert not (destination / "partial.txt").exists()

    def succeed(staged: Path) -> None:
        (staged / "complete.txt").write_text("new", encoding="utf-8")

    nuclear._atomic_publish_directory(destination, succeed)
    assert (destination / "complete.txt").read_text(encoding="utf-8") == "new"
    assert not (destination / "stale.txt").exists()


def test_cell_analysis_retains_every_frozen_ensemble_member(
    analysis_config: AnalysisConfig, tmp_path: Path
) -> None:
    """Selecting the most favorable Lamin mask would erase the declared sensitivity set."""
    dapi, reference_lamin = synthetic_nucleus(radius_um=5.0, shell_width_um=0.4)
    moved_lamin = synthetic_shell(radius_um=5.3, shell_width_um=0.4)
    source_rounds = analysis_config.cells[0].rounds
    cell = CellSpec(
        cohort="lamin",
        cell_id="SyntheticCell",
        rounds=tuple(
            replace(source_rounds[index], cohort="lamin", cell_id="SyntheticCell")
            for index in (0, 1, 2)
        ),
    )
    revised_config = replace(
        analysis_config,
        cells=(cell,),
        qc_decisions=(
            QcDecision(
                cohort="lamin",
                cell_id="SyntheticCell",
                acquisition_round=1,
                decision="accept",
                reasons=("synthetic-target-blind-acceptance",),
                reviewer="Synthetic reviewer",
                reviewed_at="2026-08-16T00:00:00Z",
            ),
            QcDecision(
                cohort="lamin",
                cell_id="SyntheticCell",
                acquisition_round=3,
                decision="accept",
                reasons=("synthetic-target-blind-acceptance",),
                reviewer="Synthetic reviewer",
                reviewed_at="2026-08-16T00:00:00Z",
            ),
        ),
    )
    artifacts = tuple(
        _write_round_artifacts(tmp_path, round_number, dapi, lamin)
        for round_number, lamin in (
            (1, reference_lamin),
            (2, reference_lamin),
            (3, moved_lamin),
        )
    )
    result = analyze_lamin_cell(cell, artifacts, revised_config, tmp_path / "out")
    combinations = {
        row.ensemble_name
        for row in result.ensemble_rows
        if row.endpoint == "lamin_jaccard"
    }
    assert combinations == {
        f"sigma={sigma:g};sbr={sbr:g}"
        for sigma in analysis_config.ensembles.gaussian_sigma_um
        for sbr in analysis_config.ensembles.threshold_sbr
    }
    for acquisition_round in (1, 2, 3):
        round_rows = [
            row
            for row in result.ensemble_rows
            if row.acquisition_round == acquisition_round
            and row.endpoint == "lamin_jaccard"
        ]
        assert len(round_rows) == 9
        assert {row.ensemble_name for row in round_rows} == combinations
    assert {
        "lamin_mask_volume",
        "lamin_mask_area_to_volume",
        "lamin_mask_sphericity",
        "lamin_fixed_to_moving_boundary_distance_p95",
        "lamin_moving_to_fixed_boundary_distance_p95",
        "lamin_surface_dice",
        "lamin_jaccard",
    } <= {row.endpoint for row in result.ensemble_rows}
    assert {row.acquisition_round for row in result.summary_rows} == {1, 2, 3}
    assert all(path.exists() for path in result.diagnostics)


def _write_round_artifacts(
    root: Path,
    acquisition_round: int,
    dapi: np.ndarray,
    lamin: np.ndarray,
) -> RoundArtifacts:
    """Write the two target-specific analysis inputs behind a validated-artifact shape."""
    round_root = root / f"round-{acquisition_round}"
    round_root.mkdir()
    dapi_path = round_root / "dapi.tif"
    lamin_path = round_root / "lamin.tif"
    tifffile.imwrite(dapi_path, dapi)
    tifffile.imwrite(lamin_path, lamin)
    return RoundArtifacts(
        acquisition_round=acquisition_round,
        workflow_round=acquisition_round,
        workflow_output_root=round_root,
        registered_dapi=dapi_path,
        registered_tubulin=round_root / "tubulin.tif",
        registered_target=lamin_path,
        affine=None,
        warp=None,
        pre_metrics=None,
        linear_metrics=None,
        nonlinear_metrics=None,
        chunk_manifest=None,
        completion_records=(),
    )
