"""Synthetic tests for consensus DAPI surfaces and Lamin normal profiles."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from cyclic_stability.cyexm_artifacts import RoundArtifacts
from cyclic_stability.models import AnalysisConfig, CellSpec, QcDecision
from cyclic_stability.nuclear import (
    analyze_lamin_cell,
    consensus_nuclear_surface,
    lamin_normal_profiles,
)
from synthetic import (
    NUCLEAR_CONFIG,
    SPACING,
    synthetic_backgrounds,
    synthetic_nucleus,
    synthetic_shell,
)


def test_lamin_profiles_recover_known_outward_displacement() -> None:
    """Flipping the outward normal would turn a known positive shell shift negative."""
    dapi, reference_lamin = synthetic_nucleus(radius_um=5.0, shell_width_um=0.4)
    moved_lamin = synthetic_shell(radius_um=5.6, shell_width_um=0.4)
    surface = consensus_nuclear_surface({2: dapi, 3: dapi}, SPACING, NUCLEAR_CONFIG)
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
    surface = consensus_nuclear_surface({2: dapi, 4: dapi}, SPACING, NUCLEAR_CONFIG)
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
    surface = consensus_nuclear_surface({3: dapi, 2: dapi}, SPACING, NUCLEAR_CONFIG)
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
    surface = consensus_nuclear_surface({2: dapi}, SPACING, NUCLEAR_CONFIG)
    outward = surface.vertices_xyz_um - surface.centroid_xyz_um
    assert np.all(np.sum(outward * surface.normals_xyz, axis=1) > 0.0)
    assert np.allclose(np.linalg.norm(surface.normals_xyz, axis=1), 1.0)


def test_consensus_rejects_face_touching_nucleus() -> None:
    """A truncated nucleus must not be summarized as a complete envelope."""
    dapi, _ = synthetic_nucleus(radius_um=5.0, shell_width_um=0.4)
    _, y, x = np.indices(dapi.shape, dtype=np.float64)
    shifted = 1.0 + 0.05 * np.sin(x * SPACING[0]) * np.cos(y * SPACING[1])
    shifted[:, :, :78] = dapi[:, :, 50:]
    with pytest.raises(ValueError, match="face"):
        consensus_nuclear_surface({2: shifted}, SPACING, NUCLEAR_CONFIG)


def test_profiles_reject_missing_acquisition_round_two() -> None:
    """A displacement has no scientific origin when round 2 is unavailable."""
    dapi, lamin = synthetic_nucleus(radius_um=5.0, shell_width_um=0.4)
    surface = consensus_nuclear_surface({2: dapi}, SPACING, NUCLEAR_CONFIG)
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
    surface = consensus_nuclear_surface({2: dapi}, SPACING, NUCLEAR_CONFIG)
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
            for index in (1, 2)
        ),
    )
    revised_config = replace(
        analysis_config,
        cells=(cell,),
        qc_decisions=(
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
        for round_number, lamin in ((2, reference_lamin), (3, moved_lamin))
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
    assert {
        "lamin_mask_volume",
        "lamin_mask_area_to_volume",
        "lamin_mask_sphericity",
        "lamin_fixed_to_moving_boundary_distance_p95",
        "lamin_moving_to_fixed_boundary_distance_p95",
        "lamin_surface_dice",
        "lamin_jaccard",
    } <= {row.endpoint for row in result.ensemble_rows}
    assert {row.acquisition_round for row in result.summary_rows} == {2, 3}
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
