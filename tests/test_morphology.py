"""Tests for physical threshold masks and resolution-normalized surfaces."""

from __future__ import annotations

import numpy as np
import pytest
from cyclic_stability.models import AnalysisConfig
from cyclic_stability.morphology import (
    gaussian_sigma_zyx,
    median_surface_metrics,
    resolution_surface_metrics,
    threshold_ensemble,
)
from cyclic_stability.units import theoretical_resolution
from scipy.ndimage import gaussian_filter
from synthetic import single_voxel, synthetic_roi, synthetic_target

SPACING = (0.1, 0.1, 0.3)
RESOLUTION = theoretical_resolution(0.640, 1.4, 1.515, 4.5)


def test_physical_gaussian_sigma_uses_zyx_voxel_order() -> None:
    """Swapping public XYZ spacing into NumPy ZYX order changes anisotropic smoothing."""
    assert gaussian_sigma_zyx(0.6, (0.1, 0.2, 0.3)) == pytest.approx((2.0, 3.0, 6.0))


def test_threshold_ensemble_retains_authoritative_frozen_members_and_source(
    analysis_config: AnalysisConfig,
) -> None:
    """Changing frozen settings, smoothing the source, or collapsing a curve is invalid."""
    source = synthetic_target()
    pristine = source.copy()
    roi = synthetic_roi()
    ensemble = threshold_ensemble(
        source,
        roi,
        analysis_config.ensembles.gaussian_sigma_um,
        analysis_config.ensembles.threshold_sbr,
        analysis_config.voxel_spacing_xyz_um,
    )
    assert len(ensemble) == 9
    assert {(item.sigma_um, item.threshold_sbr) for item in ensemble} == {
        (sigma, threshold)
        for sigma in (0.279, 0.558, 1.116)
        for threshold in (3.0, 4.0, 5.0)
    }
    assert np.array_equal(source, pristine)
    member = ensemble[0]
    expected_smoothed = gaussian_filter(
        pristine,
        sigma=gaussian_sigma_zyx(0.279, analysis_config.voxel_spacing_xyz_um),
    )
    expected_mask = (
        expected_smoothed > member.background.median + 3.0 * member.background.scale
    ) & roi
    assert np.array_equal(member.mask_zyx, expected_mask)
    assert not member.mask_zyx.flags.writeable
    assert all(
        item.mask_status == ("nonempty" if np.any(item.mask_zyx) else "empty")
        for item in ensemble
    )


@pytest.mark.parametrize(
    "invalid_value",
    (np.nan, np.inf, -1.0, 0.5, 2.0),
)
def test_threshold_ensemble_rejects_nonbinary_roi_values(invalid_value: float) -> None:
    """Casting NaN, negative, or nonbinary ROI values to bool creates false foreground."""
    roi = synthetic_roi().astype(np.float64)
    roi[0, 0, 0] = invalid_value
    with pytest.raises(ValueError, match="binary"):
        threshold_ensemble(synthetic_target(), roi, (0.279,), (3.0,), SPACING)


def test_resolution_surface_dice_treats_lateral_and_axial_offsets_differently() -> None:
    """Normalizing all axes by lateral resolution would hide axial anisotropy."""
    fixed = single_voxel((9, 9, 9), (4, 4, 4))
    lateral = single_voxel((9, 9, 9), (4, 4, 6))
    axial = single_voxel((9, 9, 9), (6, 4, 4))
    lateral_result = resolution_surface_metrics(
        fixed, lateral, SPACING, RESOLUTION, 1.0
    )
    axial_result = resolution_surface_metrics(fixed, axial, SPACING, RESOLUTION, 1.0)
    assert lateral_result.normalized_distance_p95 == pytest.approx(0.7172131)
    assert axial_result.normalized_distance_p95 == pytest.approx(0.6064356)


def test_surface_metrics_use_pooled_surface_dice_for_unequal_boundaries() -> None:
    """Averaging directed fractions overweights a smaller surface's perfect match."""
    fixed = single_voxel((11, 11, 11), (4, 4, 4))
    moving = fixed.copy()
    moving[4, 4, 8] = True
    result = resolution_surface_metrics(fixed, moving, SPACING, RESOLUTION, 0.001)
    assert result.surface_dice[0.001] == pytest.approx(2.0 / 3.0)
    assert result.jaccard == pytest.approx(0.5)


def test_surface_metrics_define_deterministic_empty_comparison_statuses(
    analysis_config: AnalysisConfig,
) -> None:
    """Empty masks must be visible, not converted into an apparent measured surface."""
    empty = np.zeros((9, 9, 9), dtype=bool)
    occupied = single_voxel((9, 9, 9), (4, 4, 4))
    tolerances = analysis_config.ensembles.surface_tolerance_resolution_units
    both_empty = resolution_surface_metrics(
        empty, empty, SPACING, RESOLUTION, tolerances
    )
    one_empty = resolution_surface_metrics(
        empty, occupied, SPACING, RESOLUTION, tolerances
    )
    assert both_empty.comparison_status == "both-empty"
    assert both_empty.normalized_distance_p95 == 0.0
    assert both_empty.surface_dice == {0.5: 1.0, 1.0: 1.0, 1.5: 1.0}
    assert both_empty.jaccard == 1.0
    assert one_empty.comparison_status == "fixed-empty"
    assert one_empty.normalized_distance_p95 is None
    assert one_empty.surface_dice == {0.5: 0.0, 1.0: 0.0, 1.5: 0.0}
    assert one_empty.jaccard == 0.0


def test_identical_nonempty_surfaces_have_exact_agreement() -> None:
    """A self-comparison must retain zero distance and perfect Dice/Jaccard."""
    fixed = single_voxel((9, 9, 9), (4, 4, 4))
    result = resolution_surface_metrics(fixed, fixed, SPACING, RESOLUTION, 1.0)
    assert result.comparison_status == "both-nonempty"
    assert result.normalized_distance_p95 == 0.0
    assert result.surface_dice == {1.0: 1.0}
    assert result.jaccard == 1.0


@pytest.mark.parametrize("invalid_value", (np.nan, np.inf, -1.0, 0.5, 2.0))
def test_surface_metrics_reject_nonbinary_masks(invalid_value: float) -> None:
    """Nonbinary surface inputs must not silently acquire foreground on casting."""
    invalid = single_voxel((9, 9, 9), (4, 4, 4)).astype(np.float64)
    invalid[0, 0, 0] = invalid_value
    with pytest.raises(ValueError, match="binary"):
        resolution_surface_metrics(invalid, invalid, SPACING, RESOLUTION, 1.0)


def test_median_surface_summary_preserves_members_and_excludes_empty_comparisons() -> (
    None
):
    """Replacing curves with one arbitrary member defeats robust ensemble reporting."""
    fixed = single_voxel((11, 11, 11), (4, 4, 4))
    same = resolution_surface_metrics(fixed, fixed, SPACING, RESOLUTION, 1.0)
    shifted = resolution_surface_metrics(
        fixed, single_voxel((11, 11, 11), (4, 4, 6)), SPACING, RESOLUTION, 1.0
    )
    empty = resolution_surface_metrics(
        np.zeros((11, 11, 11), dtype=bool), fixed, SPACING, RESOLUTION, 1.0
    )
    summary = median_surface_metrics((same, shifted, empty))
    assert summary.member_count == 3
    assert summary.measured_member_count == 2
    assert summary.median_normalized_distance_p95 == pytest.approx(
        shifted.normalized_distance_p95 / 2.0
    )
    assert summary.median_surface_dice[1.0] == pytest.approx(
        (1.0 + shifted.surface_dice[1.0]) / 2.0
    )
