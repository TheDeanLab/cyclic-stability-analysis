"""Tests for physical threshold masks and resolution-normalized surfaces."""

from __future__ import annotations

import pytest
from cyclic_stability.morphology import (
    gaussian_sigma_zyx,
    resolution_surface_metrics,
    threshold_ensemble,
)
from cyclic_stability.units import theoretical_resolution
from synthetic import single_voxel, synthetic_roi, synthetic_target

SPACING = (0.1, 0.1, 0.3)
RESOLUTION = theoretical_resolution(0.640, 1.4, 1.515, 4.5)


def test_physical_gaussian_sigma_uses_zyx_voxel_order() -> None:
    """Swapping public XYZ spacing into NumPy ZYX order changes anisotropic smoothing."""
    assert gaussian_sigma_zyx(0.6, (0.1, 0.2, 0.3)) == pytest.approx((2.0, 3.0, 6.0))


def test_threshold_ensemble_retains_every_sigma_threshold_member() -> None:
    """Collapsing the sensitivity ensemble would lose its nine configured alternatives."""
    ensemble = threshold_ensemble(
        synthetic_target(), synthetic_roi(), (0.1, 0.2, 0.3), (1.0, 2.0, 3.0), SPACING
    )
    assert len(ensemble) == 9
    assert {(item.sigma_um, item.threshold_sbr) for item in ensemble} == {
        (sigma, threshold) for sigma in (0.1, 0.2, 0.3) for threshold in (1.0, 2.0, 3.0)
    }
    assert all(item.mask_zyx.shape == synthetic_target().shape for item in ensemble)


def test_resolution_surface_dice_treats_lateral_and_axial_offsets_differently() -> None:
    """Normalizing all axes by lateral resolution would hide axial anisotropy."""
    fixed = single_voxel((9, 9, 9), (4, 4, 4))
    lateral = single_voxel((9, 9, 9), (4, 4, 5))
    axial = single_voxel((9, 9, 9), (5, 4, 4))
    lateral_result = resolution_surface_metrics(
        fixed, lateral, SPACING, RESOLUTION, 1.0
    )
    axial_result = resolution_surface_metrics(fixed, axial, SPACING, RESOLUTION, 1.0)
    assert (
        lateral_result.normalized_distance_p95 != axial_result.normalized_distance_p95
    )
