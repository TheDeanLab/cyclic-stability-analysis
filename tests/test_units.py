"""Physical-unit and deformation-summary regression tests."""

from __future__ import annotations

import numpy as np
import pytest
from cyclic_stability.registration_qc import summarize_warp
from cyclic_stability.units import theoretical_resolution
from synthetic import synthetic_folded_warp

TARGET_RESOLUTION = theoretical_resolution(0.640, 1.4, 1.515, 4.5)


def test_target_resolution_is_labeled_in_expanded_and_nominal_space() -> None:
    """Expansion-space optics must not be reported as native-space resolution."""
    value = theoretical_resolution(0.640, 1.4, 1.515, 4.5)
    assert value.lateral_expanded_um == pytest.approx(0.278857, rel=1e-5)
    assert value.axial_expanded_um == pytest.approx(0.989388, rel=1e-5)
    assert value.lateral_nominal_native_um == pytest.approx(0.061968, rel=1e-5)
    assert value.axial_nominal_native_um == pytest.approx(0.219864, rel=1e-5)
    assert value.kind == "theoretical"


def test_constant_physical_warp_has_expected_resolution_fraction() -> None:
    """A displacement equal to one lateral reference unit remains in tolerance."""
    warp = np.zeros((5, 6, 7, 3), dtype=np.float64)
    warp[..., 0] = 0.278857
    summary = summarize_warp(warp, (0.108333, 0.108333, 0.3), TARGET_RESOLUTION)
    assert summary.displacement_p95_um == pytest.approx(0.278857)
    assert summary.fraction_over_one_resolution == pytest.approx(0.0)
    assert summary.jacobian_nonpositive_count == 0


def test_folded_warp_is_not_silently_summarized_as_valid() -> None:
    """A negative local Jacobian must remain an explicit automated invalidity."""
    warp = synthetic_folded_warp(shape=(9, 9, 9))
    summary = summarize_warp(warp, (0.1, 0.1, 0.3), TARGET_RESOLUTION)
    assert summary.jacobian_nonpositive_count > 0
    assert "nonpositive-jacobian" in summary.flags
