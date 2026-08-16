"""Tests for unregistered repeated-label signal measurements."""

from __future__ import annotations

import numpy as np
from cyclic_stability.morphology import threshold_ensemble
from cyclic_stability.signal import measure_signal
from synthetic import synthetic_roi, synthetic_round_specs, synthetic_target

SPACING = (0.1, 0.1, 0.3)
ENSEMBLE = threshold_ensemble(
    synthetic_target(), synthetic_roi(), (0.1, 0.2, 0.3), (1.0, 2.0, 3.0), SPACING
)


def test_round_one_signal_is_kept_out_of_matched_setting_rows() -> None:
    """Classifying round one as matched settings would confound its exposure change."""
    rows = measure_signal(
        synthetic_round_specs()[0], synthetic_target(), synthetic_roi(), ENSEMBLE
    )
    assert all(row.comparison_set == "round-1-exposure-sensitivity" for row in rows)
    assert all(row.exposure_normalized for row in rows)


def test_later_round_signal_is_labeled_as_matched_settings() -> None:
    """Grouping rounds two through five with exposure sensitivity would misstate the design."""
    rows = measure_signal(
        synthetic_round_specs()[1], synthetic_target(), synthetic_roi(), ENSEMBLE
    )
    assert all(row.comparison_set == "matched-settings" for row in rows)
    assert not any(row.exposure_normalized for row in rows)


def test_signal_integrates_original_not_smoothed_intensities() -> None:
    """Replacing raw signal with a smoothed image changes its corrected integrated value."""
    target = synthetic_target()
    structure_roi = synthetic_roi()
    rows = measure_signal(synthetic_round_specs()[1], target, structure_roi, ENSEMBLE)
    member = ENSEMBLE[0]
    expected_mask = structure_roi & member.mask_zyx
    expected = np.clip(target - member.background.median, 0.0, None)[
        expected_mask
    ].sum()
    assert rows[0].integrated_per_ms == expected / 200.0
    assert rows[0].raw_percentiles[50] == np.percentile(target[expected_mask], 50)
