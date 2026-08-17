"""Exact display-palette contract for immutable review overlays."""

from __future__ import annotations

import numpy as np
import pytest
from cyclic_stability.review_palette import apply_review_palette


def test_palette_maps_reference_to_magenta_and_moving_to_green() -> None:
    """Reference-only, moving-only, and overlap pixels retain distinct roles."""
    source = np.array([[[20, 0, 7], [0, 30, 8], [40, 40, 9]]], dtype=np.uint8)
    original = source.copy()

    displayed = apply_review_palette(source)

    assert np.array_equal(
        displayed,
        np.array([[[20, 0, 20], [0, 30, 0], [40, 40, 40]]], dtype=np.uint8),
    )
    assert displayed.dtype == np.uint8
    assert displayed.shape == source.shape
    assert not np.shares_memory(displayed, source)
    assert np.array_equal(source, original)


@pytest.mark.parametrize(
    "invalid",
    (
        np.zeros((0, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint16),
    ),
)
def test_palette_rejects_non_rgb_uint8_images(invalid: np.ndarray) -> None:
    """Malformed decoded evidence fails before GUI or report rendering."""
    with pytest.raises(ValueError, match="nonempty H x W x 3 uint8"):
        apply_review_palette(invalid)
