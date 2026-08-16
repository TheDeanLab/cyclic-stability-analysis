"""Tests for read-only image validation and robust backgrounds."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
from cyclic_stability.images import load_volume, robust_background


def test_load_volume_returns_a_finite_zyx_volume(tmp_path) -> None:
    """A valid volume stays ZYX and is made read-only at the shared boundary."""
    path = tmp_path / "volume.tiff"
    expected = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    tifffile.imwrite(path, expected)
    actual = load_volume(path)
    assert actual.shape == (2, 3, 4)
    assert np.array_equal(actual, expected)
    assert not actual.flags.writeable


@pytest.mark.parametrize(
    "image",
    (np.zeros((2, 3)), np.full((2, 3, 4), np.nan)),
)
def test_load_volume_rejects_nonvolume_or_nonfinite_source(
    tmp_path, image: np.ndarray
) -> None:
    """Accepting a plane or NaN TIFF would make downstream metrics ill-defined."""
    path = tmp_path / "invalid.tiff"
    tifffile.imwrite(path, image, photometric="minisblack")
    with pytest.raises(ValueError, match="3D|finite"):
        load_volume(path)


def test_load_volume_falls_back_when_memory_mapping_is_unavailable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-mappable but otherwise valid TIFF must remain readable without writes."""
    path = tmp_path / "fallback.tiff"
    expected = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    tifffile.imwrite(path, expected)

    def unavailable(*args, **kwargs):
        raise ValueError("not memory-mappable")

    monkeypatch.setattr(tifffile, "memmap", unavailable)
    actual = load_volume(path)
    assert np.array_equal(actual, expected)
    assert not actual.flags.writeable


def test_robust_background_uses_low_positive_voxels_and_mad() -> None:
    """Using all intensities instead of the low-positive background changes both values."""
    image = np.arange(1.0, 11.0).reshape(2, 1, 5)
    background = robust_background(image)
    assert background.median == pytest.approx(2.0)
    assert background.scale == pytest.approx(1.4826)


@pytest.mark.parametrize(
    "image",
    (np.zeros((2, 2, 2)), np.ones((2, 2, 2))),
)
def test_robust_background_rejects_empty_or_zero_scale_data(image: np.ndarray) -> None:
    """Favorably substituting an empty or zero robust scale must remain impossible."""
    with pytest.raises(ValueError, match="background"):
        robust_background(image)
