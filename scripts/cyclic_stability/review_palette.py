"""Colorblind-friendly display conversion for immutable review overlays."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_review_palette(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Map reference red to magenta while retaining moving-round green."""
    array = np.asarray(image)
    if (
        array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[0] < 1
        or array.shape[1] < 1
        or array.shape[2] != 3
    ):
        raise ValueError("review overlay must be a nonempty H x W x 3 uint8 image")
    displayed = np.empty_like(array)
    displayed[..., 0] = array[..., 0]
    displayed[..., 1] = array[..., 1]
    displayed[..., 2] = array[..., 0]
    return displayed


__all__ = ["apply_review_palette"]
