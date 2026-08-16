"""Read-only TIFF loading and conservative robust image-background estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class BackgroundSummary:
    """Robust low-positive intensity background estimate for one 3D image."""

    median: float
    scale: float
    positive_count: int
    background_count: int


def _validated_volume(image: NDArray[np.generic]) -> NDArray[np.generic]:
    """Validate a finite numeric three-dimensional ZYX image volume."""
    if image.ndim != 3:
        raise ValueError("image must be one 3D ZYX volume")
    if not np.issubdtype(image.dtype, np.number) or np.issubdtype(
        image.dtype, np.complexfloating
    ):
        raise ValueError("image must have a real numeric dtype")
    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain only finite values")
    return image


def load_volume(path: Path) -> NDArray[np.generic]:
    """Load one finite ZYX TIFF volume without permitting source mutation.

    Memory mapping is used when TIFF layout permits it; compressed or otherwise
    non-mappable TIFFs are read normally.  Both paths return a non-writeable
    array and reject non-3D or non-finite data at the analysis boundary.
    """
    try:
        image = tifffile.memmap(path, mode="r")
    except (OSError, ValueError, tifffile.TiffFileError):
        image = tifffile.imread(path)
    result = _validated_volume(np.asarray(image))
    result.setflags(write=False)
    return result


def robust_background(image: NDArray[np.generic]) -> BackgroundSummary:
    """Estimate background from the low positive tail using a robust MAD scale.

    The lower thirty percent of strictly positive voxels supplies the background.
    Empty support or a zero/non-finite robust scale is rejected rather than
    replaced with an optimistic fallback value.
    """
    values = _validated_volume(np.asarray(image)).astype(np.float64, copy=False)
    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError("background has no positive voxels")
    upper = float(np.percentile(positive, 30.0))
    background = positive[positive <= upper]
    if background.size == 0:
        raise ValueError("background selection is empty")
    median = float(np.median(background))
    scale = float(1.4826 * np.median(np.abs(background - median)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("background robust scale must be finite and positive")
    return BackgroundSummary(
        median=median,
        scale=scale,
        positive_count=int(positive.size),
        background_count=int(background.size),
    )


__all__ = ["BackgroundSummary", "load_volume", "robust_background"]
