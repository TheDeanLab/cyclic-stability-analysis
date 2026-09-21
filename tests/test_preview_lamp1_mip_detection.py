"""Regression tests for rapid LAMP1 MIP preview rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from preview_lamp1_mip_detection import (
    _merge_candidate_footprints,
    _output_name,
    _render,
)

matplotlib.use("Agg")


def test_candidate_contour_uses_the_same_row_coordinates_as_the_mip() -> None:
    """A candidate at row one must not be rendered at its vertical mirror."""
    mip = np.zeros((6, 7), dtype=float)
    mip[1, 4] = 1.0
    admitted = np.zeros_like(mip, dtype=bool)
    admitted[1, 4] = True
    figure, axis = plt.subplots()

    _render(axis, mip, admitted, "coordinate regression")

    vertices = np.concatenate(
        [path.vertices for path in axis.collections[0].get_paths()], axis=0
    )
    assert float(np.mean(vertices[:, 0])) == pytest.approx(4.0, abs=0.25)
    assert float(np.mean(vertices[:, 1])) == pytest.approx(1.0, abs=0.25)
    plt.close(figure)


def test_connected_hollow_foreground_merges_multiple_dog_seeds() -> None:
    """A hollow lysosome with two local DoG seeds must receive one filled footprint."""
    y, x = np.indices((101, 101), dtype=float)
    radius = np.hypot(y - 50.0, x - 50.0)
    mip = np.zeros((101, 101), dtype=float)
    mip[(radius >= 23.0) & (radius <= 27.0)] = 10.0
    candidates = np.zeros_like(mip, dtype=bool)
    candidates[48:54, 22:28] = True
    candidates[48:54, 72:78] = True

    merged = _merge_candidate_footprints(mip, candidates)

    assert merged[50, 50]
    assert merged.sum() > 1_000


def test_preview_names_include_dataset_for_shared_channel_basenames() -> None:
    """Separate cells' identically named channel TIFFs must not overwrite previews."""
    alpha = Path("/results/alpha/rounds/round-002/channels/round_2_647_LAMP1.tif")
    beta = Path("/results/beta/rounds/round-002/channels/round_2_647_LAMP1.tif")

    assert _output_name(alpha) != _output_name(beta)
