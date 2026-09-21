"""Regression tests for non-destructive LAMP1 matching sensitivity analysis."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from cyclic_stability.lysosome import DetectionSet, LysosomeObject
from cyclic_stability.units import ResolutionReference

SCRIPT = Path(__file__).parents[1] / "scripts" / "preview_lamp1_matching_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("lamp1_matching_sensitivity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
lamp1_matching_sensitivity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lamp1_matching_sensitivity)


def _detections(*coordinates: tuple[float, float, float]) -> DetectionSet:
    """Construct simple LAMP1 objects with hand-defined physical centroids."""
    return DetectionSet(
        0.7,
        4.0,
        tuple(
            LysosomeObject(coordinate, 1.0, 1.0, 0.9, math.nan, math.nan)
            for coordinate in coordinates
        ),
    )


def test_rematch_reports_pairs_recovered_only_at_two_resolution_units() -> None:
    """A sensitivity report must retain the one-unit baseline while exposing recovery."""
    reference = _detections((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    moving = _detections((0.5, 0.0, 0.0), (6.5, 0.0, 0.0))
    resolution = ResolutionReference(1.0, 1.0, 1.0, 1.0, "synthetic")

    rows = lamp1_matching_sensitivity.rematch(
        reference, moving, resolution, (1.0, 1.5, 2.0)
    )

    assert [(row["match_resolution_units"], row["matched_count"]) for row in rows] == [
        (1.0, 1),
        (1.5, 2),
        (2.0, 2),
    ]


def test_overlay_match_builder_uses_each_moving_detection_set() -> None:
    """The visual-only overlay builder must pass the selected moving objects."""
    detections = {
        2: _detections((0.0, 0.0, 0.0)),
        1: _detections((0.5, 0.0, 0.0)),
    }
    resolution = ResolutionReference(1.0, 1.0, 1.0, 1.0, "synthetic")

    matches = lamp1_matching_sensitivity._matches(detections, resolution, 1.0)

    assert matches[0][2] == 1
    assert matches[0][4].matched_count == 1
