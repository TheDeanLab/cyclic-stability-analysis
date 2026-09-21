# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cy-exm[workflow] @ git+https://github.com/TheDeanLab/Cy-ExM.git@f6e0de36cd7f6b5e85f093884c2cf224225f4ec2",
#   "matplotlib>=3.8,<4",
#   "nd2==0.11.3",
#   "PyYAML>=6,<7",
#   "psutil>=5.9,<8",
#   "scikit-image>=0.22,<1",
#   "scipy>=1.12,<2",
#   "tifffile>=2025.1.10",
# ]
# ///
"""Render non-destructive LAMP1 matching-radius sensitivity overlays."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backfill_lamp1_mip_overlays import _detections, _target_paths
from cyclic_stability.config import load_analysis_config
from cyclic_stability.images import load_volume
from cyclic_stability.lysosome import (
    DetectionSet,
    ObjectMatchResult,
    _mip_unit_interval,
    _write_mip_match_overlays,
    match_lysosomes,
)
from cyclic_stability.units import ResolutionReference, theoretical_resolution


def parse_args() -> argparse.Namespace:
    """Parse existing completed attempts and a separate sensitivity output root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt-dir",
        type=Path,
        action="append",
        required=True,
        help="Completed LAMP1 attempt directory; provide once per cell.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for visual-only sensitivity outputs.",
    )
    parser.add_argument(
        "--match-units",
        type=float,
        nargs="+",
        default=(1.0, 1.5, 2.0),
        help="Resolution-unit radii to compare (default: 1.0 1.5 2.0).",
    )
    return parser.parse_args()


def rematch(
    reference: DetectionSet,
    moving: DetectionSet,
    resolution: ResolutionReference,
    match_units: tuple[float, ...],
) -> list[dict[str, float | int]]:
    """Summarize new one-to-one matches for each non-destructive radius.

    The persisted detections are retained exactly; only the maximum normalized
    centroid distance is varied.  Round-2 remains the fixed comparison member.
    """
    rows: list[dict[str, float | int]] = []
    for units in match_units:
        match = match_lysosomes(reference, moving, resolution, units)
        rows.append(
            {
                "match_resolution_units": units,
                "reference_count": reference.object_count,
                "moving_count": moving.object_count,
                "matched_count": match.matched_count,
                "lost_count": match.lost_count,
                "new_count": match.new_count,
                "matched_fraction": match.matched_fraction,
                "lost_fraction": match.lost_fraction,
                "new_fraction": match.new_fraction,
            }
        )
    return rows


def _config_for_attempt(attempt_dir: Path) -> Any:
    """Load the immutable analysis inputs frozen beside one completed attempt."""
    run_dir = attempt_dir.parents[3]
    return load_analysis_config(
        run_dir / "config" / "analysis.yaml",
        run_dir / "config" / "correspondence.csv",
        run_dir / "config" / "qc-decisions.csv",
    )


def _resolution(config: Any) -> ResolutionReference:
    """Return the explicitly labeled theoretical optical matching reference."""
    return theoretical_resolution(
        config.optics.target_wavelength_um,
        config.optics.numerical_aperture,
        config.optics.immersion_refractive_index,
        config.nominal_expansion_factor,
    )


def _representative_detections(
    attempt_dir: Path, config: Any
) -> dict[int, DetectionSet]:
    """Load only the persisted median detector-member objects for visual QC."""
    sigma_um = config.lysosome.log_sigma_um[len(config.lysosome.log_sigma_um) // 2]
    detection_sbr = config.lysosome.detection_sbr[
        len(config.lysosome.detection_sbr) // 2
    ]
    detections = _detections(
        attempt_dir / "diagnostics" / "objects.csv", sigma_um, detection_sbr
    )
    if 2 not in detections or len(detections) < 2:
        raise ValueError(f"missing round-2 or moving detections: {attempt_dir}")
    return detections


def _matches(
    detections: dict[int, DetectionSet],
    resolution: ResolutionReference,
    units: float,
) -> list[tuple[str, int, int, DetectionSet, ObjectMatchResult]]:
    """Recompute visual-only round-2 match links at one specified radius."""
    reference = detections[2]
    return [
        (
            "round2-relative",
            2,
            moving_round,
            reference,
            match_lysosomes(reference, detections[moving_round], resolution, units),
        )
        for moving_round in sorted(detections)
        if moving_round != 2
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a stable machine-readable table without modifying source results."""
    fieldnames = tuple(rows[0]) if rows else ()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Create separate LAMP1 matching-radius sensitivity figures and tables."""
    args = parse_args()
    units = tuple(float(item) for item in args.match_units)
    if not units or len(set(units)) != len(units) or any(item <= 0.0 for item in units):
        raise ValueError("--match-units must contain unique positive values")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"sensitivity output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    for raw_attempt_dir in args.attempt_dir:
        attempt_dir = raw_attempt_dir.resolve()
        config = _config_for_attempt(attempt_dir)
        resolution = _resolution(config)
        detections = _representative_detections(attempt_dir, config)
        paths = _target_paths(attempt_dir / "record.json")
        mips = {
            number: _mip_unit_interval(load_volume(path))
            for number, path in paths.items()
        }
        cell_id = attempt_dir.parent.parent.name.removeprefix("lamp1-")
        baseline: dict[int, int] = {}
        for unit in units:
            matches = _matches(detections, resolution, unit)
            setting_dir = output_dir / cell_id / f"match-units-{unit:g}"
            setting_dir.mkdir(parents=True)
            outputs = _write_mip_match_overlays(
                setting_dir,
                mips,
                {number: (item,) for number, item in detections.items()},
                matches,
                config,
            )
            for (
                _comparison,
                _reference_round,
                moving_round,
                reference,
                match,
            ) in matches:
                if unit == units[0]:
                    baseline[moving_round] = match.matched_count
                rows.append(
                    {
                        "cell_id": cell_id,
                        "attempt_dir": str(attempt_dir),
                        "moving_round": moving_round,
                        "match_resolution_units": unit,
                        "reference_count": reference.object_count,
                        "moving_count": detections[moving_round].object_count,
                        "matched_count": match.matched_count,
                        "matched_delta_vs_first_radius": match.matched_count
                        - baseline[moving_round],
                        "lost_count": match.lost_count,
                        "new_count": match.new_count,
                        "matched_fraction": match.matched_fraction,
                        "lost_fraction": match.lost_fraction,
                        "new_fraction": match.new_fraction,
                    }
                )
            provenance.append(
                {
                    "cell_id": cell_id,
                    "attempt_dir": str(attempt_dir),
                    "match_resolution_units": unit,
                    "overlay_paths": [str(path) for path in outputs],
                    "resolution_reference": {
                        "kind": resolution.kind,
                        "lateral_expanded_um": resolution.lateral_expanded_um,
                        "axial_expanded_um": resolution.axial_expanded_um,
                    },
                }
            )
        print(f"completed visual-only rematching for {cell_id}")

    _write_csv(output_dir / "match-sensitivity.csv", rows)
    (output_dir / "match-sensitivity.provenance.json").write_text(
        json.dumps(
            {
                "purpose": "visual-only matching-radius sensitivity; persisted detections and measurements unchanged",
                "reference_round": 2,
                "match_units": units,
                "cells": provenance,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} sensitivity rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
