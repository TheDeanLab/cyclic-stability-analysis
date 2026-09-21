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
"""Render LAMP1 MIP match overlays for an already-completed cell attempt."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cyclic_stability.config import load_analysis_config
from cyclic_stability.images import load_volume
from cyclic_stability.lysosome import (
    DetectionSet,
    LysosomeObject,
    ObjectMatchResult,
    _mip_unit_interval,
    _write_mip_match_overlays,
)


def parse_args() -> argparse.Namespace:
    """Parse one completed attempt directory to enrich with visual-only QC."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    return parser.parse_args()


def _target_paths(record_path: Path) -> dict[int, Path]:
    """Recover registered LAMP1 channel paths recorded by the completed attempt."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    pattern = re.compile(r"rounds/round-(\d{3})/channels/round_(\d+)_647_LAMP1\.tif$")
    paths: dict[int, Path] = {}
    for item in record["input_identities"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        path = item["path"]
        match = pattern.search(path)
        if match is None or int(match.group(1)) != int(match.group(2)):
            continue
        paths[int(match.group(1))] = Path(path)
    if 2 not in paths or len(paths) < 2:
        raise ValueError("attempt record lacks registered LAMP1 round paths")
    return paths


def _detections(
    objects_path: Path, sigma_um: float, detection_sbr: float
) -> dict[int, DetectionSet]:
    """Reconstruct the representative ensemble's detections from objects.csv."""
    groups: dict[int, list[tuple[int, LysosomeObject]]] = defaultdict(list)
    with objects_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        sigma_column = (
            "mip_smoothing_sigma_um"
            if reader.fieldnames and "mip_smoothing_sigma_um" in reader.fieldnames
            else "log_sigma_um"
        )
        for row in reader:
            if (
                float(row[sigma_column]) != sigma_um
                or float(row["detection_sbr"]) != detection_sbr
            ):
                continue
            groups[int(row["acquisition_round"])].append(
                (
                    int(row["object_index"]),
                    LysosomeObject(
                        (float(row["x_um"]), float(row["y_um"]), float(row["z_um"])),
                        float(row["volume_um3"]),
                        float(row["equivalent_diameter_um"]),
                        float(row["sphericity"]),
                        float(row["nearest_neighbor_distance_um"]),
                        float(row["nucleus_relative_distance_um"]),
                    ),
                )
            )
    return {
        number: DetectionSet(
            sigma_um,
            detection_sbr,
            tuple(object_ for _index, object_ in sorted(items)),
        )
        for number, items in groups.items()
    }


def _matches(
    matches_path: Path,
    detections: dict[int, DetectionSet],
    sigma_um: float,
    detection_sbr: float,
) -> list[tuple[str, int, int, DetectionSet, ObjectMatchResult]]:
    """Reconstruct round-2 match links from the persisted machine-readable table."""
    pairs: dict[int, list[tuple[int, int, float, float, float]]] = defaultdict(list)
    with matches_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        sigma_column = (
            "mip_smoothing_sigma_um"
            if reader.fieldnames and "mip_smoothing_sigma_um" in reader.fieldnames
            else "log_sigma_um"
        )
        for row in reader:
            if (
                row["comparison_type"] != "round2-relative"
                or float(row[sigma_column]) != sigma_um
                or float(row["detection_sbr"]) != detection_sbr
            ):
                continue
            pairs[int(row["moving_round"])].append(
                (
                    int(row["reference_object_index"]),
                    int(row["moving_object_index"]),
                    float(row["lateral_displacement_um"]),
                    float(row["axial_displacement_um"]),
                    float(row["normalized_displacement"]),
                )
            )
    reference = detections[2]
    result: list[tuple[str, int, int, DetectionSet, ObjectMatchResult]] = []
    for moving_round in sorted(number for number in detections if number != 2):
        rows = pairs[moving_round]
        moving = detections[moving_round]
        result.append(
            (
                "round2-relative",
                2,
                moving_round,
                reference,
                ObjectMatchResult(
                    tuple(
                        (reference_index, moving_index)
                        for reference_index, moving_index, *_ in rows
                    ),
                    tuple(row[4] for row in rows),
                    tuple(row[2] for row in rows),
                    tuple(row[3] for row in rows),
                    len(rows),
                    reference.object_count - len(rows),
                    moving.object_count - len(rows),
                    len(rows) / reference.object_count
                    if reference.object_count
                    else 0.0,
                    0.0,
                    0.0,
                ),
            )
        )
    return result


def main() -> int:
    """Add non-numerical representative MIP overlays beside complete diagnostics."""
    args = parse_args()
    attempt_dir = args.attempt_dir.resolve()
    run_dir = attempt_dir.parents[3]
    config = load_analysis_config(
        run_dir / "config" / "analysis.yaml",
        run_dir / "config" / "correspondence.csv",
        run_dir / "config" / "qc-decisions.csv",
    )
    sigma_um = config.lysosome.log_sigma_um[len(config.lysosome.log_sigma_um) // 2]
    detection_sbr = config.lysosome.detection_sbr[
        len(config.lysosome.detection_sbr) // 2
    ]
    diagnostics = attempt_dir / "diagnostics"
    detections = _detections(diagnostics / "objects.csv", sigma_um, detection_sbr)
    matches = _matches(diagnostics / "matches.csv", detections, sigma_um, detection_sbr)
    paths = _target_paths(attempt_dir / "record.json")
    mips = {
        number: _mip_unit_interval(load_volume(path)) for number, path in paths.items()
    }
    outputs = _write_mip_match_overlays(
        diagnostics,
        mips,
        {number: (item,) for number, item in detections.items()},
        matches,
        config,
    )
    print(f"wrote {len(outputs)} LAMP1 MIP match overlays to {diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
