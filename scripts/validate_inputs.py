# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cy-exm[workflow] @ git+https://github.com/TheDeanLab/Cy-ExM.git@c8b6a9f4a2f5f5fab3296fdc81190a7324da2a52",
#   "nd2==0.11.3",
#   "numpy>=1.26,<3",
#   "tifffile>=2025.1.10",
#   "PyYAML>=6,<7",
# ]
# ///
"""Validate cyclic-stability source correspondence and Cy-ExM provenance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cyclic_stability.config import load_analysis_config  # noqa: E402
from cyclic_stability.cyexm_artifacts import resolve_round_artifacts  # noqa: E402
from cyclic_stability.provenance import compare_nd2_export  # noqa: E402
from cyexm.errors import InputValidationError  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--correspondence", type=Path, required=True)
    parser.add_argument("--qc-decisions", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--skip-nd2-pixel-check", action="store_true")
    return parser


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.skip_nd2_pixel_check
        and os.environ.get("CYCLIC_STABILITY_SYNTHETIC_TEST") != "1"
    ):
        raise InputValidationError(
            "--skip-nd2-pixel-check is allowed only for synthetic tests"
        )
    config = load_analysis_config(
        args.analysis_config, args.correspondence, args.qc_decisions
    )
    cell = config.cell(args.cohort, args.cell)
    matches = (
        ()
        if args.skip_nd2_pixel_check
        else tuple(
            match
            for round_spec in cell.rounds
            for match in compare_nd2_export(round_spec)
        )
    )
    if any(not match.exact for match in matches):
        raise InputValidationError("ND2/TIFF exact source correspondence failed")
    artifacts = tuple(resolve_round_artifacts(round_spec) for round_spec in cell.rounds)
    payload: dict[str, object] = {
        "cohort": cell.cohort,
        "cell_id": cell.cell_id,
        "config_digest": config.config_digest,
        "skip_nd2_pixel_check": args.skip_nd2_pixel_check,
        "source_channel_checks": [
            {
                "channel": match.channel,
                "plane_indices": list(match.plane_indices),
                "exact": match.exact,
            }
            for match in matches
        ],
        "rounds": [
            {
                "acquisition_round": artifact.acquisition_round,
                "workflow_round": artifact.workflow_round,
                "workflow_output_root": str(artifact.workflow_output_root),
                "registered_dapi": str(artifact.registered_dapi),
                "registered_tubulin": str(artifact.registered_tubulin),
                "registered_target": str(artifact.registered_target),
                "completion_record_count": len(artifact.completion_records),
            }
            for artifact in artifacts
        ],
    }
    output = args.run_dir / "validation" / "inputs.json"
    _write_json(output, payload)
    print(f"validated {len(matches)} source channel checks and {len(artifacts)} rounds")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
