"""Run-specific PDF replay; preserve terminal attempts and their checksum records."""

# ruff: noqa: E402
from __future__ import annotations

import ast
import csv
import gc
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from cyclic_stability import nuclear
from cyclic_stability.config import load_analysis_config
from cyclic_stability.images import load_volume
from scipy.ndimage import find_objects

RUN = ROOT / "results/full-lamin-watershed-20260921/20260921T213601Z"
OUT = Path(__file__).resolve().parent
CONFIG = load_analysis_config(
    RUN / "config/analysis.yaml",
    RUN / "config/correspondence.csv",
    RUN / "config/qc-decisions.csv",
)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def message(text):
    print(datetime.now(timezone.utc).isoformat(), text, flush=True)


def replay(cell):
    terminal = RUN / "tasks" / f"lamin-{cell}" / "terminal.json"
    pointer = json.loads(terminal.read_text())
    assert pointer["status"] == "complete"
    attempt = terminal.parent / "attempts" / pointer["attempt_id"]
    source_record = json.loads((attempt / "record.json").read_text())
    original_stats = {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in attempt.rglob("*")
        if p.is_file()
    }
    with (attempt / "nuclear-component-qc.csv").open() as handle:
        rows = tuple(
            nuclear.NuclearComponentQc(
                r["nucleus_id"],
                r["accepted"] == "True",
                ast.literal_eval(r["rejection_reasons"]),
                float(r["expanded_volume_um3"]),
                float(r["nominal_volume_um3"]),
                float(r["sphericity"]),
                ast.literal_eval(r["centroid_xyz_um"]),
                ast.literal_eval(r["face_contact_fractions"]),
            )
            for r in csv.DictReader(handle)
        )
    label_path = attempt / "diagnostics/consensus-nuclear-labels.npz"
    for relative in (
        "nuclear-component-qc.csv",
        "diagnostics/consensus-nuclear-labels.npz",
        "diagnostics/nuclear-component-inventory.pdf",
    ):
        source = attempt / relative
        identity = next(
            item["identity"]
            for item in json.loads((attempt / "output-manifest.json").read_text())[
                "outputs"
            ]
            if item["relative"] == relative
        )
        assert sha(source) == identity["sha256"], source
    with np.load(label_path) as saved:
        labels = saved["labels_zyx"]
        ids = saved["nucleus_ids"].tolist()
        spacing = tuple(saved["spacing_xyz_um"])
    assert spacing == CONFIG.voxel_spacing_xyz_um
    accepted_rounds = sorted(
        {2}
        | {
            d.acquisition_round
            for d in CONFIG.qc_decisions
            if d.cohort == "lamin" and d.cell_id == cell and d.decision == "accept"
        }
    )
    recorded_dapi = [
        item
        for item in source_record["input_identities"]
        if "/channels/" in item["path"] and item["path"].endswith("DAPI.tif")
    ]
    inputs = []
    dapi = {}
    for round_number in accepted_rounds:
        identity = next(
            item
            for item in recorded_dapi
            if Path(item["path"]).name == f"round_{round_number}_405_DAPI.tif"
        )
        path = Path(identity["path"])
        assert (path.stat().st_size, path.stat().st_mtime_ns) == (
            identity["st_size"],
            identity["st_mtime_ns"],
        )
        dapi[round_number] = load_volume(path)
        inputs.append(identity)
    message(f"{cell}: loaded recorded DAPI rounds {accepted_rounds}")
    if any(not row.accepted for row in rows):
        message(
            f"{cell}: reconstructing rejected outlines with unchanged DAPI adjudication"
        )
        adjudication = nuclear.adjudicate_nuclear_components(
            dapi, spacing, CONFIG.nuclear, CONFIG.nominal_expansion_factor
        )
        adjudication = nuclear._apply_lamin_nuclear_exclusions(
            adjudication,
            tuple(
                rule for rule in CONFIG.lamin_nuclear_exclusions if rule.cell_id == cell
            ),
        )
        assert adjudication.component_qc == rows, (
            "Replayed QC differs from saved scientific evidence"
        )
        for candidate, row in zip(adjudication._candidates, rows, strict=True):
            if row.accepted:
                geometry = candidate.accepted_geometry
                index = ids.index(row.nucleus_id) + 1
                assert np.array_equal(
                    labels[geometry.region_zyx] == index, geometry.mask_zyx
                )
                assert np.count_nonzero(labels == index) == np.count_nonzero(
                    geometry.mask_zyx
                )
        diagnostics = adjudication.component_diagnostics
        del adjudication
        replay_method = "DAPI-only adjudication; every QC value and accepted voxel mask verified equal"
        message(f"{cell}: all {len(rows)} QC rows and all accepted masks exactly match")
    else:
        regions = find_objects(labels, max_label=len(ids))
        diagnostics = []
        for row in rows:
            index = ids.index(row.nucleus_id) + 1
            region = nuclear._expanded_component_slice(regions[index - 1], labels.shape)
            mask = labels[region] == index
            assert float(mask.sum()) * spacing[0] * spacing[1] * spacing[
                2
            ] == row.expanded_volume_um3 or np.isclose(
                float(mask.sum()) * np.prod(spacing),
                row.expanded_volume_um3,
                rtol=1e-14,
            )
            projections = nuclear._component_diagnostic_projections(
                mask, region, spacing
            )
            diagnostics.append(
                nuclear.NuclearComponentDiagnostic(
                    row.nucleus_id, row.accepted, row.rejection_reasons, projections
                )
            )
        diagnostics = tuple(diagnostics)
        replay_method = "Outlines reconstructed directly from saved accepted labels; QC table reused verbatim"
    gc.collect()
    message(f"{cell}: rebuilding median-DAPI projections")
    normalized = [
        nuclear._normalized_dapi(nuclear._volume(image, "DAPI"))
        for image in dapi.values()
    ]
    # Identical element-wise median to the writer, with bounded temporary storage.
    shape = labels.shape
    xy = np.zeros(shape[1:], dtype=np.float64)
    xz = np.empty((shape[0], shape[2]), dtype=np.float64)
    yz = np.empty((shape[0], shape[1]), dtype=np.float64)
    for start in range(0, shape[0], 4):
        stop = min(start + 4, shape[0])
        block = np.median(np.stack([image[start:stop] for image in normalized]), axis=0)
        np.maximum(xy, block.max(axis=0), out=xy)
        xz[start:stop] = block.max(axis=1)
        yz[start:stop] = block.max(axis=2)
    del normalized, block, dapi
    gc.collect()
    consensus = nuclear.NuclearConsensus(
        (),
        labels > 0,
        labels,
        rows,
        diagnostics,
        spacing,
        CONFIG.nominal_expansion_factor,
    )
    directory = OUT / cell
    pdf = nuclear._write_nuclear_component_inventory(
        directory, cell, {"XY": xy, "XZ": xz, "YZ": yz}, consensus
    )
    after = {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in attempt.rglob("*")
        if p.is_file()
    }
    assert after == original_stats, "A completed attempt changed during replay"
    evidence = {
        "cell_id": cell,
        "source_attempt": str(attempt),
        "source_record_sha256": sha(attempt / "record.json"),
        "source_pdf_sha256": sha(
            attempt / "diagnostics/nuclear-component-inventory.pdf"
        ),
        "source_qc_sha256": sha(attempt / "nuclear-component-qc.csv"),
        "source_labels_sha256": sha(label_path),
        "input_identities": inputs,
        "input_identity_check": "recorded size and mtime; original recorded SHA256 retained",
        "accepted_rounds": accepted_rounds,
        "outline_replay": replay_method,
        "component_count": len(rows),
        "accepted_count": sum(row.accepted for row in rows),
        "component_qc": [asdict(row) for row in rows],
        "outline_point_counts": {
            d.nucleus_id: {p.plane: p.retained_point_count for p in d.projections}
            for d in diagnostics
        },
        "original_attempt_files_unchanged": len(original_stats),
        "renderer_sha256": sha(ROOT / "scripts/cyclic_stability/nuclear.py"),
        "regeneration_script_sha256": sha(Path(__file__)),
        "pdf": str(pdf),
        "pdf_sha256": sha(pdf),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (directory / "provenance.json").write_text(json.dumps(evidence, indent=2) + "\n")
    message(f"{cell}: completed {pdf}")


if __name__ == "__main__":
    for cell in sys.argv[1:]:
        replay(cell)
