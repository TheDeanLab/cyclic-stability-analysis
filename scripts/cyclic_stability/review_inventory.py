"""Strict inventory construction for the target-blind registration-QC reviewer."""

from __future__ import annotations

import csv
import hashlib
import io
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import tifffile

from cyclic_stability.review_evidence import (
    list_verified_directory,
    overlay_qc_root,
    read_verified_evidence,
)
from cyclic_stability.review_models import (
    CaseKey,
    CaseRevision,
    EvidenceIdentity,
    OverlayEvidence,
    Presentation,
    ReviewManifest,
)

OVERLAY_NAME = re.compile(
    r"^(01|03|04|05)-(dapi|tubulin)-(pre|affine|nonlinear)-(xy|xz|yz)\.tif$"
)
_DECISION_COLUMNS = frozenset(
    {
        "cohort",
        "cell_id",
        "acquisition_round",
        "automated_valid",
        "automated_reason",
        "decision",
    }
)
_REGISTRATION_COLUMNS = frozenset(
    {
        "cohort",
        "cell_id",
        "acquisition_round",
        "workflow_round",
        "is_reference",
        "automated_valid",
        "invalid_reason",
    }
)
_DEFORMATION_COLUMNS = frozenset(
    {"cohort", "cell_id", "acquisition_round", "region", "warp_flags"}
)
_REGIONS = frozenset({"whole", "seam", "interior"})
_EXPECTED_CASE_KEYS = frozenset(
    CaseKey(cohort, f"Cell{cell_number}", acquisition_round)
    for cohort in ("golgi", "lamin")
    for cell_number in range(1, 6)
    for acquisition_round in (1, 3, 4, 5)
)
_TECHNICAL_CASE = CaseKey("lamin", "Cell1", 1)


def _csv_rows(
    path: Path, required_columns: frozenset[str]
) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required_columns <= set(
                reader.fieldnames
            ):
                raise ValueError(f"{path.name} has an invalid schema")
            rows = tuple(dict(row) for row in reader)
    except FileNotFoundError as error:
        raise ValueError(f"missing required QC table: {path.name}") from error
    if any(set(row) != set(reader.fieldnames) or None in row.values() for row in rows):
        raise ValueError(f"{path.name} contains malformed rows")
    return rows


def _boolean(value: str, name: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{name} must be True or False")


def _case_key(row: Mapping[str, str]) -> CaseKey:
    try:
        acquisition_round = int(row["acquisition_round"])
    except (KeyError, ValueError) as error:
        raise ValueError("acquisition_round must be an integer") from error
    return CaseKey(row["cohort"], row["cell_id"], acquisition_round)


def _index_rows(
    rows: Iterable[Mapping[str, str]], name: str
) -> dict[CaseKey, Mapping[str, str]]:
    indexed: dict[CaseKey, Mapping[str, str]] = {}
    for row in rows:
        key = _case_key(row)
        if key in indexed:
            raise ValueError(f"duplicate {name} row for {key}")
        indexed[key] = row
    return indexed


def _overlay_paths(
    qc_root: Path, directory: Path, round_number: int
) -> tuple[tuple[Path, ...], tuple[bool, ...]]:
    expected = tuple(
        directory / f"{round_number:02d}-{channel}-{stage}-{view}.tif"
        for channel in ("dapi", "tubulin")
        for stage in ("pre", "affine", "nonlinear")
        for view in ("xy", "xz", "yz")
    )
    try:
        names = set(list_verified_directory(qc_root, directory))
    except FileNotFoundError:
        names = set()
    unexpected = sorted(
        name
        for name in names
        if name.endswith(".tif") and OVERLAY_NAME.fullmatch(name) is None
    )
    if unexpected:
        raise ValueError(
            f"overlay basename outside target-blind allowlist: {unexpected}"
        )
    return expected, tuple(path.name in names for path in expected)


def _overlay_evidence(qc_root: Path, path: Path) -> OverlayEvidence:
    match = OVERLAY_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(
            f"overlay basename outside target-blind allowlist: {path.name}"
        )
    try:
        verified = read_verified_evidence(qc_root, path)
    except FileNotFoundError as error:
        raise ValueError(f"missing required overlay: {path.name}") from error
    try:
        decoded = tifffile.imread(io.BytesIO(verified.content))
    except (OSError, ValueError, tifffile.TiffFileError) as error:
        raise ValueError(
            f"overlay must be a fully decoded RGB uint8 image: {path.name}"
        ) from error
    if (
        decoded.ndim != 3
        or decoded.shape[-1] != 3
        or any(dimension < 1 for dimension in decoded.shape)
        or decoded.dtype != np.dtype(np.uint8)
    ):
        raise ValueError(
            f"overlay must be a fully decoded RGB uint8 image: {path.name}"
        )
    round_text, channel, stage, view = match.groups()
    del round_text
    identity = EvidenceIdentity(
        str(verified.path), verified.size, verified.mtime_ns, verified.sha256
    )
    return OverlayEvidence(channel, stage, view, identity)


def read_overlay_bytes(overlay: OverlayEvidence) -> bytes:
    """Return bytes whose path, metadata, digest, and role still match one overlay."""
    path = Path(overlay.identity.path)
    match = OVERLAY_NAME.fullmatch(path.name)
    if (
        match is None
        or match.group(2) != overlay.channel
        or match.group(3) != overlay.stage
        or match.group(4) != overlay.view
    ):
        raise ValueError("review evidence basename is no longer allowlisted")
    verified = read_verified_evidence(overlay_qc_root(path), path)
    current = EvidenceIdentity(
        str(verified.path), verified.size, verified.mtime_ns, verified.sha256
    )
    if current != overlay.identity:
        raise ValueError("reviewed evidence identity changed")
    return verified.content


def _digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _case_token(seed: int, case_key: CaseKey) -> str:
    return _digest(str(seed), repr(case_key.to_dict()))[:20]


def _revision_id(
    case_key: CaseKey,
    overlays: tuple[OverlayEvidence, ...],
    automated_valid: bool,
    automated_reason: str,
    deformation_flags: tuple[tuple[str, str], ...],
) -> str:
    overlay_evidence = ",".join(
        f"{item.channel}:{item.stage}:{item.view}:{item.identity.sha256}"
        for item in overlays
    )
    technical_evidence = repr(
        {
            "automated_valid": automated_valid,
            "automated_reason": automated_reason,
            "deformation_flags": deformation_flags,
        }
    )
    return f"REV-{_digest(repr(case_key.to_dict()), overlay_evidence, technical_evidence)[:20]}"


def read_revision_cases(qc_root: Path) -> dict[CaseKey, CaseRevision]:
    """Read a complete 40-case QC root into strictly validated case revisions."""
    qc_root = qc_root.resolve()
    decision_rows = _index_rows(
        _csv_rows(qc_root / "qc" / "decision-requests.csv", _DECISION_COLUMNS),
        "decision request",
    )
    registration_rows = _index_rows(
        tuple(
            row
            for row in _csv_rows(
                qc_root / "qc" / "registration-qc.csv", _REGISTRATION_COLUMNS
            )
            if row["is_reference"] == "False"
        ),
        "registration QC",
    )
    deformation_by_case: defaultdict[CaseKey, list[Mapping[str, str]]] = defaultdict(
        list
    )
    for row in _csv_rows(qc_root / "qc" / "deformation.csv", _DEFORMATION_COLUMNS):
        deformation_by_case[_case_key(row)].append(row)
    if (
        set(decision_rows) != _EXPECTED_CASE_KEYS
        or set(registration_rows) != _EXPECTED_CASE_KEYS
    ):
        raise ValueError(
            "QC inventory must contain the expected ten-cell moving-round key set"
        )

    revisions: dict[CaseKey, CaseRevision] = {}
    visual_count = 0
    for case_key in sorted(decision_rows):
        decision = decision_rows[case_key]
        registration = registration_rows[case_key]
        automated_valid = _boolean(
            decision["automated_valid"], "decision automated_valid"
        )
        registration_valid = _boolean(
            registration["automated_valid"], "registration automated_valid"
        )
        if automated_valid != registration_valid:
            raise ValueError("decision and registration validity disagree")
        decision_reason = decision["automated_reason"]
        registration_reason = registration["invalid_reason"]
        if decision_reason != registration_reason:
            raise ValueError("decision and registration automated reasons disagree")
        if automated_valid and decision_reason:
            raise ValueError("valid cases require empty automated reasons")
        if not automated_valid and not decision_reason:
            raise ValueError("invalid cases require an automated reason")
        directory = (
            qc_root / "cells" / case_key.cohort / case_key.cell_id / "qc" / "overlays"
        )
        expected_paths, overlay_exists = _overlay_paths(
            qc_root, directory, case_key.acquisition_round
        )
        deformation_rows = deformation_by_case.pop(case_key, [])
        if any(overlay_exists) and not all(overlay_exists):
            missing = expected_paths[overlay_exists.index(False)]
            raise ValueError(f"missing required overlay: {missing.name}")
        if all(overlay_exists):
            overlays = tuple(
                _overlay_evidence(qc_root, path) for path in expected_paths
            )
            visual_count += 1
            if {row["region"] for row in deformation_rows} != _REGIONS or len(
                deformation_rows
            ) != 3:
                raise ValueError(
                    "visual case requires whole, seam, and interior deformation rows"
                )
        else:
            overlays = ()
            if case_key != _TECHNICAL_CASE:
                raise ValueError("only Lamin Cell1 round 1 may be the technical case")
            if automated_valid or registration_valid or deformation_rows:
                raise ValueError(
                    "technical case must be invalid in both source tables and have no visual evidence"
                )
        reason = decision_reason
        flags = tuple(
            sorted((row["region"], row["warp_flags"]) for row in deformation_rows)
        )
        revisions[case_key] = CaseRevision(
            case_key,
            _case_token(0, case_key),
            _revision_id(case_key, overlays, automated_valid, reason, flags),
            overlays,
            automated_valid,
            reason,
            flags,
        )
    if deformation_by_case:
        raise ValueError("deformation table contains unknown QC cases")
    if visual_count != 39:
        raise ValueError("QC inventory must contain exactly 39 visual cases")
    return revisions


def build_review_manifest(
    qc_root: Path, *, seed: int, repeat_count: int = 4
) -> ReviewManifest:
    """Validate one QC root and create its deterministic, blinded presentation order."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(repeat_count, int) or isinstance(repeat_count, bool):
        raise TypeError("repeat_count must be an integer")
    cases = read_revision_cases(qc_root)
    revisions = tuple(
        CaseRevision(
            revision.case_key,
            _case_token(seed, revision.case_key),
            revision.revision_id,
            revision.overlays,
            revision.automated_valid,
            revision.automated_reason,
            revision.deformation_flags,
        )
        for revision in (cases[key] for key in sorted(cases))
    )
    visual_cases = [item for item in revisions if item.overlays]
    if repeat_count < 0 or repeat_count > len(visual_cases):
        raise ValueError("repeat_count must select distinct visual cases")
    generator = random.Random(seed)
    generator.shuffle(visual_cases)
    repeat_indexes = generator.sample(range(len(visual_cases)), repeat_count)
    descriptors: list[tuple[str, int, CaseRevision]] = [
        ("base", index, revision) for index, revision in enumerate(visual_cases)
    ]
    descriptors.extend(
        ("repeat", index, visual_cases[index]) for index in repeat_indexes
    )
    generator.shuffle(descriptors)
    assigned = tuple(
        (
            kind,
            base_index,
            revision,
            "QC-"
            + _digest(
                str(seed),
                "presentation",
                str(position),
                kind,
                str(base_index),
                revision.revision_id,
            )[:16],
        )
        for position, (kind, base_index, revision) in enumerate(descriptors, start=1)
    )
    base_ids = {
        base_index: opaque_id
        for kind, base_index, _revision, opaque_id in assigned
        if kind == "base"
    }
    presentations = [
        Presentation(
            opaque_id,
            revision.case_token,
            revision.revision_id,
            revision.overlays,
            base_ids[base_index] if kind == "repeat" else None,
        )
        for kind, base_index, revision, opaque_id in assigned
    ]
    presentation_cases = {
        opaque_id: revision.case_key
        for _kind, _base_index, revision, opaque_id in assigned
    }
    return ReviewManifest(seed, revisions, tuple(presentations), presentation_cases)


__all__ = [
    "OVERLAY_NAME",
    "build_review_manifest",
    "read_overlay_bytes",
    "read_revision_cases",
]
