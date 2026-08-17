"""Target-blind review inventory contract tests."""

from __future__ import annotations

import csv
import io
import re
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import tifffile
from cyclic_stability import review_evidence
from cyclic_stability.review_inventory import build_review_manifest
from cyclic_stability.review_models import CaseKey, CaseRevision, Presentation
from synthetic import write_review_qc_fixture


def _rewrite_review_rows(
    path: Path,
    predicate: Callable[[dict[str, str]], bool],
    updates: dict[str, str],
) -> None:
    """Apply one controlled table mutation to a synthetic review fixture."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        rows = list(reader)
        columns = reader.fieldnames
    assert any(predicate(row) for row in rows)
    for row in rows:
        if predicate(row):
            row.update(updates)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _technical_row(row: dict[str, str]) -> bool:
    return (
        row["cohort"] == "lamin"
        and row["cell_id"] == "Cell1"
        and row["acquisition_round"] == "1"
    )


def _visual_row(row: dict[str, str]) -> bool:
    return (
        row["cohort"] == "golgi"
        and row["cell_id"] == "Cell1"
        and row["acquisition_round"] == "1"
    )


def test_manifest_is_complete_randomized_and_target_blind(tmp_path: Path) -> None:
    """Inventory exposes each case through opaque randomized presentations."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")

    manifest = build_review_manifest(qc_root, seed=8142026)

    assert len(manifest.case_revisions) == 40
    assert len(manifest.visual_presentations) == 43
    assert (
        sum(item.repeat_of is not None for item in manifest.visual_presentations) == 4
    )
    assert (
        len(
            {item.repeat_of for item in manifest.visual_presentations if item.repeat_of}
        )
        == 4
    )
    assert all(
        item.opaque_id.startswith("QC-") for item in manifest.visual_presentations
    )
    assert all("Cell" not in item.opaque_id for item in manifest.visual_presentations)


def test_manifest_randomization_is_stable_and_repeats_are_distinct(
    tmp_path: Path,
) -> None:
    """A seed controls ordering while repeats are distinct presentation records."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")

    first = build_review_manifest(qc_root, seed=8142026)
    same_seed = build_review_manifest(qc_root, seed=8142026)
    different_seed = build_review_manifest(qc_root, seed=7)

    first_ids = tuple(item.opaque_id for item in first.visual_presentations)
    assert first_ids == tuple(item.opaque_id for item in same_seed.visual_presentations)
    assert first_ids != tuple(
        item.opaque_id for item in different_seed.visual_presentations
    )
    assert len(first_ids) == len(set(first_ids))


def test_all_presentation_ids_use_one_nonordinal_blinding_scheme(
    tmp_path: Path,
) -> None:
    """Removing a repeat must not be inferable from an ID range or grammar."""
    manifest = build_review_manifest(
        write_review_qc_fixture(tmp_path / "qc-only"), seed=8142026
    )

    identifiers = {item.opaque_id for item in manifest.visual_presentations}
    repeat_ids = {
        item.opaque_id
        for item in manifest.visual_presentations
        if item.repeat_of is not None
    }

    assert len(identifiers) == 43
    assert len(repeat_ids) == 4
    assert all(re.fullmatch(r"QC-[0-9a-f]{16}", item) for item in identifiers)
    assert all(
        item.repeat_of in identifiers
        for item in manifest.visual_presentations
        if item.repeat_of is not None
    )


def test_inventory_rejects_non_allowlisted_tiff(tmp_path: Path) -> None:
    """Target-channel filenames cannot enter the allowlisted evidence tree."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    extra = qc_root / "cells/golgi/Cell1/qc/overlays/01-647-golgi-nonlinear-xy.tif"
    extra.write_bytes(b"target must never be read")

    with pytest.raises(
        ValueError, match="overlay basename outside target-blind allowlist"
    ):
        build_review_manifest(qc_root, seed=1)


def test_inventory_rejects_missing_overlay(tmp_path: Path) -> None:
    """Every visual case must supply every fixed evidence panel."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    missing = qc_root / "cells/golgi/Cell1/qc/overlays/01-dapi-pre-xy.tif"
    missing.unlink()

    with pytest.raises(ValueError, match="missing required overlay"):
        build_review_manifest(qc_root, seed=1)


def test_inventory_rejects_non_rgb_overlay(tmp_path: Path) -> None:
    """Only tiny RGB overlays are valid reviewer evidence."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    overlay = qc_root / "cells/golgi/Cell1/qc/overlays/01-dapi-pre-xy.tif"
    tifffile.imwrite(overlay, [[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="RGB"):
        build_review_manifest(qc_root, seed=1)


def test_inventory_rejects_truncated_rgb_payload(tmp_path: Path) -> None:
    """Metadata-valid RGB TIFFs must fully decode before entering the manifest."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    overlay = qc_root / "cells/golgi/Cell1/qc/overlays/01-dapi-pre-xy.tif"
    content = overlay.read_bytes()
    overlay.write_bytes(content[:-32])

    with pytest.raises(ValueError, match="fully decoded RGB"):
        build_review_manifest(qc_root, seed=1)


def test_inventory_rejects_final_overlay_symlink_without_reading_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A final-component symlink is rejected before its target bytes are opened."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    overlay = qc_root / "cells/golgi/Cell1/qc/overlays/01-dapi-pre-xy.tif"
    target = tmp_path / "target-overlay.tif"
    shutil.copy2(overlay, target)
    overlay.unlink()
    overlay.symlink_to(target)
    target_reads: list[Path] = []
    real_open = io.open
    real_read = review_evidence.os.read

    def guarded_open(file: object, *args: object, **kwargs: object):
        if not isinstance(file, int) and Path(file).resolve() == target.resolve():
            target_reads.append(Path(file))
        return real_open(file, *args, **kwargs)

    def guarded_read(descriptor: int, count: int) -> bytes:
        opened = Path(review_evidence.os.readlink(f"/proc/self/fd/{descriptor}"))
        if opened.is_absolute() and opened == target.resolve():
            target_reads.append(target)
        return real_read(descriptor, count)

    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(review_evidence.os, "read", guarded_read)
    with pytest.raises(ValueError, match="symlink"):
        build_review_manifest(qc_root, seed=1)
    assert target_reads == []


def test_inventory_rejects_symlinked_overlay_ancestor_without_reading_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlinked overlay directory is rejected before any target file is opened."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    overlay_dir = qc_root / "cells/golgi/Cell1/qc/overlays"
    target_dir = tmp_path / "target-overlays"
    overlay_dir.rename(target_dir)
    overlay_dir.symlink_to(target_dir, target_is_directory=True)
    target_reads: list[Path] = []
    real_open = io.open
    real_read = review_evidence.os.read

    def guarded_open(file: object, *args: object, **kwargs: object):
        if not isinstance(file, int) and Path(file).resolve().is_relative_to(
            target_dir.resolve()
        ):
            target_reads.append(Path(file))
        return real_open(file, *args, **kwargs)

    def guarded_read(descriptor: int, count: int) -> bytes:
        opened = Path(review_evidence.os.readlink(f"/proc/self/fd/{descriptor}"))
        if opened.is_absolute() and opened.is_relative_to(target_dir.resolve()):
            target_reads.append(target_dir)
        return real_read(descriptor, count)

    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(review_evidence.os, "read", guarded_read)
    with pytest.raises(ValueError, match="symlink"):
        build_review_manifest(qc_root, seed=1)
    assert target_reads == []


def test_inventory_requires_the_exact_ten_cell_moving_round_keys(
    tmp_path: Path,
) -> None:
    """The only technical case cannot be moved to an arbitrary cell identifier."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    _rewrite_review_rows(
        qc_root / "qc" / "decision-requests.csv",
        _technical_row,
        {"cell_id": "Cell6"},
    )
    _rewrite_review_rows(
        qc_root / "qc" / "registration-qc.csv",
        _technical_row,
        {"cell_id": "Cell6"},
    )

    with pytest.raises(ValueError, match="expected ten-cell moving-round key set"):
        build_review_manifest(qc_root, seed=1)


def test_inventory_preserves_automated_invalid_visual_case(tmp_path: Path) -> None:
    """Complete visual evidence remains reviewable with its automated warning."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    _rewrite_review_rows(
        qc_root / "qc" / "decision-requests.csv",
        _visual_row,
        {
            "automated_valid": "False",
            "automated_reason": "nonpositive-jacobian",
        },
    )
    _rewrite_review_rows(
        qc_root / "qc" / "registration-qc.csv",
        _visual_row,
        {
            "automated_valid": "False",
            "invalid_reason": "nonpositive-jacobian",
        },
    )

    manifest = build_review_manifest(qc_root, seed=1)

    revision = next(
        item
        for item in manifest.case_revisions
        if item.case_key == CaseKey("golgi", "Cell1", 1)
    )
    assert len(revision.overlays) == 18
    assert not revision.automated_valid
    assert revision.automated_reason == "nonpositive-jacobian"


def test_inventory_rejects_non_hard_invalid_technical_case(tmp_path: Path) -> None:
    """The no-overlay technical case must be invalid in both source tables."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    _rewrite_review_rows(
        qc_root / "qc" / "decision-requests.csv",
        _technical_row,
        {"automated_valid": "True"},
    )

    with pytest.raises(ValueError, match="decision and registration validity disagree"):
        build_review_manifest(qc_root, seed=1)


def test_inventory_rejects_decision_registration_disagreement(tmp_path: Path) -> None:
    """Decision and registration validity cannot diverge for a visual case."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    _rewrite_review_rows(
        qc_root / "qc" / "decision-requests.csv",
        _visual_row,
        {"automated_valid": "False"},
    )

    with pytest.raises(ValueError, match="decision and registration validity disagree"):
        build_review_manifest(qc_root, seed=1)


def test_inventory_rejects_automated_reason_disagreement(tmp_path: Path) -> None:
    """Equal invalid booleans cannot conceal conflicting producer reasons."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    _rewrite_review_rows(
        qc_root / "qc" / "decision-requests.csv",
        _visual_row,
        {
            "automated_valid": "False",
            "automated_reason": "nonpositive-jacobian",
        },
    )
    _rewrite_review_rows(
        qc_root / "qc" / "registration-qc.csv",
        _visual_row,
        {
            "automated_valid": "False",
            "invalid_reason": "different-raw-reason",
        },
    )

    with pytest.raises(ValueError, match="automated reasons disagree"):
        build_review_manifest(qc_root, seed=1)


def test_inventory_requires_empty_reasons_for_valid_case(tmp_path: Path) -> None:
    """A valid producer row cannot carry a stale matching invalid reason."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    _rewrite_review_rows(
        qc_root / "qc" / "decision-requests.csv",
        _visual_row,
        {"automated_reason": "stale-producer-reason"},
    )
    _rewrite_review_rows(
        qc_root / "qc" / "registration-qc.csv",
        _visual_row,
        {"invalid_reason": "stale-producer-reason"},
    )

    with pytest.raises(ValueError, match="valid cases require empty automated reasons"):
        build_review_manifest(qc_root, seed=1)


def test_inventory_requires_reason_for_invalid_case(tmp_path: Path) -> None:
    """Matching empty strings are not evidence for an automated-invalid case."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    for filename in ("decision-requests.csv", "registration-qc.csv"):
        _rewrite_review_rows(
            qc_root / "qc" / filename,
            _visual_row,
            {"automated_valid": "False"},
        )

    with pytest.raises(ValueError, match="invalid cases require an automated reason"):
        build_review_manifest(qc_root, seed=1)


def test_inventory_rejects_grayscale_overlay_with_width_three(tmp_path: Path) -> None:
    """An X dimension of three is not evidence of RGB TIFF samples."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    overlay = qc_root / "cells/golgi/Cell1/qc/overlays/01-dapi-pre-xy.tif"
    tifffile.imwrite(overlay, [[1, 2, 3], [4, 5, 6]])

    with pytest.raises(ValueError, match="RGB"):
        build_review_manifest(qc_root, seed=1)


def test_review_models_reject_duplicate_or_incomplete_overlay_descriptors(
    tmp_path: Path,
) -> None:
    """Serialized review values must retain every unique evidence panel."""
    manifest = build_review_manifest(
        write_review_qc_fixture(tmp_path / "qc-only"), seed=1
    )
    revision = next(item for item in manifest.case_revisions if item.overlays)
    presentation = manifest.visual_presentations[0]

    with pytest.raises(ValueError, match="exact 2 x 3 x 3 overlay product"):
        replace(revision, overlays=(revision.overlays[0],) * 18)
    with pytest.raises(ValueError, match="exact 2 x 3 x 3 overlay product"):
        replace(presentation, overlays=presentation.overlays[:-1])

    duplicate_revision = revision.to_dict()
    duplicate_revision["overlays"] = [duplicate_revision["overlays"][0]] * 18
    with pytest.raises(ValueError, match="exact 2 x 3 x 3 overlay product"):
        CaseRevision.from_dict(duplicate_revision)

    incomplete_presentation = presentation.to_dict()
    incomplete_presentation["overlays"] = incomplete_presentation["overlays"][:-1]
    with pytest.raises(ValueError, match="exact 2 x 3 x 3 overlay product"):
        Presentation.from_dict(incomplete_presentation)


def test_case_revision_roundtrips_empty_warp_flags(tmp_path: Path) -> None:
    """An empty production warp-flags cell is a valid reviewed summary."""
    manifest = build_review_manifest(
        write_review_qc_fixture(tmp_path / "qc-only"), seed=1
    )
    revision = next(item for item in manifest.case_revisions if item.overlays)

    assert CaseRevision.from_dict(revision.to_dict()) == revision
