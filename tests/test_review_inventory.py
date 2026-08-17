"""Target-blind review inventory contract tests."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import tifffile
from cyclic_stability.review_inventory import build_review_manifest
from cyclic_stability.review_models import CaseRevision, Presentation
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


def test_inventory_requires_the_exact_ten_cell_moving_round_keys(
    tmp_path: Path,
) -> None:
    """The only technical case cannot be moved to an arbitrary cell identifier."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    technical = lambda row: (
        row["cohort"] == "lamin"
        and row["cell_id"] == "Cell1"
        and row["acquisition_round"] == "1"
    )
    _rewrite_review_rows(
        qc_root / "qc" / "decision-requests.csv", technical, {"cell_id": "Cell6"}
    )
    _rewrite_review_rows(
        qc_root / "qc" / "registration-qc.csv",
        technical,
        {"cell_id": "Cell6"},
    )

    with pytest.raises(ValueError, match="expected ten-cell moving-round key set"):
        build_review_manifest(qc_root, seed=1)


def test_inventory_rejects_invalid_visual_case(tmp_path: Path) -> None:
    """A visual case must be valid in both decision and registration evidence."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    visual = lambda row: (
        row["cohort"] == "golgi"
        and row["cell_id"] == "Cell1"
        and row["acquisition_round"] == "1"
    )
    for filename in ("decision-requests.csv", "registration-qc.csv"):
        _rewrite_review_rows(
            qc_root / "qc" / filename, visual, {"automated_valid": "False"}
        )

    with pytest.raises(ValueError, match="visual case must be valid"):
        build_review_manifest(qc_root, seed=1)


def test_inventory_rejects_non_hard_invalid_technical_case(tmp_path: Path) -> None:
    """The no-overlay technical case must be invalid in both source tables."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    technical = lambda row: (
        row["cohort"] == "lamin"
        and row["cell_id"] == "Cell1"
        and row["acquisition_round"] == "1"
    )
    _rewrite_review_rows(
        qc_root / "qc" / "decision-requests.csv",
        technical,
        {"automated_valid": "True"},
    )

    with pytest.raises(ValueError, match="decision and registration validity disagree"):
        build_review_manifest(qc_root, seed=1)


def test_inventory_rejects_decision_registration_disagreement(tmp_path: Path) -> None:
    """Decision and registration validity cannot diverge for a visual case."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    visual = lambda row: (
        row["cohort"] == "golgi"
        and row["cell_id"] == "Cell1"
        and row["acquisition_round"] == "1"
    )
    _rewrite_review_rows(
        qc_root / "qc" / "decision-requests.csv",
        visual,
        {"automated_valid": "False"},
    )

    with pytest.raises(ValueError, match="decision and registration validity disagree"):
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
