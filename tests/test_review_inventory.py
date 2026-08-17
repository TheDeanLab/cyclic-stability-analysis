"""Target-blind review inventory contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import tifffile
from cyclic_stability.review_inventory import build_review_manifest
from synthetic import write_review_qc_fixture


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
