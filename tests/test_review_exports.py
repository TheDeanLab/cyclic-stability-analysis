"""Remediation and final export contract tests."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from cyclic_stability import review_exports, review_store
from cyclic_stability.review_exports import (
    CONSISTENCY_COLUMNS,
    FINAL_COLUMNS,
    REMEDIATION_COLUMNS,
    attach_revision_root,
    finalize_decisions,
    write_operational_exports,
)
from cyclic_stability.review_inventory import build_review_manifest
from cyclic_stability.review_models import CaseKey, RepositoryIdentity
from cyclic_stability.review_store import ReviewStore
from synthetic import (
    completed_review_store,
    fixed_clock,
    fresh_review_store,
    rewrite_review_case_pixels,
    write_review_qc_fixture,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _technical_case(manifest) -> CaseKey:
    return next(item.case_key for item in manifest.case_revisions if not item.overlays)


def _complete_visual_assessments(
    store: ReviewStore,
    manifest,
    *,
    statuses: dict[str, str] | None = None,
) -> None:
    statuses = statuses or {}
    for presentation in manifest.visual_presentations:
        status = statuses.get(presentation.opaque_id, "accept")
        reasons = (
            ("z-misalignment",)
            if status in {"exclude", "needs-adjudication"}
            else ("dapi-aligned",)
        )
        store.assess(presentation.opaque_id, status, reasons, status)


def test_remediation_is_nonfinal_and_new_revision_preserves_history(
    tmp_path: Path,
) -> None:
    """A changed immutable revision re-enters review without erasing provenance."""
    qc_root = write_review_qc_fixture(tmp_path / "qc-v1")
    manifest = build_review_manifest(qc_root, seed=9)
    store = ReviewStore.create(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    presentation = manifest.visual_presentations[0]
    store.assess(
        presentation.opaque_id,
        "needs-registration-improvement",
        ("persistent-interior-misalignment",),
        "retry registration",
    )
    with pytest.raises(ValueError, match="registration improvement remains unresolved"):
        finalize_decisions(store, manifest, tmp_path / "review")
    revised_root = write_review_qc_fixture(tmp_path / "qc-v2")
    rewrite_review_case_pixels(
        revised_root, manifest.presentation_cases[presentation.opaque_id]
    )
    attach_revision_root(store, revised_root)
    assert len(store.state.case_histories[presentation.case_token].revisions) == 2
    assert (
        store.state.assessment_history[presentation.opaque_id][0].notes
        == "retry registration"
    )


def test_final_csv_has_one_binary_row_per_biological_case(tmp_path: Path) -> None:
    """Final output contains forty unique biological decisions and no repeats."""
    store, manifest = completed_review_store(tmp_path)
    output = finalize_decisions(store, manifest, tmp_path / "review")
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert len(rows) == 40
    assert set(rows[0]) == {
        "cohort",
        "cell_id",
        "acquisition_round",
        "decision",
        "reasons",
        "reviewer",
        "reviewed_at",
    }
    assert {row["decision"] for row in rows} <= {"accept", "exclude"}


def test_operational_exports_are_deterministic_unblinded_and_latest_only(
    tmp_path: Path,
) -> None:
    """Operational files locate unresolved work and preserve repeat assessments."""
    store, manifest = fresh_review_store(tmp_path)
    assert (tmp_path / "review/remediation-requests.csv").is_file()
    assert (tmp_path / "review/consistency-report.csv").is_file()
    pending = manifest.visual_presentations[0]
    resolved = manifest.visual_presentations[1]
    store.assess(
        pending.opaque_id,
        "needs-registration-improvement",
        ("persistent-interior-misalignment", "z-misalignment"),
        "rerun",
    )
    store.assess(
        resolved.opaque_id,
        "needs-registration-improvement",
        ("z-misalignment",),
        "old request",
    )
    store.assess(resolved.opaque_id, "accept", ("dapi-aligned",), "resolved")

    remediation = tmp_path / "review/remediation-requests.csv"
    consistency = tmp_path / "review/consistency-report.csv"
    automatic_remediation = remediation.read_bytes()
    automatic_consistency = consistency.read_bytes()
    explicit_remediation, explicit_consistency = write_operational_exports(
        store.state, manifest, tmp_path / "review"
    )
    assert explicit_remediation == remediation
    assert explicit_consistency == consistency
    assert remediation.read_bytes() == automatic_remediation
    assert consistency.read_bytes() == automatic_consistency

    remediation_rows = _read_rows(remediation)
    assert tuple(remediation_rows[0]) == REMEDIATION_COLUMNS
    assert remediation_rows == [
        {
            "cohort": manifest.presentation_cases[pending.opaque_id].cohort,
            "cell_id": manifest.presentation_cases[pending.opaque_id].cell_id,
            "acquisition_round": str(
                manifest.presentation_cases[pending.opaque_id].acquisition_round
            ),
            "revision_id": pending.revision_id,
            "reasons": "persistent-interior-misalignment;z-misalignment",
            "notes": "rerun",
            "reviewer": "Reviewer",
            "reviewed_at": fixed_clock().isoformat(),
        }
    ]
    consistency_rows = _read_rows(consistency)
    assert tuple(consistency_rows[0]) == CONSISTENCY_COLUMNS
    assert len(consistency_rows) == 4
    assert (
        remediation.read_bytes()
        == write_operational_exports(store.state, manifest, tmp_path / "review")[
            0
        ].read_bytes()
    )


def test_revision_attachment_rejects_unchanged_evidence_atomically(
    tmp_path: Path,
) -> None:
    """Every pending case must change content before any revision is committed."""
    store, manifest = fresh_review_store(tmp_path)
    pending = manifest.visual_presentations[:2]
    for presentation in pending:
        store.assess(
            presentation.opaque_id,
            "needs-registration-improvement",
            ("persistent-interior-misalignment",),
            "retry",
        )
    revised_root = write_review_qc_fixture(tmp_path / "qc-v2")
    rewrite_review_case_pixels(
        revised_root, manifest.presentation_cases[pending[0].opaque_id]
    )
    before_state = (tmp_path / "review/review-state.json").read_bytes()
    before_manifest = store.manifest

    with pytest.raises(ValueError, match="different evidence identity"):
        attach_revision_root(store, revised_root)

    assert store.manifest == before_manifest
    assert (tmp_path / "review/review-state.json").read_bytes() == before_state
    assert all(
        len(store.state.case_histories[item.case_token].revisions) == 1
        for item in pending
    )


def test_revision_attachment_is_atomic_when_state_promotion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed state promotion leaves memory and durable review history unchanged."""
    store, manifest = fresh_review_store(tmp_path)
    pending = manifest.visual_presentations[0]
    store.assess(
        pending.opaque_id,
        "needs-registration-improvement",
        ("persistent-interior-misalignment",),
        "retry",
    )
    revised_root = write_review_qc_fixture(tmp_path / "qc-v2")
    rewrite_review_case_pixels(
        revised_root, manifest.presentation_cases[pending.opaque_id]
    )
    current = tmp_path / "review/review-state.json"
    before = current.read_bytes()
    original = review_store._atomic_pretty_json

    def fail_current(path: Path, payload: dict[str, object]) -> None:
        if path == current:
            raise OSError("injected revision promotion failure")
        original(path, payload)

    monkeypatch.setattr(review_store, "_atomic_pretty_json", fail_current)
    with pytest.raises(OSError, match="revision promotion failure"):
        attach_revision_root(store, revised_root)

    assert current.read_bytes() == before
    assert store.manifest == manifest
    assert len(store.state.case_histories[pending.case_token].revisions) == 1


def test_attached_revision_updates_active_manifest_and_survives_reopen(
    tmp_path: Path,
) -> None:
    """State and reconstructed active mapping contain the same appended revision."""
    store, manifest = fresh_review_store(tmp_path, seed=11)
    pending = manifest.visual_presentations[0]
    store.assess(
        pending.opaque_id,
        "needs-registration-improvement",
        ("persistent-interior-misalignment",),
        "retry",
    )
    revised_root = write_review_qc_fixture(tmp_path / "qc-v2")
    rewrite_review_case_pixels(
        revised_root, manifest.presentation_cases[pending.opaque_id]
    )

    attach_revision_root(store, revised_root)

    appended = store.state.case_histories[pending.case_token].revisions[-1]
    new_presentation = next(
        item
        for item in store.manifest.visual_presentations
        if item.revision_id == appended.revision_id
    )
    assert store.manifest.case_revisions[-1] == appended
    assert (
        store.manifest.presentation_cases[new_presentation.opaque_id]
        == appended.case_key
    )
    reopened = ReviewStore.open(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    assert reopened.manifest == store.manifest
    assert reopened.state == store.state
    reopened.assess(new_presentation.opaque_id, "accept", ("dapi-aligned",), "fixed")


def test_multiple_and_repeated_revision_attachments_reopen_exactly(
    tmp_path: Path,
) -> None:
    """Canonical active-manifest ordering survives batch and later attachments."""
    store, manifest = fresh_review_store(tmp_path, seed=13)
    pending = manifest.visual_presentations[:2]
    for presentation in pending:
        store.assess(
            presentation.opaque_id,
            "needs-registration-improvement",
            ("persistent-interior-misalignment",),
            "retry",
        )
    revised_root = write_review_qc_fixture(tmp_path / "qc-v2")
    for presentation in pending:
        rewrite_review_case_pixels(
            revised_root, manifest.presentation_cases[presentation.opaque_id]
        )

    attach_revision_root(store, revised_root)

    first_new, second_new = store.manifest.visual_presentations[-2:]
    store.assess(
        first_new.opaque_id,
        "needs-registration-improvement",
        ("persistent-interior-misalignment",),
        "retry again",
    )
    store.assess(second_new.opaque_id, "accept", ("dapi-aligned",), "fixed")
    third_root = write_review_qc_fixture(tmp_path / "qc-v3")
    first_key = store.manifest.presentation_cases[first_new.opaque_id]
    rewrite_review_case_pixels(third_root, first_key)
    first_overlay = next(
        (
            third_root / "cells" / first_key.cohort / first_key.cell_id / "qc/overlays"
        ).glob(f"{first_key.acquisition_round:02d}-*.tif")
    )
    tifffile.imwrite(
        first_overlay,
        np.full((8, 12, 3), 251, dtype=np.uint8),
        photometric="rgb",
    )

    attach_revision_root(store, third_root)

    assert len(store.state.case_histories[first_new.case_token].revisions) == 3
    reopened = ReviewStore.open(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    assert reopened.state == store.state
    assert reopened.manifest == store.manifest


def test_revision_attachment_rejects_target_image_without_reading_it(
    tmp_path: Path,
) -> None:
    """A target-channel basename fails the allowlist at revision discovery."""
    store, manifest = fresh_review_store(tmp_path)
    pending = manifest.visual_presentations[0]
    store.assess(
        pending.opaque_id,
        "needs-registration-improvement",
        ("persistent-interior-misalignment",),
        "retry",
    )
    revised_root = write_review_qc_fixture(tmp_path / "qc-v2")
    case_key = manifest.presentation_cases[pending.opaque_id]
    rewrite_review_case_pixels(revised_root, case_key)
    target = (
        revised_root
        / "cells"
        / case_key.cohort
        / case_key.cell_id
        / "qc/overlays"
        / f"{case_key.acquisition_round:02d}-647-target-nonlinear-xy.tif"
    )
    target.write_bytes(b"must not be opened")

    with pytest.raises(ValueError, match="outside target-blind allowlist"):
        attach_revision_root(store, revised_root)


def test_repeat_disagreement_requires_current_revision_adjudication(
    tmp_path: Path,
) -> None:
    """A biological adjudication resolves repeats without erasing either assessment."""
    store, manifest = fresh_review_store(tmp_path)
    repeated = next(
        item for item in manifest.visual_presentations if item.repeat_of is not None
    )
    assert repeated.repeat_of is not None
    _complete_visual_assessments(
        store,
        manifest,
        statuses={repeated.opaque_id: "exclude"},
    )
    store.adjudicate(
        _technical_case(manifest),
        "exclude",
        ("invalid-affine-reflection",),
        "hard invalid",
    )

    with pytest.raises(ValueError, match="concealed repeat disagreement"):
        finalize_decisions(store, manifest, tmp_path / "review")

    case_key = manifest.presentation_cases[repeated.opaque_id]
    store.adjudicate(case_key, "exclude", ("z-misalignment",), "reconciled")
    output = finalize_decisions(store, manifest, tmp_path / "review")
    row = next(
        item
        for item in _read_rows(output)
        if (item["cohort"], item["cell_id"], item["acquisition_round"])
        == (case_key.cohort, case_key.cell_id, str(case_key.acquisition_round))
    )
    assert row["decision"] == "exclude"
    consistency = _read_rows(tmp_path / "review/consistency-report.csv")
    report_row = next(
        item for item in consistency if item["repeat_opaque_id"] == repeated.opaque_id
    )
    assert report_row["base_status"] == "accept"
    assert report_row["repeat_status"] == "exclude"
    assert report_row["agreement"] == "disagree"
    assert report_row["adjudicated_decision"] == "exclude"


def test_hard_invalid_technical_case_requires_exclusion(tmp_path: Path) -> None:
    """The no-overlay affine failure is final only through explicit exclusion."""
    store, manifest = fresh_review_store(tmp_path)
    _complete_visual_assessments(store, manifest)
    technical = _technical_case(manifest)

    with pytest.raises(ValueError, match="hard-invalid technical case"):
        finalize_decisions(store, manifest, tmp_path / "review")
    with pytest.raises(
        ValueError, match="hard-invalid technical case must be excluded"
    ):
        store.adjudicate(technical, "accept", ("dapi-aligned",), "incorrect")
    with pytest.raises(ValueError, match="invalid-affine-reflection"):
        store.adjudicate(technical, "exclude", ("dapi-aligned",), "wrong reason")


def test_finalization_revalidates_current_evidence_identities(tmp_path: Path) -> None:
    """Changed bytes at a reviewed path block export instead of becoming exclusion."""
    store, manifest = completed_review_store(tmp_path)
    overlay = Path(manifest.visual_presentations[0].overlays[0].path)
    tifffile.imwrite(
        overlay,
        np.full((8, 12, 3), 255, dtype=np.uint8),
        photometric="rgb",
    )

    with pytest.raises(ValueError, match="reviewed evidence identity changed"):
        finalize_decisions(store, manifest, tmp_path / "review")
    assert not (tmp_path / "review/qc-decisions.csv").exists()
    assert store.state.export_history == ()


def test_finalization_revalidates_repository_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repository drift after review creation blocks a provenance-bound export."""
    store, manifest = completed_review_store(tmp_path)
    monkeypatch.setattr(
        review_store,
        "_repository_identity",
        lambda: RepositoryIdentity("a" * 40, False, None),
    )

    with pytest.raises(ValueError, match="repository identity changed"):
        finalize_decisions(store, manifest, tmp_path / "review")


def test_finalization_cannot_write_outside_the_owned_review_directory(
    tmp_path: Path,
) -> None:
    """The configuration-compatible CSV remains inside the review directory."""
    store, manifest = completed_review_store(tmp_path)

    with pytest.raises(ValueError, match="owned review directory"):
        finalize_decisions(store, manifest, tmp_path / "elsewhere")

    assert not (tmp_path / "elsewhere/qc-decisions.csv").exists()


def test_finalization_preserves_previous_valid_csv_and_records_validated_export(
    tmp_path: Path,
) -> None:
    """Repeated finalization backs up valid bytes and appends their verified digest."""
    store, manifest = completed_review_store(tmp_path)
    output = finalize_decisions(store, manifest, tmp_path / "review")
    first = output.read_bytes()
    assert tuple(_read_rows(output)[0]) == FINAL_COLUMNS
    assert store.state.export_history[-1].sha256 == hashlib.sha256(first).hexdigest()
    assert store.state.export_history[-1].exported_at == fixed_clock()

    finalize_decisions(store, manifest, tmp_path / "review")

    assert (tmp_path / "review/qc-decisions.previous.csv").read_bytes() == first
    reopened = ReviewStore.open(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    assert reopened.state.export_history == store.state.export_history
    assert len(reopened.state.export_history) == 2


def test_invalid_reopened_csv_is_not_recorded_or_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Export history advances only after the staged CSV is reopened and validated."""
    store, manifest = completed_review_store(tmp_path)

    def write_invalid(path: Path, columns, rows) -> None:
        del columns, rows
        path.write_text("cohort,cell_id\ngolgi,Cell1\n", encoding="utf-8")

    monkeypatch.setattr(review_exports, "_atomic_csv", write_invalid)
    with pytest.raises(ValueError, match="final decision CSV"):
        finalize_decisions(store, manifest, tmp_path / "review")

    assert store.state.export_history == ()


def test_reopened_csv_requires_the_exact_biological_case_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full-schema CSV with a substituted biological key is not recordable."""
    store, manifest = completed_review_store(tmp_path)
    original = review_exports._atomic_csv

    def substitute_case(path: Path, columns, rows) -> None:
        materialized = [dict(row) for row in rows]
        if path.name == "qc-decisions.csv":
            materialized[0]["cohort"] = "unknown"
        original(path, columns, materialized)

    monkeypatch.setattr(review_exports, "_atomic_csv", substitute_case)
    with pytest.raises(ValueError, match="40 unique cases"):
        finalize_decisions(store, manifest, tmp_path / "review")

    assert store.state.export_history == ()


def test_state_rejects_forged_export_history(tmp_path: Path) -> None:
    """Persisted export records require exact SHA-256 and UTC timestamp fields."""
    store, manifest = completed_review_store(tmp_path)
    finalize_decisions(store, manifest, tmp_path / "review")
    payload = json.loads((tmp_path / "review/review-state.json").read_text())
    payload["export_history"][0]["sha256"] = "not-a-digest"
    for name in ("review-state.json", "review-state.previous.json"):
        (tmp_path / "review" / name).write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="review state is malformed"):
        ReviewStore.open(tmp_path / "review", manifest, "Reviewer", now=fixed_clock)
