"""Persistent target-blind review-state contract tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import tifffile
from cyclic_stability import review_store
from cyclic_stability.review_inventory import build_review_manifest
from cyclic_stability.review_models import RepositoryIdentity, ReviewState
from cyclic_stability.review_store import ReviewStore
from synthetic import write_review_qc_fixture


def fixed_clock() -> datetime:
    """Return the deterministic UTC time used by review-store tests."""
    return datetime(2026, 8, 17, 12, 0, 0, tzinfo=UTC)


def test_review_state_is_pretty_atomic_and_resumable(tmp_path: Path) -> None:
    """An assessment survives a strict reopen in readable pretty JSON."""
    manifest = build_review_manifest(write_review_qc_fixture(tmp_path / "qc"), seed=7)
    store = ReviewStore.create(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    opaque_id = manifest.visual_presentations[0].opaque_id

    store.assess(opaque_id, "accept", ("dapi-aligned", "tubulin-aligned"), "clear")

    text = (tmp_path / "review/review-state.json").read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert '\n  "assessment_history":' in text
    payload = json.loads(text)
    assert payload["reviewer"] == "Reviewer"
    assert payload["schema_version"] == 1
    assert payload["repository_identity"]["commit"]
    if payload["repository_identity"]["dirty"]:
        assert payload["repository_identity"]["diff_sha256"]
    reopened = ReviewStore.open(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    assert reopened.state.assessment_history[opaque_id][-1].status == "accept"


def test_changed_overlay_identity_blocks_resume(tmp_path: Path) -> None:
    """A reviewed overlay cannot be silently substituted during resume."""
    qc_root = write_review_qc_fixture(tmp_path / "qc")
    manifest = build_review_manifest(qc_root, seed=7)
    ReviewStore.create(tmp_path / "review", manifest, "Reviewer", now=fixed_clock)
    tifffile.imwrite(
        Path(manifest.visual_presentations[0].overlays[0].path),
        np.full((8, 12, 3), 255, dtype=np.uint8),
        photometric="rgb",
    )
    changed = build_review_manifest(qc_root, seed=7)

    with pytest.raises(ValueError, match="reviewed evidence identity changed"):
        ReviewStore.open(tmp_path / "review", changed, "Reviewer", now=fixed_clock)


def test_assessment_failure_before_promotion_keeps_last_valid_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted save cannot expose an in-memory-only assessment."""
    manifest = build_review_manifest(write_review_qc_fixture(tmp_path / "qc"), seed=7)
    store = ReviewStore.create(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    current = tmp_path / "review/review-state.json"
    before = current.read_bytes()
    original = review_store._atomic_pretty_json

    def fail_current(path: Path, payload: dict[str, object]) -> None:
        if path == current:
            raise OSError("injected interruption before promotion")
        original(path, payload)

    monkeypatch.setattr(review_store, "_atomic_pretty_json", fail_current)
    with pytest.raises(OSError, match="injected interruption"):
        store.assess(
            manifest.visual_presentations[0].opaque_id,
            "accept",
            ("dapi-aligned",),
            "clear",
        )

    assert current.read_bytes() == before
    reopened = ReviewStore.open(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    assert reopened.state.assessment_history == {}


def test_open_recovers_previous_valid_state_after_malformed_current(
    tmp_path: Path,
) -> None:
    """Recovery uses the complete previous snapshot rather than partial JSON."""
    manifest = build_review_manifest(write_review_qc_fixture(tmp_path / "qc"), seed=7)
    store = ReviewStore.create(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    opaque_id = manifest.visual_presentations[0].opaque_id
    store.assess(opaque_id, "accept", ("dapi-aligned",), "clear")
    (tmp_path / "review/review-state.json").write_text("{ malformed", encoding="utf-8")

    reopened = ReviewStore.open(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )

    assert reopened.state.assessment_history == {}
    assert len(reopened.state.recovery_events) == 1


def test_open_rejects_when_current_and_previous_are_malformed(tmp_path: Path) -> None:
    """Two malformed snapshots fail closed rather than merging untrusted state."""
    manifest = build_review_manifest(write_review_qc_fixture(tmp_path / "qc"), seed=7)
    store = ReviewStore.create(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    store.assess(
        manifest.visual_presentations[0].opaque_id,
        "accept",
        ("dapi-aligned",),
        "clear",
    )
    for name in ("review-state.json", "review-state.previous.json"):
        (tmp_path / "review" / name).write_text("not JSON", encoding="utf-8")

    with pytest.raises(ValueError, match="review state is malformed"):
        ReviewStore.open(tmp_path / "review", manifest, "Reviewer", now=fixed_clock)


def test_open_rejects_unknown_state_keys(tmp_path: Path) -> None:
    """Unexpected serialized fields cannot change review-state semantics."""
    manifest = build_review_manifest(write_review_qc_fixture(tmp_path / "qc"), seed=7)
    ReviewStore.create(tmp_path / "review", manifest, "Reviewer", now=fixed_clock)
    current = tmp_path / "review/review-state.json"
    payload = json.loads(current.read_text(encoding="utf-8"))
    payload["untrusted"] = True
    current.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="review state is malformed"):
        ReviewStore.open(tmp_path / "review", manifest, "Reviewer", now=fixed_clock)


def test_open_rejects_changed_dirty_repository_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume is refused if the exact dirty-tree diff is no longer identical."""
    manifest = build_review_manifest(write_review_qc_fixture(tmp_path / "qc"), seed=7)
    first = RepositoryIdentity("a" * 40, True, "1" * 64)
    second = RepositoryIdentity("a" * 40, True, "2" * 64)
    monkeypatch.setattr(review_store, "_repository_identity", lambda: first)
    ReviewStore.create(tmp_path / "review", manifest, "Reviewer", now=fixed_clock)
    monkeypatch.setattr(review_store, "_repository_identity", lambda: second)

    with pytest.raises(ValueError, match="repository identity changed"):
        ReviewStore.open(tmp_path / "review", manifest, "Reviewer", now=fixed_clock)


def test_assessments_append_and_adjudication_preserves_visual_history(
    tmp_path: Path,
) -> None:
    """A final case decision adds an event without replacing its visual assessment."""
    manifest = build_review_manifest(write_review_qc_fixture(tmp_path / "qc"), seed=7)
    store = ReviewStore.create(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    presentation = manifest.visual_presentations[0]
    case_key = manifest.presentation_cases[presentation.opaque_id]
    store.assess(
        presentation.opaque_id,
        "needs-adjudication",
        ("z-misalignment",),
        "check repeats",
    )

    state = store.adjudicate(case_key, "exclude", ("z-misalignment",), "confirmed")

    assert (
        state.assessment_history[presentation.opaque_id][0].status
        == "needs-adjudication"
    )
    assert (
        state.case_histories[presentation.case_token].adjudications[-1].status
        == "exclude"
    )


def test_assessment_rejects_nonopaque_ids_and_invalid_reasons(tmp_path: Path) -> None:
    """Only known opaque presentations may receive allowlisted reasoned assessments."""
    manifest = build_review_manifest(write_review_qc_fixture(tmp_path / "qc"), seed=7)
    store = ReviewStore.create(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    presentation = manifest.visual_presentations[0]
    biological_id = manifest.presentation_cases[presentation.opaque_id].cell_id

    with pytest.raises(ValueError, match="opaque"):
        store.assess(biological_id, "accept", ("dapi-aligned",), "clear")
    with pytest.raises(ValueError, match="nonempty"):
        store.assess(presentation.opaque_id, "accept", (), "clear")
    with pytest.raises(ValueError, match="unique"):
        store.assess(
            presentation.opaque_id,
            "accept",
            ("dapi-aligned", "dapi-aligned"),
            "clear",
        )
    with pytest.raises(ValueError, match="allowlisted"):
        store.assess(presentation.opaque_id, "accept", ("invented",), "clear")


def test_adjudication_requires_a_prior_visual_assessment(tmp_path: Path) -> None:
    """Reconciliation cannot create a case outcome without visual evidence."""
    manifest = build_review_manifest(write_review_qc_fixture(tmp_path / "qc"), seed=7)
    store = ReviewStore.create(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    case_key = manifest.presentation_cases[manifest.visual_presentations[0].opaque_id]

    with pytest.raises(ValueError, match="visual assessment"):
        store.adjudicate(case_key, "accept", ("dapi-aligned",), "premature")


def test_open_rejects_reviewer_mismatch(tmp_path: Path) -> None:
    """A review directory has one explicit reviewer identity."""
    manifest = build_review_manifest(write_review_qc_fixture(tmp_path / "qc"), seed=7)
    ReviewStore.create(tmp_path / "review", manifest, "Reviewer", now=fixed_clock)

    with pytest.raises(ValueError, match="reviewer"):
        ReviewStore.open(tmp_path / "review", manifest, "Other", now=fixed_clock)


def test_state_model_rejects_duplicate_event_ids_and_non_utc_timestamps(
    tmp_path: Path,
) -> None:
    """Serialized histories cannot reuse events or accept local-time timestamps."""
    manifest = build_review_manifest(write_review_qc_fixture(tmp_path / "qc"), seed=7)
    store = ReviewStore.create(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    opaque_id = manifest.visual_presentations[0].opaque_id
    store.assess(opaque_id, "accept", ("dapi-aligned",), "first")
    store.assess(opaque_id, "exclude", ("z-misalignment",), "second")
    payload = store.state.to_dict()
    assessments = payload["assessment_history"][opaque_id]
    assessments[1]["event_id"] = assessments[0]["event_id"]

    with pytest.raises(ValueError, match="duplicate event IDs"):
        ReviewState.from_dict(payload, manifest)

    assessments[1]["event_id"] = "different-event"
    assessments[1]["assessed_at"] = "2026-08-17T12:00:00"
    with pytest.raises(ValueError, match="UTC timestamp"):
        ReviewState.from_dict(payload, manifest)


def test_state_model_rejects_biological_history_absent_from_manifest(
    tmp_path: Path,
) -> None:
    """State cannot discard or introduce biological case histories on decode."""
    manifest = build_review_manifest(write_review_qc_fixture(tmp_path / "qc"), seed=7)
    store = ReviewStore.create(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    payload = store.state.to_dict()
    payload["case_histories"].pop(next(iter(payload["case_histories"])))

    with pytest.raises(ValueError, match="biological keys absent"):
        ReviewState.from_dict(payload, manifest)
