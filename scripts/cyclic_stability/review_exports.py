"""Atomic remediation, consistency, and final target-blind review exports."""

from __future__ import annotations

import csv
import hashlib
import os
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from cyclic_stability.review_inventory import OVERLAY_NAME, read_revision_cases
from cyclic_stability.review_models import (
    FINAL_STATUSES,
    REASON_TOKENS,
    Adjudication,
    Assessment,
    CaseHistory,
    CaseRevision,
    EvidenceIdentity,
    Presentation,
    ReviewManifest,
    ReviewState,
)
from cyclic_stability.review_store import ReviewStore

REMEDIATION_COLUMNS = (
    "cohort",
    "cell_id",
    "acquisition_round",
    "revision_id",
    "reasons",
    "notes",
    "reviewer",
    "reviewed_at",
)
CONSISTENCY_COLUMNS = (
    "base_opaque_id",
    "repeat_opaque_id",
    "base_status",
    "repeat_status",
    "agreement",
    "adjudicated_decision",
)
FINAL_COLUMNS = (
    "cohort",
    "cell_id",
    "acquisition_round",
    "decision",
    "reasons",
    "reviewer",
    "reviewed_at",
)
_EXPECTED_CASE_KEYS = frozenset(
    (cohort, f"Cell{cell_number}", acquisition_round)
    for cohort in ("golgi", "lamin")
    for cell_number in range(1, 6)
    for acquisition_round in (1, 3, 4, 5)
)


@dataclass(frozen=True, slots=True)
class _Outcome:
    decision: str
    reasons: tuple[str, ...]
    reviewer: str
    reviewed_at: datetime


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_csv(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Durably replace one CSV with exact columns and no partial visibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, content: bytes) -> None:
    """Durably replace one byte-for-byte backup file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _latest_assessment(state: ReviewState, opaque_id: str) -> Assessment | None:
    history = state.assessment_history.get(opaque_id, ())
    return history[-1] if history else None


def _latest_adjudication(history: CaseHistory, revision_id: str) -> Adjudication | None:
    return next(
        (
            item
            for item in reversed(history.adjudications)
            if item.revision_id == revision_id
        ),
        None,
    )


def _presentations_for_revision(
    manifest: ReviewManifest, revision: CaseRevision
) -> tuple[Presentation, ...]:
    return tuple(
        item
        for item in manifest.visual_presentations
        if item.case_token == revision.case_token
        and item.revision_id == revision.revision_id
    )


def _validated_active_manifest(
    state: ReviewState, manifest: ReviewManifest
) -> tuple[ReviewState, ReviewManifest]:
    validated = ReviewState.from_dict(state.to_dict(), manifest)
    return validated, validated.active_manifest(manifest)


def _remediation_rows(
    state: ReviewState, active: ReviewManifest
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for history in sorted(
        state.case_histories.values(), key=lambda item: item.case_key
    ):
        revision = history.revisions[-1]
        pending = next(
            (
                assessment
                for presentation in _presentations_for_revision(active, revision)
                if (assessment := _latest_assessment(state, presentation.opaque_id))
                is not None
                and assessment.status == "needs-registration-improvement"
            ),
            None,
        )
        if pending is None:
            continue
        rows.append(
            {
                "cohort": history.case_key.cohort,
                "cell_id": history.case_key.cell_id,
                "acquisition_round": history.case_key.acquisition_round,
                "revision_id": revision.revision_id,
                "reasons": ";".join(pending.reasons),
                "notes": pending.notes,
                "reviewer": pending.reviewer,
                "reviewed_at": pending.assessed_at.isoformat(),
            }
        )
    return rows


def _consistency_rows(
    state: ReviewState, active: ReviewManifest
) -> list[dict[str, object]]:
    presentations = {item.opaque_id: item for item in active.visual_presentations}
    rows: list[dict[str, object]] = []
    repeats = sorted(
        (item for item in active.visual_presentations if item.repeat_of is not None),
        key=lambda item: item.opaque_id,
    )
    for repeat in repeats:
        assert repeat.repeat_of is not None
        base = presentations[repeat.repeat_of]
        base_assessment = _latest_assessment(state, base.opaque_id)
        repeat_assessment = _latest_assessment(state, repeat.opaque_id)
        statuses = (
            base_assessment.status if base_assessment else "",
            repeat_assessment.status if repeat_assessment else "",
        )
        agreement = (
            ""
            if not all(statuses)
            else "agree"
            if statuses[0] == statuses[1]
            else "disagree"
        )
        history = state.case_histories[repeat.case_token]
        adjudication = _latest_adjudication(history, repeat.revision_id)
        rows.append(
            {
                "base_opaque_id": base.opaque_id,
                "repeat_opaque_id": repeat.opaque_id,
                "base_status": statuses[0],
                "repeat_status": statuses[1],
                "agreement": agreement,
                "adjudicated_decision": (
                    adjudication.status if adjudication is not None else ""
                ),
            }
        )
    return rows


def write_operational_exports(
    state: ReviewState, manifest: ReviewManifest, review_dir: Path
) -> tuple[Path, Path]:
    """Atomically write the current remediation queue and repeat report."""
    validated, active = _validated_active_manifest(state, manifest)
    remediation = review_dir / "remediation-requests.csv"
    consistency = review_dir / "consistency-report.csv"
    _atomic_csv(remediation, REMEDIATION_COLUMNS, _remediation_rows(validated, active))
    _atomic_csv(consistency, CONSISTENCY_COLUMNS, _consistency_rows(validated, active))
    return remediation, consistency


def _revision_opaque_id(seed: int, revision: CaseRevision) -> str:
    digest = hashlib.sha256(
        "\0".join((str(seed), revision.revision_id, revision.case_token)).encode(
            "utf-8"
        )
    ).hexdigest()
    return f"QC-R-{digest[:16]}"


def _revision_evidence_signature(revision: CaseRevision) -> tuple[object, ...]:
    """Return exact content identities and technical values for change detection."""
    return (
        tuple(
            (
                item.channel,
                item.stage,
                item.view,
                item.identity.sha256,
            )
            for item in revision.overlays
        ),
        revision.automated_valid,
        revision.automated_reason,
        revision.deformation_flags,
    )


def _pending_case_histories(
    state: ReviewState, active: ReviewManifest
) -> tuple[CaseHistory, ...]:
    pending: list[CaseHistory] = []
    for history in state.case_histories.values():
        revision = history.revisions[-1]
        if any(
            assessment is not None
            and assessment.status == "needs-registration-improvement"
            for presentation in _presentations_for_revision(active, revision)
            if (assessment := _latest_assessment(state, presentation.opaque_id))
            is not None
        ):
            pending.append(history)
    return tuple(sorted(pending, key=lambda item: item.case_key))


def attach_revision_root(store: ReviewStore, revision_root: Path) -> ReviewState:
    """Attach changed evidence for every currently unresolved remediation case."""
    state = store.validate_current_bindings()
    active = store.manifest
    pending = _pending_case_histories(state, active)
    if not pending:
        raise ValueError("no cases await registration improvement")
    discovered = read_revision_cases(revision_root)
    revisions: list[CaseRevision] = []
    presentations: list[Presentation] = []
    for history in pending:
        candidate = discovered.get(history.case_key)
        if candidate is None or candidate.case_key != history.case_key:
            raise ValueError("revision case identity changed or is missing")
        current = history.revisions[-1]
        if _revision_evidence_signature(candidate) == _revision_evidence_signature(
            current
        ):
            raise ValueError(
                f"revision for {history.case_key} requires a different evidence identity"
            )
        candidate = replace(candidate, case_token=history.case_token)
        presentation = Presentation(
            _revision_opaque_id(active.seed, candidate),
            candidate.case_token,
            candidate.revision_id,
            candidate.overlays,
            None,
        )
        revisions.append(candidate)
        presentations.append(presentation)
    result = store.attach_revisions(tuple(revisions), tuple(presentations))
    return result


def _validate_current_evidence(active: ReviewManifest) -> None:
    seen: set[EvidenceIdentity] = set()
    for revision in active.case_revisions:
        for overlay in revision.overlays:
            identity = overlay.identity
            if identity in seen:
                continue
            seen.add(identity)
            path = Path(identity.path)
            if OVERLAY_NAME.fullmatch(path.name) is None:
                raise ValueError(
                    "reviewed evidence basename is outside target-blind allowlist"
                )
            try:
                current = EvidenceIdentity.from_path(path)
            except OSError as error:
                raise ValueError("reviewed evidence identity changed") from error
            if current != identity:
                raise ValueError("reviewed evidence identity changed")


def _case_outcome(
    state: ReviewState,
    active: ReviewManifest,
    history: CaseHistory,
) -> _Outcome:
    revision = history.revisions[-1]
    adjudication = _latest_adjudication(history, revision.revision_id)
    presentations = _presentations_for_revision(active, revision)
    if not presentations:
        if (
            revision.overlays
            or revision.automated_valid
            or revision.automated_reason != "invalid-affine-reflection"
            or adjudication is None
            or adjudication.status != "exclude"
        ):
            raise ValueError("hard-invalid technical case requires explicit exclusion")
        return _Outcome(
            adjudication.status,
            adjudication.reasons,
            adjudication.reviewer,
            adjudication.adjudicated_at,
        )
    assessments = tuple(
        _latest_assessment(state, presentation.opaque_id)
        for presentation in presentations
    )
    if any(item is None for item in assessments):
        raise ValueError("original case lacks a final assessment")
    complete = tuple(item for item in assessments if item is not None)
    if any(item.status == "needs-registration-improvement" for item in complete):
        raise ValueError("registration improvement remains unresolved")
    statuses = {item.status for item in complete}
    needs_adjudication = "needs-adjudication" in statuses
    final_statuses = statuses & FINAL_STATUSES
    if len(final_statuses) > 1:
        if adjudication is None:
            raise ValueError("concealed repeat disagreement remains unresolved")
    elif needs_adjudication and adjudication is None:
        raise ValueError("adjudication remains unresolved")
    elif not final_statuses and adjudication is None:
        raise ValueError("original case lacks a final decision")
    if adjudication is not None:
        return _Outcome(
            adjudication.status,
            adjudication.reasons,
            adjudication.reviewer,
            adjudication.adjudicated_at,
        )
    base = next(
        (
            assessment
            for presentation, assessment in zip(presentations, complete)
            if presentation.repeat_of is None
        ),
        complete[0],
    )
    return _Outcome(base.status, base.reasons, base.reviewer, base.assessed_at)


def _final_rows(state: ReviewState, active: ReviewManifest) -> list[dict[str, object]]:
    histories = tuple(
        sorted(state.case_histories.values(), key=lambda item: item.case_key)
    )
    if len(histories) != 40 or len({item.case_key for item in histories}) != 40:
        raise ValueError("review must contain exactly 40 original cases")
    presentations = {item.opaque_id: item for item in active.visual_presentations}
    if any(
        assessment_history[-1].status == "needs-registration-improvement"
        and assessment_history[-1].revision_id
        == state.case_histories[presentations[opaque_id].case_token]
        .revisions[-1]
        .revision_id
        for opaque_id, assessment_history in state.assessment_history.items()
        if assessment_history
    ):
        raise ValueError("registration improvement remains unresolved")
    rows: list[dict[str, object]] = []
    for history in histories:
        outcome = _case_outcome(state, active, history)
        rows.append(
            {
                "cohort": history.case_key.cohort,
                "cell_id": history.case_key.cell_id,
                "acquisition_round": history.case_key.acquisition_round,
                "decision": outcome.decision,
                "reasons": ";".join(outcome.reasons),
                "reviewer": outcome.reviewer,
                "reviewed_at": outcome.reviewed_at.isoformat(),
            }
        )
    return rows


def _read_final_csv(
    path: Path, expected: Sequence[Mapping[str, object]] | None = None
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FINAL_COLUMNS:
                raise ValueError("final decision CSV has an invalid schema")
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise ValueError("final decision CSV could not be reopened") from error
    keys: set[tuple[str, str, int]] = set()
    for row in rows:
        if set(row) != set(FINAL_COLUMNS) or None in row.values():
            raise ValueError("final decision CSV contains malformed rows")
        try:
            key = (row["cohort"], row["cell_id"], int(row["acquisition_round"]))
            timestamp = datetime.fromisoformat(row["reviewed_at"])
        except (ValueError, TypeError) as error:
            raise ValueError("final decision CSV contains malformed rows") from error
        reasons = row["reasons"].split(";") if row["reasons"] else []
        if (
            row["decision"] not in FINAL_STATUSES
            or not row["reviewer"]
            or timestamp.tzinfo is None
            or timestamp.utcoffset() != UTC.utcoffset(timestamp)
            or not reasons
            or len(reasons) != len(set(reasons))
            or not set(reasons) <= REASON_TOKENS
            or key in keys
        ):
            raise ValueError("final decision CSV contains invalid decisions")
        keys.add(key)
    if keys != _EXPECTED_CASE_KEYS:
        raise ValueError("final decision CSV must contain 40 unique cases")
    if expected is not None:
        expected_rows = [
            {column: str(row[column]) for column in FINAL_COLUMNS} for row in expected
        ]
        if rows != expected_rows:
            raise ValueError("final decision CSV does not match finalized decisions")
    return rows


def finalize_decisions(
    store: ReviewStore, manifest: ReviewManifest, review_dir: Path
) -> Path:
    """Validate all current evidence and atomically publish final decisions."""
    if review_dir.resolve() != store.review_dir:
        raise ValueError("final output must remain in the owned review directory")
    state = store.validate_current_bindings(manifest)
    active = state.active_manifest(manifest)
    _validate_current_evidence(active)
    rows = _final_rows(state, active)
    write_operational_exports(state, manifest, review_dir)
    output = review_dir / "qc-decisions.csv"
    previous = review_dir / "qc-decisions.previous.csv"
    if output.exists():
        _read_final_csv(output)
        _atomic_bytes(previous, output.read_bytes())
    _atomic_csv(output, FINAL_COLUMNS, rows)
    _read_final_csv(output, rows)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    store.record_export(digest)
    return output


__all__ = [
    "CONSISTENCY_COLUMNS",
    "FINAL_COLUMNS",
    "REMEDIATION_COLUMNS",
    "attach_revision_root",
    "finalize_decisions",
    "write_operational_exports",
]
