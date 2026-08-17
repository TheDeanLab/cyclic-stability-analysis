"""Crash-safe, append-preserving persistence for target-blind QC review state."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from cyclic_stability.review_models import (
    Adjudication,
    Assessment,
    AssessmentStatus,
    CaseHistory,
    CaseKey,
    Presentation,
    RecoveryEvent,
    RepositoryIdentity,
    ReviewManifest,
    ReviewState,
)

_CURRENT_NAME = "review-state.json"
_PREVIOUS_NAME = "review-state.previous.json"
_MANIFEST_NAME = "randomization-manifest.json"


def _pretty_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Return canonical, human-readable JSON bytes with a final newline."""
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _atomic_pretty_json(path: Path, payload: Mapping[str, object]) -> None:
    """Durably publish one canonical JSON payload without partial replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(_pretty_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _repository_identity() -> RepositoryIdentity:
    """Return the current commit plus the exact binary diff from that commit."""
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = (
            subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=repository,
                check=True,
                capture_output=True,
            )
            .stdout.decode("ascii")
            .strip()
        )
        diff = subprocess.run(
            ("git", "diff", "--binary", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as error:
        raise ValueError("could not determine repository identity") from error
    return RepositoryIdentity(
        commit,
        bool(diff),
        hashlib.sha256(diff).hexdigest() if diff else None,
    )


def _read_payload(path: Path) -> Mapping[str, object]:
    """Read one JSON-object payload, preserving strict schema validation upstream."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("review state is malformed") from error
    if not isinstance(payload, Mapping):
        raise TypeError("review state JSON must be an object")
    return payload


class ReviewStore:
    """Own one durable review state and append-only assessment transitions."""

    def __init__(
        self,
        review_dir: Path,
        manifest: ReviewManifest,
        reviewer: str,
        now: Callable[[], datetime],
        state: ReviewState,
    ) -> None:
        self._review_dir = review_dir
        self._manifest = manifest
        self._reviewer = reviewer
        self._now = now
        self._state = state

    @property
    def state(self) -> ReviewState:
        """Return the immutable latest state successfully committed to disk."""
        return self._state

    @classmethod
    def create(
        cls,
        review_dir: Path,
        manifest: ReviewManifest,
        reviewer: str,
        *,
        now: Callable[[], datetime],
    ) -> ReviewStore:
        """Initialize a new review directory bound to one immutable manifest."""
        if not isinstance(reviewer, str) or not reviewer:
            raise ValueError("reviewer must be a nonempty string")
        current = review_dir / _CURRENT_NAME
        previous = review_dir / _PREVIOUS_NAME
        if current.exists() or previous.exists():
            raise ValueError("review directory already contains review state")
        created_at = _utc_now(now)
        histories = {
            revision.case_token: CaseHistory(
                revision.case_key,
                revision.case_token,
                (revision,),
                (),
            )
            for revision in manifest.case_revisions
        }
        state = ReviewState(
            1,
            1,
            manifest.manifest_identity,
            reviewer,
            _repository_identity(),
            created_at,
            {},
            histories,
            (),
        )
        _atomic_pretty_json(review_dir / _MANIFEST_NAME, manifest.to_dict())
        _atomic_pretty_json(current, state.to_dict())
        return cls(review_dir, manifest, reviewer, now, state)

    @classmethod
    def open(
        cls,
        review_dir: Path,
        manifest: ReviewManifest,
        reviewer: str,
        *,
        now: Callable[[], datetime],
    ) -> ReviewStore:
        """Open a strictly validated state, recovering only from its last snapshot."""
        if not isinstance(reviewer, str) or not reviewer:
            raise ValueError("reviewer must be a nonempty string")
        current_path = review_dir / _CURRENT_NAME
        previous_path = review_dir / _PREVIOUS_NAME
        current_state, current_error = _load_state(current_path, manifest)
        previous_state, previous_error = _load_state(previous_path, manifest)
        recovered = False
        if current_state is not None:
            state = current_state
        elif previous_state is not None:
            state = previous_state
            recovered = True
        elif current_error is None and previous_error is None:
            raise ValueError("review state is missing")
        elif current_error is not None and str(current_error) == (
            "reviewed evidence identity changed"
        ):
            raise current_error
        elif previous_error is not None and str(previous_error) == (
            "reviewed evidence identity changed"
        ):
            raise previous_error
        else:
            raise ValueError("review state is malformed")
        if state.reviewer != reviewer:
            raise ValueError("reviewer does not match persisted review state")
        if state.repository_identity != _repository_identity():
            raise ValueError("repository identity changed since review creation")
        if recovered:
            state = replace(
                state,
                recovery_events=state.recovery_events
                + (
                    RecoveryEvent(
                        _utc_now(now),
                        "recovered previous valid state after malformed or absent current state",
                    ),
                ),
            )
            _atomic_pretty_json(current_path, state.to_dict())
        return cls(review_dir, manifest, reviewer, now, state)

    def assess(
        self,
        opaque_id: str,
        status: AssessmentStatus,
        reasons: tuple[str, ...],
        notes: str,
    ) -> ReviewState:
        """Append and durably persist one blinded assessment for an opaque ID."""
        presentation = self._presentations().get(opaque_id)
        if presentation is None:
            raise ValueError("assessment must use a known opaque presentation ID")
        assessment = Assessment(
            uuid.uuid4().hex,
            opaque_id,
            presentation.revision_id,
            status,
            reasons,
            notes,
            self._reviewer,
            _utc_now(self._now),
        )
        history = dict(self._state.assessment_history)
        history[opaque_id] = history.get(opaque_id, ()) + (assessment,)
        state = replace(self._state, assessment_history=history)
        self._persist(state)
        return state

    def adjudicate(
        self,
        case_key: CaseKey,
        decision: Literal["accept", "exclude"],
        reasons: tuple[str, ...],
        notes: str,
    ) -> ReviewState:
        """Append a final reconciliation event after at least one visual assessment."""
        if not isinstance(case_key, CaseKey):
            raise TypeError("adjudication requires a CaseKey")
        case_token = next(
            (
                token
                for token, history in self._state.case_histories.items()
                if history.case_key == case_key
            ),
            None,
        )
        if case_token is None:
            raise ValueError("adjudication case is absent from immutable manifest")
        if not any(
            self._manifest.presentation_cases[opaque_id] == case_key
            for opaque_id in self._state.assessment_history
        ):
            raise ValueError("adjudication requires a visual assessment")
        case_history = self._state.case_histories[case_token]
        adjudication = Adjudication(
            uuid.uuid4().hex,
            case_key,
            case_history.revisions[-1].revision_id,
            decision,
            reasons,
            notes,
            self._reviewer,
            _utc_now(self._now),
        )
        histories = dict(self._state.case_histories)
        histories[case_token] = replace(
            case_history,
            adjudications=case_history.adjudications + (adjudication,),
        )
        state = replace(self._state, case_histories=histories)
        self._persist(state)
        return state

    def _persist(self, state: ReviewState) -> None:
        """Write the backup before promotion and update memory only after success."""
        current_path = self._review_dir / _CURRENT_NAME
        if current_path.exists():
            current_payload = _read_payload(current_path)
            ReviewState.from_dict(current_payload, self._manifest)
            _atomic_pretty_json(self._review_dir / _PREVIOUS_NAME, current_payload)
        _atomic_pretty_json(current_path, state.to_dict())
        self._state = state

    def _presentations(self) -> dict[str, Presentation]:
        """Return the immutable opaque presentation lookup without biological keys."""
        return {
            presentation.opaque_id: presentation
            for presentation in self._manifest.visual_presentations
        }


def _utc_now(now: Callable[[], datetime]) -> datetime:
    """Read and validate one UTC clock value at the persistence boundary."""
    timestamp = now()
    if not isinstance(timestamp, datetime):
        raise TypeError("review clock must return a datetime")
    if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
        raise ValueError("review clock must return a UTC timestamp")
    return timestamp


def _load_state(
    path: Path, manifest: ReviewManifest
) -> tuple[ReviewState | None, ValueError | None]:
    """Independently load a state candidate, retaining its strict parse error."""
    if not path.exists():
        return None, None
    try:
        return ReviewState.from_dict(_read_payload(path), manifest), None
    except TypeError:
        return None, ValueError("review state is malformed")
    except ValueError as error:
        if str(error) == "reviewed evidence identity changed":
            return None, error
        return None, ValueError("review state is malformed")


__all__ = ["ReviewStore"]
