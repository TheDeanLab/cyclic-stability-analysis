"""Crash-safe, append-preserving persistence for target-blind QC review state."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from cyclic_stability.review_models import (
    Adjudication,
    Assessment,
    AssessmentStatus,
    CaseHistory,
    CaseKey,
    CaseRevision,
    ExportRecord,
    Presentation,
    RecoveryEvent,
    RepositoryIdentity,
    ReviewManifest,
    ReviewState,
    ReviewStateBindingError,
)

_CURRENT_NAME = "review-state.json"
_PREVIOUS_NAME = "review-state.previous.json"
_MANIFEST_NAME = "randomization-manifest.json"


class _Digest(Protocol):
    """Minimal hash interface used to construct repository identity bytes."""

    def update(self, data: bytes, /) -> None:
        """Append bytes to the digest input."""


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
    """Return commit plus tracked-diff and untracked-content dirty evidence."""
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
        untracked_paths = tuple(
            sorted(
                path
                for path in subprocess.run(
                    ("git", "ls-files", "--others", "--exclude-standard", "-z"),
                    cwd=repository,
                    check=True,
                    capture_output=True,
                ).stdout.split(b"\0")
                if path
            )
        )
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as error:
        raise ValueError("could not determine repository identity") from error
    digest = hashlib.sha256()
    _digest_component(digest, b"tracked-diff", diff)
    for relative_path in untracked_paths:
        try:
            path = repository / os.fsdecode(relative_path)
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                kind = b"symlink"
                content = os.fsencode(os.readlink(path))
            elif stat.S_ISREG(mode):
                kind = b"regular-file"
                content = path.read_bytes()
            else:
                raise ValueError(
                    "untracked repository path is not a regular file or symlink"
                )
        except OSError as error:
            raise ValueError("could not read untracked repository content") from error
        _digest_component(digest, b"untracked-kind", kind)
        _digest_component(digest, b"untracked-path", relative_path)
        _digest_component(digest, b"untracked-content", content)
    dirty = bool(diff or untracked_paths)
    return RepositoryIdentity(
        commit,
        dirty,
        digest.hexdigest() if dirty else None,
    )


def _digest_component(digest: _Digest, label: bytes, value: bytes) -> None:
    """Append one length-delimited dirty-tree identity component."""
    digest.update(label)
    digest.update(b"\0")
    digest.update(len(value).to_bytes(8, byteorder="big"))
    digest.update(value)


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
        self._initial_manifest = manifest
        self._manifest = state.active_manifest(manifest)
        self._reviewer = reviewer
        self._now = now
        self._state = state

    @property
    def state(self) -> ReviewState:
        """Return the immutable latest state successfully committed to disk."""
        return self._state

    @property
    def manifest(self) -> ReviewManifest:
        """Return the exact active manifest validated against persisted state."""
        return self._manifest

    @property
    def review_dir(self) -> Path:
        """Return the resolved directory owned by this review lifecycle."""
        return self._review_dir.resolve()

    def validate_current_bindings(
        self, manifest: ReviewManifest | None = None
    ) -> ReviewState:
        """Revalidate manifest, reviewer, repository, and active-state bindings."""
        if manifest is None:
            manifest = self._initial_manifest
        if manifest != self._initial_manifest:
            raise ValueError("initial review manifest does not match review store")
        state = ReviewState.from_dict(self._state.to_dict(), manifest)
        if state.reviewer != self._reviewer:
            raise ValueError("reviewer does not match persisted review state")
        if state.repository_identity != _repository_identity():
            raise ValueError("repository identity changed since review creation")
        if state.active_manifest(manifest) != self._manifest:
            raise ValueError("active review manifest does not match persisted state")
        return state

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
            manifest.manifest_identity,
            reviewer,
            _repository_identity(),
            created_at,
            {},
            histories,
            (),
            (),
            (),
        )
        _atomic_pretty_json(review_dir / _MANIFEST_NAME, manifest.to_dict())
        _atomic_pretty_json(current, state.to_dict())
        store = cls(review_dir, manifest, reviewer, now, state)
        store._write_operational_exports()
        return store

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
        if isinstance(current_error, ReviewStateBindingError):
            raise current_error
        if isinstance(previous_error, ReviewStateBindingError):
            raise previous_error
        if current_state is not None:
            state = current_state
        elif previous_state is not None:
            state = previous_state
            recovered = True
        elif current_error is None and previous_error is None:
            raise ValueError("review state is missing")
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
        store = cls(review_dir, manifest, reviewer, now, state)
        store._write_operational_exports()
        return store

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
        case_history = self._state.case_histories[case_token]
        latest_revision = case_history.revisions[-1]
        technical = not latest_revision.overlays and not latest_revision.automated_valid
        if technical:
            if decision != "exclude":
                raise ValueError("hard-invalid technical case must be excluded")
            if reasons != ("invalid-affine-reflection",):
                raise ValueError(
                    "hard-invalid technical exclusion requires invalid-affine-reflection"
                )
        elif not any(
            self._manifest.presentation_cases[opaque_id] == case_key
            and history[-1].revision_id == latest_revision.revision_id
            for opaque_id, history in self._state.assessment_history.items()
            if history
        ):
            raise ValueError("adjudication requires a visual assessment")
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

    def attach_revisions(
        self,
        revisions: tuple[CaseRevision, ...],
        presentations: tuple[Presentation, ...],
    ) -> ReviewState:
        """Atomically append validated remediation revisions and presentations."""
        if not revisions or len(revisions) != len(presentations):
            raise ValueError("revision attachment must contain paired revisions")
        ordered = tuple(
            sorted(
                zip(revisions, presentations),
                key=lambda pair: pair[0].case_token,
            )
        )
        revisions = tuple(pair[0] for pair in ordered)
        presentations = tuple(pair[1] for pair in ordered)
        if len({item.case_token for item in revisions}) != len(revisions):
            raise ValueError("revision attachment contains duplicate cases")
        pairs = dict(zip((item.case_token for item in revisions), presentations))
        histories = dict(self._state.case_histories)
        for revision in revisions:
            history = histories.get(revision.case_token)
            presentation = pairs[revision.case_token]
            if history is None or revision.case_key != history.case_key:
                raise ValueError("revision case identity changed")
            if revision.revision_id in {item.revision_id for item in history.revisions}:
                raise ValueError("revision evidence identity is not new")
            if (
                presentation.case_token != revision.case_token
                or presentation.revision_id != revision.revision_id
                or presentation.overlays != revision.overlays
                or presentation.repeat_of is not None
            ):
                raise ValueError("revision presentation does not match evidence")
            pending = any(
                assessment_history[-1].status == "needs-registration-improvement"
                and assessment_history[-1].revision_id
                == history.revisions[-1].revision_id
                for opaque_id, assessment_history in self._state.assessment_history.items()
                if assessment_history
                and self._manifest.presentation_cases[opaque_id] == history.case_key
            )
            if not pending:
                raise ValueError(
                    "revision case is not awaiting registration improvement"
                )
            histories[revision.case_token] = replace(
                history, revisions=history.revisions + (revision,)
            )
        attached_presentations = self._state.attached_presentations + presentations
        cases = dict(self._initial_manifest.presentation_cases)
        cases.update(
            {
                presentation.opaque_id: histories[presentation.case_token].case_key
                for presentation in attached_presentations
            }
        )
        appended_revisions = tuple(
            revision
            for token in sorted(histories)
            for revision in histories[token].revisions[1:]
        )
        active = ReviewManifest(
            self._initial_manifest.seed,
            self._initial_manifest.case_revisions + appended_revisions,
            self._initial_manifest.visual_presentations + attached_presentations,
            cases,
        )
        state = replace(
            self._state,
            manifest_identity=active.manifest_identity,
            case_histories=histories,
            attached_presentations=attached_presentations,
        )
        ReviewState.from_dict(state.to_dict(), self._initial_manifest)
        self._persist(state)
        return state

    def record_export(self, sha256: str) -> ReviewState:
        """Append one validated final-export identity and persist it durably."""
        record = ExportRecord(sha256, _utc_now(self._now))
        state = replace(
            self._state, export_history=self._state.export_history + (record,)
        )
        self._persist(state)
        return state

    def _persist(self, state: ReviewState) -> None:
        """Write the backup before promotion and update memory only after success."""
        current_path = self._review_dir / _CURRENT_NAME
        if current_path.exists():
            current_payload = _read_payload(current_path)
            ReviewState.from_dict(current_payload, self._initial_manifest)
            _atomic_pretty_json(self._review_dir / _PREVIOUS_NAME, current_payload)
        _atomic_pretty_json(current_path, state.to_dict())
        self._state = state
        self._manifest = state.active_manifest(self._initial_manifest)
        self._write_operational_exports()

    def _write_operational_exports(self) -> None:
        """Synchronize deterministic derived queues after a committed state change."""
        from cyclic_stability.review_exports import write_operational_exports

        write_operational_exports(self._state, self._initial_manifest, self._review_dir)

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
    except ReviewStateBindingError as error:
        return None, error
    except TypeError:
        return None, ValueError("review state is malformed")
    except ValueError:
        return None, ValueError("review state is malformed")


__all__ = ["ReviewStore"]
