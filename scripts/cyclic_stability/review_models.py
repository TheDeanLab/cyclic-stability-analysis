"""Immutable, validated values for the target-blind registration-QC review."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

_COHORTS = frozenset({"golgi", "lamin"})
_ROUNDS = frozenset({1, 3, 4, 5})
_CHANNELS = frozenset({"dapi", "tubulin"})
_STAGES = frozenset({"pre", "affine", "nonlinear"})
_VIEWS = frozenset({"xy", "xz", "yz"})
_OVERLAY_PRODUCT = frozenset(
    (channel, stage, view)
    for channel in _CHANNELS
    for stage in _STAGES
    for view in _VIEWS
)

AssessmentStatus = Literal[
    "accept",
    "needs-registration-improvement",
    "exclude",
    "needs-adjudication",
]
FINAL_STATUSES = frozenset({"accept", "exclude"})
_ASSESSMENT_STATUSES = frozenset(
    {
        "accept",
        "needs-registration-improvement",
        "exclude",
        "needs-adjudication",
    }
)
REASON_TOKENS = frozenset(
    {
        "dapi-aligned",
        "tubulin-aligned",
        "no-interior-artifact",
        "persistent-interior-misalignment",
        "nonlinear-worse-than-affine",
        "seam-artifact",
        "tearing-or-duplication",
        "z-misalignment",
        "identity-mismatch",
        "insufficient-common-support",
        "invalid-affine-reflection",
    }
)


class ReviewStateBindingError(ValueError):
    """A structurally valid state conflicts with immutable review bindings."""


def _mapping(payload: object, keys: frozenset[str], name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise ValueError(f"{name} has an invalid schema")
    return payload


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha256(value: object, name: str) -> str:
    value = _string(value, name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _utc_timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must be a UTC timestamp")
    return value


def _timestamp_from_dict(value: object, name: str) -> datetime:
    text = _string(value, name)
    try:
        return _utc_timestamp(datetime.fromisoformat(text), name)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp") from error


def _reason_tokens(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError("assessment reasons must be nonempty")
    if any(not isinstance(item, str) for item in value):
        raise TypeError("assessment reasons must be strings")
    if len(value) != len(set(value)):
        raise ValueError("assessment reasons must be unique")
    if not set(value) <= REASON_TOKENS:
        raise ValueError("assessment reasons must be allowlisted")
    return value


def _validate_overlay_product(
    overlays: tuple[OverlayEvidence, ...], *, allow_empty: bool
) -> None:
    if allow_empty and not overlays:
        return
    descriptors = {(item.channel, item.stage, item.view) for item in overlays}
    if len(overlays) != 18 or descriptors != _OVERLAY_PRODUCT:
        raise ValueError("overlays must be the exact 2 x 3 x 3 overlay product")


@dataclass(frozen=True, slots=True, order=True)
class CaseKey:
    """One biological cohort, cell, and moving acquisition round."""

    cohort: Literal["golgi", "lamin"]
    cell_id: str
    acquisition_round: Literal[1, 3, 4, 5]

    def __post_init__(self) -> None:
        if self.cohort not in _COHORTS:
            raise ValueError("cohort must be golgi or lamin")
        _string(self.cell_id, "cell_id")
        if self.acquisition_round not in _ROUNDS:
            raise ValueError("acquisition_round must be one of 1, 3, 4, or 5")

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible representation."""
        return {
            "cohort": self.cohort,
            "cell_id": self.cell_id,
            "acquisition_round": self.acquisition_round,
        }

    @classmethod
    def from_dict(cls, payload: object) -> CaseKey:
        """Parse one strictly validated JSON-compatible representation."""
        data = _mapping(
            payload, frozenset({"cohort", "cell_id", "acquisition_round"}), "case key"
        )
        cohort = _string(data["cohort"], "cohort")
        acquisition_round = data["acquisition_round"]
        if not isinstance(acquisition_round, int) or isinstance(
            acquisition_round, bool
        ):
            raise TypeError("acquisition_round must be an integer")
        return cls(cohort, _string(data["cell_id"], "cell_id"), acquisition_round)


@dataclass(frozen=True, slots=True)
class EvidenceIdentity:
    """Resolved path metadata and complete content hash for one evidence file."""

    path: str
    size: int
    mtime_ns: int
    sha256: str

    def __post_init__(self) -> None:
        _string(self.path, "path")
        if self.size < 0 or self.mtime_ns < 0:
            raise ValueError("evidence size and mtime_ns must be nonnegative")
        if len(self.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.sha256
        ):
            raise ValueError("evidence sha256 must be a lowercase SHA-256 digest")

    @classmethod
    def from_path(cls, path: Path) -> EvidenceIdentity:
        """Return a content identity for an existing evidence file."""
        from cyclic_stability.review_evidence import (
            overlay_qc_root,
            read_verified_evidence,
        )

        verified = read_verified_evidence(overlay_qc_root(path), path)
        return cls(
            str(verified.path),
            verified.size,
            verified.mtime_ns,
            verified.sha256,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible representation."""
        return {
            "path": self.path,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: object) -> EvidenceIdentity:
        """Parse one strictly validated JSON-compatible representation."""
        data = _mapping(
            payload,
            frozenset({"path", "size", "mtime_ns", "sha256"}),
            "evidence identity",
        )
        for name in ("size", "mtime_ns"):
            if not isinstance(data[name], int) or isinstance(data[name], bool):
                raise TypeError(f"{name} must be an integer")
        return cls(
            _string(data["path"], "path"),
            data["size"],
            data["mtime_ns"],
            _string(data["sha256"], "sha256"),
        )


@dataclass(frozen=True, slots=True)
class OverlayEvidence:
    """One allowlisted RGB overlay and its immutable identity."""

    channel: Literal["dapi", "tubulin"]
    stage: Literal["pre", "affine", "nonlinear"]
    view: Literal["xy", "xz", "yz"]
    identity: EvidenceIdentity

    def __post_init__(self) -> None:
        if self.channel not in _CHANNELS:
            raise ValueError("overlay channel must be dapi or tubulin")
        if self.stage not in _STAGES:
            raise ValueError("overlay stage is not allowlisted")
        if self.view not in _VIEWS:
            raise ValueError("overlay view is not allowlisted")

    @property
    def path(self) -> str:
        """Return the resolved evidence path without biological identity."""
        return self.identity.path

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible representation."""
        return {
            "channel": self.channel,
            "stage": self.stage,
            "view": self.view,
            "identity": self.identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> OverlayEvidence:
        """Parse one strictly validated JSON-compatible representation."""
        data = _mapping(
            payload, frozenset({"channel", "stage", "view", "identity"}), "overlay"
        )
        return cls(
            _string(data["channel"], "channel"),
            _string(data["stage"], "stage"),
            _string(data["view"], "view"),
            EvidenceIdentity.from_dict(data["identity"]),
        )


@dataclass(frozen=True, slots=True)
class CaseRevision:
    """Immutable automated evidence for one biological case revision."""

    case_key: CaseKey
    case_token: str
    revision_id: str
    overlays: tuple[OverlayEvidence, ...]
    automated_valid: bool
    automated_reason: str
    deformation_flags: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _string(self.case_token, "case_token")
        _string(self.revision_id, "revision_id")
        _validate_overlay_product(self.overlays, allow_empty=True)
        if not isinstance(self.automated_valid, bool):
            raise TypeError("automated_valid must be a boolean")
        if not isinstance(self.automated_reason, str):
            raise TypeError("automated_reason must be a string")
        if tuple(sorted(region for region, _ in self.deformation_flags)) not in (
            (),
            ("interior", "seam", "whole"),
        ):
            raise ValueError("deformation flags must cover whole, seam, and interior")

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible representation."""
        return {
            "case_key": self.case_key.to_dict(),
            "case_token": self.case_token,
            "revision_id": self.revision_id,
            "overlays": [item.to_dict() for item in self.overlays],
            "automated_valid": self.automated_valid,
            "automated_reason": self.automated_reason,
            "deformation_flags": [list(item) for item in self.deformation_flags],
        }

    @classmethod
    def from_dict(cls, payload: object) -> CaseRevision:
        """Parse one strictly validated JSON-compatible representation."""
        data = _mapping(
            payload,
            frozenset(
                {
                    "case_key",
                    "case_token",
                    "revision_id",
                    "overlays",
                    "automated_valid",
                    "automated_reason",
                    "deformation_flags",
                }
            ),
            "case revision",
        )
        if not isinstance(data["overlays"], list) or not isinstance(
            data["deformation_flags"], list
        ):
            raise TypeError("case revision collections must be lists")
        flags: list[tuple[str, str]] = []
        for item in data["deformation_flags"]:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("deformation flag must be a two-item list")
            if not isinstance(item[1], str):
                raise TypeError("warp_flags must be a string")
            flags.append((_string(item[0], "region"), item[1]))
        return cls(
            CaseKey.from_dict(data["case_key"]),
            _string(data["case_token"], "case_token"),
            _string(data["revision_id"], "revision_id"),
            tuple(OverlayEvidence.from_dict(item) for item in data["overlays"]),
            data["automated_valid"]
            if isinstance(data["automated_valid"], bool)
            else _invalid_bool(),
            data["automated_reason"]
            if isinstance(data["automated_reason"], str)
            else _invalid_string(),
            tuple(flags),
        )


def _invalid_bool() -> bool:
    raise TypeError("automated_valid must be a boolean")


def _invalid_string() -> str:
    raise TypeError("automated_reason must be a string")


@dataclass(frozen=True, slots=True)
class Presentation:
    """Opaque reviewer-facing presentation with only allowlisted overlays."""

    opaque_id: str
    case_token: str
    revision_id: str
    overlays: tuple[OverlayEvidence, ...]
    repeat_of: str | None

    def __post_init__(self) -> None:
        _string(self.opaque_id, "opaque_id")
        _string(self.case_token, "case_token")
        _string(self.revision_id, "revision_id")
        _validate_overlay_product(self.overlays, allow_empty=False)
        if self.repeat_of is not None:
            _string(self.repeat_of, "repeat_of")

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible representation."""
        return {
            "opaque_id": self.opaque_id,
            "case_token": self.case_token,
            "revision_id": self.revision_id,
            "overlays": [item.to_dict() for item in self.overlays],
            "repeat_of": self.repeat_of,
        }

    @classmethod
    def from_dict(cls, payload: object) -> Presentation:
        """Parse one strictly validated JSON-compatible representation."""
        data = _mapping(
            payload,
            frozenset(
                {"opaque_id", "case_token", "revision_id", "overlays", "repeat_of"}
            ),
            "presentation",
        )
        if not isinstance(data["overlays"], list):
            raise TypeError("presentation overlays must be a list")
        repeat_of = data["repeat_of"]
        if repeat_of is not None and not isinstance(repeat_of, str):
            raise ValueError("repeat_of must be a string or null")
        return cls(
            _string(data["opaque_id"], "opaque_id"),
            _string(data["case_token"], "case_token"),
            _string(data["revision_id"], "revision_id"),
            tuple(OverlayEvidence.from_dict(item) for item in data["overlays"]),
            repeat_of,
        )


@dataclass(frozen=True, slots=True)
class ReviewManifest:
    """Immutable inventory, randomized visual order, and concealed case mapping."""

    seed: int
    case_revisions: tuple[CaseRevision, ...]
    visual_presentations: tuple[Presentation, ...]
    presentation_cases: Mapping[str, CaseKey]

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        identifiers = tuple(item.opaque_id for item in self.visual_presentations)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("presentation opaque IDs must be unique")
        if set(identifiers) != set(self.presentation_cases):
            raise ValueError("presentation mapping must exactly match presentations")
        revisions: dict[tuple[str, str], CaseRevision] = {}
        token_cases: dict[str, CaseKey] = {}
        for revision in self.case_revisions:
            revision_key = (revision.case_token, revision.revision_id)
            if revision_key in revisions:
                raise ValueError("manifest revision IDs must be unique per case")
            if (
                revision.case_token in token_cases
                and token_cases[revision.case_token] != revision.case_key
            ):
                raise ValueError(
                    "manifest case token maps to multiple biological cases"
                )
            revisions[revision_key] = revision
            token_cases[revision.case_token] = revision.case_key
        presentations = {
            presentation.opaque_id: presentation
            for presentation in self.visual_presentations
        }
        for presentation in self.visual_presentations:
            revision = revisions.get(
                (presentation.case_token, presentation.revision_id)
            )
            if revision is None or presentation.overlays != revision.overlays:
                raise ValueError("presentation revision is absent from manifest")
            if self.presentation_cases[presentation.opaque_id] != revision.case_key:
                raise ValueError("presentation mapping does not match its revision")
            if presentation.repeat_of is not None:
                base = presentations.get(presentation.repeat_of)
                if (
                    base is None
                    or base.repeat_of is not None
                    or base.case_token != presentation.case_token
                    or base.revision_id != presentation.revision_id
                ):
                    raise ValueError(
                        "concealed repeat does not match a base presentation"
                    )
        object.__setattr__(
            self, "presentation_cases", MappingProxyType(dict(self.presentation_cases))
        )

    @property
    def manifest_identity(self) -> str:
        """Return the SHA-256 identity of the entire immutable manifest."""
        return hashlib.sha256(repr(self.to_dict()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible representation."""
        return {
            "seed": self.seed,
            "case_revisions": [item.to_dict() for item in self.case_revisions],
            "visual_presentations": [
                item.to_dict() for item in self.visual_presentations
            ],
            "presentation_cases": {
                opaque_id: case_key.to_dict()
                for opaque_id, case_key in sorted(self.presentation_cases.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: object) -> ReviewManifest:
        """Parse one strictly validated JSON-compatible representation."""
        data = _mapping(
            payload,
            frozenset(
                {"seed", "case_revisions", "visual_presentations", "presentation_cases"}
            ),
            "review manifest",
        )
        if not isinstance(data["seed"], int) or isinstance(data["seed"], bool):
            raise TypeError("seed must be an integer")
        if (
            not isinstance(data["case_revisions"], list)
            or not isinstance(data["visual_presentations"], list)
            or not isinstance(data["presentation_cases"], Mapping)
        ):
            raise TypeError("review manifest collections have invalid types")
        presentation_cases = {
            _string(opaque_id, "opaque_id"): CaseKey.from_dict(case_key)
            for opaque_id, case_key in data["presentation_cases"].items()
        }
        return cls(
            data["seed"],
            tuple(CaseRevision.from_dict(item) for item in data["case_revisions"]),
            tuple(
                Presentation.from_dict(item) for item in data["visual_presentations"]
            ),
            presentation_cases,
        )


@dataclass(frozen=True, slots=True)
class RepositoryIdentity:
    """Immutable Git commit and exact dirty-tree binary-diff identity."""

    commit: str
    dirty: bool
    diff_sha256: str | None

    def __post_init__(self) -> None:
        commit = _string(self.commit, "repository commit")
        if len(commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.commit
        ):
            raise ValueError("repository commit must be a lowercase Git SHA-1")
        if not isinstance(self.dirty, bool):
            raise TypeError("repository dirty must be a boolean")
        if self.dirty:
            if self.diff_sha256 is None:
                raise ValueError("dirty repository identity requires a diff_sha256")
            _sha256(self.diff_sha256, "repository diff_sha256")
        elif self.diff_sha256 is not None:
            raise ValueError("clean repository identity cannot contain a diff_sha256")

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible representation."""
        return {
            "commit": self.commit,
            "dirty": self.dirty,
            "diff_sha256": self.diff_sha256,
        }

    @classmethod
    def from_dict(cls, payload: object) -> RepositoryIdentity:
        """Parse one strictly validated JSON-compatible representation."""
        data = _mapping(
            payload,
            frozenset({"commit", "dirty", "diff_sha256"}),
            "repository identity",
        )
        if not isinstance(data["dirty"], bool):
            raise TypeError("repository dirty must be a boolean")
        if data["diff_sha256"] is not None and not isinstance(data["diff_sha256"], str):
            raise TypeError("repository diff_sha256 must be a string or null")
        return cls(
            _string(data["commit"], "repository commit"),
            data["dirty"],
            data["diff_sha256"],
        )


@dataclass(frozen=True, slots=True)
class Assessment:
    """One immutable, blinded visual assessment of a presentation revision."""

    event_id: str
    opaque_id: str
    revision_id: str
    status: AssessmentStatus
    reasons: tuple[str, ...]
    notes: str
    reviewer: str
    assessed_at: datetime

    def __post_init__(self) -> None:
        _string(self.event_id, "assessment event_id")
        _string(self.opaque_id, "assessment opaque_id")
        _string(self.revision_id, "assessment revision_id")
        if self.status not in _ASSESSMENT_STATUSES:
            raise ValueError("assessment status is not allowlisted")
        _reason_tokens(self.reasons)
        if not isinstance(self.notes, str):
            raise TypeError("assessment notes must be a string")
        _string(self.reviewer, "assessment reviewer")
        _utc_timestamp(self.assessed_at, "assessment timestamp")

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible representation."""
        return {
            "event_id": self.event_id,
            "opaque_id": self.opaque_id,
            "revision_id": self.revision_id,
            "status": self.status,
            "reasons": list(self.reasons),
            "notes": self.notes,
            "reviewer": self.reviewer,
            "assessed_at": self.assessed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> Assessment:
        """Parse one strictly validated JSON-compatible representation."""
        data = _mapping(
            payload,
            frozenset(
                {
                    "event_id",
                    "opaque_id",
                    "revision_id",
                    "status",
                    "reasons",
                    "notes",
                    "reviewer",
                    "assessed_at",
                }
            ),
            "assessment",
        )
        if not isinstance(data["reasons"], list):
            raise TypeError("assessment reasons must be a list")
        return cls(
            _string(data["event_id"], "assessment event_id"),
            _string(data["opaque_id"], "assessment opaque_id"),
            _string(data["revision_id"], "assessment revision_id"),
            cast(AssessmentStatus, _string(data["status"], "assessment status")),
            tuple(data["reasons"]),
            data["notes"] if isinstance(data["notes"], str) else _invalid_string(),
            _string(data["reviewer"], "assessment reviewer"),
            _timestamp_from_dict(data["assessed_at"], "assessment timestamp"),
        )


@dataclass(frozen=True, slots=True)
class Adjudication:
    """One immutable reconciliation decision for a biological case history."""

    event_id: str
    case_key: CaseKey
    revision_id: str
    assessment_event_ids: tuple[str, ...]
    status: Literal["accept", "exclude"]
    reasons: tuple[str, ...]
    notes: str
    reviewer: str
    adjudicated_at: datetime

    def __post_init__(self) -> None:
        _string(self.event_id, "adjudication event_id")
        _string(self.revision_id, "adjudication revision_id")
        if any(
            not isinstance(item, str) or not item for item in self.assessment_event_ids
        ):
            raise ValueError(
                "adjudication assessment event IDs must be nonempty strings"
            )
        if len(self.assessment_event_ids) != len(set(self.assessment_event_ids)):
            raise ValueError("adjudication assessment event IDs must be unique")
        if self.status not in FINAL_STATUSES:
            raise ValueError("adjudication status must be accept or exclude")
        _reason_tokens(self.reasons)
        if not isinstance(self.notes, str):
            raise TypeError("adjudication notes must be a string")
        _string(self.reviewer, "adjudication reviewer")
        _utc_timestamp(self.adjudicated_at, "adjudication timestamp")

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible representation."""
        return {
            "event_id": self.event_id,
            "case_key": self.case_key.to_dict(),
            "revision_id": self.revision_id,
            "assessment_event_ids": list(self.assessment_event_ids),
            "status": self.status,
            "reasons": list(self.reasons),
            "notes": self.notes,
            "reviewer": self.reviewer,
            "adjudicated_at": self.adjudicated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> Adjudication:
        """Parse one strictly validated JSON-compatible representation."""
        data = _mapping(
            payload,
            frozenset(
                {
                    "event_id",
                    "case_key",
                    "revision_id",
                    "assessment_event_ids",
                    "status",
                    "reasons",
                    "notes",
                    "reviewer",
                    "adjudicated_at",
                }
            ),
            "adjudication",
        )
        if not isinstance(data["reasons"], list) or not isinstance(
            data["assessment_event_ids"], list
        ):
            raise TypeError("adjudication collections must be lists")
        return cls(
            _string(data["event_id"], "adjudication event_id"),
            CaseKey.from_dict(data["case_key"]),
            _string(data["revision_id"], "adjudication revision_id"),
            tuple(data["assessment_event_ids"]),
            cast(
                Literal["accept", "exclude"],
                _string(data["status"], "adjudication status"),
            ),
            tuple(data["reasons"]),
            data["notes"] if isinstance(data["notes"], str) else _invalid_string(),
            _string(data["reviewer"], "adjudication reviewer"),
            _timestamp_from_dict(data["adjudicated_at"], "adjudication timestamp"),
        )


@dataclass(frozen=True, slots=True)
class CaseHistory:
    """Immutable revisions and reconciliation events for one biological case."""

    case_key: CaseKey
    case_token: str
    revisions: tuple[CaseRevision, ...]
    adjudications: tuple[Adjudication, ...]

    def __post_init__(self) -> None:
        _string(self.case_token, "case history case_token")
        if not self.revisions:
            raise ValueError("case history must contain at least one revision")
        if any(
            revision.case_key != self.case_key or revision.case_token != self.case_token
            for revision in self.revisions
        ):
            raise ValueError("case history revisions must match its biological key")
        if any(item.case_key != self.case_key for item in self.adjudications):
            raise ValueError("case history adjudications must match its biological key")
        revision_ids = tuple(item.revision_id for item in self.revisions)
        if len(revision_ids) != len(set(revision_ids)):
            raise ValueError("case history revision IDs must be unique")
        if any(item.revision_id not in revision_ids for item in self.adjudications):
            raise ValueError("adjudication revision is absent from case history")

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible representation."""
        return {
            "case_key": self.case_key.to_dict(),
            "case_token": self.case_token,
            "revisions": [item.to_dict() for item in self.revisions],
            "adjudications": [item.to_dict() for item in self.adjudications],
        }

    @classmethod
    def from_dict(cls, payload: object) -> CaseHistory:
        """Parse one strictly validated JSON-compatible representation."""
        data = _mapping(
            payload,
            frozenset({"case_key", "case_token", "revisions", "adjudications"}),
            "case history",
        )
        if not isinstance(data["revisions"], list) or not isinstance(
            data["adjudications"], list
        ):
            raise TypeError("case history collections must be lists")
        return cls(
            CaseKey.from_dict(data["case_key"]),
            _string(data["case_token"], "case history case_token"),
            tuple(CaseRevision.from_dict(item) for item in data["revisions"]),
            tuple(Adjudication.from_dict(item) for item in data["adjudications"]),
        )


@dataclass(frozen=True, slots=True)
class RecoveryEvent:
    """One timestamped recovery of the last valid review-state snapshot."""

    recovered_at: datetime
    message: str

    def __post_init__(self) -> None:
        _utc_timestamp(self.recovered_at, "recovery timestamp")
        _string(self.message, "recovery message")

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible representation."""
        return {"recovered_at": self.recovered_at.isoformat(), "message": self.message}

    @classmethod
    def from_dict(cls, payload: object) -> RecoveryEvent:
        """Parse one strictly validated JSON-compatible representation."""
        data = _mapping(
            payload, frozenset({"recovered_at", "message"}), "recovery event"
        )
        return cls(
            _timestamp_from_dict(data["recovered_at"], "recovery timestamp"),
            _string(data["message"], "recovery message"),
        )


@dataclass(frozen=True, slots=True)
class ExportRecord:
    """One terminal export bound to its exact pre-export review-state digest."""

    sha256: str
    finalized_state_sha256: str
    pdf_sha256: str | None
    summary_sha256: str | None
    exported_at: datetime

    def __post_init__(self) -> None:
        _sha256(self.sha256, "export sha256")
        _sha256(self.finalized_state_sha256, "finalized state sha256")
        if (self.pdf_sha256 is None) != (self.summary_sha256 is None):
            raise ValueError("final report digests must be recorded together")
        if self.pdf_sha256 is not None:
            _sha256(self.pdf_sha256, "final PDF sha256")
            _sha256(self.summary_sha256, "final Markdown sha256")
        _utc_timestamp(self.exported_at, "export timestamp")

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible representation."""
        return {
            "sha256": self.sha256,
            "finalized_state_sha256": self.finalized_state_sha256,
            "pdf_sha256": self.pdf_sha256,
            "summary_sha256": self.summary_sha256,
            "exported_at": self.exported_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> ExportRecord:
        """Parse one strictly validated JSON-compatible representation."""
        data = _mapping(
            payload,
            frozenset(
                {
                    "sha256",
                    "finalized_state_sha256",
                    "pdf_sha256",
                    "summary_sha256",
                    "exported_at",
                }
            ),
            "export record",
        )
        for name in ("pdf_sha256", "summary_sha256"):
            if data[name] is not None and not isinstance(data[name], str):
                raise TypeError(f"{name} must be a string or null")
        return cls(
            _sha256(data["sha256"], "export sha256"),
            _sha256(data["finalized_state_sha256"], "finalized state sha256"),
            data["pdf_sha256"],
            data["summary_sha256"],
            _timestamp_from_dict(data["exported_at"], "export timestamp"),
        )


@dataclass(frozen=True, slots=True)
class ReviewState:
    """Strict append-preserving state with one validated active manifest."""

    schema_version: int
    format_version: int
    initial_manifest_identity: str
    manifest_identity: str
    reviewer: str
    repository_identity: RepositoryIdentity
    created_at: datetime
    assessment_history: Mapping[str, tuple[Assessment, ...]]
    case_histories: Mapping[str, CaseHistory]
    attached_presentations: tuple[Presentation, ...]
    recovery_events: tuple[RecoveryEvent, ...]
    export_history: tuple[ExportRecord, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.format_version != 1:
            raise ValueError("review state schema or format version is unsupported")
        _sha256(self.initial_manifest_identity, "initial manifest identity")
        _sha256(self.manifest_identity, "active manifest identity")
        _string(self.reviewer, "reviewer")
        _utc_timestamp(self.created_at, "review state creation timestamp")
        if not isinstance(self.repository_identity, RepositoryIdentity):
            raise TypeError(
                "review state repository_identity must be a RepositoryIdentity"
            )
        histories = dict(self.case_histories)
        if any(
            not isinstance(case_token, str)
            or not isinstance(history, CaseHistory)
            or case_token != history.case_token
            for case_token, history in histories.items()
        ):
            raise ValueError("case histories have an invalid schema")
        assessments = {
            opaque_id: tuple(history)
            for opaque_id, history in self.assessment_history.items()
        }
        event_ids = [
            assessment.event_id
            for opaque_id, history in assessments.items()
            for assessment in history
            if isinstance(opaque_id, str)
            and assessment.opaque_id == opaque_id
            and isinstance(assessment, Assessment)
        ]
        if len(event_ids) != sum(len(history) for history in assessments.values()):
            raise ValueError("assessment history has an invalid schema")
        event_ids.extend(
            adjudication.event_id
            for history in histories.values()
            for adjudication in history.adjudications
        )
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("review state has duplicate event IDs")
        if any(not isinstance(event, RecoveryEvent) for event in self.recovery_events):
            raise ValueError("recovery events have an invalid schema")
        if any(
            not isinstance(presentation, Presentation)
            for presentation in self.attached_presentations
        ):
            raise ValueError("attached presentations have an invalid schema")
        attached_ids = tuple(
            presentation.opaque_id for presentation in self.attached_presentations
        )
        if len(attached_ids) != len(set(attached_ids)):
            raise ValueError("attached presentation opaque IDs must be unique")
        if any(not isinstance(record, ExportRecord) for record in self.export_history):
            raise ValueError("export history has an invalid schema")
        if len(self.export_history) > 1:
            raise ValueError("a review may contain only one terminal export record")
        object.__setattr__(self, "assessment_history", MappingProxyType(assessments))
        object.__setattr__(self, "case_histories", MappingProxyType(histories))
        if (
            self.export_history
            and self.export_history[0].finalized_state_sha256
            != self.finalizable_state_sha256
        ):
            raise ValueError("export record does not match the finalized review state")

    @property
    def finalizable_state_sha256(self) -> str:
        """Return the canonical digest of this state without terminal export metadata."""
        payload = self.to_dict()
        payload["export_history"] = []
        content = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "format_version": self.format_version,
            "initial_manifest_identity": self.initial_manifest_identity,
            "manifest_identity": self.manifest_identity,
            "reviewer": self.reviewer,
            "repository_identity": self.repository_identity.to_dict(),
            "created_at": self.created_at.isoformat(),
            "assessment_history": {
                opaque_id: [item.to_dict() for item in history]
                for opaque_id, history in sorted(self.assessment_history.items())
            },
            "case_histories": {
                case_token: history.to_dict()
                for case_token, history in sorted(self.case_histories.items())
            },
            "attached_presentations": [
                item.to_dict() for item in self.attached_presentations
            ],
            "recovery_events": [item.to_dict() for item in self.recovery_events],
            "export_history": [item.to_dict() for item in self.export_history],
        }

    @classmethod
    def from_dict(cls, payload: object, manifest: ReviewManifest) -> ReviewState:
        """Parse state strictly and verify it against an immutable manifest."""
        data = _mapping(
            payload,
            frozenset(
                {
                    "schema_version",
                    "format_version",
                    "initial_manifest_identity",
                    "manifest_identity",
                    "reviewer",
                    "repository_identity",
                    "created_at",
                    "assessment_history",
                    "case_histories",
                    "attached_presentations",
                    "recovery_events",
                    "export_history",
                }
            ),
            "review state",
        )
        if not isinstance(data["schema_version"], int) or isinstance(
            data["schema_version"], bool
        ):
            raise TypeError("review state schema_version must be an integer")
        if not isinstance(data["format_version"], int) or isinstance(
            data["format_version"], bool
        ):
            raise TypeError("review state format_version must be an integer")
        if (
            not isinstance(data["assessment_history"], Mapping)
            or not isinstance(data["case_histories"], Mapping)
            or not isinstance(data["attached_presentations"], list)
            or not isinstance(data["recovery_events"], list)
            or not isinstance(data["export_history"], list)
        ):
            raise TypeError("review state collections have invalid types")
        assessment_history: dict[str, tuple[Assessment, ...]] = {}
        for opaque_id, history in data["assessment_history"].items():
            if not isinstance(history, list):
                raise TypeError("assessment history entries must be lists")
            assessment_history[_string(opaque_id, "assessment opaque_id")] = tuple(
                Assessment.from_dict(item) for item in history
            )
        state = cls(
            data["schema_version"],
            data["format_version"],
            _sha256(data["initial_manifest_identity"], "initial manifest identity"),
            _sha256(data["manifest_identity"], "active manifest identity"),
            _string(data["reviewer"], "reviewer"),
            RepositoryIdentity.from_dict(data["repository_identity"]),
            _timestamp_from_dict(data["created_at"], "review state creation timestamp"),
            assessment_history,
            {
                _string(case_token, "case history case_token"): CaseHistory.from_dict(
                    history
                )
                for case_token, history in data["case_histories"].items()
            },
            tuple(
                Presentation.from_dict(item) for item in data["attached_presentations"]
            ),
            tuple(RecoveryEvent.from_dict(item) for item in data["recovery_events"]),
            tuple(ExportRecord.from_dict(item) for item in data["export_history"]),
        )
        state._validate_manifest(manifest)
        return state

    def active_manifest(self, manifest: ReviewManifest) -> ReviewManifest:
        """Reconstruct the exact active manifest from initial and appended state."""
        if self.initial_manifest_identity != manifest.manifest_identity:
            raise ReviewStateBindingError("reviewed evidence identity changed")
        initial_revisions = {item.case_token: item for item in manifest.case_revisions}
        if set(self.case_histories) != set(initial_revisions):
            raise ReviewStateBindingError(
                "case histories contain biological keys absent from immutable manifest"
            )
        appended_revisions: list[CaseRevision] = []
        for token in sorted(self.case_histories):
            history = self.case_histories[token]
            initial = initial_revisions[token]
            if history.case_key != initial.case_key or history.revisions[0] != initial:
                raise ReviewStateBindingError(
                    "case history does not match immutable manifest"
                )
            appended_revisions.extend(history.revisions[1:])
        appended_by_key = {
            (item.case_token, item.revision_id): item for item in appended_revisions
        }
        if len(appended_by_key) != len(appended_revisions):
            raise ReviewStateBindingError(
                "case history revisions do not match active manifest"
            )
        attached_by_key = {
            (item.case_token, item.revision_id): item
            for item in self.attached_presentations
        }
        if len(attached_by_key) != len(self.attached_presentations):
            raise ReviewStateBindingError(
                "attached presentations do not match active manifest"
            )
        if set(attached_by_key) != set(appended_by_key):
            raise ReviewStateBindingError(
                "case history revisions do not match active manifest"
            )
        for key, presentation in attached_by_key.items():
            revision = appended_by_key[key]
            if presentation.overlays != revision.overlays or presentation.repeat_of:
                raise ReviewStateBindingError(
                    "attached presentations do not match active manifest"
                )
        cases = dict(manifest.presentation_cases)
        cases.update(
            {
                presentation.opaque_id: self.case_histories[
                    presentation.case_token
                ].case_key
                for presentation in self.attached_presentations
            }
        )
        try:
            active = ReviewManifest(
                manifest.seed,
                manifest.case_revisions + tuple(appended_revisions),
                manifest.visual_presentations + self.attached_presentations,
                cases,
            )
        except (TypeError, ValueError) as error:
            raise ReviewStateBindingError(
                "attached revisions do not match active manifest"
            ) from error
        if self.manifest_identity != active.manifest_identity:
            raise ReviewStateBindingError("reviewed evidence identity changed")
        return active

    def _validate_manifest(self, manifest: ReviewManifest) -> None:
        active = self.active_manifest(manifest)
        presentations = {
            presentation.opaque_id: presentation
            for presentation in active.visual_presentations
        }
        if not set(self.assessment_history) <= set(presentations):
            raise ReviewStateBindingError(
                "assessment history contains unknown opaque IDs"
            )
        for opaque_id, history in self.assessment_history.items():
            presentation = presentations[opaque_id]
            if any(
                assessment.revision_id != presentation.revision_id
                for assessment in history
            ):
                raise ReviewStateBindingError(
                    "assessment revision is absent from immutable manifest"
                )
            if any(assessment.reviewer != self.reviewer for assessment in history):
                raise ReviewStateBindingError(
                    "assessment reviewer does not match persisted review state"
                )
        assessments_by_id = {
            assessment.event_id: assessment
            for assessment_history in self.assessment_history.values()
            for assessment in assessment_history
        }
        for history in self.case_histories.values():
            if any(
                adjudication.reviewer != self.reviewer
                for adjudication in history.adjudications
            ):
                raise ReviewStateBindingError(
                    "adjudication reviewer does not match persisted review state"
                )
            for adjudication in history.adjudications:
                for event_id in adjudication.assessment_event_ids:
                    assessment = assessments_by_id.get(event_id)
                    presentation = (
                        presentations.get(assessment.opaque_id)
                        if assessment is not None
                        else None
                    )
                    if (
                        assessment is None
                        or presentation is None
                        or assessment.revision_id != adjudication.revision_id
                        or presentation.case_token != history.case_token
                    ):
                        raise ReviewStateBindingError(
                            "adjudication assessment snapshot does not match its case revision"
                        )
