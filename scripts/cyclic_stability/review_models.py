"""Immutable, validated values for the target-blind registration-QC review."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

_COHORTS = frozenset({"golgi", "lamin"})
_ROUNDS = frozenset({1, 3, 4, 5})
_CHANNELS = frozenset({"dapi", "tubulin"})
_STAGES = frozenset({"pre", "affine", "nonlinear"})
_VIEWS = frozenset({"xy", "xz", "yz"})


def _mapping(payload: object, keys: frozenset[str], name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise ValueError(f"{name} has an invalid schema")
    return payload


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


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
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stat = path.stat()
        return cls(str(path.resolve()), stat.st_size, stat.st_mtime_ns, digest)

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
        if len(self.overlays) not in (0, 18):
            raise ValueError("a case revision must contain zero or eighteen overlays")
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
            flags.append((_string(item[0], "region"), _string(item[1], "warp_flags")))
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
        if len(self.overlays) != 18:
            raise ValueError("a visual presentation must contain eighteen overlays")
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
