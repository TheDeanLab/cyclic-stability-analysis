"""Durable recovery journal for the owned final-review output bundle."""

from __future__ import annotations

import base64
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

FINALIZATION_BUNDLE_NAMES = (
    "review-state.json",
    "review-state.previous.json",
    "remediation-requests.csv",
    "consistency-report.csv",
    "qc-decisions.csv",
    "qc-decisions.previous.csv",
    "review-record.pdf",
    "review-summary.md",
)
PENDING_FINALIZATION_NAME = "pending-finalization.json"


@dataclass(frozen=True, slots=True)
class PendingFinalization:
    """Exact prior owned bytes made authoritative before finalization mutates live files."""

    files: Mapping[str, bytes | None]
    staged_names: frozenset[str]


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _payload(checkpoint: PendingFinalization) -> dict[str, object]:
    return {
        "schema_version": 1,
        "authority": "restore-exact-prior-owned-finalization-bundle",
        "files": {
            name: None if content is None else base64.b64encode(content).decode("ascii")
            for name, content in checkpoint.files.items()
        },
        "staged_names": sorted(checkpoint.staged_names),
    }


def _from_payload(payload: object) -> PendingFinalization:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "authority",
        "files",
        "staged_names",
    }:
        raise ValueError("pending finalization journal is malformed")
    if (
        payload["schema_version"] != 1
        or payload["authority"] != "restore-exact-prior-owned-finalization-bundle"
        or not isinstance(payload["files"], Mapping)
        or set(payload["files"]) != set(FINALIZATION_BUNDLE_NAMES)
        or not isinstance(payload["staged_names"], list)
        or any(not isinstance(item, str) for item in payload["staged_names"])
    ):
        raise ValueError("pending finalization journal is malformed")
    files: dict[str, bytes | None] = {}
    try:
        for name in FINALIZATION_BUNDLE_NAMES:
            encoded = payload["files"][name]
            if encoded is None:
                files[name] = None
            elif isinstance(encoded, str):
                files[name] = base64.b64decode(encoded, validate=True)
            else:
                raise ValueError
    except (ValueError, TypeError) as error:
        raise ValueError("pending finalization journal is malformed") from error
    return PendingFinalization(files, frozenset(payload["staged_names"]))


def begin_pending_finalization(review_dir: Path) -> PendingFinalization:
    """Durably make the exact prior bundle authoritative before live mutation."""
    journal = review_dir / PENDING_FINALIZATION_NAME
    if journal.exists():
        raise ValueError("an interrupted finalization must be recovered before retry")
    checkpoint = PendingFinalization(
        {
            name: path.read_bytes() if (path := review_dir / name).exists() else None
            for name in FINALIZATION_BUNDLE_NAMES
        },
        frozenset(path.name for path in review_dir.glob(".*tmp")),
    )
    _atomic_json(journal, _payload(checkpoint))
    return checkpoint


def _restore_file(path: Path, content: bytes | None) -> None:
    if content is None:
        if path.exists() or path.is_symlink():
            path.unlink()
            _fsync_directory(path.parent)
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.rollback.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def restore_pending_finalization(
    review_dir: Path, checkpoint: PendingFinalization | None = None
) -> bool:
    """Restore the journal-authoritative prior bundle, then durably remove the marker."""
    journal = review_dir / PENDING_FINALIZATION_NAME
    if checkpoint is None:
        if not journal.exists():
            return False
        try:
            checkpoint = _from_payload(json.loads(journal.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("pending finalization journal is malformed") from error
    for name, content in checkpoint.files.items():
        _restore_file(review_dir / name, content)
    for path in review_dir.glob(".*tmp"):
        if path.name not in checkpoint.staged_names:
            path.unlink()
    _fsync_directory(review_dir)
    journal.unlink(missing_ok=True)
    _fsync_directory(review_dir)
    return True


def complete_pending_finalization(review_dir: Path) -> None:
    """Durably discard the prior-bundle authority after full live validation."""
    journal = review_dir / PENDING_FINALIZATION_NAME
    if not journal.exists():
        raise ValueError("pending finalization journal is missing")
    journal.unlink()
    _fsync_directory(review_dir)


__all__ = [
    "FINALIZATION_BUNDLE_NAMES",
    "PENDING_FINALIZATION_NAME",
    "PendingFinalization",
    "begin_pending_finalization",
    "complete_pending_finalization",
    "restore_pending_finalization",
]
