"""Immutable, restart-safe marker-cell and generic artifact task attempts.

This module deliberately imports no scientific/image library at import time.  A
worker first applies its native-thread policy and only then imports the cohort
analyzer that may load NumPy, ANTs, or a BLAS implementation.

Use :func:`run_cell_task` for configured marker analysis and
:func:`run_artifact_task` with :class:`ArtifactTask` for another file-producing
operation that needs the same terminal authority. Generic producers do not own
discovery: their :class:`ArtifactSnapshot` callback must admit every exact input
and an immutable consumer context after the running terminal is published.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import BinaryIO, Literal

import psutil
from cyexm.registration.artifacts import FileIdentity

from cyclic_stability.models import AnalysisConfig, CellSpec

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_THREAD_ENVIRONMENT = (
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
_STABLE_CONTENT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
CYEXM_PIN = "f6e0de36cd7f6b5e85f093884c2cf224225f4ec2"


@dataclass(frozen=True, slots=True)
class ContentIdentity:
    """Exact file metadata plus authoritative SHA-256 content identity.

    Parameters
    ----------
    path : str
        Resolved absolute file path.
    st_size : int
        Byte size captured with the digest.
    st_mtime_ns : int
        Nanosecond modification time captured with the digest.
    sha256 : str
        Lower-case SHA-256 digest of the complete file content.
    """

    path: str
    st_size: int
    st_mtime_ns: int
    sha256: str

    @classmethod
    def from_path(cls, path: Path) -> ContentIdentity:
        """Snapshot stable metadata and content through a no-follow descriptor."""
        identity, _ = cls._read(path, retain_bytes=False)
        return identity

    @classmethod
    def read_bytes(cls, path: Path) -> tuple[ContentIdentity, bytes]:
        """Return one stable identity and the exact bytes hashed for that identity."""
        identity, payload = cls._read(path, retain_bytes=True)
        assert payload is not None
        return identity, payload

    @classmethod
    def read_bytes_many(
        cls, paths: tuple[Path, ...]
    ) -> tuple[tuple[ContentIdentity, bytes], ...]:
        """Snapshot related files while retaining one descriptor per lexical parent.

        A directory alias may be retargeted while files are being read. Grouping
        siblings behind one already-open directory descriptor guarantees that one
        call cannot mix entries from two parent generations.
        """
        candidates = tuple(Path(path) for path in paths)
        parents = {
            candidate.parent if candidate.parent != Path("") else Path(".")
            for candidate in candidates
        }
        descriptors: dict[Path, int] = {}
        try:
            for parent in parents:
                descriptors[parent] = os.open(
                    parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
                )
            snapshots: list[tuple[ContentIdentity, bytes]] = []
            for candidate in candidates:
                parent = candidate.parent if candidate.parent != Path("") else Path(".")
                identity, payload = cls._read_from_parent_descriptor(
                    candidate,
                    descriptors[parent],
                    retain_bytes=True,
                )
                assert payload is not None
                snapshots.append((identity, payload))
            return tuple(snapshots)
        finally:
            for descriptor in descriptors.values():
                os.close(descriptor)

    @classmethod
    def _read(
        cls, path: Path, *, retain_bytes: bool
    ) -> tuple[ContentIdentity, bytes | None]:
        candidate = Path(path)
        parent = candidate.parent if candidate.parent != Path("") else Path(".")
        parent_fd = os.open(
            parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        try:
            return cls._read_from_parent_descriptor(
                candidate, parent_fd, retain_bytes=retain_bytes
            )
        finally:
            os.close(parent_fd)

    @classmethod
    def _read_from_parent_descriptor(
        cls,
        candidate: Path,
        parent_fd: int,
        *,
        retain_bytes: bool,
    ) -> tuple[ContentIdentity, bytes | None]:
        identity, payload, descriptor, _stable_stat = cls._open_from_parent_descriptor(
            candidate,
            parent_fd,
            retain_bytes=retain_bytes,
        )
        os.close(descriptor)
        return identity, payload

    @classmethod
    def _open_from_parent_descriptor(
        cls,
        candidate: Path,
        parent_fd: int,
        *,
        retain_bytes: bool,
    ) -> tuple[ContentIdentity, bytes | None, int, tuple[int, ...]]:
        """Open and hash one stable entry, transferring its descriptor on success."""
        descriptor = os.open(
            candidate.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            before = os.fstat(descriptor)
            entry_before = os.stat(
                candidate.name, dir_fd=parent_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(before.st_mode)
                or not stat.S_ISREG(entry_before.st_mode)
                or (before.st_dev, before.st_ino)
                != (entry_before.st_dev, entry_before.st_ino)
            ):
                raise ValueError(f"file is not a stable regular entry: {candidate}")
            canonical_parent = Path(os.readlink(f"/proc/self/fd/{parent_fd}"))
            if canonical_parent.name.endswith(" (deleted)"):
                raise ValueError(f"file parent changed while opening: {candidate}")
            resolved = canonical_parent / candidate.name
            digest = hashlib.sha256()
            retained: list[bytes] | None = [] if retain_bytes else None
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                if retained is not None:
                    retained.append(chunk)
            after = os.fstat(descriptor)
            entry_after = os.stat(
                candidate.name, dir_fd=parent_fd, follow_symlinks=False
            )
            if any(
                getattr(before, name) != getattr(after, name)
                for name in _STABLE_CONTENT_FIELDS
            ) or (after.st_dev, after.st_ino) != (
                entry_after.st_dev,
                entry_after.st_ino,
            ):
                raise ValueError(f"file changed while hashing: {resolved}")
            identity = cls(
                str(resolved),
                after.st_size,
                after.st_mtime_ns,
                digest.hexdigest(),
            )
            os.lseek(descriptor, 0, os.SEEK_SET)
            return (
                identity,
                None if retained is None else b"".join(retained),
                descriptor,
                tuple(int(getattr(after, name)) for name in _STABLE_CONTENT_FIELDS),
            )
        except BaseException:
            os.close(descriptor)
            raise

    def matches_path(self) -> bool:
        """Return whether the same path still has this exact content identity."""
        try:
            return type(self).from_path(Path(self.path)) == self
        except (OSError, ValueError):
            return False

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object], *, name: str = "content identity"
    ) -> ContentIdentity:
        """Validate and reconstruct an identity from persisted JSON."""
        if set(payload) != {"path", "st_size", "st_mtime_ns", "sha256"}:
            raise ValueError(f"{name} has invalid fields")
        metadata = FileIdentity.from_dict(
            {key: payload[key] for key in ("path", "st_size", "st_mtime_ns")},
            name=name,
        )
        digest = payload["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{name} has invalid sha256")
        return cls(metadata.path, metadata.st_size, metadata.st_mtime_ns, digest)

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON representation."""
        return {
            "path": self.path,
            "st_size": self.st_size,
            "st_mtime_ns": self.st_mtime_ns,
            "sha256": self.sha256,
        }


class RetainedContent:
    """Owned no-follow descriptor for the exact bytes admitted to a producer.

    The caller path is retained only in :attr:`identity` for provenance. Consumers
    open :attr:`consumer_path`, a process-local descriptor path that cannot be
    redirected by replacing the caller's directory entry. The task lifecycle owns
    and closes instances transferred through :class:`ArtifactSnapshot`.
    """

    __slots__ = ("_descriptor", "_stable_stat", "identity")

    def __init__(
        self,
        identity: ContentIdentity,
        descriptor: int,
        stable_stat: tuple[int, ...],
    ) -> None:
        self.identity = identity
        self._descriptor = descriptor
        self._stable_stat = stable_stat

    @classmethod
    def from_path(cls, path: Path) -> RetainedContent:
        """Securely hash and retain one canonical regular-file descriptor."""
        candidate = Path(path)
        parent = candidate.parent if candidate.parent != Path("") else Path(".")
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            identity, _payload, descriptor, stable_stat = (
                ContentIdentity._open_from_parent_descriptor(
                    candidate,
                    parent_fd,
                    retain_bytes=False,
                )
            )
        finally:
            os.close(parent_fd)
        return cls(identity, descriptor, stable_stat)

    @property
    def closed(self) -> bool:
        """Return whether lifecycle ownership has closed the descriptor."""
        return self._descriptor < 0

    @property
    def consumer_path(self) -> Path:
        """Return the process-local path that reopens exactly this descriptor."""
        if self.closed:
            raise ValueError("retained input descriptor is closed")
        return Path(f"/proc/self/fd/{self._descriptor}")

    def open_binary(self) -> BinaryIO:
        """Open an independent binary reader for the retained inode."""
        if self.closed:
            raise ValueError("retained input descriptor is closed")
        return self.consumer_path.open("rb")

    def validate(self) -> None:
        """Require the retained inode and its full content to remain unchanged."""
        if self.closed:
            raise ValueError("retained input descriptor is closed")
        try:
            before = os.fstat(self._descriptor)
            digest = hashlib.sha256()
            os.lseek(self._descriptor, 0, os.SEEK_SET)
            while True:
                chunk = os.read(self._descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(self._descriptor)
            os.lseek(self._descriptor, 0, os.SEEK_SET)
        except OSError as error:
            raise ValueError(
                f"retained input changed during execution: {self.identity.path}"
            ) from error
        before_stat = tuple(
            int(getattr(before, name)) for name in _STABLE_CONTENT_FIELDS
        )
        after_stat = tuple(int(getattr(after, name)) for name in _STABLE_CONTENT_FIELDS)
        if (
            before_stat != self._stable_stat
            or after_stat != self._stable_stat
            or digest.hexdigest() != self.identity.sha256
        ):
            raise ValueError(
                f"retained input changed during execution: {self.identity.path}"
            )

    def close(self) -> None:
        """Close the owned descriptor idempotently."""
        if self.closed:
            return
        descriptor = self._descriptor
        self._descriptor = -1
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class TaskOutput:
    """Compact per-cell tables and paths produced by one analyzer."""

    tables: Mapping[str, tuple[Mapping[str, object], ...]]
    diagnostics: tuple[Path, ...] = ()


TaskAnalyzer = Callable[["CellTask", Path], TaskOutput]


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    """One authoritative, under-lock snapshot for a generic artifact attempt.

    Parameters
    ----------
    context : object
        Immutable producer context parsed only from the bytes represented by
        ``input_identities``. The producer must consume task inputs exclusively
        through this value.
    config_digest : str
        Scientific identity derived from the snapshotted configuration bytes.
    input_identities : tuple[ContentIdentity, ...]
        Complete canonical content identities consumed by the producer.
    native_threads : int, default=1
        Native thread policy to apply after snapshot admission.
    scratch_root : pathlib.Path or None, default=None
        Optional external scratch root selected by the snapshotted execution
        profile.
    owned_inputs : tuple[RetainedContent, ...], default=()
        Open exact-content descriptors whose process-local ``consumer_path`` values
        are supplied through ``context``. Ownership transfers to
        :func:`run_artifact_task`, which validates and closes every descriptor on
        success, failure, and reuse.
    """

    context: object
    config_digest: str
    input_identities: tuple[ContentIdentity, ...]
    native_threads: int = 1
    scratch_root: Path | None = None
    owned_inputs: tuple[RetainedContent, ...] = ()


ArtifactSnapshotter = Callable[[Path, Path], ArtifactSnapshot]
ArtifactProducer = Callable[[Path, object], tuple[Path, ...]]


@dataclass(frozen=True, slots=True)
class ArtifactTask:
    """One immutable file-producing request using the shared attempt store.

    This is the supported generic extension interface for the task lifecycle.
    :func:`run_artifact_task` owns locking, terminal transitions, reuse, resource
    accounting, and atomic publication. The supplied snapshot owns input discovery
    and parsing; the producer owns only output construction from the admitted
    context.

    Parameters
    ----------
    key : str
        Stable path-safe task identity within a run.
    software_identity : str
        Frozen software and repository identity used to authorize reuse.
    snapshot : Callable[[pathlib.Path, pathlib.Path], ArtifactSnapshot]
        Callback run under the task lock after ``running`` becomes authoritative.
        It receives the canonical run and task roots, in that order, and securely
        discovers, hashes, and parses every producer input.
    producer : Callable[[Path, object], tuple[Path, ...]]
        Function that writes outputs below its unique attempt directory and
        returns every primary output path. Its second argument is exactly the
        immutable context returned by ``snapshot``; it must not retain or consume
        pre-snapshot scientific objects. The shared store reopens all files,
        records their identities, and publishes completion atomically.
    native_threads : int, default=1
        Native thread policy applied before production.
    force : bool, default=False
        If true, bypass an otherwise reusable complete attempt.
    config_digest : str, internal
        Populated only from the admitted snapshot.
    input_identities : tuple[ContentIdentity, ...], internal
        Populated only from the admitted snapshot.
    context : object or None, internal
        Producer context populated only from the admitted snapshot.

    See Also
    --------
    run_artifact_task :
        Admit and execute this request through the authoritative task lifecycle.
    ArtifactSnapshot :
        Carries the exact input identities and producer-only immutable context.

    Notes
    -----
    ``config_digest``, ``input_identities``, and ``context`` are internal admitted
    state and must be empty on caller-created requests. The snapshot callback runs
    under the task lock only after a new ``running`` terminal is authoritative and
    receives no caller path alias. Producers must not retain or read pre-snapshot
    scientific objects or aliases.
    """

    key: str
    software_identity: str
    snapshot: ArtifactSnapshotter
    producer: ArtifactProducer
    native_threads: int = 1
    force: bool = False
    scratch_root: Path | None = None
    config_digest: str = ""
    input_identities: tuple[ContentIdentity, ...] = ()
    context: object | None = None


@dataclass(frozen=True, slots=True)
class PreflightSnapshot:
    """Picklable validated source/provenance state consumed by one task attempt."""

    rounds: tuple[object, ...]
    input_identities: tuple[ContentIdentity, ...]


@dataclass(frozen=True, slots=True)
class CellTask:
    """One exact cohort/cell task owning all five acquisition rounds."""

    cell: CellSpec
    config: AnalysisConfig | None
    config_digest: str
    input_identities: tuple[ContentIdentity, ...]
    analyzer: TaskAnalyzer
    software_identity: str = CYEXM_PIN
    native_threads: int = 1
    force: bool = False
    scratch_root: Path | None = None
    preflight: PreflightSnapshot | None = None

    @property
    def key(self) -> str:
        """Return the path-safe, exact cohort-cell task identity."""
        if not _SAFE_COMPONENT.fullmatch(
            self.cell.cohort
        ) or not _SAFE_COMPONENT.fullmatch(self.cell.cell_id):
            raise ValueError("cohort and cell_id must be path-safe components")
        return f"{self.cell.cohort}-{self.cell.cell_id}"


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """Completion or failed-attempt record; resource fields are operational only."""

    key: str
    attempt_id: str
    attempt_dir: Path
    config_digest: str
    input_identities: tuple[ContentIdentity, ...]
    software_identity: str
    output_identities: tuple[ContentIdentity, ...]
    status: Literal["complete", "failed"]
    started_at: float
    completed_at: float
    peak_rss_bytes: int
    elapsed_seconds: float
    process_read_bytes: int
    scratch_bytes: int
    reused: bool = False

    @property
    def software_versions(self) -> tuple[str, ...]:
        """Return the immutable pinned software identity in record-friendly form."""
        return (self.software_identity,)


@dataclass(frozen=True, slots=True)
class _TerminalSnapshot:
    """One parsed terminal pointer bound to the exact bytes read from disk."""

    identity: ContentIdentity
    payload: bytes
    pointer: Mapping[str, object]


def configure_native_threads(native_threads: int) -> None:
    """Set every supported native runtime thread variable before scientific imports."""
    if isinstance(native_threads, bool) or native_threads < 1:
        raise ValueError("native_threads must be a positive integer")
    value = str(native_threads)
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _fsync_directory(path: Path) -> None:
    """Durably publish a directory-entry change."""
    directory = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _remove_pointer(path: Path) -> None:
    """Remove one pointer without following it and fsync the owning directory."""
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _identity_payload(
    values: tuple[ContentIdentity, ...],
) -> list[dict[str, object]]:
    return [dict(value.to_dict()) for value in values]


def _identity_values(value: object) -> tuple[ContentIdentity, ...]:
    if not isinstance(value, list):
        raise TypeError("identities must be a list")
    if not all(isinstance(item, Mapping) for item in value):
        raise TypeError("task identities must be mappings")
    return tuple(
        ContentIdentity.from_dict(item, name="task content identity") for item in value
    )


def _record_payload(record: TaskRecord) -> dict[str, object]:
    return {
        "key": record.key,
        "attempt_id": record.attempt_id,
        "attempt_dir": str(record.attempt_dir),
        "config_digest": record.config_digest,
        "input_identities": _identity_payload(record.input_identities),
        "software_identity": record.software_identity,
        "output_identities": _identity_payload(record.output_identities),
        "status": record.status,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "peak_rss_bytes": record.peak_rss_bytes,
        "elapsed_seconds": record.elapsed_seconds,
        "process_read_bytes": record.process_read_bytes,
        "scratch_bytes": record.scratch_bytes,
    }


def _record_from_payload(payload: Mapping[str, object]) -> TaskRecord:
    required = {
        "key",
        "attempt_id",
        "attempt_dir",
        "config_digest",
        "input_identities",
        "software_identity",
        "output_identities",
        "status",
        "started_at",
        "completed_at",
        "peak_rss_bytes",
        "elapsed_seconds",
        "process_read_bytes",
        "scratch_bytes",
    }
    if set(payload) != required or payload["status"] not in {"complete", "failed"}:
        raise ValueError("invalid task completion record")
    if not isinstance(payload["attempt_id"], str) or not re.fullmatch(
        r"[0-9a-f]{32}", payload["attempt_id"]
    ):
        raise ValueError("task record attempt_id must be a UUID hex component")
    numeric_names = (
        "started_at",
        "completed_at",
        "peak_rss_bytes",
        "elapsed_seconds",
        "process_read_bytes",
        "scratch_bytes",
    )
    if any(
        isinstance(payload[name], bool)
        or not isinstance(payload[name], int | float)
        or not math.isfinite(float(payload[name]))
        or float(payload[name]) < 0.0
        for name in numeric_names
    ):
        raise ValueError("task record numeric fields must be finite and nonnegative")
    if float(payload["completed_at"]) < float(payload["started_at"]):
        raise ValueError("task record timestamps must be ordered")
    return TaskRecord(
        key=str(payload["key"]),
        attempt_id=str(payload["attempt_id"]),
        attempt_dir=Path(str(payload["attempt_dir"])),
        config_digest=str(payload["config_digest"]),
        input_identities=_identity_values(payload["input_identities"]),
        software_identity=str(payload["software_identity"]),
        output_identities=_identity_values(payload["output_identities"]),
        status=payload["status"],
        started_at=float(payload["started_at"]),
        completed_at=float(payload["completed_at"]),
        peak_rss_bytes=int(payload["peak_rss_bytes"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        process_read_bytes=int(payload["process_read_bytes"]),
        scratch_bytes=int(payload["scratch_bytes"]),
    )


def _current_identities_match(values: tuple[ContentIdentity, ...]) -> bool:
    return all(value.matches_path() for value in values)


def _reusable_input_identities(
    task: CellTask | ArtifactTask, record: TaskRecord
) -> tuple[ContentIdentity, ...] | None:
    """Return the exact identities that authorize reuse for this task invocation."""
    if record.input_identities != task.input_identities:
        return None
    return task.input_identities


def _terminal_snapshot(terminal: Path) -> _TerminalSnapshot | None:
    """Read, parse, and recheck a terminal while its path remains authoritative."""
    try:
        identity, payload = ContentIdentity.read_bytes(terminal)
        pointer = json.loads(payload.decode("utf-8"))
        if not isinstance(pointer, Mapping) or not identity.matches_path():
            return None
    except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError):
        return None
    return _TerminalSnapshot(identity, payload, pointer)


def _read_reusable(
    task: CellTask | ArtifactTask,
    terminal: Path,
    *,
    terminal_snapshot: _TerminalSnapshot | None = None,
    recheck_terminal: bool = True,
) -> TaskRecord | None:
    if terminal_snapshot is None:
        terminal_snapshot = _terminal_snapshot(terminal)
        if terminal_snapshot is None:
            return None
    try:
        pointer = terminal_snapshot.pointer
        if (
            set(pointer)
            != {
                "status",
                "attempt_id",
                "record",
                "record_identity",
                "output_manifest",
                "output_manifest_identity",
            }
            or pointer["status"] != "complete"
        ):
            return None
        attempt_id = str(pointer["attempt_id"])
        if (
            not re.fullmatch(r"[0-9a-f]{32}", attempt_id)
            or pointer["record"] != "record.json"
            or pointer["output_manifest"] != "output-manifest.json"
        ):
            return None
        attempts_root = (terminal.parent / "attempts").resolve(strict=True)
        attempt_entry = attempts_root / attempt_id
        if attempt_entry.is_symlink() or not stat.S_ISDIR(attempt_entry.stat().st_mode):
            return None
        attempt_dir = attempt_entry.resolve(strict=True)
        if attempt_dir.parent != attempts_root:
            return None
        record_path = attempt_dir / "record.json"
        actual_record_identity, record_bytes = ContentIdentity.read_bytes(record_path)
        record = _record_from_payload(json.loads(record_bytes.decode("utf-8")))
        manifest_path = attempt_dir / "output-manifest.json"
        actual_manifest_identity, manifest_bytes = ContentIdentity.read_bytes(
            manifest_path
        )
        record_identity = ContentIdentity.from_dict(
            pointer["record_identity"], name="terminal record identity"
        )
        manifest_identity = ContentIdentity.from_dict(
            pointer["output_manifest_identity"], name="terminal manifest identity"
        )
        if (
            actual_record_identity != record_identity
            or actual_manifest_identity != manifest_identity
        ):
            return None
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if set(manifest) != {"outputs"} or not isinstance(manifest["outputs"], list):
            return None
        expected: list[ContentIdentity] = []
        expected_relative: set[str] = set()
        for entry in manifest["outputs"]:
            if not isinstance(entry, Mapping) or set(entry) != {"relative", "identity"}:
                return None
            relative = entry["relative"]
            if not isinstance(relative, str) or Path(relative).is_absolute():
                return None
            entry_path = attempt_dir / relative
            if entry_path.is_symlink():
                return None
            path = entry_path.resolve(strict=True)
            if (
                attempt_dir not in path.parents
                or path.is_symlink()
                or not stat.S_ISREG(path.stat().st_mode)
            ):
                return None
            expected_relative.add(relative)
            identity_payload = entry["identity"]
            if not isinstance(identity_payload, Mapping):
                return None
            expected.append(ContentIdentity.from_dict(identity_payload))
        actual_relative = {
            path.relative_to(attempt_dir).as_posix()
            for path in attempt_dir.rglob("*")
            if path.relative_to(attempt_dir).as_posix()
            not in {"record.json", "output-manifest.json"}
            and not path.is_symlink()
            and stat.S_ISREG(path.stat().st_mode)
        }
        if (
            expected_relative != actual_relative
            or tuple(expected) != record.output_identities
        ):
            return None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    reusable_inputs = _reusable_input_identities(task, record)
    if (
        record.status != "complete"
        or record.key != task.key
        or record.attempt_id != attempt_id
        or record.config_digest != task.config_digest
        or record.software_identity != task.software_identity
        or reusable_inputs is None
        or record.attempt_dir.resolve(strict=True) != attempt_dir
        or (recheck_terminal and not terminal_snapshot.identity.matches_path())
        or not actual_record_identity.matches_path()
        or not actual_manifest_identity.matches_path()
        or not record.output_identities
        or any(
            path.is_symlink()
            or not stat.S_ISREG(path.stat().st_mode)
            or attempt_dir not in path.resolve(strict=True).parents
            for path in (Path(identity.path) for identity in record.output_identities)
        )
        or not _current_identities_match(reusable_inputs)
        or not _current_identities_match(record.output_identities)
    ):
        return None
    return replace(record, reused=True)


def _csv_table(path: Path, rows: tuple[Mapping[str, object], ...]) -> None:
    columns = tuple(sorted({key for row in rows for key in row}))
    if not columns:
        columns = ("status",)
        rows = ({"status": "empty"},)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
        handle.flush()
        os.fsync(handle.fileno())


def _validate_output(path: Path) -> None:
    """Reopen staged output content before immutable publication."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"output is not a nonempty regular file: {path.name}")
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header or any(len(row) != len(header) for row in reader):
                raise ValueError(f"invalid CSV schema: {path.name}")
    elif path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".npz":
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            for value in archive.values():
                _ = value.shape
    elif path.suffix in {".png", ".tif", ".tiff", ".pdf", ".svg"}:
        return
    else:
        raise ValueError(f"unrecognized output diagnostic type: {path.name}")


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if not item.is_symlink() and stat.S_ISREG(item.stat().st_mode)
    )


def _sample_resources(
    process: psutil.Process,
    scratch_dir: Path,
    stop: threading.Event,
    values: dict[str, int],
) -> None:
    """Continuously retain bounded operational RSS and scratch high-water marks."""
    while not stop.wait(0.02):
        values["rss"] = max(values["rss"], process.memory_info().rss)
        values["scratch"] = max(values["scratch"], _directory_size(scratch_dir))


def _qc_gate(task: CellTask) -> None:
    if task.config is None:
        return
    decisions = {
        (item.cohort, item.cell_id, item.acquisition_round): item
        for item in task.config.qc_decisions
    }
    missing = [
        item.acquisition_round
        for item in task.cell.rounds
        if item.acquisition_round != task.config.reference_acquisition_round
        and (item.cohort, item.cell_id, item.acquisition_round) not in decisions
    ]
    if missing:
        raise ValueError(
            f"target-blind QC decisions are missing for moving rounds: {missing}"
        )


def _preflight_task(task: CellTask) -> CellTask:
    """Validate current source/Cy-ExM evidence before reuse or target analysis."""
    if task.config is None:
        return task
    _qc_gate(task)
    from cyclic_stability.cyexm_artifacts import validate_cell_inputs
    from cyclic_stability.provenance import channel_tiff_paths

    report = validate_cell_inputs(task.cell)
    identities: list[ContentIdentity] = []
    identities_by_path: dict[Path, ContentIdentity] = {}
    canonical_rounds: list[object] = []
    canonical_artifacts: list[object] = []

    def retain_identity(identity: ContentIdentity) -> Path:
        path = Path(identity.path)
        prior = identities_by_path.get(path)
        if prior is not None and prior != identity:
            raise ValueError(f"conflicting source snapshots for canonical path: {path}")
        if prior is None:
            identities_by_path[path] = identity
            identities.append(identity)
        return path

    def snapshot_path(path: Path) -> Path:
        return retain_identity(ContentIdentity.from_path(path))

    decisions = {
        (item.cohort, item.cell_id, item.acquisition_round): item.decision
        for item in task.config.qc_decisions
    }
    native_rounds = (
        {
            round_spec.acquisition_round
            for round_spec in task.cell.rounds
            if round_spec.acquisition_round == task.config.reference_acquisition_round
            or decisions[
                (
                    round_spec.cohort,
                    round_spec.cell_id,
                    round_spec.acquisition_round,
                )
            ]
            == "accept"
        }
        if task.cell.cohort in {"golgi", "lamp1"}
        else set()
    )
    for round_spec, artifacts in zip(task.cell.rounds, report.rounds, strict=True):
        if len(artifacts.completion_record_paths) != len(
            artifacts.completion_records
        ) or len(artifacts.completion_record_identities) != len(
            artifacts.completion_records
        ):
            raise ValueError("validated artifact record paths are incomplete")
        canonical_nd2 = snapshot_path(round_spec.nd2_path)
        canonical_manifest = snapshot_path(round_spec.dataset_manifest)
        profile_identity = getattr(artifacts, "execution_profile_identity", None)
        if profile_identity is None:
            canonical_profile = snapshot_path(round_spec.execution_profile)
        elif not isinstance(profile_identity, ContentIdentity):
            raise ValueError("validated execution-profile identity is inconsistent")
        else:
            canonical_profile = retain_identity(profile_identity)
            if round_spec.execution_profile.resolve(strict=True) != canonical_profile:
                raise ValueError("validated execution-profile identity is inconsistent")
        canonical_tiffs = tuple(
            snapshot_path(path) for _, path in channel_tiff_paths(round_spec)
        )
        tiff_parents = {path.parent for path in canonical_tiffs}
        if len(tiff_parents) != 1:
            raise ValueError(
                "source TIFF channels do not share one canonical directory"
            )
        canonical_tiff_directory = next(iter(tiff_parents))

        artifact_sources = tuple(
            (name, path)
            for name, path in (
                ("registered_dapi", artifacts.registered_dapi),
                ("registered_tubulin", artifacts.registered_tubulin),
                ("registered_target", artifacts.registered_target),
                ("affine", artifacts.affine),
                ("warp", artifacts.warp),
                ("pre_metrics", artifacts.pre_metrics),
                ("linear_metrics", artifacts.linear_metrics),
                ("nonlinear_metrics", artifacts.nonlinear_metrics),
                ("chunk_manifest", artifacts.chunk_manifest),
            )
            if path is not None
        )
        artifact_identities = getattr(artifacts, "artifact_identities", ())
        if artifact_identities and len(artifact_identities) != len(artifact_sources):
            raise ValueError("validated artifact identities are incomplete")
        artifact_paths: dict[str, Path | None] = {
            name: None
            for name in (
                "affine",
                "warp",
                "pre_metrics",
                "linear_metrics",
                "nonlinear_metrics",
                "chunk_manifest",
            )
        }
        for index, (name, path) in enumerate(artifact_sources):
            if artifact_identities:
                identity = artifact_identities[index]
                if not isinstance(identity, ContentIdentity) or path.resolve(
                    strict=True
                ) != Path(identity.path):
                    raise ValueError("validated artifact identity is inconsistent")
                artifact_paths[name] = retain_identity(identity)
            else:
                artifact_paths[name] = snapshot_path(path)
        artifact_paths["moving_crop"] = (
            None
            if artifacts.moving_crop is None
            else snapshot_path(artifacts.moving_crop)
        )
        canonical_record_paths: list[Path] = []
        for record_path, identity in zip(
            artifacts.completion_record_paths,
            artifacts.completion_record_identities,
            strict=True,
        ):
            if not isinstance(identity, ContentIdentity) or record_path.resolve(
                strict=True
            ) != Path(identity.path):
                raise ValueError("validated artifact record identity is inconsistent")
            canonical_record_paths.append(retain_identity(identity))
        canonical_native_target = artifacts.native_target
        native_identity = artifacts.native_target_identity
        if canonical_native_target is not None and isinstance(
            native_identity, ContentIdentity
        ):
            if canonical_native_target.resolve(strict=True) != Path(
                native_identity.path
            ):
                raise ValueError("validated native target identity is inconsistent")
            canonical_native_target = Path(native_identity.path)
        if (
            round_spec.acquisition_round in native_rounds
            and canonical_native_target is not None
        ):
            if not isinstance(native_identity, ContentIdentity):
                raise ValueError("validated native target identity is inconsistent")
            retain_identity(native_identity)
        canonical_rounds.append(
            replace(
                round_spec,
                nd2_path=canonical_nd2,
                tiff_directory=canonical_tiff_directory,
                dataset_manifest=canonical_manifest,
                execution_profile=canonical_profile,
            )
        )
        canonical_artifacts.append(
            replace(
                artifacts,
                workflow_output_root=artifacts.workflow_output_root.resolve(
                    strict=True
                ),
                **artifact_paths,
                completion_record_paths=tuple(canonical_record_paths),
                native_target=canonical_native_target,
            )
        )
    canonical_cell = replace(task.cell, rounds=tuple(canonical_rounds))
    matching_cells = tuple(
        cell
        for cell in task.config.cells
        if (cell.cohort, cell.cell_id) == (task.cell.cohort, task.cell.cell_id)
    )
    if len(matching_cells) != 1:
        raise ValueError("task cell is not unique in the analysis configuration")
    canonical_config = replace(
        task.config,
        cells=tuple(
            canonical_cell
            if (cell.cohort, cell.cell_id) == (task.cell.cohort, task.cell.cell_id)
            else cell
            for cell in task.config.cells
        ),
    )
    snapshot = PreflightSnapshot(tuple(canonical_artifacts), tuple(identities))
    return replace(
        task,
        cell=canonical_cell,
        config=canonical_config,
        input_identities=(*task.input_identities, *snapshot.input_identities),
        preflight=snapshot,
    )


def default_marker_analyzer(task: CellTask, output_dir: Path) -> TaskOutput:
    """Compose the existing provenance, marker, and QC analyzers for one cell."""
    _qc_gate(task)  # Must precede any analyzer import or target-volume opening.
    if task.config is None:
        raise ValueError("production marker analysis requires an AnalysisConfig")
    from cyclic_stability.golgi import analyze_golgi_cell
    from cyclic_stability.images import load_volume
    from cyclic_stability.lysosome import analyze_lamp1_cell
    from cyclic_stability.morphology import threshold_ensemble
    from cyclic_stability.nuclear import analyze_lamin_cell
    from cyclic_stability.provenance import channel_tiff_paths
    from cyclic_stability.registration_qc import analyze_registration_qc
    from cyclic_stability.signal import measure_signal, raw_structure_roi

    if task.preflight is None:
        raise ValueError("production task lacks validated preflight snapshot")
    artifacts = task.preflight.rounds
    qc = analyze_registration_qc(task.cell, artifacts, output_dir / "target-blind-qc")
    analyzer = {
        "golgi": analyze_golgi_cell,
        "lamin": analyze_lamin_cell,
        "lamp1": analyze_lamp1_cell,
    }.get(task.cell.cohort)
    if analyzer is None:
        raise ValueError(f"unsupported cohort: {task.cell.cohort}")
    result = analyzer(task.cell, artifacts, task.config, output_dir / "diagnostics")
    decisions = {
        (item.cohort, item.cell_id, item.acquisition_round): item.decision
        for item in task.config.qc_decisions
    }
    signal_rows: list[Mapping[str, object]] = []
    for round_spec in task.cell.rounds:
        if (
            round_spec.acquisition_round != task.config.reference_acquisition_round
            and decisions[
                (round_spec.cohort, round_spec.cell_id, round_spec.acquisition_round)
            ]
            != "accept"
        ):
            continue
        paths = dict(channel_tiff_paths(round_spec))
        raw_dapi = load_volume(paths["dapi"])
        raw_target = load_volume(paths["target"])
        roi = raw_structure_roi(round_spec.cohort, raw_dapi, task.config)
        members = threshold_ensemble(
            raw_target,
            roi,
            task.config.ensembles.gaussian_sigma_um,
            task.config.ensembles.threshold_sbr,
            task.config.voxel_spacing_xyz_um,
        )
        signal_rows.extend(
            {
                field: dict(value) if isinstance(value, Mapping) else value
                for field in row.__dataclass_fields__
                for value in (getattr(row, field),)
            }
            for row in measure_signal(round_spec, raw_target, roi, members)
        )
    return TaskOutput(
        tables={
            "registration-qc": qc.registration_rows,
            "deformation-qc": qc.deformation_rows,
            "summary": tuple(asdict(item) for item in result.summary_rows),
            "ensemble": tuple(asdict(item) for item in result.ensemble_rows),
            "raw-signal": tuple(signal_rows),
            "nuclear-component-qc": tuple(
                dict(row) for row in result.component_qc_rows
            ),
        },
        diagnostics=(*qc.overlay_paths, *result.diagnostics),
    )


def _task_directory(task: CellTask | ArtifactTask, run_root: Path) -> Path:
    """Return one validated task directory in the shared immutable store."""
    if not _SAFE_COMPONENT.fullmatch(task.key):
        raise ValueError("task key must be a path-safe component")
    tasks_root = run_root / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    if tasks_root.is_symlink() or not stat.S_ISDIR(tasks_root.stat().st_mode):
        raise ValueError("task root must be a real nonsymlink directory")
    task_dir = tasks_root / task.key
    task_dir.mkdir(parents=True, exist_ok=True)
    if task_dir.is_symlink() or not stat.S_ISDIR(task_dir.stat().st_mode):
        raise ValueError("task directory must be a real nonsymlink directory")
    attempts_root = task_dir / "attempts"
    if attempts_root.exists() and (
        attempts_root.is_symlink() or not stat.S_ISDIR(attempts_root.stat().st_mode)
    ):
        raise ValueError("attempts directory must be a real nonsymlink directory")
    attempts_root.mkdir(exist_ok=True)
    return task_dir


def _produce_cell_outputs(task: CellTask, attempt_dir: Path) -> tuple[Path, ...]:
    """Serialize one marker analyzer result inside its unique attempt."""
    output = task.analyzer(task, attempt_dir)
    table_paths: list[Path] = []
    for name, rows in sorted(output.tables.items()):
        if not _SAFE_COMPONENT.fullmatch(name):
            raise ValueError("table names must be path-safe")
        path = attempt_dir / f"{name}.csv"
        _csv_table(path, rows)
        table_paths.append(path)
    return tuple(table_paths) + tuple(output.diagnostics)


def run_cell_task(task: CellTask, run_dir: Path) -> TaskRecord:
    """Run one immutable cell attempt or reuse an exactly matching completion."""
    configure_native_threads(task.native_threads)
    if any(
        not isinstance(identity, ContentIdentity) for identity in task.input_identities
    ) or (task.config is None and not task.input_identities):
        raise TypeError("cell task inputs require content-bound identities")
    run_root = run_dir.resolve()
    task_dir = _task_directory(task, run_root)
    with (task_dir / ".task.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _run_task_locked(task, task_dir, run_root)


def run_artifact_task(task: ArtifactTask, run_dir: Path) -> TaskRecord:
    """Run or exactly reuse one generic immutable file-producing attempt.

    This is the canonical entry point for non-cell artifacts that require the same
    fail-closed terminal and SHA-bound reuse contract as scientific cell tasks. It
    publishes ``running`` under the task lock, invokes ``task.snapshot``, decides
    reuse from that current snapshot, and passes only its context to the producer.

    Parameters
    ----------
    task : ArtifactTask
        Unadmitted request. Its internal digest, identity, and context fields must
        be empty; the snapshot callback supplies them under lock.
    run_dir : pathlib.Path
        Run root receiving the task attempt tree. Existing immutable attempts are
        retained and only the atomic terminal pointer changes authority.

    Returns
    -------
    TaskRecord
        Complete, reused, or failed attempt record. Snapshot and producer failures
        are represented by a failed record and authoritative failed terminal.

    Raises
    ------
    TypeError
        If the caller attempts to prepopulate admitted identity or context fields.
    ValueError
        If the run task tree is unsafe or invalid before an attempt can be admitted.

    See Also
    --------
    ArtifactTask :
        Defines the snapshot and producer responsibilities for this lifecycle.
    run_cell_task :
        Executes normal configured marker-cell analysis with strict preflight.

    Notes
    -----
    Once a running terminal is published, all snapshot or producer exceptions are
    persisted as failed terminal state rather than re-exposing prior success.
    """
    configure_native_threads(task.native_threads)
    if task.config_digest or task.input_identities or task.context is not None:
        raise TypeError("artifact task identity and context must come from snapshot")
    run_root = run_dir.resolve()
    task_dir = _task_directory(task, run_root)
    with (task_dir / ".task.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _run_task_locked(task, task_dir, run_root)


def _run_task_locked(
    task: CellTask | ArtifactTask,
    task_dir: Path,
    run_root: Path,
) -> TaskRecord:
    """Execute under the kernel lock that serializes reuse and publication."""
    terminal_path = task_dir / "terminal.json"
    _remove_pointer(task_dir / "complete.json")
    requires_current_preflight = isinstance(task, ArtifactTask) or (
        isinstance(task, CellTask) and task.config is not None
    )
    prior_terminal_snapshot: _TerminalSnapshot | None = None
    if requires_current_preflight and not task.force:
        prior_terminal_snapshot = _terminal_snapshot(terminal_path)
    elif not task.force:
        reusable = _read_reusable(task, terminal_path)
        if reusable is not None:
            return reusable
    _remove_pointer(terminal_path)
    attempt_id = uuid.uuid4().hex
    attempt_dir = task_dir / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(attempt_dir.parent)
    _atomic_bytes(
        terminal_path,
        _json_bytes({"status": "running", "attempt_id": attempt_id}),
    )
    running_terminal_identity = ContentIdentity.from_path(terminal_path)
    process = psutil.Process()
    before = process.io_counters()
    started = time.time()
    started_monotonic = time.monotonic()
    peak = process.memory_info().rss
    samples = {"rss": peak, "scratch": 0}
    stop = threading.Event()
    sampler: threading.Thread | None = None
    scratch_dir = attempt_dir / "scratch"
    previous_temp: dict[str, str | None] = {}
    owned_inputs: tuple[RetainedContent, ...] = ()
    try:
        if isinstance(task, ArtifactTask):
            admitted = task.snapshot(run_root, task_dir)
            if isinstance(admitted, ArtifactSnapshot):
                owned_inputs = tuple(
                    item
                    for item in admitted.owned_inputs
                    if isinstance(item, RetainedContent)
                )
            if (
                not isinstance(admitted, ArtifactSnapshot)
                or not admitted.config_digest
                or not admitted.input_identities
                or not all(
                    isinstance(identity, ContentIdentity)
                    for identity in admitted.input_identities
                )
                or len(owned_inputs) != len(admitted.owned_inputs)
                or any(
                    item.identity not in admitted.input_identities
                    for item in owned_inputs
                )
            ):
                raise TypeError(
                    "artifact snapshot requires a digest and owned content-bound identities"
                )
            for item in owned_inputs:
                item.validate()
            configure_native_threads(admitted.native_threads)
            task = replace(
                task,
                config_digest=admitted.config_digest,
                input_identities=admitted.input_identities,
                context=admitted.context,
                native_threads=admitted.native_threads,
                scratch_root=admitted.scratch_root,
            )
        elif isinstance(task, CellTask):
            task = _preflight_task(task)
            if not task.input_identities or not all(
                isinstance(identity, ContentIdentity)
                for identity in task.input_identities
            ):
                raise TypeError("cell task inputs require content-bound identities")
            _qc_gate(task)
        scratch_dir = (
            (task.scratch_root / task.key / attempt_id)
            if task.scratch_root is not None
            else attempt_dir / "scratch"
        )
        scratch_dir.mkdir(parents=True, exist_ok=False)
        sampler = threading.Thread(
            target=_sample_resources,
            args=(process, scratch_dir, stop, samples),
            daemon=True,
        )
        previous_temp = {
            name: os.environ.get(name) for name in ("TMPDIR", "TEMP", "TMP")
        }
        for name in previous_temp:
            os.environ[name] = str(scratch_dir)
        sampler.start()
        if prior_terminal_snapshot is not None:
            reusable = _read_reusable(
                task,
                terminal_path,
                terminal_snapshot=prior_terminal_snapshot,
                recheck_terminal=False,
            )
            if reusable is not None:
                stop.set()
                sampler.join()
                shutil.rmtree(scratch_dir)
                if not _current_identities_match(task.input_identities):
                    raise ValueError(
                        "one or more immutable task inputs changed before reuse"
                    )
                for item in owned_inputs:
                    item.validate()
                if not running_terminal_identity.matches_path():
                    raise ValueError(
                        "authoritative running terminal changed before reuse"
                    )
                _atomic_bytes(terminal_path, prior_terminal_snapshot.payload)
                return reusable
        if not _current_identities_match(task.input_identities):
            raise ValueError(
                "one or more immutable task inputs changed before execution"
            )
        for item in owned_inputs:
            item.validate()
        if isinstance(task, CellTask):
            output_paths = _produce_cell_outputs(task, attempt_dir)
        else:
            output_paths = task.producer(attempt_dir, task.context)
        for item in owned_inputs:
            item.validate()
        if not _current_identities_match(task.input_identities):
            raise ValueError(
                "one or more immutable task inputs changed during execution"
            )
        if not output_paths or not all(
            path.is_file()
            and not path.is_symlink()
            and attempt_dir in path.resolve(strict=True).parents
            for path in output_paths
        ):
            raise ValueError("analyzer did not produce valid task outputs")
        after = process.io_counters()
        stop.set()
        sampler.join()
        peak = max(samples["rss"], process.memory_info().rss)
        scratch_bytes = max(samples["scratch"], _directory_size(scratch_dir))
        shutil.rmtree(scratch_dir)
        if not all(
            path.is_file()
            and not path.is_symlink()
            and attempt_dir in path.resolve(strict=True).parents
            for path in output_paths
        ):
            raise ValueError("scratch cleanup removed a declared task output")
        manifest_paths = tuple(
            sorted(
                (
                    path
                    for path in attempt_dir.rglob("*")
                    if path.is_file() and not path.is_symlink()
                ),
                key=lambda path: path.relative_to(attempt_dir).as_posix(),
            )
        )
        for path in manifest_paths:
            _validate_output(path)
        record = TaskRecord(
            task.key,
            attempt_id,
            attempt_dir,
            task.config_digest,
            task.input_identities,
            task.software_identity,
            tuple(ContentIdentity.from_path(path) for path in manifest_paths),
            "complete",
            started,
            time.time(),
            peak,
            time.monotonic() - started_monotonic,
            max(0, after.read_bytes - before.read_bytes),
            scratch_bytes,
        )
        manifest = {
            "outputs": [
                {
                    "relative": path.relative_to(attempt_dir).as_posix(),
                    "identity": identity.to_dict(),
                }
                for path, identity in zip(
                    manifest_paths, record.output_identities, strict=True
                )
            ]
        }
        manifest_path = attempt_dir / "output-manifest.json"
        _atomic_bytes(manifest_path, _json_bytes(manifest))
        record_path = attempt_dir / "record.json"
        _atomic_bytes(record_path, _json_bytes(_record_payload(record)))
        terminal_payload = _json_bytes(
            {
                "status": "complete",
                "attempt_id": attempt_id,
                "record": "record.json",
                "record_identity": ContentIdentity.from_path(record_path).to_dict(),
                "output_manifest": "output-manifest.json",
                "output_manifest_identity": ContentIdentity.from_path(
                    manifest_path
                ).to_dict(),
            }
        )
        try:
            for item in owned_inputs:
                item.validate()
        except Exception:
            _remove_pointer(record_path)
            _remove_pointer(manifest_path)
            raise
        if not _current_identities_match(task.input_identities):
            _remove_pointer(record_path)
            _remove_pointer(manifest_path)
            raise ValueError(
                "one or more immutable task inputs changed before completion publication"
            )
        _atomic_bytes(terminal_path, terminal_payload)
        return record
    except Exception as error:  # noqa: BLE001 - persist analyzer failures, preserve interrupts
        stop.set()
        if sampler is not None:
            sampler.join()
        after = process.io_counters()
        peak = max(samples["rss"], process.memory_info().rss)
        failure = {
            "attempt_id": attempt_id,
            "config_digest": task.config_digest,
            "error_type": type(error).__name__,
            "input_identities": _identity_payload(task.input_identities),
            "key": task.key,
            "message": str(error),
            "software_identity": task.software_identity,
            "traceback": traceback.format_exc(),
            "started_at": started,
            "failed_at": time.time(),
        }
        failure_path = attempt_dir / "failure.json"
        _atomic_bytes(failure_path, _json_bytes(failure))
        _atomic_bytes(
            terminal_path,
            _json_bytes(
                {
                    "status": "failed",
                    "attempt_id": attempt_id,
                    "failure": "failure.json",
                    "failure_identity": ContentIdentity.from_path(
                        failure_path
                    ).to_dict(),
                }
            ),
        )
        return TaskRecord(
            task.key,
            attempt_id,
            attempt_dir,
            task.config_digest,
            task.input_identities,
            task.software_identity,
            (),
            "failed",
            started,
            time.time(),
            peak,
            time.monotonic() - started_monotonic,
            max(0, after.read_bytes - before.read_bytes),
            max(samples["scratch"], _directory_size(scratch_dir)),
        )
    finally:
        for item in owned_inputs:
            item.close()
        for name, value in previous_temp.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def build_tasks(
    config: AnalysisConfig,
    cohorts: tuple[str, ...],
    *,
    cells: tuple[CellSpec, ...] | None = None,
) -> tuple[CellTask, ...]:
    """Build sorted five-round tasks, failing closed when source identities are absent."""
    selected = (
        list(cells)
        if cells is not None
        else [cell for cell in config.cells if not cohorts or cell.cohort in cohorts]
    )
    tasks: list[CellTask] = []
    for cell in sorted(selected, key=lambda value: (value.cohort, value.cell_id)):
        if len(cell.rounds) != 5:
            raise ValueError("every CellTask must own exactly five rounds")
        # This snapshots paths only; it deliberately performs no target-pixel I/O.
        # The QC gate in ``run_cell_task`` remains ahead of every target load.
        from cyclic_stability.provenance import channel_tiff_paths

        for round_spec in sorted(
            cell.rounds, key=lambda value: value.acquisition_round
        ):
            FileIdentity.from_path(round_spec.nd2_path)
            for _, path in channel_tiff_paths(round_spec):
                FileIdentity.from_path(path)
        tasks.append(
            CellTask(
                cell,
                config,
                config.config_digest,
                (),
                default_marker_analyzer,
            )
        )
    return tuple(tasks)


__all__ = [
    "CYEXM_PIN",
    "ArtifactSnapshot",
    "ArtifactTask",
    "CellTask",
    "ContentIdentity",
    "RetainedContent",
    "TaskOutput",
    "TaskRecord",
    "build_tasks",
    "configure_native_threads",
    "default_marker_analyzer",
    "run_artifact_task",
    "run_cell_task",
]
