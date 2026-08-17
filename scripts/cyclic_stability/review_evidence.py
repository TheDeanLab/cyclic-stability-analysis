"""Descriptor-relative, no-follow reads for target-blind review evidence."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VerifiedEvidenceBytes:
    """One regular evidence file read exactly once through a verified descriptor."""

    path: Path
    size: int
    mtime_ns: int
    sha256: str
    content: bytes


def overlay_qc_root(path: Path) -> Path:
    """Return the lexical QC root encoded by the fixed overlay namespace."""
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.parents[5]
    except IndexError as error:
        raise ValueError("reviewed evidence path has an invalid namespace") from error


def _root_and_relative(trusted_root: Path, path: Path) -> tuple[Path, Path]:
    root_absolute = Path(os.path.abspath(trusted_root))
    try:
        root = root_absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError("trusted QC root is unavailable") from error
    candidate_absolute = Path(os.path.abspath(path))
    try:
        relative = candidate_absolute.relative_to(root_absolute)
    except ValueError:
        try:
            relative = candidate_absolute.relative_to(root)
        except ValueError as error:
            raise ValueError("review evidence escapes the trusted QC root") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("review evidence has an invalid relative path")
    return root, relative


def _open_child(parent: int, name: str, *, directory: bool) -> int:
    try:
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError("review evidence is unavailable") from error
    if stat.S_ISLNK(before.st_mode):
        raise ValueError("review evidence path contains a symlink")
    required = stat.S_ISDIR if directory else stat.S_ISREG
    if not required(before.st_mode):
        kind = "directory" if directory else "regular file"
        raise ValueError(f"review evidence path component is not a {kind}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except OSError as error:
        raise ValueError(
            "review evidence is unavailable without following links"
        ) from error
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        os.close(descriptor)
        raise ValueError("review evidence changed while it was opened")
    return descriptor


@contextmanager
def _open_directory(trusted_root: Path, directory: Path) -> Iterator[tuple[int, Path]]:
    root, relative = _root_and_relative(trusted_root, directory)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(root, flags)
    try:
        for part in relative.parts:
            child = _open_child(descriptor, part, directory=True)
            os.close(descriptor)
            descriptor = child
        resolved = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("review evidence escapes the trusted QC root")
        yield descriptor, root / relative
    finally:
        os.close(descriptor)


def list_verified_directory(trusted_root: Path, directory: Path) -> tuple[str, ...]:
    """List one real descendant directory without following any path symlink."""
    with _open_directory(trusted_root, directory) as (descriptor, _canonical):
        return tuple(os.listdir(descriptor))


def read_verified_evidence(trusted_root: Path, path: Path) -> VerifiedEvidenceBytes:
    """Read one regular descendant once after no-follow containment checks."""
    root, relative = _root_and_relative(trusted_root, path)
    if len(relative.parts) < 2:
        raise ValueError("review evidence must be below the trusted QC root")
    with _open_directory(root, root / Path(*relative.parts[:-1])) as (
        parent,
        _canonical_parent,
    ):
        descriptor = _open_child(parent, relative.parts[-1], directory=False)
        try:
            before = os.fstat(descriptor)
            resolved = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError("review evidence escapes the trusted QC root")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise ValueError("review evidence changed while it was read")
    content = b"".join(chunks)
    if len(content) != before.st_size:
        raise ValueError("review evidence changed while it was read")
    return VerifiedEvidenceBytes(
        root / relative,
        before.st_size,
        before.st_mtime_ns,
        hashlib.sha256(content).hexdigest(),
        content,
    )


__all__ = [
    "VerifiedEvidenceBytes",
    "list_verified_directory",
    "overlay_qc_root",
    "read_verified_evidence",
]
