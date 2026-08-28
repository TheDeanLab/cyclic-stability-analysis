# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cy-exm[workflow] @ git+https://github.com/TheDeanLab/Cy-ExM.git@f6e0de36cd7f6b5e85f093884c2cf224225f4ec2",
#   "nd2==0.11.3",
#   "numpy>=1.26,<3",
#   "scipy>=1.12,<2",
#   "scikit-image>=0.22,<1",
#   "tifffile>=2025.1.10",
#   "PyYAML>=6,<7",
#   "pandas>=2.2,<3",
#   "matplotlib>=3.8,<4",
#   "seaborn>=0.13,<1",
#   "Jinja2>=3.1,<4",
#   "psutil>=5.9,<8",
#   "Pillow>=10,<13",
# ]
# ///
"""Launch or validate the target-blind registration-QC review lifecycle."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

if TYPE_CHECKING:
    from cyclic_stability.review_models import ReviewManifest
    from cyclic_stability.review_store import ReviewStore

_MANIFEST_NAME = "randomization-manifest.json"
_STATE_NAMES = ("review-state.json", "review-state.previous.json")


def _load_x11() -> ctypes.CDLL:
    library = ctypes.util.find_library("X11") or "libX11.so.6"
    return ctypes.CDLL(library)


def _initialize_x11_threads() -> None:
    """Enable Xlib thread safety before any graphical or imaging imports."""
    if not sys.platform.startswith("linux"):
        return
    initializer = _load_x11().XInitThreads
    initializer.argtypes = []
    initializer.restype = ctypes.c_int
    if initializer() == 0:
        raise OSError("XInitThreads failed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-root", type=Path)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--add-revision-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _now() -> datetime:
    return datetime.now(UTC)


def _has_review_state(review_dir: Path) -> bool:
    return (review_dir / _MANIFEST_NAME).exists() or any(
        (review_dir / name).exists() for name in _STATE_NAMES
    )


def _load_manifest(path: Path) -> ReviewManifest:
    from cyclic_stability.review_models import ReviewManifest

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("randomization manifest is malformed") from error
    return ReviewManifest.from_dict(payload)


def _manifest_qc_root(manifest: ReviewManifest) -> Path:
    roots: set[Path] = set()
    for revision in manifest.case_revisions:
        for overlay in revision.overlays:
            path = Path(overlay.identity.path)
            try:
                roots.add(path.parents[5])
            except IndexError as error:
                raise ValueError(
                    "manifest evidence path has an invalid namespace"
                ) from error
    if len(roots) != 1:
        raise ValueError("initial manifest must use exactly one QC root")
    return roots.pop().resolve()


def _validate_initial_manifest(
    manifest: ReviewManifest, requested_qc_root: Path | None
) -> ReviewManifest:
    from cyclic_stability.review_inventory import build_review_manifest

    derived_root = _manifest_qc_root(manifest)
    if requested_qc_root is not None and requested_qc_root.resolve() != derived_root:
        raise ValueError("--qc-root does not match the persisted review manifest")
    rebuilt = build_review_manifest(derived_root, seed=manifest.seed)
    if rebuilt != manifest:
        raise ValueError("reviewed evidence identity changed")
    return rebuilt


def _validate_active_evidence(manifest: ReviewManifest) -> None:
    from cyclic_stability.review_inventory import read_overlay_bytes

    validated: set[tuple[str, str]] = set()
    for presentation in manifest.visual_presentations:
        for overlay in presentation.overlays:
            identity = overlay.identity
            key = (identity.path, identity.sha256)
            if key in validated:
                continue
            validated.add(key)
            read_overlay_bytes(overlay)


def _open_lifecycle(args: argparse.Namespace) -> tuple[ReviewStore, ReviewManifest]:
    from cyclic_stability.review_exports import attach_revision_root
    from cyclic_stability.review_inventory import build_review_manifest
    from cyclic_stability.review_store import ReviewStore

    review_dir: Path = args.review_dir
    existing = _has_review_state(review_dir)
    if args.add_revision_root is not None and not existing:
        raise ValueError("--add-revision-root requires existing state")
    if existing:
        if args.seed is not None:
            raise ValueError("--seed cannot be used when resuming existing state")
        manifest = _load_manifest(review_dir / _MANIFEST_NAME)
        manifest = _validate_initial_manifest(manifest, args.qc_root)
        store = ReviewStore.open(review_dir, manifest, args.reviewer, now=_now)
    else:
        if args.qc_root is None:
            raise ValueError("--qc-root is required for first review creation")
        if review_dir.exists() and any(review_dir.iterdir()):
            raise ValueError("new review directory must be empty")
        seed = args.seed if args.seed is not None else secrets.randbits(63)
        manifest = build_review_manifest(args.qc_root, seed=seed)
        store = ReviewStore.create(review_dir, manifest, args.reviewer, now=_now)
    if args.add_revision_root is not None:
        attach_revision_root(store, args.add_revision_root)
    store.validate_current_bindings(manifest)
    _validate_active_evidence(store.manifest)
    return store, manifest


def _validated_counts(manifest: ReviewManifest) -> tuple[int, int, int, int, int, int]:
    cases = len(manifest.case_revisions)
    visual = sum(
        presentation.repeat_of is None for presentation in manifest.visual_presentations
    )
    repeats = sum(
        presentation.repeat_of is not None
        for presentation in manifest.visual_presentations
    )
    presentations = len(manifest.visual_presentations)
    technical = sum(not revision.overlays for revision in manifest.case_revisions)
    overlays = sum(len(revision.overlays) for revision in manifest.case_revisions)
    return cases, visual, repeats, presentations, technical, overlays


def main(argv: list[str] | None = None) -> int:
    """Create, resume, preflight, or display one target-blind QC review.

    This is the canonical operator entry point for the review lifecycle.  It owns
    creation and reopening of durable state under ``--review-dir``, optional
    attachment of a previously generated target-blind QC revision, and graphical
    application startup after validation.

    It does not generate QC evidence, run or modify registration, configure a VNC
    or other display server, open target-channel images, or promote finalized
    decisions into scientific configuration.  ``--validate-only`` is the supported
    headless inventory and state preflight and imports no Tkinter presentation code.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments without the program name.  ``None`` reads the
        process command line.  ``--qc-root`` is required only for first creation;
        resume derives and revalidates it from the immutable manifest.  ``--seed``
        is creation-only, and ``--add-revision-root`` is resume-only.

    Returns
    -------
    int
        Zero after successful validation or normal GUI shutdown; nonzero when
        lifecycle validation or graphical startup fails.

    See Also
    --------
    cyclic_stability.review_inventory.build_review_manifest :
        Validate target-blind QC evidence and create the initial randomization.
    cyclic_stability.review_store.ReviewStore :
        Own durable, append-preserving review state.
    cyclic_stability.review_gui.ReviewApplication :
        Own widgets after this entry point completes headless-safe validation.

    Notes
    -----
    Normal graphical launch requires an existing GUI-enabled session with Tkinter
    8.6.  This command deliberately provides no VNC, browser, scheduler, server, or
    target-image lifecycle.

    Examples
    --------
    Validate a new review without creating widgets::

        main(["--qc-root", "results/qc-only", "--review-dir", "/tmp/review",
              "--reviewer", "Reviewer", "--seed", "8142026", "--validate-only"])
    """
    args = _parser().parse_args(argv)
    if not args.validate_only:
        try:
            _initialize_x11_threads()
        except (AttributeError, OSError) as error:
            print(
                f"GUI unavailable: could not initialize X11 thread safety: {error}",
                file=sys.stderr,
            )
            return 2
    try:
        store, manifest = _open_lifecycle(args)
    except (OSError, TypeError, ValueError) as error:
        print(f"review preflight failed: {error}", file=sys.stderr)
        return 2
    (
        case_count,
        visual_count,
        repeat_count,
        presentation_count,
        technical_count,
        overlay_count,
    ) = _validated_counts(manifest)
    print(
        f"validated {case_count} cases, {visual_count} visual cases, "
        f"and {repeat_count} repeats; {presentation_count} uniformly blinded "
        f"presentations, {technical_count} technical case, {overlay_count} fully "
        "decoded allowlisted overlays, 0 target basenames, and 0 symlinks"
    )
    if args.validate_only:
        return 0

    try:
        import tkinter as tk
    except ImportError:
        print(
            "GUI unavailable: Tkinter 8.6 is required in a GUI-enabled session.",
            file=sys.stderr,
        )
        return 2
    try:
        root = tk.Tk()
    except tk.TclError:
        print(
            "GUI unavailable: launch this command inside a GUI-enabled session.",
            file=sys.stderr,
        )
        return 2
    try:
        from cyclic_stability.review_gui import ReviewApplication

        ReviewApplication(root, store, manifest)
        root.mainloop()
    except (OSError, TypeError, ValueError) as error:
        root.destroy()
        print(f"review application failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
