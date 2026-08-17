"""Thin Tkinter presentation layer for target-blind registration-QC review.

``ReviewApplication`` is the sole widget owner.  It composes the durable
``ReviewStore`` and the existing export/report functions; it does not own QC
inventory generation, registration, VNC, or target-channel image lifecycles.
The widget-free ``ReviewViewModel`` and ``ReviewController`` keep blinded state
and navigation behavior independently testable.
"""

from __future__ import annotations

import hashlib
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Literal, cast

from PIL import Image, ImageTk

from cyclic_stability.review_inventory import OVERLAY_NAME
from cyclic_stability.review_models import (
    REASON_TOKENS,
    AssessmentStatus,
    CaseKey,
    OverlayEvidence,
    Presentation,
    ReviewManifest,
)
from cyclic_stability.review_store import ReviewStore

_CHANNELS = ("dapi", "tubulin")
_STAGES = ("pre", "affine", "nonlinear")
_VIEWS = ("xy", "xz", "yz")
_STATUSES: tuple[AssessmentStatus, ...] = (
    "accept",
    "needs-registration-improvement",
    "exclude",
    "needs-adjudication",
)
_STATUS_LABELS = {
    "accept": "Accept",
    "needs-registration-improvement": "Needs registration improvement",
    "exclude": "Exclude",
    "needs-adjudication": "Needs adjudication",
}
_COMMON_REASONS = (
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
)
_REASON_KEYS = dict(zip("1234567890", _COMMON_REASONS, strict=True))
_STATUS_KEYS: dict[str, AssessmentStatus] = {
    "a": "accept",
    "r": "needs-registration-improvement",
    "e": "exclude",
    "u": "needs-adjudication",
}

KeyboardAction = tuple[Literal["status", "navigate", "reason"], str]
EvidenceTokenGrid = dict[str, dict[str, dict[str, str]]]


class IncompleteAssessmentError(ValueError):
    """A dirty visual assessment is missing a status or structured reason."""


@dataclass(frozen=True, slots=True)
class VisualPresentation:
    """One path-free, biologically blinded visual-review presentation."""

    opaque_id: str
    progress_position: int
    progress_total: int
    revision_label: str
    evidence_tokens: EvidenceTokenGrid
    status: AssessmentStatus | None
    reasons: tuple[str, ...]
    notes: str


@dataclass(frozen=True, slots=True)
class ReviewViewModel:
    """Immutable path-free values for the visual pass.

    This model is the presentation boundary used by ``ReviewApplication``.  It
    intentionally excludes biological keys, filesystem paths, automated flags,
    repeat identity, and technical reasons.  Durable state remains owned by
    ``ReviewStore``.

    Attributes
    ----------
    presentations : tuple of VisualPresentation
        Randomized visual values in the active manifest's display order.
    """

    presentations: tuple[VisualPresentation, ...]

    @classmethod
    def from_store(
        cls,
        store: ReviewStore,
        manifest: ReviewManifest | None = None,
    ) -> ReviewViewModel:
        """Build blinded values from the store's immutable active manifest.

        Parameters
        ----------
        store : ReviewStore
            Durable review owner.  Its active manifest is always the evidence
            source, including attached remediation revisions.
        manifest : ReviewManifest, optional
            Compatibility argument for callers that already hold the initial
            manifest.  It is never used in place of ``store.manifest``.

        Returns
        -------
        ReviewViewModel
            Path-free values safe to bind to visual-review widgets.

        Notes
        -----
        This operation does not read image bytes or resolve evidence paths.
        """
        del manifest
        active = store.manifest
        total = len(active.visual_presentations)
        values: list[VisualPresentation] = []
        for position, presentation in enumerate(active.visual_presentations, start=1):
            history = store.state.assessment_history.get(presentation.opaque_id, ())
            assessment = (
                history[-1]
                if history and history[-1].revision_id == presentation.revision_id
                else None
            )
            case_history = store.state.case_histories[presentation.case_token]
            revision_number = next(
                index
                for index, revision in enumerate(case_history.revisions, start=1)
                if revision.revision_id == presentation.revision_id
            )
            grid: EvidenceTokenGrid = {
                channel: {stage: {} for stage in _STAGES} for channel in _CHANNELS
            }
            for overlay in presentation.overlays:
                grid[overlay.channel][overlay.stage][overlay.view] = _evidence_token(
                    presentation, overlay
                )
            values.append(
                VisualPresentation(
                    presentation.opaque_id,
                    position,
                    total,
                    f"Revision {revision_number}",
                    grid,
                    assessment.status if assessment is not None else None,
                    assessment.reasons if assessment is not None else (),
                    assessment.notes if assessment is not None else "",
                )
            )
        return cls(tuple(values))


@dataclass(frozen=True, slots=True)
class _ResolvedEvidence:
    basename: str
    overlay: OverlayEvidence


def _evidence_token(presentation: Presentation, overlay: OverlayEvidence) -> str:
    parts = (
        "review-evidence-v1",
        presentation.opaque_id,
        presentation.revision_id,
        overlay.channel,
        overlay.stage,
        overlay.view,
        overlay.identity.sha256,
    )
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


class EvidenceResolver:
    """Privately resolve opaque tokens through one immutable active manifest.

    The resolver owns no paths beyond those already recorded in the supplied
    manifest.  It rejects raw paths and revalidates the allowlisted basename and
    complete SHA-256 digest immediately before returning a path to an image
    loader.
    """

    def __init__(self, manifest: ReviewManifest) -> None:
        self._evidence: dict[str, _ResolvedEvidence] = {}
        for presentation in manifest.visual_presentations:
            for overlay in presentation.overlays:
                token = _evidence_token(presentation, overlay)
                if token in self._evidence:
                    raise ValueError("opaque evidence tokens must be unique")
                self._evidence[token] = _ResolvedEvidence(
                    Path(overlay.identity.path).name, overlay
                )

    def resolve(self, token: str) -> Path:
        """Return a revalidated allowlisted path for one opaque token."""
        record = self._evidence.get(token)
        if record is None:
            raise ValueError("unknown opaque evidence token")
        overlay = record.overlay
        path = Path(overlay.identity.path)
        match = OVERLAY_NAME.fullmatch(path.name)
        if (
            path.name != record.basename
            or match is None
            or match.group(2) != overlay.channel
            or match.group(3) != overlay.stage
            or match.group(4) != overlay.view
        ):
            raise ValueError("review evidence basename is no longer allowlisted")
        try:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as error:
            raise ValueError("review evidence is unavailable") from error
        if digest != overlay.identity.sha256:
            raise ValueError("review evidence identity changed")
        return path

    def load_image(self, token: str) -> Image.Image:
        """Load one revalidated RGB TIFF as an independent Pillow image."""
        path = self.resolve(token)
        try:
            with Image.open(path) as source:
                source.load()
                image = source.copy()
        except (OSError, SyntaxError) as error:
            raise ValueError("review evidence is not a readable image") from error
        if (
            image.mode != "RGB"
            or image.getbands() != ("R", "G", "B")
            or image.width < 1
            or image.height < 1
        ):
            raise ValueError("review evidence must be a nonempty RGB image")
        return image

    def load_thumbnail(self, token: str, maximum_size: tuple[int, int]) -> Image.Image:
        """Return an aspect-preserving thumbnail of one revalidated RGB TIFF."""
        if len(maximum_size) != 2 or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 1
            for item in maximum_size
        ):
            raise ValueError("thumbnail bounds must be two positive integers")
        image = self.load_image(token)
        image.thumbnail(maximum_size, Image.Resampling.LANCZOS)
        return image


def keyboard_action(key: str) -> KeyboardAction | None:
    """Map one Tk keysym to a widget-free visual-review action."""
    normalized = key.lower()
    if normalized in _STATUS_KEYS:
        return "status", _STATUS_KEYS[normalized]
    if normalized == "left":
        return "navigate", "previous"
    if normalized == "right":
        return "navigate", "next"
    if key in _REASON_KEYS:
        return "reason", _REASON_KEYS[key]
    return None


class ReviewController:
    """Own widget-free draft, persistence, and navigation behavior.

    ``ReviewController`` appends valid drafts through ``ReviewStore`` before
    changing position.  It does not own widgets, dialogs, evidence paths,
    reconciliation, or application shutdown.
    """

    def __init__(self, store: ReviewStore, view_model: ReviewViewModel) -> None:
        if not view_model.presentations:
            raise ValueError("visual review requires at least one presentation")
        self._store = store
        self._view_model = view_model
        self._position = 0
        self._dirty = False
        self._status: AssessmentStatus | None = None
        self._reasons: tuple[str, ...] = ()
        self._draft_notes = ""
        self._load_current_draft()

    @property
    def store(self) -> ReviewStore:
        """Return the durable review owner used for persistence."""
        return self._store

    @property
    def position(self) -> int:
        """Return the zero-based current visual position."""
        return self._position

    @property
    def current(self) -> VisualPresentation:
        """Return the current path-free presentation."""
        return self._view_model.presentations[self._position]

    @property
    def dirty(self) -> bool:
        """Return whether the current draft differs from its persisted value."""
        return self._dirty

    @property
    def draft(
        self,
    ) -> tuple[AssessmentStatus | None, tuple[str, ...], str]:
        """Return the current widget-free provisional draft."""
        return self._status, self._reasons, self._draft_notes

    @property
    def completed_count(self) -> int:
        """Return the number of active visual presentations with assessments."""
        return sum(
            presentation.status is not None
            for presentation in self._view_model.presentations
        )

    @property
    def all_assessed(self) -> bool:
        """Return whether every active visual presentation has an assessment."""
        return self.completed_count == len(self._view_model.presentations)

    def select_status(
        self,
        status: AssessmentStatus,
        reasons: tuple[str, ...],
        notes: str,
    ) -> None:
        """Replace the current draft and mark it dirty."""
        if status not in _STATUSES:
            raise ValueError("assessment status is not allowlisted")
        if not isinstance(notes, str):
            raise TypeError("assessment notes must be a string")
        if any(not isinstance(reason, str) for reason in reasons):
            raise TypeError("assessment reasons must be strings")
        self._status = status
        self._reasons = reasons
        self._draft_notes = notes
        self._dirty = True

    def toggle_reason(self, reason: str) -> None:
        """Toggle one standardized reason in the current draft."""
        if reason not in REASON_TOKENS:
            raise ValueError("assessment reason is not allowlisted")
        selected = set(self._reasons)
        if reason in selected:
            selected.remove(reason)
        else:
            selected.add(reason)
        self._reasons = tuple(item for item in _COMMON_REASONS if item in selected)
        self._dirty = True

    def set_notes(self, notes: str) -> None:
        """Replace current notes and mark the draft dirty."""
        if not isinstance(notes, str):
            raise TypeError("assessment notes must be a string")
        self._draft_notes = notes
        self._dirty = True

    def persist_current(self) -> None:
        """Append the valid dirty draft through the durable review store."""
        if not self._dirty:
            return
        if self._status is None:
            raise IncompleteAssessmentError("select a status before leaving this case")
        if not self._reasons:
            raise IncompleteAssessmentError(
                "select at least one structured reason before leaving this case"
            )
        self._store.assess(
            self.current.opaque_id,
            self._status,
            self._reasons,
            self._draft_notes,
        )
        self._view_model = ReviewViewModel.from_store(self._store)
        self._dirty = False
        self._load_current_draft()

    def next_case(self) -> None:
        """Persist first, then advance and request a render."""
        self.persist_current()
        if self._position < len(self._view_model.presentations) - 1:
            self._position += 1
            self._load_current_draft()
        self.render_current("next")

    def previous_case(self) -> None:
        """Persist first, then move backward and request a render."""
        self.persist_current()
        if self._position > 0:
            self._position -= 1
            self._load_current_draft()
        self.render_current("previous")

    def request_close(self) -> bool:
        """Persist a valid dirty draft before permitting application shutdown."""
        self.persist_current()
        return True

    def refresh_from_store(self, *, first_unassessed: bool = False) -> None:
        """Refresh from attached revisions without replacing durable ownership."""
        self._view_model = ReviewViewModel.from_store(self._store)
        if first_unassessed:
            self._position = next(
                (
                    index
                    for index, item in enumerate(self._view_model.presentations)
                    if item.status is None
                ),
                min(self._position, len(self._view_model.presentations) - 1),
            )
        else:
            self._position = min(
                self._position, len(self._view_model.presentations) - 1
            )
        self._dirty = False
        self._load_current_draft()

    def render_current(self, transition: str = "current") -> None:
        """Handle a render request in a concrete application or test harness."""
        del transition

    def _load_current_draft(self) -> None:
        current = self.current
        self._status = current.status
        self._reasons = current.reasons
        self._draft_notes = current.notes


class ReviewApplication(ReviewController):
    """Own and run the Tkinter review interface over an existing review store.

    This is the sole widget owner for visual review, target-blind reconciliation,
    revision attachment, and explicit finalization.  The supplied store continues
    to own persistence.  Inventory construction, registration execution, VNC or
    display setup, and promotion into scientific configuration remain outside the
    application.

    Parameters
    ----------
    root : tkinter.Tk
        Existing graphical root owned by the command-line entry point.
    store : ReviewStore
        Open durable review lifecycle.
    manifest : ReviewManifest
        Immutable initial manifest used by final export and report validation.

    Notes
    -----
    Evidence paths are resolved only by ``EvidenceResolver`` immediately before
    image loading.  Free-text notes are persisted but the report layer keeps them
    off blinded PDF pages.
    """

    def __init__(
        self, root: tk.Tk, store: ReviewStore, manifest: ReviewManifest
    ) -> None:
        store.validate_current_bindings(manifest)
        self._root = root
        self._initial_manifest = manifest
        self._resolver = EvidenceResolver(store.manifest)
        self._rendering = False
        self._photo_images: dict[tuple[str, str, str], ImageTk.PhotoImage] = {}
        self._viewer_images: dict[tk.Toplevel, ImageTk.PhotoImage] = {}
        self._status_variable = tk.StringVar(root)
        self._reason_variables = {
            reason: tk.BooleanVar(root) for reason in _COMMON_REASONS
        }
        self._opaque_variable = tk.StringVar(root)
        self._progress_variable = tk.StringVar(root)
        self._save_variable = tk.StringVar(root, "Saved")
        super().__init__(store, ReviewViewModel.from_store(store))
        self._build_widgets()
        self.render_current()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_widgets(self) -> None:
        self._root.title("Target-blind registration QC review")
        self._root.minsize(1050, 780)
        outer = ttk.Frame(self._root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self._root.rowconfigure(0, weight=1)
        self._root.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, textvariable=self._opaque_variable).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(header, textvariable=self._progress_variable).grid(row=0, column=1)
        ttk.Label(header, textvariable=self._save_variable).grid(
            row=0, column=2, sticky="e"
        )

        self._notebook = ttk.Notebook(outer)
        self._notebook.grid(row=1, column=0, sticky="nsew")
        self._image_labels: dict[tuple[str, str, str], ttk.Label] = {}
        for channel in _CHANNELS:
            tab = ttk.Frame(self._notebook, padding=8)
            self._notebook.add(tab, text=channel.upper())
            for column, stage in enumerate(_STAGES, start=1):
                ttk.Label(tab, text=stage.title()).grid(row=0, column=column, pady=4)
                tab.columnconfigure(column, weight=1)
            for row, view in enumerate(_VIEWS, start=1):
                ttk.Label(tab, text=view.upper()).grid(row=row, column=0, padx=4)
                tab.rowconfigure(row, weight=1)
                for column, stage in enumerate(_STAGES, start=1):
                    label = ttk.Label(tab, anchor="center", cursor="hand2")
                    label.grid(
                        row=row,
                        column=column,
                        padx=4,
                        pady=4,
                        sticky="nsew",
                    )
                    label.bind(
                        "<Button-1>",
                        lambda _event, c=channel, s=stage, v=view: self._open_viewer(
                            c, s, v
                        ),
                    )
                    self._image_labels[(channel, stage, view)] = label

        controls = ttk.LabelFrame(outer, text="Provisional assessment", padding=8)
        controls.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for column, status in enumerate(_STATUSES):
            ttk.Radiobutton(
                controls,
                text=f"{_STATUS_LABELS[status]} ({'AREU'[column]})",
                value=status,
                variable=self._status_variable,
                command=self._form_changed,
            ).grid(row=0, column=column, padx=4, sticky="w")
        for index, reason in enumerate(_COMMON_REASONS):
            key = "1234567890"[index]
            ttk.Checkbutton(
                controls,
                text=f"{key}: {reason}",
                variable=self._reason_variables[reason],
                command=self._form_changed,
            ).grid(row=1 + index // 5, column=index % 5, sticky="w", padx=4)
        ttk.Label(controls, text="Notes (unblinded provenance only):").grid(
            row=3, column=0, sticky="nw", pady=(6, 0)
        )
        self._notes_widget = tk.Text(controls, height=3, wrap="word")
        self._notes_widget.grid(row=3, column=1, columnspan=4, sticky="ew", pady=(6, 0))
        self._notes_widget.bind("<KeyRelease>", self._notes_changed)
        controls.columnconfigure(4, weight=1)

        actions = ttk.Frame(outer)
        actions.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Previous", command=self._previous).pack(side="left")
        ttk.Button(actions, text="Save", command=self._save).pack(side="left", padx=5)
        ttk.Button(actions, text="Next", command=self._next).pack(side="left")
        self._reconcile_button = ttk.Button(
            actions, text="Reconcile / finalize", command=self._open_reconciliation
        )
        self._reconcile_button.pack(side="right")
        self._root.bind_all("<KeyPress>", self._on_key, add="+")

    def render_current(self, transition: str = "current") -> None:
        """Render one path-free presentation after preloading all 18 images."""
        del transition
        current = self.current
        loaded: dict[tuple[str, str, str], ImageTk.PhotoImage] = {}
        for channel in _CHANNELS:
            for stage in _STAGES:
                for view in _VIEWS:
                    token = current.evidence_tokens[channel][stage][view]
                    thumbnail = self._resolver.load_thumbnail(token, (300, 190))
                    loaded[(channel, stage, view)] = ImageTk.PhotoImage(
                        thumbnail, master=self._root
                    )
        self._rendering = True
        try:
            self._opaque_variable.set(
                f"{current.opaque_id}  •  {current.revision_label}"
            )
            self._progress_variable.set(
                f"Case {current.progress_position}/{current.progress_total}  •  "
                f"{self.completed_count} completed"
            )
            status, reasons, notes = self.draft
            self._status_variable.set(status or "")
            for reason, variable in self._reason_variables.items():
                variable.set(reason in reasons)
            self._notes_widget.delete("1.0", "end")
            self._notes_widget.insert("1.0", notes)
            for descriptor, photo in loaded.items():
                self._image_labels[descriptor].configure(image=photo)
            self._photo_images = loaded
            self._save_variable.set("Unsaved" if self.dirty else "Saved")
            self._reconcile_button.configure(
                state="normal" if self.all_assessed else "disabled"
            )
        finally:
            self._rendering = False

    def _sync_draft(self) -> None:
        status_text = self._status_variable.get()
        if status_text not in _STATUSES:
            self._status = None
            self._reasons = tuple(
                reason
                for reason in _COMMON_REASONS
                if self._reason_variables[reason].get()
            )
            self._draft_notes = self._notes_widget.get("1.0", "end-1c")
            self._dirty = True
            return
        self.select_status(
            cast(AssessmentStatus, status_text),
            tuple(
                reason
                for reason in _COMMON_REASONS
                if self._reason_variables[reason].get()
            ),
            self._notes_widget.get("1.0", "end-1c"),
        )

    def _form_changed(self) -> None:
        if self._rendering:
            return
        self._sync_draft()
        self._save_variable.set("Unsaved")

    def _notes_changed(self, _event: tk.Event[tk.Misc]) -> None:
        self._form_changed()

    def _save(self) -> None:
        try:
            self._sync_draft()
            self.persist_current()
            self.render_current()
        except (IncompleteAssessmentError, ValueError, OSError) as error:
            messagebox.showwarning("Review not saved", str(error), parent=self._root)

    def _next(self) -> None:
        try:
            if self._dirty or self._form_has_changed():
                self._sync_draft()
            self.next_case()
        except (IncompleteAssessmentError, ValueError, OSError) as error:
            messagebox.showwarning("Cannot leave case", str(error), parent=self._root)

    def _previous(self) -> None:
        try:
            if self._dirty or self._form_has_changed():
                self._sync_draft()
            self.previous_case()
        except (IncompleteAssessmentError, ValueError, OSError) as error:
            messagebox.showwarning("Cannot leave case", str(error), parent=self._root)

    def _form_has_changed(self) -> bool:
        status, reasons, notes = self.draft
        return (
            self._status_variable.get() != (status or "")
            or tuple(
                reason
                for reason in _COMMON_REASONS
                if self._reason_variables[reason].get()
            )
            != reasons
            or self._notes_widget.get("1.0", "end-1c") != notes
        )

    def _on_close(self) -> None:
        try:
            if self._dirty or self._form_has_changed():
                self._sync_draft()
            self.request_close()
        except (IncompleteAssessmentError, ValueError, OSError) as error:
            messagebox.showwarning(
                "Incomplete unsaved edit",
                f"The window remains open: {error}",
                parent=self._root,
            )
            return
        self._root.destroy()

    def _on_key(self, event: tk.Event[tk.Misc]) -> str | None:
        if isinstance(event.widget, (tk.Text, ttk.Entry)):
            return None
        action = keyboard_action(event.keysym)
        if action is None:
            return None
        kind, value = action
        if kind == "status":
            self._status_variable.set(value)
            self._form_changed()
        elif kind == "reason":
            variable = self._reason_variables[value]
            variable.set(not variable.get())
            self._form_changed()
        elif value == "previous":
            self._previous()
        else:
            self._next()
        return "break"

    def _open_viewer(self, channel: str, stage: str, view: str) -> None:
        token = self.current.evidence_tokens[channel][stage][view]
        try:
            source = self._resolver.load_image(token)
        except ValueError as error:
            messagebox.showerror("Evidence unavailable", str(error), parent=self._root)
            return
        window = tk.Toplevel(self._root)
        window.title(f"{channel.upper()} • {stage} • {view.upper()}")
        canvas = tk.Canvas(window, highlightthickness=0)
        horizontal = ttk.Scrollbar(window, orient="horizontal", command=canvas.xview)
        vertical = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        canvas.grid(row=1, column=0, sticky="nsew")
        horizontal.grid(row=2, column=0, sticky="ew")
        vertical.grid(row=1, column=1, sticky="ns")
        window.rowconfigure(1, weight=1)
        window.columnconfigure(0, weight=1)
        scale = tk.DoubleVar(window, 1.0)

        def redraw() -> None:
            factor = scale.get()
            size = (
                max(1, round(source.width * factor)),
                max(1, round(source.height * factor)),
            )
            resized = source.resize(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized, master=window)
            self._viewer_images[window] = photo
            canvas.delete("all")
            canvas.create_image(0, 0, image=photo, anchor="nw")
            canvas.configure(scrollregion=(0, 0, *size))

        zoom = ttk.Frame(window)
        zoom.grid(row=0, column=0, sticky="w")
        ttk.Button(
            zoom,
            text="−",
            command=lambda: (scale.set(max(0.1, scale.get() / 1.25)), redraw()),
        ).pack(side="left")
        ttk.Button(
            zoom,
            text="+",
            command=lambda: (scale.set(min(8.0, scale.get() * 1.25)), redraw()),
        ).pack(side="left")
        canvas.bind("<ButtonPress-1>", lambda event: canvas.scan_mark(event.x, event.y))
        canvas.bind(
            "<B1-Motion>", lambda event: canvas.scan_dragto(event.x, event.y, gain=1)
        )
        window.protocol(
            "WM_DELETE_WINDOW",
            lambda: (self._viewer_images.pop(window, None), window.destroy()),
        )
        redraw()

    def _open_reconciliation(self) -> None:
        if not self.all_assessed:
            messagebox.showwarning(
                "Visual review incomplete",
                "Assess every visual presentation before reconciliation.",
                parent=self._root,
            )
            return
        window = tk.Toplevel(self._root)
        window.title("Target-blind technical reconciliation")
        window.geometry("1150x650")
        columns = (
            "review_ids",
            "automated",
            "seam",
            "interior",
            "provisional",
            "repeat",
        )
        tree = ttk.Treeview(window, columns=columns, show="headings")
        for name, label in zip(
            columns,
            (
                "Opaque presentation(s)",
                "Automated QC",
                "Seam summary",
                "Interior summary",
                "Visual state(s)",
                "Repeat agreement",
            ),
            strict=True,
        ):
            tree.heading(name, text=label)
            tree.column(name, width=180, stretch=True)
        tree.pack(fill="both", expand=True, padx=8, pady=8)
        case_keys = self._populate_reconciliation(tree)

        controls = ttk.Frame(window, padding=8)
        controls.pack(fill="x")
        decision = tk.StringVar(window, "accept")
        for status in ("accept", "exclude"):
            ttk.Radiobutton(
                controls, text=status.title(), variable=decision, value=status
            ).pack(side="left")
        reason = ttk.Combobox(
            controls,
            values=tuple(sorted(REASON_TOKENS)),
            state="readonly",
            width=38,
        )
        reason.set("dapi-aligned")
        reason.pack(side="left", padx=6)
        notes = ttk.Entry(controls, width=42)
        notes.pack(side="left", padx=6, fill="x", expand=True)

        def adjudicate() -> None:
            selection = tree.selection()
            if not selection:
                messagebox.showwarning(
                    "No case selected", "Select a row to adjudicate.", parent=window
                )
                return
            case_token = selection[0]
            try:
                self.store.adjudicate(
                    case_keys[case_token],
                    cast(Literal["accept", "exclude"], decision.get()),
                    (reason.get(),),
                    notes.get(),
                )
            except (ValueError, OSError) as error:
                messagebox.showerror("Adjudication failed", str(error), parent=window)
                return
            tree.delete(*tree.get_children())
            case_keys.update(self._populate_reconciliation(tree))

        ttk.Button(controls, text="Adjudicate selected", command=adjudicate).pack(
            side="left"
        )
        action_row = ttk.Frame(window, padding=(8, 0, 8, 8))
        action_row.pack(fill="x")
        ttk.Button(
            action_row,
            text="Write remediation exports",
            command=lambda: self._write_remediation(window),
        ).pack(side="left")
        ttk.Button(
            action_row,
            text="Attach revision root",
            command=lambda: self._attach_revision(window),
        ).pack(side="left", padx=6)
        ttk.Button(
            action_row,
            text="Finalize decisions and report",
            command=lambda: self._finalize(window),
        ).pack(side="right")

    def _populate_reconciliation(self, tree: ttk.Treeview) -> dict[str, CaseKey]:
        active = self.store.manifest
        mappings: dict[str, CaseKey] = {}
        for case_token, history in sorted(self.store.state.case_histories.items()):
            revision = history.revisions[-1]
            presentations = tuple(
                item
                for item in active.visual_presentations
                if item.case_token == case_token
                and item.revision_id == revision.revision_id
            )
            identifiers = tuple(item.opaque_id for item in presentations)
            statuses = tuple(
                assessment_history[-1].status
                for item in presentations
                if (
                    assessment_history := self.store.state.assessment_history.get(
                        item.opaque_id, ()
                    )
                )
            )
            repeated = tuple(
                item for item in presentations if item.repeat_of is not None
            )
            agreements: list[str] = []
            for item in repeated:
                base = self.store.state.assessment_history.get(item.repeat_of or "", ())
                repeat = self.store.state.assessment_history.get(item.opaque_id, ())
                if base and repeat:
                    agreements.append(
                        "agree" if base[-1].status == repeat[-1].status else "disagree"
                    )
            flags = dict(revision.deformation_flags)
            automated = (
                "valid" if revision.automated_valid else revision.automated_reason
            )
            label = ", ".join(identifiers) if identifiers else "Technical affine case"
            tree.insert(
                "",
                "end",
                iid=case_token,
                values=(
                    label,
                    automated,
                    flags.get("seam", "not available"),
                    flags.get("interior", "not available"),
                    ", ".join(statuses) if statuses else "requires adjudication",
                    ", ".join(agreements) if agreements else "not repeated",
                ),
            )
            mappings[case_token] = history.case_key
        return mappings

    def _write_remediation(self, parent: tk.Misc) -> None:
        from cyclic_stability.review_exports import write_operational_exports

        try:
            paths = write_operational_exports(
                self.store.state, self._initial_manifest, self.store.review_dir
            )
        except (ValueError, OSError) as error:
            messagebox.showerror("Export failed", str(error), parent=parent)
            return
        messagebox.showinfo(
            "Exports written",
            f"Updated {paths[0].name} and {paths[1].name}.",
            parent=parent,
        )

    def _attach_revision(self, parent: tk.Toplevel) -> None:
        from cyclic_stability.review_exports import attach_revision_root

        selected = filedialog.askdirectory(
            parent=parent, title="Select a new target-blind QC revision root"
        )
        if not selected:
            return
        try:
            attach_revision_root(self.store, Path(selected))
            self._resolver = EvidenceResolver(self.store.manifest)
            self.refresh_from_store(first_unassessed=True)
            self.render_current()
        except (ValueError, OSError) as error:
            messagebox.showerror(
                "Revision attachment failed", str(error), parent=parent
            )
            return
        parent.destroy()

    def _finalize(self, parent: tk.Misc) -> None:
        from cyclic_stability.review_exports import finalize_decisions
        from cyclic_stability.review_report import write_review_report

        try:
            decisions = finalize_decisions(
                self.store, self._initial_manifest, self.store.review_dir
            )
            pdf, summary = write_review_report(
                self.store.state,
                self._initial_manifest,
                self.store.review_dir / "review-record.pdf",
                self.store.review_dir / "review-summary.md",
            )
        except (ValueError, OSError) as error:
            messagebox.showerror("Finalization failed", str(error), parent=parent)
            return
        messagebox.showinfo(
            "Review finalized",
            f"Wrote {decisions.name}, {pdf.name}, and {summary.name}.",
            parent=parent,
        )


__all__ = [
    "IncompleteAssessmentError",
    "ReviewApplication",
    "ReviewController",
    "ReviewViewModel",
    "VisualPresentation",
    "keyboard_action",
]
