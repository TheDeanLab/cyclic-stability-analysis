"""Target-blind GUI controller and command-line lifecycle contract tests."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import tifffile
from cyclic_stability import review_evidence, review_exports, review_gui, review_report
from cyclic_stability.review_exports import finalize_decisions
from cyclic_stability.review_gui import (
    EvidenceResolver,
    IncompleteAssessmentError,
    ReviewApplication,
    ReviewController,
    ReviewViewModel,
    keyboard_action,
)
from cyclic_stability.review_store import ReviewStore
from synthetic import (
    completed_review_store,
    fresh_review_store,
    write_review_qc_fixture,
)

_BUNDLE_FILES = (
    "review-state.json",
    "review-state.previous.json",
    "remediation-requests.csv",
    "consistency-report.csv",
    "qc-decisions.csv",
    "qc-decisions.previous.csv",
    "review-record.pdf",
    "review-summary.md",
)


class RecordingReviewApplication(ReviewController):
    """Widget-free controller harness that records persistence and rendering."""

    def __init__(self, store: ReviewStore) -> None:
        super().__init__(store, ReviewViewModel.from_store(store))
        self.calls: list[str] = []

    def persist_current(self) -> None:
        """Record the save boundary while retaining real store persistence."""
        self.calls.append("persist")
        super().persist_current()

    def render_current(self, transition: str = "current") -> None:
        """Record which controller transition requested a render."""
        self.calls.append(f"render-{transition}")


class AtomicRenderApplication(ReviewController):
    """Widget-free harness whose visible evidence changes only after preload."""

    def __init__(self, store: ReviewStore) -> None:
        super().__init__(store, ReviewViewModel.from_store(store))
        self.visible_opaque_id = self.current.opaque_id
        self.fail_opaque_id: str | None = None

    def render_current(self, transition: str = "current") -> None:
        """Inject one candidate preload failure before visible evidence changes."""
        del transition
        candidate = self.current.opaque_id
        if candidate == self.fail_opaque_id:
            raise ValueError("injected candidate render failure")
        self.visible_opaque_id = candidate


class RecordingTree:
    """Minimal Treeview boundary used to inspect reconciliation row values."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, ...]] = {}

    def insert(
        self,
        _parent: str,
        _index: str,
        *,
        iid: str,
        values: tuple[str, ...],
    ) -> None:
        """Record one reconciliation row without creating Tk widgets."""
        self.rows[iid] = values


@pytest.fixture
def fake_app(tmp_path: Path) -> RecordingReviewApplication:
    """Return a concrete widget-free application around a real review store."""
    store, _manifest = fresh_review_store(tmp_path)
    return RecordingReviewApplication(store)


def _entry_point_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts/review_registration_qc.py"
    spec = importlib.util.spec_from_file_location("review_registration_qc", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle_bytes(review_dir: Path) -> dict[str, bytes | None]:
    return {
        name: path.read_bytes() if (path := review_dir / name).exists() else None
        for name in _BUNDLE_FILES
    }


def test_visual_view_model_exposes_no_biological_identity(tmp_path: Path) -> None:
    """Visual presentation values cannot reveal biology or evidence locations."""
    store, manifest = fresh_review_store(tmp_path)

    view = ReviewViewModel.from_store(store, manifest)

    payload = json.dumps(asdict(view.presentations[0]), sort_keys=True)
    assert "Cell" not in payload
    assert "golgi" not in payload
    assert "lamin" not in payload
    assert str((tmp_path / "qc").resolve()) not in payload
    assert "automated" not in payload
    assert "repeat" not in payload


def test_view_model_uses_attached_active_manifest(tmp_path: Path) -> None:
    """A remediation presentation comes from the store's active manifest."""
    from cyclic_stability.review_exports import attach_revision_root
    from synthetic import rewrite_review_case_pixels

    store, initial = fresh_review_store(tmp_path)
    pending = initial.visual_presentations[0]
    store.assess(
        pending.opaque_id,
        "needs-registration-improvement",
        ("persistent-interior-misalignment",),
        "retry",
    )
    revised_root = write_review_qc_fixture(tmp_path / "qc-v2")
    rewrite_review_case_pixels(
        revised_root, initial.presentation_cases[pending.opaque_id]
    )
    attach_revision_root(store, revised_root)

    view = ReviewViewModel.from_store(store, initial)

    assert len(view.presentations) == len(store.manifest.visual_presentations)
    assert view.presentations[-1].revision_label == "Revision 2"


def test_evidence_resolver_accepts_only_tokens_and_revalidates_digest(
    tmp_path: Path,
) -> None:
    """A token resolves once, while a raw path and drifted evidence fail closed."""
    store, _manifest = fresh_review_store(tmp_path)
    view = ReviewViewModel.from_store(store)
    resolver = EvidenceResolver(store.manifest)
    token = view.presentations[0].evidence_tokens["dapi"]["pre"]["xy"]
    overlay = next(
        item
        for item in store.manifest.visual_presentations[0].overlays
        if (item.channel, item.stage, item.view) == ("dapi", "pre", "xy")
    )
    path = Path(overlay.path)

    content = resolver.resolve(token)

    assert isinstance(content, bytes)
    assert content.startswith((b"II", b"MM"))
    assert re.fullmatch(r"(01|03|04|05)-dapi-pre-xy\.tif", path.name)
    with pytest.raises(ValueError, match="opaque evidence token"):
        resolver.resolve(str(path))
    tifffile.imwrite(
        path,
        np.full((8, 12, 3), 255, dtype=np.uint8),
        photometric="rgb",
    )
    with pytest.raises(ValueError, match="identity changed"):
        resolver.resolve(token)


def test_evidence_thumbnail_is_rgb_and_preserves_aspect_ratio(tmp_path: Path) -> None:
    """Thumbnail loading retains the source aspect ratio without exposing paths."""
    store, _manifest = fresh_review_store(tmp_path)
    presentation = ReviewViewModel.from_store(store).presentations[0]
    resolver = EvidenceResolver(store.manifest)
    token = presentation.evidence_tokens["tubulin"]["nonlinear"]["yz"]

    thumbnail = resolver.load_thumbnail(token, (6, 6))

    assert thumbnail.mode == "RGB"
    assert thumbnail.size == (6, 4)


def test_evidence_resolver_rejects_symlinked_ancestor_without_target_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GUI image loading rejects an ancestor symlink before target bytes are read."""
    store, _manifest = fresh_review_store(tmp_path)
    presentation = store.manifest.visual_presentations[0]
    overlay = presentation.overlays[0]
    path = Path(overlay.path)
    overlay_dir = path.parent
    target_dir = tmp_path / "gui-target-overlays"
    overlay_dir.rename(target_dir)
    overlay_dir.symlink_to(target_dir, target_is_directory=True)
    token = (
        ReviewViewModel.from_store(store)
        .presentations[0]
        .evidence_tokens[overlay.channel][overlay.stage][overlay.view]
    )
    target_reads: list[Path] = []
    real_open = io.open
    real_read = review_evidence.os.read

    def guarded_open(file: object, *args: object, **kwargs: object):
        if not isinstance(file, int) and Path(file).resolve().is_relative_to(
            target_dir.resolve()
        ):
            target_reads.append(Path(file))
        return real_open(file, *args, **kwargs)

    def guarded_read(descriptor: int, count: int) -> bytes:
        opened = Path(review_evidence.os.readlink(f"/proc/self/fd/{descriptor}"))
        if opened.is_absolute() and opened.is_relative_to(target_dir.resolve()):
            target_reads.append(target_dir)
        return real_read(descriptor, count)

    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(review_evidence.os, "read", guarded_read)
    with pytest.raises(ValueError, match="symlink"):
        EvidenceResolver(store.manifest).load_image(token)
    assert target_reads == []


@pytest.mark.parametrize(
    ("key", "expected"),
    (
        ("A", ("status", "accept")),
        ("r", ("status", "needs-registration-improvement")),
        ("E", ("status", "exclude")),
        ("u", ("status", "needs-adjudication")),
        ("Left", ("navigate", "previous")),
        ("Right", ("navigate", "next")),
        ("1", ("reason", "dapi-aligned")),
        ("0", ("reason", "insufficient-common-support")),
        ("F1", None),
    ),
)
def test_keyboard_shortcuts_map_to_review_actions(
    key: str, expected: tuple[str, str] | None
) -> None:
    """Exact status, navigation, and common-reason keys map to pure actions."""
    assert keyboard_action(key) == expected


def test_navigation_persists_before_changing_case(
    fake_app: RecordingReviewApplication,
) -> None:
    """Navigation durably saves an edit before the next presentation is rendered."""
    opaque_id = fake_app.current.opaque_id
    fake_app.select_status("accept", ("dapi-aligned",), "reviewed")

    fake_app.next_case()

    assert fake_app.calls[:2] == ["persist", "render-next"]
    assert fake_app.store.state.assessment_history[opaque_id][-1].notes == "reviewed"


def test_candidate_render_failure_restores_case_draft_and_visible_binding(
    tmp_path: Path,
) -> None:
    """A failed candidate preload cannot bind old evidence to a new current ID."""
    store, _manifest = fresh_review_store(tmp_path)
    app = AtomicRenderApplication(store)
    prior_id = app.current.opaque_id
    app.select_status("accept", ("dapi-aligned",), "saved before navigation")
    candidate_id = app._view_model.presentations[1].opaque_id
    app.fail_opaque_id = candidate_id

    with pytest.raises(ValueError, match="candidate render failure"):
        app.next_case()

    assert app.position == 0
    assert app.current.opaque_id == prior_id
    assert app.visible_opaque_id == prior_id
    assert app.draft == ("accept", ("dapi-aligned",), "saved before navigation")
    app.fail_opaque_id = None
    app.select_status("exclude", ("z-misalignment",), "saved after failure")
    app.persist_current()
    assert store.state.assessment_history[prior_id][-1].status == "exclude"
    assert candidate_id not in store.state.assessment_history


def test_unchanged_save_preserves_event_count_and_current_adjudication(
    tmp_path: Path,
) -> None:
    """Saving an unchanged form does not stale its case adjudication snapshot."""
    store, manifest = fresh_review_store(tmp_path)
    first = manifest.visual_presentations[0]
    store.assess(first.opaque_id, "accept", ("dapi-aligned",), "unchanged")
    case_key = manifest.presentation_cases[first.opaque_id]
    store.adjudicate(case_key, "accept", ("dapi-aligned",), "current")
    app = RecordingReviewApplication(store)
    event_count = len(store.state.assessment_history[first.opaque_id])
    adjudication = store.state.case_histories[first.case_token].adjudications[-1]

    app.select_status("accept", ("dapi-aligned",), "unchanged")
    app.persist_current()

    assert len(store.state.assessment_history[first.opaque_id]) == event_count
    assert (
        store.state.assessment_history[first.opaque_id][-1].event_id
        in adjudication.assessment_event_ids
    )


def test_incomplete_dirty_edit_blocks_navigation_and_close(
    fake_app: RecordingReviewApplication,
) -> None:
    """A status without a reason is neither discarded nor navigated away from."""
    fake_app.select_status("accept", (), "unfinished")

    with pytest.raises(IncompleteAssessmentError, match="reason"):
        fake_app.next_case()
    assert fake_app.position == 0
    assert fake_app.dirty
    with pytest.raises(IncompleteAssessmentError, match="reason"):
        fake_app.request_close()
    assert fake_app.dirty


def test_valid_dirty_edit_is_saved_on_close(
    fake_app: RecordingReviewApplication,
) -> None:
    """Closing a valid dirty form persists it before allowing destruction."""
    opaque_id = fake_app.current.opaque_id
    fake_app.select_status("exclude", ("z-misalignment",), "exclude")

    assert fake_app.request_close()
    assert not fake_app.dirty
    assert fake_app.store.state.assessment_history[opaque_id][-1].status == "exclude"


def test_reconciliation_persists_changed_draft_before_unlock(tmp_path: Path) -> None:
    """A changed valid assessment is durable before reconciliation can open."""
    store, manifest = fresh_review_store(tmp_path)
    for presentation in manifest.visual_presentations:
        store.assess(presentation.opaque_id, "accept", ("dapi-aligned",), "initial")
    app = RecordingReviewApplication(store)
    opaque_id = app.current.opaque_id
    app.select_status("exclude", ("z-misalignment",), "changed")

    app.prepare_reconciliation()

    assert app.calls == ["persist"]
    assert store.state.assessment_history[opaque_id][-1].status == "exclude"
    assert store.state.assessment_history[opaque_id][-1].notes == "changed"
    assert app.all_assessed


def test_reconciliation_blocks_incomplete_dirty_edit(tmp_path: Path) -> None:
    """Persisted completeness cannot hide an incomplete changed visual draft."""
    store, manifest = fresh_review_store(tmp_path)
    for presentation in manifest.visual_presentations:
        store.assess(presentation.opaque_id, "accept", ("dapi-aligned",), "initial")
    app = RecordingReviewApplication(store)
    opaque_id = app.current.opaque_id
    event_count = len(store.state.assessment_history[opaque_id])
    app.select_status("exclude", (), "unfinished change")

    with pytest.raises(IncompleteAssessmentError, match="reason"):
        app.prepare_reconciliation()

    assert app.calls == ["persist"]
    assert len(store.state.assessment_history[opaque_id]) == event_count
    assert app.dirty


def test_reconciliation_rows_show_current_and_stale_adjudication_details(
    tmp_path: Path,
) -> None:
    """Technical and repeat rows expose controlled adjudication state and reasons."""
    store, manifest = fresh_review_store(tmp_path)
    technical = next(item for item in manifest.case_revisions if not item.overlays)
    store.adjudicate(
        technical.case_key,
        "exclude",
        ("invalid-affine-reflection",),
        "hidden note",
    )
    repeated = next(
        item for item in manifest.visual_presentations if item.repeat_of is not None
    )
    assert repeated.repeat_of is not None
    store.assess(repeated.repeat_of, "accept", ("dapi-aligned",), "base")
    store.assess(repeated.opaque_id, "exclude", ("z-misalignment",), "repeat")
    repeated_case = manifest.presentation_cases[repeated.opaque_id]
    store.adjudicate(repeated_case, "exclude", ("z-misalignment",), "hidden")
    application = object.__new__(ReviewApplication)
    application._store = store
    current_tree = RecordingTree()

    application._populate_reconciliation(current_tree)

    technical_text = " | ".join(current_tree.rows[technical.case_token])
    repeat_text = " | ".join(current_tree.rows[repeated.case_token])
    assert "current" in technical_text
    assert "exclude" in technical_text
    assert "invalid-affine-reflection" in technical_text
    assert "hidden note" not in technical_text
    assert "current" in repeat_text
    assert "z-misalignment" in repeat_text

    store.assess(repeated.opaque_id, "accept", ("dapi-aligned",), "changed")
    stale_tree = RecordingTree()
    application._populate_reconciliation(stale_tree)
    stale_text = " | ".join(stale_tree.rows[repeated.case_token])
    assert "stale" in stale_text
    assert "exclude" in stale_text
    assert "z-misalignment" in stale_text


def test_finalization_bundle_rolls_back_every_layer_after_report_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-CSV report failure restores exact durable and in-memory state."""
    store, manifest = completed_review_store(tmp_path)
    review_dir = store.review_dir
    review_exports._atomic_csv(
        review_dir / "qc-decisions.csv",
        review_exports.FINAL_COLUMNS,
        review_exports._final_rows(store.state, store.manifest),
    )
    (review_dir / "qc-decisions.previous.csv").write_bytes(b"prior decision backup\n")
    (review_dir / "review-record.pdf").write_bytes(b"prior PDF\n")
    (review_dir / "review-summary.md").write_text("prior Markdown\n", encoding="utf-8")
    prior_state = store.state
    prior_manifest = store.manifest
    prior_files = _bundle_bytes(review_dir)
    prior_staged = {path.name for path in review_dir.glob(".*tmp")}

    def fail_report(*_args: object, **_kwargs: object) -> None:
        assert store.state != prior_state
        (review_dir / "review-record.pdf").write_bytes(b"partial replacement PDF\n")
        (review_dir / "review-summary.md").write_text(
            "partial replacement Markdown\n", encoding="utf-8"
        )
        (review_dir / ".review-record.pdf.injected.tmp").write_bytes(b"staged debris\n")
        raise OSError("injected combined report failure")

    monkeypatch.setattr(review_report, "write_review_report", fail_report)

    with pytest.raises(OSError, match="combined report failure"):
        review_gui.finalize_review_bundle(store, manifest)

    assert store.state == prior_state
    assert store.manifest == prior_manifest
    assert _bundle_bytes(review_dir) == prior_files
    assert {path.name for path in review_dir.glob(".*tmp")} == prior_staged


def test_finalization_bundle_publishes_current_outputs_and_export_history(
    tmp_path: Path,
) -> None:
    """A successful bundle publishes all three outputs and records their CSV."""
    store, manifest = completed_review_store(tmp_path)

    decisions, pdf, summary = review_gui.finalize_review_bundle(store, manifest)

    assert decisions == store.review_dir / "qc-decisions.csv"
    assert pdf == store.review_dir / "review-record.pdf"
    assert summary == store.review_dir / "review-summary.md"
    assert pdf.read_bytes().startswith(b"%PDF")
    assert summary.read_text(encoding="utf-8").startswith("# Blinded QC review record")
    assert (
        store.state.export_history[-1].sha256
        == hashlib.sha256(decisions.read_bytes()).hexdigest()
    )
    assert (
        store.state.export_history[-1].pdf_sha256
        == hashlib.sha256(pdf.read_bytes()).hexdigest()
    )
    assert (
        store.state.export_history[-1].summary_sha256
        == hashlib.sha256(summary.read_bytes()).hexdigest()
    )
    assert not (store.review_dir / "pending-finalization.json").exists()


@pytest.mark.parametrize("partial_report", (False, True))
def test_reopen_recovers_pending_finalization_after_uncaught_process_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial_report: bool,
) -> None:
    """A durable journal restores the prior bundle after two crash phases."""
    store, manifest = completed_review_store(tmp_path)
    review_dir = store.review_dir
    prior_state = store.state
    prior_files = _bundle_bytes(review_dir)

    def crash_report(
        _state: object,
        _manifest: object,
        pdf_path: Path,
        summary_path: Path,
    ) -> None:
        if partial_report:
            pdf_path.write_bytes(b"%PDF partial crash output")
            summary_path.write_text("partial crash output\n", encoding="utf-8")
        raise SystemExit("injected node/process termination")

    monkeypatch.setattr(review_report, "write_review_report", crash_report)
    with pytest.raises(SystemExit, match="node/process termination"):
        review_gui.finalize_review_bundle(store, manifest)

    assert (review_dir / "pending-finalization.json").exists()
    assert store.state != prior_state
    reopened = ReviewStore.open(
        review_dir,
        manifest,
        "Reviewer",
        now=store._now,
    )
    assert reopened.state == prior_state
    assert _bundle_bytes(review_dir) == prior_files
    assert not (review_dir / "pending-finalization.json").exists()


def test_finalized_controller_disables_every_mutating_route(tmp_path: Path) -> None:
    """A finalized GUI remains viewable but exposes no assessment mutation state."""
    store, manifest = completed_review_store(tmp_path)
    finalize_decisions(store, manifest, store.review_dir)
    controller = ReviewController(store, ReviewViewModel.from_store(store))

    assert controller.finalized
    assert not controller.mutation_controls_enabled
    with pytest.raises(ValueError, match="finalized"):
        controller.select_status("accept", ("dapi-aligned",), "late edit")
    with pytest.raises(ValueError, match="finalized"):
        controller.prepare_reconciliation()


def test_validate_only_creates_and_resumes_without_tk(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Headless preflight validates exact counts and resumes its saved manifest."""
    module = _entry_point_module()
    qc_root = write_review_qc_fixture(tmp_path / "qc")
    review_dir = tmp_path / "review"
    create_args = [
        "--qc-root",
        str(qc_root),
        "--review-dir",
        str(review_dir),
        "--reviewer",
        "Reviewer",
        "--seed",
        "8142026",
        "--validate-only",
    ]

    assert module.main(create_args) == 0
    assert (
        "validated 40 cases, 39 visual cases, and 4 repeats" in capsys.readouterr().out
    )
    assert (
        module.main(
            [
                "--review-dir",
                str(review_dir),
                "--reviewer",
                "Reviewer",
                "--validate-only",
            ]
        )
        == 0
    )


def test_cli_owns_create_resume_and_revision_argument_lifecycle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Creation needs QC, resume rejects a seed, and attachment needs saved state."""
    module = _entry_point_module()
    qc_root = write_review_qc_fixture(tmp_path / "qc")
    review_dir = tmp_path / "review"
    base = ["--review-dir", str(review_dir), "--reviewer", "Reviewer"]

    assert module.main([*base, "--validate-only"]) != 0
    assert "--qc-root is required" in capsys.readouterr().err
    assert (
        module.main(
            [
                "--qc-root",
                str(qc_root),
                *base,
                "--seed",
                "3",
                "--validate-only",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert module.main([*base, "--seed", "3", "--validate-only"]) != 0
    assert "--seed cannot be used when resuming" in capsys.readouterr().err
    assert (
        module.main(
            [
                "--review-dir",
                str(tmp_path / "missing-review"),
                "--reviewer",
                "Reviewer",
                "--add-revision-root",
                str(tmp_path / "revision"),
                "--validate-only",
            ]
        )
        != 0
    )
    assert "--add-revision-root requires existing state" in capsys.readouterr().err


def test_validate_only_production_import_order_is_headless(tmp_path: Path) -> None:
    """Validate-only succeeds even when importing Tkinter is made fatal."""
    qc_root = write_review_qc_fixture(tmp_path / "qc")
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "tkinter.py").write_text(
        "raise RuntimeError('Tkinter imported during headless validation')\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts/review_registration_qc.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(poison), environment.get("PYTHONPATH", ""))
    )

    result = subprocess.run(
        (
            sys.executable,
            str(script),
            "--qc-root",
            str(qc_root),
            "--review-dir",
            str(tmp_path / "review"),
            "--reviewer",
            "Reviewer",
            "--seed",
            "5",
            "--validate-only",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "validated 40 cases, 39 visual cases, and 4 repeats" in result.stdout


def test_normal_launch_without_display_is_concise(tmp_path: Path) -> None:
    """A non-headless launch without DISPLAY exits with actionable guidance."""
    qc_root = write_review_qc_fixture(tmp_path / "qc")
    script = Path(__file__).resolve().parents[1] / "scripts/review_registration_qc.py"
    environment = os.environ.copy()
    environment.pop("DISPLAY", None)

    result = subprocess.run(
        (
            sys.executable,
            str(script),
            "--qc-root",
            str(qc_root),
            "--review-dir",
            str(tmp_path / "review"),
            "--reviewer",
            "Reviewer",
            "--seed",
            "5",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert "GUI-enabled session" in result.stderr
    assert "Traceback" not in result.stderr


def test_readme_documents_review_remediation_and_finalization() -> None:
    """The operator guide documents the complete blinded-review lifecycle."""
    text = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    assert "review_registration_qc.py" in text
    assert "Needs registration improvement" in text
    assert "--add-revision-root" in text
    assert "qc-decisions.csv" in text
    assert "VNC" in text and "does not configure" in text
