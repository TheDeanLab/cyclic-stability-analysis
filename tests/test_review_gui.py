"""Target-blind GUI controller and command-line lifecycle contract tests."""

from __future__ import annotations

import importlib.util
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
from cyclic_stability.review_gui import (
    EvidenceResolver,
    IncompleteAssessmentError,
    ReviewController,
    ReviewViewModel,
    keyboard_action,
)
from cyclic_stability.review_store import ReviewStore
from synthetic import fresh_review_store, write_review_qc_fixture


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

    path = resolver.resolve(token)

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
