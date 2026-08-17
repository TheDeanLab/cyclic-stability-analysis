"""Archival blinded-review record contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import pytest
import tifffile
from cyclic_stability import review_report
from cyclic_stability.review_exports import (
    _consistency_rows,
    _final_rows,
    attach_revision_root,
)
from cyclic_stability.review_report import write_review_report
from synthetic import (
    completed_review_store,
    fresh_review_store,
    rewrite_review_case_pixels,
    write_review_qc_fixture,
)


def test_report_is_target_blind_and_uses_truetype_fonts(tmp_path: Path) -> None:
    """Rendered records omit target identifiers before their provenance appendix."""
    store, manifest = completed_review_store(tmp_path)

    pdf, summary = write_review_report(
        store.state,
        manifest,
        tmp_path / "review/review-record.pdf",
        tmp_path / "review/review-summary.md",
    )

    text = summary.read_text(encoding="utf-8")
    appendix = text.index("## Unblinded provenance appendix")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert "target channel" not in text.lower()
    assert "Cell" not in text[:appendix]
    assert "Cell1" in text[appendix:]
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42


def test_report_opens_only_allowlisted_overlays_and_maps_biology_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report loads manifest evidence only and defers biological mapping."""
    store, manifest = completed_review_store(tmp_path)
    opened: list[str] = []
    original = tifffile.imread

    def spy(path: str | Path, *args: object, **kwargs: object):
        opened.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(tifffile, "imread", spy)
    _pdf, summary = write_review_report(
        store.state,
        manifest,
        tmp_path / "review/review-record.pdf",
        tmp_path / "review/review-summary.md",
    )

    assert len(opened) == len(manifest.visual_presentations) * 18
    assert all(
        re.fullmatch(
            r"(01|03|04|05)-(dapi|tubulin)-(pre|affine|nonlinear)-(xy|xz|yz)\.tif",
            name,
        )
        for name in opened
    )
    text = summary.read_text(encoding="utf-8")
    assert text.rindex("## Unblinded provenance appendix") < text.index("Cell1")


def test_report_rejects_unfinalized_state_without_publication(tmp_path: Path) -> None:
    """An incomplete review cannot replace a previously valid archival record."""
    completed, manifest = completed_review_store(tmp_path)
    pdf_path = tmp_path / "review/review-record.pdf"
    summary_path = tmp_path / "review/review-summary.md"
    write_review_report(completed.state, manifest, pdf_path, summary_path)
    before = pdf_path.read_bytes(), summary_path.read_bytes()
    pending, pending_manifest = fresh_review_store(tmp_path / "pending")

    with pytest.raises(ValueError, match="final"):
        write_review_report(pending.state, pending_manifest, pdf_path, summary_path)

    assert (pdf_path.read_bytes(), summary_path.read_bytes()) == before


def test_blinded_page_content_excludes_notes_and_unblinded_appendix_keeps_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Free text never reaches visual builders and is appended after them."""
    store, manifest = completed_review_store(tmp_path)
    presentation = manifest.visual_presentations[0]
    store.assess(
        presentation.opaque_id,
        "accept",
        ("dapi-aligned",),
        "LAMP1 puncta remain offset",
    )
    active = store.state.active_manifest(manifest)
    final_rows = _final_rows(store.state, active)
    decisions = review_report._final_decisions(final_rows)

    blinded = review_report._blinded_presentation_lines(
        store.state, active, presentation, decisions
    )
    appendix = review_report._unblinded_provenance_lines(store.state, active)
    assert "LAMP1 puncta remain offset" not in "\n".join(blinded)
    assert "LAMP1 puncta remain offset" in "\n".join(appendix)

    calls: list[str] = []
    monkeypatch.setattr(
        review_report, "_render_visual_page", lambda *args: calls.append("visual")
    )
    monkeypatch.setattr(
        review_report,
        "_render_technical_summary",
        lambda *args: calls.append("technical"),
    )
    monkeypatch.setattr(
        review_report,
        "_render_unblinded_provenance",
        lambda *args: calls.append("unblinded"),
    )
    review_report._render_pdf(
        tmp_path / "ordered.pdf",
        store.state,
        active,
        final_rows,
        _consistency_rows(store.state, active),
    )
    assert calls == ["visual"] * len(active.visual_presentations) + [
        "technical",
        "unblinded",
    ]


def test_unblinded_markdown_records_exclusions_remediation_and_repository_identity(
    tmp_path: Path,
) -> None:
    """Archival Markdown contains the complete final provenance record."""
    store, manifest = completed_review_store(tmp_path)
    active = store.state.active_manifest(manifest)
    final_rows = _final_rows(store.state, active)
    summary = review_report._summary_text(
        store.state, active, final_rows, _consistency_rows(store.state, active)
    )
    repository = store.state.repository_identity

    assert "### Exclusion details" in summary
    assert "lamin | Cell1 | 1 | exclude | invalid-affine-reflection" in summary
    assert (
        f"- Repository dirty diff SHA-256: {repository.diff_sha256 or 'none'}"
        in summary
    )

    pending, initial = fresh_review_store(tmp_path / "remediation")
    presentation = initial.visual_presentations[0]
    pending.assess(
        presentation.opaque_id,
        "needs-registration-improvement",
        ("persistent-interior-misalignment",),
        "rerun needed",
    )
    revised_root = write_review_qc_fixture(tmp_path / "revised")
    rewrite_review_case_pixels(
        revised_root, initial.presentation_cases[presentation.opaque_id]
    )
    attach_revision_root(pending, revised_root)
    remediation_summary = review_report._summary_text(
        pending.state,
        pending.manifest,
        (),
        _consistency_rows(pending.state, pending.manifest),
    )
    history = pending.state.case_histories[presentation.case_token]

    assert "### Remediation revision history" in remediation_summary
    assert history.revisions[0].revision_id in remediation_summary
    assert history.revisions[1].revision_id in remediation_summary
    assert "rerun needed" in remediation_summary


@pytest.mark.parametrize("staged_writer", ("_render_pdf", "_write_summary"))
def test_staged_report_failure_preserves_prior_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_writer: str
) -> None:
    """A failed staged report never promotes either replacement report."""
    store, manifest = completed_review_store(tmp_path)
    pdf_path = tmp_path / "review/review-record.pdf"
    summary_path = tmp_path / "review/review-summary.md"
    write_review_report(store.state, manifest, pdf_path, summary_path)
    before = pdf_path.read_bytes(), summary_path.read_bytes()

    def fail_staging(*args: object, **kwargs: object) -> None:
        raise OSError("injected report staging failure")

    monkeypatch.setattr(review_report, staged_writer, fail_staging)
    with pytest.raises(OSError, match="report staging failure"):
        write_review_report(store.state, manifest, pdf_path, summary_path)

    assert (pdf_path.read_bytes(), summary_path.read_bytes()) == before
    assert not list(pdf_path.parent.glob(".*review-*tmp"))
