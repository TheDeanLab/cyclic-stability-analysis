"""Archival blinded-review record contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import pytest
import tifffile
from cyclic_stability import review_report
from cyclic_stability.review_report import write_review_report
from synthetic import completed_review_store, fresh_review_store


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


def test_report_rejects_biological_text_in_blinded_page_notes(tmp_path: Path) -> None:
    """Reviewer notes cannot leak a cohort or cell identifier onto visual pages."""
    store, manifest = completed_review_store(tmp_path)
    presentation = manifest.visual_presentations[0]
    store.assess(
        presentation.opaque_id,
        "accept",
        ("dapi-aligned",),
        "Golgi Cell1 remains aligned",
    )

    with pytest.raises(ValueError, match="target-blind"):
        write_review_report(
            store.state,
            manifest,
            tmp_path / "review/review-record.pdf",
            tmp_path / "review/review-summary.md",
        )


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
