"""Archival blinded-review record contract tests."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import matplotlib
import pytest
from cyclic_stability import review_evidence, review_report
from cyclic_stability.review_exports import (
    _consistency_rows,
    _final_rows,
    attach_revision_root,
)
from cyclic_stability.review_inventory import build_review_manifest
from cyclic_stability.review_report import write_review_report
from cyclic_stability.review_store import ReviewStore
from synthetic import (
    completed_review_store,
    fixed_clock,
    fresh_review_store,
    rewrite_review_case_pixels,
    write_review_qc_fixture,
)

_RAW_AFFINE_REFLECTION_REASON = (
    "unreadable or invalid affine geometry: affine matrix is improper "
    "(contains a reflection)"
)


def _replace_case_reason(
    path: Path,
    *,
    cohort: str,
    cell_id: str,
    acquisition_round: int,
    reason_column: str,
    reason: str,
) -> None:
    """Set one controlled producer reason in a synthetic QC table."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        rows = list(reader)
        columns = reader.fieldnames
    matched = False
    for row in rows:
        if (
            row["cohort"] == cohort
            and row["cell_id"] == cell_id
            and row["acquisition_round"] == str(acquisition_round)
        ):
            row["automated_valid"] = "False"
            row[reason_column] = reason
            matched = True
    assert matched
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


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
    original = review_report.read_overlay_bytes

    def spy(overlay):
        opened.append(Path(overlay.path).name)
        return original(overlay)

    monkeypatch.setattr(review_report, "read_overlay_bytes", spy)
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


def test_report_rejects_symlinked_overlay_ancestor_without_target_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Archival rendering rejects an ancestor symlink before target bytes are read."""
    store, manifest = completed_review_store(tmp_path)
    path = Path(manifest.visual_presentations[0].overlays[0].path)
    overlay_dir = path.parent
    target_dir = tmp_path / "report-target-overlays"
    overlay_dir.rename(target_dir)
    overlay_dir.symlink_to(target_dir, target_is_directory=True)
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
        write_review_report(
            store.state,
            manifest,
            tmp_path / "review/review-record.pdf",
            tmp_path / "review/review-summary.md",
        )
    assert target_reads == []


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


def test_target_blind_technical_rows_cover_all_cases_and_adjudication_states(
    tmp_path: Path,
) -> None:
    """Every active case retains raw technical evidence without pre-appendix biology."""
    qc_root = write_review_qc_fixture(tmp_path / "qc")
    for filename, reason_column in (
        ("decision-requests.csv", "automated_reason"),
        ("registration-qc.csv", "invalid_reason"),
    ):
        table = qc_root / "qc" / filename
        _replace_case_reason(
            table,
            cohort="golgi",
            cell_id="Cell1",
            acquisition_round=1,
            reason_column=reason_column,
            reason="nonpositive-jacobian",
        )
        _replace_case_reason(
            table,
            cohort="lamin",
            cell_id="Cell1",
            acquisition_round=1,
            reason_column=reason_column,
            reason=_RAW_AFFINE_REFLECTION_REASON,
        )
    manifest = build_review_manifest(qc_root, seed=7)
    store = ReviewStore.create(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    visual = tuple(manifest.visual_presentations)
    for presentation in visual:
        store.assess(
            presentation.opaque_id,
            "accept",
            ("dapi-aligned",),
            f"private note {presentation.opaque_id}",
        )
    technical = next(item for item in manifest.case_revisions if not item.overlays)
    store.adjudicate(
        technical.case_key,
        "exclude",
        ("invalid-affine-reflection",),
        "private technical note",
    )
    current_presentation = next(
        item
        for item in visual
        if all(
            candidate.repeat_of != item.opaque_id
            and candidate.opaque_id != item.repeat_of
            for candidate in visual
        )
    )
    current_case = manifest.presentation_cases[current_presentation.opaque_id]
    store.adjudicate(
        current_case,
        "accept",
        ("dapi-aligned",),
        "private current note",
    )
    stale_presentation = next(
        item
        for item in visual
        if manifest.presentation_cases[item.opaque_id] != current_case
        and item.case_token != technical.case_token
    )
    stale_case = manifest.presentation_cases[stale_presentation.opaque_id]
    store.adjudicate(
        stale_case,
        "accept",
        ("dapi-aligned",),
        "private stale note",
    )
    store.assess(
        stale_presentation.opaque_id,
        "exclude",
        ("z-misalignment",),
        "private changed note",
    )

    lines = review_report._technical_reconciliation_lines(store.state, store.manifest)
    summary = review_report._summary_text(
        store.state,
        store.manifest,
        (),
        _consistency_rows(store.state, store.manifest),
    )
    blinded, appendix = summary.split("## Unblinded provenance appendix", maxsplit=1)
    invalid_visual = next(
        item
        for item in manifest.case_revisions
        if item.automated_reason == "nonpositive-jacobian"
    )

    assert len(lines) == 40
    assert any(
        invalid_visual.case_token in line
        and "automated_valid=False" in line
        and "automated_reason=nonpositive-jacobian" in line
        for line in lines
    )
    assert any(
        technical.case_token in line
        and f"automated_reason={_RAW_AFFINE_REFLECTION_REASON}" in line
        and "adjudication=current" in line
        and "adjudicated_decision=exclude" in line
        and "adjudicated_reasons=invalid-affine-reflection" in line
        for line in lines
    )
    assert any(
        current_presentation.case_token in line and "adjudication=current" in line
        for line in lines
    )
    assert any(
        stale_presentation.case_token in line
        and "adjudication=stale" in line
        and "adjudicated_decision=accept" in line
        for line in lines
    )
    assert "## Target-blind technical reconciliation" in blinded
    assert "Cell" not in blinded
    assert "golgi" not in blinded
    assert "lamin" not in blinded
    assert "private note" not in blinded
    assert "private technical note" not in blinded
    assert "private note" in appendix
    assert "private technical note" in appendix


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
