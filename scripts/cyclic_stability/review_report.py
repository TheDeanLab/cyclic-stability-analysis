"""Atomic archival PDF and Markdown records for finalized blinded reviews."""

from __future__ import annotations

import io
import os
import textwrap
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.backends.backend_pdf import PdfPages

from cyclic_stability.review_exports import (
    _consistency_rows,
    _final_rows,
    _latest_assessment,
)
from cyclic_stability.review_inventory import read_overlay_bytes
from cyclic_stability.review_models import Presentation, ReviewManifest, ReviewState

_CHANNELS = ("dapi", "tubulin")
_STAGES = ("pre", "affine", "nonlinear")
_VIEWS = ("xy", "xz", "yz")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _latest_activity(state: ReviewState) -> str:
    timestamps = [state.created_at]
    timestamps.extend(
        assessment.assessed_at
        for history in state.assessment_history.values()
        for assessment in history
    )
    timestamps.extend(
        adjudication.adjudicated_at
        for history in state.case_histories.values()
        for adjudication in history.adjudications
    )
    return max(timestamps).isoformat()


def _final_decisions(
    final_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, int], str]:
    return {
        (
            str(row["cohort"]),
            str(row["cell_id"]),
            int(row["acquisition_round"]),
        ): str(row["decision"])
        for row in final_rows
    }


def _blinded_presentation_lines(
    state: ReviewState,
    active: ReviewManifest,
    presentation: Presentation,
    decisions: Mapping[tuple[str, str, int], str],
) -> tuple[str, ...]:
    """Build visual-page text entirely from opaque and structured fields."""
    assessment = _latest_assessment(state, presentation.opaque_id)
    assert assessment is not None
    case_key = active.presentation_cases[presentation.opaque_id]
    final = decisions[(case_key.cohort, case_key.cell_id, case_key.acquisition_round)]
    reasons = ", ".join(assessment.reasons)
    return (
        f"{presentation.opaque_id}  |  revision {presentation.revision_id}",
        f"Provisional decision: {assessment.status}; final decision: {final}",
        f"Structured reasons: {reasons or 'none'}",
    )


def _render_visual_page(
    pdf: PdfPages,
    state: ReviewState,
    active: ReviewManifest,
    presentation: Presentation,
    decisions: Mapping[tuple[str, str, int], str],
) -> None:
    """Render one blind visual presentation from its explicit overlay allowlist."""
    overlays = {
        (item.channel, item.stage, item.view): item for item in presentation.overlays
    }
    figure, axes = plt.subplots(2, 1, figsize=(11, 14), constrained_layout=True)
    figure.suptitle(
        "\n".join(_blinded_presentation_lines(state, active, presentation, decisions))
    )
    for channel_index, channel in enumerate(_CHANNELS):
        images: list[np.ndarray] = []
        for stage_index, stage in enumerate(_STAGES):
            for view_index, view in enumerate(_VIEWS):
                overlay = overlays[(channel, stage, view)]
                images.append(
                    np.asarray(tifffile.imread(io.BytesIO(read_overlay_bytes(overlay))))
                )
        axis = axes[channel_index]
        mosaic = _overlay_mosaic(images)
        axis.imshow(mosaic)
        axis.set_xticks(
            tuple((index + 0.5) * mosaic.shape[1] / 3 for index in range(3)),
            _VIEWS,
        )
        axis.set_yticks(
            tuple((index + 0.5) * mosaic.shape[0] / 3 for index in range(3)),
            _STAGES,
        )
        axis.set_title(f"{channel.upper()} overlays: stage by view", fontsize=9)
    pdf.savefig(figure)
    plt.close(figure)


def _overlay_mosaic(images: Sequence[np.ndarray]) -> np.ndarray:
    """Arrange nine RGB overlays into a padded three-by-three image grid."""
    if len(images) != 9:
        raise ValueError("a channel section requires exactly nine overlays")
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    padded = [
        np.pad(
            image,
            ((0, height - image.shape[0]), (0, width - image.shape[1]), (0, 0)),
        )
        for image in images
    ]
    return np.concatenate(
        tuple(
            np.concatenate(tuple(padded[index : index + 3]), axis=1)
            for index in range(0, 9, 3)
        ),
        axis=0,
    )


def _render_technical_summary(
    pdf: PdfPages,
    state: ReviewState,
    active: ReviewManifest,
    consistency: Sequence[Mapping[str, object]],
    final_rows: Sequence[Mapping[str, object]],
) -> None:
    """Render blind technical and concealed-repeat summaries after visual pages."""
    accepted = sum(row["decision"] == "accept" for row in final_rows)
    excluded = len(final_rows) - accepted
    figure, axis = plt.subplots(figsize=(11, 8.5))
    axis.set_axis_off()
    repository = state.repository_identity
    lines = [
        "Technical review summary",
        "",
        f"Visual presentations: {len(active.visual_presentations)}",
        f"Final cases: {len(final_rows)} (accept {accepted}; exclude {excluded})",
        f"Reviewer: {state.reviewer}",
        f"Review created: {state.created_at.isoformat()}",
        f"Last review activity: {_latest_activity(state)}",
        f"Repository commit: {repository.commit}",
        f"Repository dirty at review creation: {repository.dirty}",
        "",
        "Concealed-repeat agreement",
    ]
    lines.extend(
        " | ".join(
            (
                str(row["base_opaque_id"]),
                str(row["repeat_opaque_id"]),
                str(row["agreement"] or "pending"),
                str(row["adjudicated_decision"] or "not needed"),
            )
        )
        for row in consistency
    )
    axis.text(0.03, 0.97, "\n".join(lines), va="top", family="monospace", fontsize=10)
    pdf.savefig(figure)
    plt.close(figure)
    wrapped: list[str] = []
    for line in _technical_reconciliation_lines(state, active):
        wrapped.extend(
            textwrap.wrap(
                line,
                width=145,
                subsequent_indent="  ",
                break_long_words=False,
                break_on_hyphens=False,
            )
            or ("",)
        )
    for page_number, start in enumerate(range(0, len(wrapped), 34), start=1):
        figure, axis = plt.subplots(figsize=(11, 8.5))
        axis.set_axis_off()
        page = [
            f"Target-blind per-case technical reconciliation — page {page_number}",
            "",
            *wrapped[start : start + 34],
        ]
        axis.text(
            0.02,
            0.98,
            "\n".join(page),
            va="top",
            family="monospace",
            fontsize=6.5,
        )
        pdf.savefig(figure)
        plt.close(figure)


def _technical_reconciliation_lines(
    state: ReviewState, active: ReviewManifest
) -> tuple[str, ...]:
    """Return one path-free, note-free technical row per active case revision."""
    lines: list[str] = []
    for case_token, history in sorted(state.case_histories.items()):
        revision = history.revisions[-1]
        presentations = tuple(
            item
            for item in active.visual_presentations
            if item.case_token == case_token
            and item.revision_id == revision.revision_id
        )
        assessment_event_ids = tuple(
            assessment.event_id
            for presentation in presentations
            if (assessment := _latest_assessment(state, presentation.opaque_id))
            is not None
        )
        adjudication = next(
            (
                item
                for item in reversed(history.adjudications)
                if item.revision_id == revision.revision_id
            ),
            None,
        )
        if adjudication is None:
            adjudication_state = "not-adjudicated"
            adjudicated_decision = "none"
            adjudicated_reasons = "none"
        else:
            adjudication_state = (
                "current"
                if adjudication.assessment_event_ids == assessment_event_ids
                else "stale"
            )
            adjudicated_decision = adjudication.status
            adjudicated_reasons = ";".join(adjudication.reasons)
        flags = dict(revision.deformation_flags)
        lines.append(
            " | ".join(
                (
                    f"case_token={case_token}",
                    "presentations="
                    + (",".join(item.opaque_id for item in presentations) or "none"),
                    f"revision={revision.revision_id}",
                    f"automated_valid={revision.automated_valid}",
                    f"automated_reason={revision.automated_reason or 'none'}",
                    f"whole={flags.get('whole', 'not-available')}",
                    f"seam={flags.get('seam', 'not-available')}",
                    f"interior={flags.get('interior', 'not-available')}",
                    f"adjudication={adjudication_state}",
                    f"adjudicated_decision={adjudicated_decision}",
                    f"adjudicated_reasons={adjudicated_reasons}",
                )
            )
        )
    return tuple(lines)


def _unblinded_provenance_lines(
    state: ReviewState, active: ReviewManifest
) -> tuple[str, ...]:
    """Build the sole free-text and biological identity section of the record."""
    lines = [
        "UNBLINDED PROVENANCE APPENDIX",
        "",
        "Opaque ID | cohort | cell | round | notes",
    ]
    for presentation in active.visual_presentations:
        case_key = active.presentation_cases[presentation.opaque_id]
        assessment = _latest_assessment(state, presentation.opaque_id)
        lines.append(
            " | ".join(
                (
                    presentation.opaque_id,
                    case_key.cohort,
                    case_key.cell_id,
                    str(case_key.acquisition_round),
                    assessment.notes if assessment is not None else "",
                )
            )
        )
    lines.extend(
        (
            "",
            "Adjudication notes",
            "cohort | cell | round | revision | decision | notes",
        )
    )
    for history in sorted(
        state.case_histories.values(), key=lambda item: item.case_key
    ):
        for adjudication in history.adjudications:
            lines.append(
                " | ".join(
                    (
                        history.case_key.cohort,
                        history.case_key.cell_id,
                        str(history.case_key.acquisition_round),
                        adjudication.revision_id,
                        adjudication.status,
                        adjudication.notes,
                    )
                )
            )
    return tuple(lines)


def _render_unblinded_provenance(
    pdf: PdfPages, state: ReviewState, active: ReviewManifest
) -> None:
    """Render the sole biological mapping, after every blinded PDF page."""
    lines = _unblinded_provenance_lines(state, active)
    for page_number, start in enumerate(range(0, len(lines), 42), start=1):
        figure, axis = plt.subplots(figsize=(11, 8.5))
        axis.set_axis_off()
        page = list(lines[start : start + 42])
        if start:
            page = [
                f"UNBLINDED PROVENANCE APPENDIX — page {page_number}",
                "",
                *page,
            ]
        axis.text(
            0.03,
            0.97,
            "\n".join(page),
            va="top",
            family="monospace",
            fontsize=8,
        )
        pdf.savefig(figure)
        plt.close(figure)


def _render_pdf(
    path: Path,
    state: ReviewState,
    active: ReviewManifest,
    final_rows: Sequence[Mapping[str, object]],
    consistency: Sequence[Mapping[str, object]],
) -> None:
    decisions = _final_decisions(final_rows)
    with PdfPages(path) as pdf:
        for presentation in active.visual_presentations:
            _render_visual_page(pdf, state, active, presentation, decisions)
        _render_technical_summary(pdf, state, active, consistency, final_rows)
        _render_unblinded_provenance(pdf, state, active)


def _summary_text(
    state: ReviewState,
    active: ReviewManifest,
    final_rows: Sequence[Mapping[str, object]],
    consistency: Sequence[Mapping[str, object]],
) -> str:
    accepted = sum(row["decision"] == "accept" for row in final_rows)
    excluded = len(final_rows) - accepted
    remediation_history = sum(
        max(0, len(history.revisions) - 1) for history in state.case_histories.values()
    )
    repository = state.repository_identity
    lines = [
        "# Blinded QC review record",
        "",
        "## Finalized review summary",
        "",
        f"- Visual presentations: {len(active.visual_presentations)}",
        f"- Final cases: {len(final_rows)}",
        f"- Accepted: {accepted}",
        f"- Excluded: {excluded}",
        f"- Remediation revisions: {remediation_history}",
        f"- Reviewer: {state.reviewer}",
        f"- Review created: {state.created_at.isoformat()}",
        f"- Last review activity: {_latest_activity(state)}",
        f"- Repository commit: {repository.commit}",
        f"- Repository dirty at review creation: {repository.dirty}",
        f"- Repository dirty diff SHA-256: {repository.diff_sha256 or 'none'}",
        "",
        "## Final revision IDs",
        "",
    ]
    lines.extend(
        f"- {history.case_token}: {history.revisions[-1].revision_id}"
        for history in sorted(
            state.case_histories.values(), key=lambda item: item.case_token
        )
    )
    lines.extend(("", "## Concealed-repeat consistency", ""))
    lines.extend(
        "- "
        + " | ".join(
            (
                str(row["base_opaque_id"]),
                str(row["repeat_opaque_id"]),
                str(row["agreement"] or "pending"),
                str(row["adjudicated_decision"] or "not needed"),
            )
        )
        for row in consistency
    )
    lines.extend(("", "## Target-blind technical reconciliation", ""))
    lines.extend(f"- {line}" for line in _technical_reconciliation_lines(state, active))
    lines.extend(("", "## Unblinded provenance appendix", ""))
    lines.extend(("### Presentation provenance", ""))
    lines.extend(f"- {line}" for line in _unblinded_provenance_lines(state, active)[2:])
    lines.extend(("", "### Exclusion details", ""))
    lines.extend(
        "- "
        + " | ".join(
            (
                str(row["cohort"]),
                str(row["cell_id"]),
                str(row["acquisition_round"]),
                str(row["decision"]),
                str(row["reasons"]),
                str(row["reviewer"]),
                str(row["reviewed_at"]),
            )
        )
        for row in final_rows
        if row["decision"] == "exclude"
    )
    lines.extend(("", "### Remediation revision history", ""))
    for history in sorted(
        state.case_histories.values(), key=lambda item: item.case_key
    ):
        if len(history.revisions) < 2:
            continue
        case_key = history.case_key
        lines.append(
            f"- Case: {case_key.cohort} | {case_key.cell_id} | {case_key.acquisition_round}"
        )
        for revision in history.revisions:
            lines.append(
                "  - "
                + " | ".join(
                    (
                        revision.revision_id,
                        f"automated_valid={revision.automated_valid}",
                        f"automated_reason={revision.automated_reason or 'none'}",
                        f"deformation_flags={dict(revision.deformation_flags)}",
                    )
                )
            )
            for presentation in active.visual_presentations:
                if (
                    presentation.case_token != history.case_token
                    or presentation.revision_id != revision.revision_id
                ):
                    continue
                assessment = _latest_assessment(state, presentation.opaque_id)
                if assessment is not None:
                    lines.append(
                        "    - "
                        + " | ".join(
                            (
                                presentation.opaque_id,
                                assessment.status,
                                ";".join(assessment.reasons),
                                assessment.notes,
                                assessment.reviewer,
                                assessment.assessed_at.isoformat(),
                            )
                        )
                    )
    return "\n".join(lines) + "\n"


def _write_summary(path: Path, content: str) -> None:
    """Write one already-validated Markdown record to its staging path."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_staged_reports(pdf_path: Path, summary_path: Path) -> None:
    if not pdf_path.read_bytes().startswith(b"%PDF"):
        raise ValueError("staged review PDF is invalid")
    if not summary_path.read_text(encoding="utf-8").strip():
        raise ValueError("staged review Markdown is empty")


def write_review_report(
    state: ReviewState,
    manifest: ReviewManifest,
    pdf_path: Path,
    summary_path: Path,
) -> tuple[Path, Path]:
    """Atomically publish blinded visual and unblinded provenance review records.

    The supplied state must reconstruct exactly from the initial manifest, have
    final decisions for every case, and still match every explicitly allowlisted
    overlay identity.  Biological mappings are rendered only in the final
    unblinded provenance appendix.
    """
    if pdf_path.resolve() == summary_path.resolve():
        raise ValueError("PDF and Markdown report paths must differ")
    matplotlib.rcParams.update(
        {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
    )
    validated = ReviewState.from_dict(state.to_dict(), manifest)
    active = validated.active_manifest(manifest)
    final_rows = _final_rows(validated, active)
    consistency = _consistency_rows(validated, active)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    staged_pdf = pdf_path.with_name(f".{pdf_path.name}.{uuid.uuid4().hex}.tmp")
    staged_summary = summary_path.with_name(
        f".{summary_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        _render_pdf(staged_pdf, validated, active, final_rows, consistency)
        _fsync_file(staged_pdf)
        _write_summary(
            staged_summary, _summary_text(validated, active, final_rows, consistency)
        )
        _validate_staged_reports(staged_pdf, staged_summary)
        os.replace(staged_pdf, pdf_path)
        _fsync_directory(pdf_path.parent)
        os.replace(staged_summary, summary_path)
        _fsync_directory(summary_path.parent)
    finally:
        staged_pdf.unlink(missing_ok=True)
        staged_summary.unlink(missing_ok=True)
    return pdf_path, summary_path


__all__ = ["write_review_report"]
