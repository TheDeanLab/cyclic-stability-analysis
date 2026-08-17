# Target-blind Registration QC Reviewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone PEP 723 Tkinter application that presents randomized DAPI/tubulin registration evidence, preserves pretty resumable review state, supports immutable registration remediation, and exports finalized target-blind QC decisions.

**Architecture:** Keep strict inventory/randomization, state transitions, exports, reporting, and Tk widgets in separate focused modules. The GUI receives an opaque presentation view model and never receives biological identifiers or target-image paths during visual review. Every saved assessment is bound to SHA-256 identities of immutable QC inputs, while remediation imports a new QC root as a new review revision instead of replacing prior evidence.

**Tech Stack:** Python 3.12, standard-library `tkinter`, Pillow/ImageTk, tifffile, Matplotlib/PdfPages, Cy-ExM `FileIdentity`-compatible provenance concepts, pytest, Ruff.

## Global Constraints

- Supported Python is exactly `>=3.12,<3.13`; the selected runtime provides Tkinter 8.6.
- The application is launched inside an existing GUI-enabled HPC/VNC session and contains no VNC, browser, server, or remote-display management.
- The only image basenames admitted are `RR-(dapi|tubulin)-(pre|affine|nonlinear)-(xy|xz|yz).tif`, where `RR` is acquisition round `01`, `03`, `04`, or `05`.
- Never discover, open, hash for display, thumbnail, or report a Golgi, Lamin, LAMP1, 647-nm, or other target image; cohort names in parent directories and metadata do not authorize image access.
- The initial inventory is exactly 40 moving-round cases: 39 visual cases and the hard-invalid Lamin Cell1 round-1 technical case.
- Create exactly four concealed, deterministic repeat presentations from distinct visual cases; never identify repeats in visual-review widget text.
- Visual decisions remain provisional until explicit finalization. `needs-registration-improvement` and `needs-adjudication` block final export.
- Improved registrations are imported only from a new immutable QC root and receive new opaque presentation identifiers; prior evidence and assessments remain preserved.
- JSON uses two-space indentation, sorted keys, UTF-8, ISO-8601 timestamps, one trailing newline, atomic promotion, directory fsync where supported, and a single previous-valid backup.
- Final CSV uses exactly `cohort,cell_id,acquisition_round,decision,reasons,reviewer,reviewed_at`; reason tokens are semicolon-separated.
- Matplotlib PDF and PostScript font types are exactly 42.
- Before every `uv` command export:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
```

## File Structure

- Create `scripts/cyclic_stability/review_models.py`: immutable review-domain values, strict serializers, and validators.
- Create `scripts/cyclic_stability/review_inventory.py`: fixed-schema CSV loading, overlay allowlisting, RGB validation, SHA-256 evidence identities, deterministic randomization, and remediation-revision discovery.
- Create `scripts/cyclic_stability/review_store.py`: pretty JSON persistence, backup/recovery, assessment history, reconciliation, and revision attachment.
- Create `scripts/cyclic_stability/review_exports.py`: remediation queue, consistency report, final CSV, and Markdown summary.
- Create `scripts/cyclic_stability/review_report.py`: target-blind archival PDF pages and final provenance appendix.
- Create `scripts/cyclic_stability/review_gui.py`: opaque view model, Tkinter widgets, image viewer, keyboard navigation, and reconciliation/finalization screens.
- Create `scripts/review_registration_qc.py`: PEP 723 command-line entry point and display preflight.
- Modify `scripts/run_tests.py`: add the same Pillow runtime range used by the reviewer.
- Modify `tests/synthetic.py`: add a complete tiny RGB QC fixture builder.
- Create `tests/test_review_inventory.py`, `tests/test_review_store.py`, `tests/test_review_exports.py`, `tests/test_review_report.py`, and `tests/test_review_gui.py`.
- Modify `README.md`: document launch, provisional review, remediation, resume, and final export.

## Reuse analysis

- Closest existing symbols: `scripts/run_registration_qc.py`, `cyclic_stability.registration_qc.publish_qc_outputs`, `cyclic_stability.config.load_analysis_config`, `cyclic_stability.task._atomic_bytes`, and the existing decision-request and final-decision CSV schemas.
- Decision: consume the published QC directory and reuse the strict final CSV schema without changing scientific analysis configuration. Add a separate interactive reviewer because evidence generation does not own human randomization, GUI state, assessment history, remediation, or finalization.
- Private atomic helpers are not reused: they do not provide pretty JSON, previous-valid recovery, strict reopen validation, or assessment-history semantics. `review_store.py` will implement that distinct lifecycle and test interruption recovery directly.
- Rejected alternative: adding GUI behavior to `run_registration_qc.py` would couple immutable evidence generation to subjective review and make headless QC generation depend on Tk.
- Rejected alternative: a browser application adds a server and browser-state boundary despite Tkinter 8.6 being available in the selected workflow.
- Public documentation: give the PEP 723 CLI a complete Numpydoc module/`main` contract, document ownership and non-ownership, and add the canonical launch to `README.md`.

---

### Task 1: Strict Target-blind Inventory and Randomized Manifest

**Files:**
- Create: `scripts/cyclic_stability/review_models.py`
- Create: `scripts/cyclic_stability/review_inventory.py`
- Modify: `tests/synthetic.py`
- Create: `tests/test_review_inventory.py`

**Interfaces:**
- Produces: `CaseKey`, `EvidenceIdentity`, `OverlayEvidence`, `CaseRevision`, `Presentation`, and `ReviewManifest` frozen dataclasses.
- Produces: `build_review_manifest(qc_root: Path, *, seed: int, repeat_count: int = 4) -> ReviewManifest`.
- Produces: `read_revision_cases(qc_root: Path) -> dict[CaseKey, CaseRevision]` for Task 3.
- Consumes: existing `decision-requests.csv`, `registration-qc.csv`, `deformation.csv`, and RGB overlays.

- [ ] **Step 1: Add the complete synthetic QC fixture**

Add `REVIEW_DECISION_COLUMNS`, `REVIEW_REGISTRATION_COLUMNS`, and `REVIEW_DEFORMATION_COLUMNS` constants plus a private `_write_review_rows(path: Path, columns: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> None` CSV helper. Add `write_review_qc_fixture(root: Path) -> Path`, which writes ten cells, four moving rounds per cell, one hard-invalid row, and 702 tiny RGB overlays using production filenames. The table helper writes:

- `decision-requests.csv`: `cohort,cell_id,acquisition_round,automated_valid,automated_reason,decision`;
- `registration-qc.csv`: `cohort,cell_id,acquisition_round,workflow_round,is_reference,automated_valid,invalid_reason`; and
- `deformation.csv`: `cohort,cell_id,acquisition_round,region,warp_flags`.

For every visual case, the registration row is valid and the deformation table contains `whole`, `seam`, and `interior` rows. The Lamin Cell1 round-1 registration row is hard-invalid and has no overlay or deformation rows.

```python
def write_review_qc_fixture(root: Path) -> Path:
    """Write the complete 40-case target-blind QC inventory."""
    import tifffile

    qc = root / "qc"
    qc.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for cohort in ("golgi", "lamin"):
        for cell_number in range(1, 6):
            for round_number in (1, 3, 4, 5):
                hard_invalid = (
                    cohort == "lamin" and cell_number == 1 and round_number == 1
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "cell_id": f"Cell{cell_number}",
                        "acquisition_round": round_number,
                        "automated_valid": not hard_invalid,
                        "automated_reason": "invalid-affine-reflection"
                        if hard_invalid
                        else "",
                        "decision": "",
                    }
                )
                if hard_invalid:
                    continue
                overlay_dir = (
                    root / "cells" / cohort / f"Cell{cell_number}" / "qc" / "overlays"
                )
                overlay_dir.mkdir(parents=True, exist_ok=True)
                for channel in ("dapi", "tubulin"):
                    for stage in ("pre", "affine", "nonlinear"):
                        for view in ("xy", "xz", "yz"):
                            image = np.full((8, 12, 3), round_number, dtype=np.uint8)
                            tifffile.imwrite(
                                overlay_dir
                                / f"{round_number:02d}-{channel}-{stage}-{view}.tif",
                                image,
                                photometric="rgb",
                            )
    _write_review_tables(qc, rows)
    return root
```

Define `_write_review_tables(qc: Path, decision_rows: Sequence[Mapping[str, object]]) -> None` in the same fixture module. It derives the exact registration/deformation rows described above and delegates all three files to `_write_review_rows`; no test-local schema or row builders remain implicit.

- [ ] **Step 2: Write failing inventory tests**

Cover exact counts, opaque IDs, stable randomization, four distinct repeats, RGB rejection, missing overlay rejection, and prohibited overlay basenames:

```python
def test_manifest_is_complete_randomized_and_target_blind(tmp_path: Path) -> None:
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    manifest = build_review_manifest(qc_root, seed=8142026)
    assert len(manifest.case_revisions) == 40
    assert len(manifest.visual_presentations) == 43
    assert (
        sum(item.repeat_of is not None for item in manifest.visual_presentations) == 4
    )
    assert (
        len(
            {item.repeat_of for item in manifest.visual_presentations if item.repeat_of}
        )
        == 4
    )
    assert all(
        item.opaque_id.startswith("QC-") for item in manifest.visual_presentations
    )
    assert all("Cell" not in item.opaque_id for item in manifest.visual_presentations)


def test_inventory_rejects_non_allowlisted_tiff(tmp_path: Path) -> None:
    qc_root = write_review_qc_fixture(tmp_path / "qc-only")
    extra = qc_root / "cells/golgi/Cell1/qc/overlays/01-647-golgi-nonlinear-xy.tif"
    extra.write_bytes(b"target must never be read")
    with pytest.raises(
        ValueError, match="overlay basename outside target-blind allowlist"
    ):
        build_review_manifest(qc_root, seed=1)
```

- [ ] **Step 3: Run the inventory tests to verify failure**

Run:

```bash
uv run --script scripts/run_tests.py tests/test_review_inventory.py -q
```

Expected: collection fails because `cyclic_stability.review_inventory` does not exist.

- [ ] **Step 4: Implement immutable review models**

Define strict frozen values and explicit `to_dict`/`from_dict` validation. Use full SHA-256 content identity rather than only path metadata:

```python
@dataclass(frozen=True, slots=True, order=True)
class CaseKey:
    cohort: Literal["golgi", "lamin"]
    cell_id: str
    acquisition_round: Literal[1, 3, 4, 5]


@dataclass(frozen=True, slots=True)
class EvidenceIdentity:
    path: str
    size: int
    mtime_ns: int
    sha256: str

    @classmethod
    def from_path(cls, path: Path) -> EvidenceIdentity:
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        stat = path.stat()
        return cls(str(path.resolve()), stat.st_size, stat.st_mtime_ns, digest)
```

`Presentation` contains only `opaque_id`, `revision_id`, overlay evidence, and `repeat_of`; biological mapping remains in `ReviewManifest.presentation_cases` and is never copied into the GUI view model.

- [ ] **Step 5: Implement fixed-schema inventory validation and randomization**

Use an exact basename regular expression and explicitly construct all 18 overlay paths. Validate with tifffile without scanning adjacent image trees:

```python
OVERLAY_NAME = re.compile(
    r"^(01|03|04|05)-(dapi|tubulin)-(pre|affine|nonlinear)-(xy|xz|yz)\.tif$"
)


def _overlay_paths(directory: Path, round_number: int) -> tuple[Path, ...]:
    expected = tuple(
        directory / f"{round_number:02d}-{channel}-{stage}-{view}.tif"
        for channel in ("dapi", "tubulin")
        for stage in ("pre", "affine", "nonlinear")
        for view in ("xy", "xz", "yz")
    )
    unexpected = sorted(
        path.name
        for path in directory.glob("*.tif")
        if OVERLAY_NAME.fullmatch(path.name) is None
    )
    if unexpected:
        raise ValueError(
            f"overlay basename outside target-blind allowlist: {unexpected}"
        )
    return expected
```

Use `random.Random(seed).shuffle`, select four distinct base presentations with `sample`, derive opaque IDs from shuffled ordinal plus a seed-bound digest, and enforce 39 visual plus one technical case in production.

- [ ] **Step 6: Run focused tests and quality gates**

Run:

```bash
uv run --script scripts/run_tests.py tests/test_review_inventory.py -q
uvx ruff check scripts/cyclic_stability/review_models.py scripts/cyclic_stability/review_inventory.py tests/synthetic.py tests/test_review_inventory.py
uvx ruff format --check scripts/cyclic_stability/review_models.py scripts/cyclic_stability/review_inventory.py tests/synthetic.py tests/test_review_inventory.py
git diff --check
```

Expected: all inventory tests pass and all gates are clean.

- [ ] **Step 7: Commit Task 1**

```bash
git add scripts/cyclic_stability/review_models.py scripts/cyclic_stability/review_inventory.py tests/synthetic.py tests/test_review_inventory.py
git commit -m "feat: validate blinded QC review inventory"
```

---

### Task 2: Pretty Atomic Review State and Assessment History

**Files:**
- Create: `scripts/cyclic_stability/review_store.py`
- Create: `tests/test_review_store.py`

**Interfaces:**
- Consumes: `ReviewManifest` and presentation IDs from Task 1.
- Produces: `AssessmentStatus`, `Assessment`, `CaseHistory`, and `ReviewState` values in `review_models.py`.
- Produces: `ReviewStore.create(review_dir: Path, manifest: ReviewManifest, reviewer: str, *, now: Callable[[], datetime]) -> ReviewStore`.
- Produces: `ReviewStore.open(review_dir: Path, manifest: ReviewManifest, reviewer: str, *, now: Callable[[], datetime]) -> ReviewStore`.
- Produces: `ReviewStore.assess(opaque_id: str, status: AssessmentStatus, reasons: tuple[str, ...], notes: str) -> ReviewState`.
- Produces: `ReviewStore.adjudicate(case_key: CaseKey, decision: Literal["accept", "exclude"], reasons: tuple[str, ...], notes: str) -> ReviewState`.

- [ ] **Step 1: Write failing state tests**

Define `fixed_clock() -> datetime` in this test module, returning the timezone-aware constant `2026-08-17T12:00:00+00:00`. In addition to the examples below, add fault-injection tests for failure before promotion, malformed-current recovery from the previous valid state, rejection when both copies are malformed, strict unknown-key rejection, and dirty-repository identity changes.

```python
def test_review_state_is_pretty_atomic_and_resumable(tmp_path: Path) -> None:
    manifest = build_review_manifest(write_review_qc_fixture(tmp_path / "qc"), seed=7)
    store = ReviewStore.create(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    opaque_id = manifest.visual_presentations[0].opaque_id
    store.assess(opaque_id, "accept", ("dapi-aligned", "tubulin-aligned"), "clear")
    text = (tmp_path / "review/review-state.json").read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert '\n  "assessment_history":' in text
    payload = json.loads(text)
    assert payload["reviewer"] == "Reviewer"
    assert payload["schema_version"] == 1
    assert payload["repository_identity"]["commit"]
    if payload["repository_identity"]["dirty"]:
        assert payload["repository_identity"]["diff_sha256"]
    reopened = ReviewStore.open(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    assert reopened.state.assessment_history[opaque_id][-1].status == "accept"


def test_changed_overlay_identity_blocks_resume(tmp_path: Path) -> None:
    qc_root = write_review_qc_fixture(tmp_path / "qc")
    manifest = build_review_manifest(qc_root, seed=7)
    ReviewStore.create(tmp_path / "review", manifest, "Reviewer", now=fixed_clock)
    Path(manifest.visual_presentations[0].overlays[0].path).write_bytes(b"changed")
    changed = build_review_manifest(qc_root, seed=7)
    with pytest.raises(ValueError, match="reviewed evidence identity changed"):
        ReviewStore.open(tmp_path / "review", changed, "Reviewer", now=fixed_clock)
```

- [ ] **Step 2: Run state tests to verify failure**

Run:

```bash
uv run --script scripts/run_tests.py tests/test_review_store.py -q
```

Expected: collection fails because `cyclic_stability.review_store` does not exist.

- [ ] **Step 3: Add strict assessment and state models**

Use literal statuses and an allowlisted reason set:

```python
AssessmentStatus = Literal[
    "accept",
    "needs-registration-improvement",
    "exclude",
    "needs-adjudication",
]
FINAL_STATUSES = frozenset({"accept", "exclude"})
REASON_TOKENS = frozenset(
    {
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
        "invalid-affine-reflection",
    }
)
```

Every assessment records opaque presentation ID, revision ID, status, nonempty unique reason tokens, notes, reviewer, and UTC timestamp. State validation rejects unknown keys, duplicate event IDs, missing manifest identity, invalid timestamps, and biological keys not present in the immutable manifest. Add `RepositoryIdentity(commit: str, dirty: bool, diff_sha256: str | None)`; obtain the commit and exact binary diff digest at state creation, require the digest whenever dirty is true, and reject resume when either commit or dirty-tree digest changes.

- [ ] **Step 4: Implement pretty atomic persistence and recovery**

Use a unique temporary path in the destination directory, fsync the file, rotate the current valid state to `review-state.previous.json`, promote the new file, then fsync the directory:

```python
def _pretty_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _atomic_pretty_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        stream.write(_pretty_json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
```

Before replacing a valid current state, copy its parsed canonical payload through the same atomic writer to the previous-state path. On reopen, parse and strictly validate current and previous independently; recover from previous only when current is absent or malformed, and record the recovery event.

- [ ] **Step 5: Implement append-preserving assessments and adjudication**

`assess` appends an immutable event and writes before returning. `adjudicate` is allowed only after a visual assessment exists and records a distinct event; it never mutates or deletes the earlier entry. Reject empty reasons, unknown reason tokens, reviewer mismatch, and biological identifiers passed where opaque IDs are required.

- [ ] **Step 6: Run focused tests and quality gates**

```bash
uv run --script scripts/run_tests.py tests/test_review_store.py tests/test_review_inventory.py -q
uvx ruff check scripts/cyclic_stability/review_models.py scripts/cyclic_stability/review_store.py tests/test_review_store.py
uvx ruff format --check scripts/cyclic_stability/review_models.py scripts/cyclic_stability/review_store.py tests/test_review_store.py
git diff --check
```

Expected: state and inventory tests pass; formatting and lint are clean.

- [ ] **Step 7: Commit Task 2**

```bash
git add scripts/cyclic_stability/review_models.py scripts/cyclic_stability/review_store.py tests/test_review_store.py
git commit -m "feat: persist resumable blinded reviews"
```

---

### Task 3: Remediation Revisions, Reconciliation, and Final CSV

**Files:**
- Modify: `scripts/cyclic_stability/review_inventory.py`
- Modify: `scripts/cyclic_stability/review_store.py`
- Create: `scripts/cyclic_stability/review_exports.py`
- Create: `tests/test_review_exports.py`

**Interfaces:**
- Consumes: `ReviewStore`, `ReviewState`, `ReviewManifest`, and `read_revision_cases`.
- Produces: `attach_revision_root(store: ReviewStore, revision_root: Path) -> ReviewState`.
- Produces: `write_operational_exports(state: ReviewState, manifest: ReviewManifest, review_dir: Path) -> tuple[Path, Path]` for remediation and consistency CSVs.
- Produces: `finalize_decisions(store: ReviewStore, manifest: ReviewManifest, review_dir: Path) -> Path`.

- [ ] **Step 1: Write failing remediation and export tests**

Extend `tests/synthetic.py` with these exact reusable fixture interfaces before adding the tests:

- `rewrite_review_case_pixels(qc_root: Path, case_key: CaseKey) -> None`: rewrite all 18 allowlisted overlays for one case with distinct RGB pixels;
- `fresh_review_store(tmp_path: Path, *, seed: int = 7) -> tuple[ReviewStore, ReviewManifest]`: build a fixture manifest and new store using `fixed_clock`; and
- `completed_review_store(tmp_path: Path) -> tuple[ReviewStore, ReviewManifest]`: assess all 43 visual presentations, adjudicate any concealed-repeat inconsistency deterministically, and assign the hard-invalid technical exclusion so finalization is permitted.

The helpers use only public reviewer APIs and are imported by later report and GUI tests.

```python
def test_remediation_is_nonfinal_and_new_revision_preserves_history(
    tmp_path: Path,
) -> None:
    qc_root = write_review_qc_fixture(tmp_path / "qc-v1")
    manifest = build_review_manifest(qc_root, seed=9)
    store = ReviewStore.create(
        tmp_path / "review", manifest, "Reviewer", now=fixed_clock
    )
    presentation = manifest.visual_presentations[0]
    store.assess(
        presentation.opaque_id,
        "needs-registration-improvement",
        ("persistent-interior-misalignment",),
        "retry registration",
    )
    with pytest.raises(ValueError, match="registration improvement remains unresolved"):
        finalize_decisions(store, manifest, tmp_path / "review")
    revised_root = write_review_qc_fixture(tmp_path / "qc-v2")
    rewrite_review_case_pixels(
        revised_root, manifest.presentation_cases[presentation.opaque_id]
    )
    attach_revision_root(store, revised_root)
    assert len(store.state.case_histories[presentation.case_token].revisions) == 2
    assert (
        store.state.assessment_history[presentation.opaque_id][0].notes
        == "retry registration"
    )


def test_final_csv_has_one_binary_row_per_biological_case(tmp_path: Path) -> None:
    store, manifest = completed_review_store(tmp_path)
    output = finalize_decisions(store, manifest, tmp_path / "review")
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert len(rows) == 40
    assert set(rows[0]) == {
        "cohort",
        "cell_id",
        "acquisition_round",
        "decision",
        "reasons",
        "reviewer",
        "reviewed_at",
    }
    assert {row["decision"] for row in rows} <= {"accept", "exclude"}
```

- [ ] **Step 2: Run export tests to verify failure**

```bash
uv run --script scripts/run_tests.py tests/test_review_exports.py -q
```

Expected: collection fails because `cyclic_stability.review_exports` does not exist.

- [ ] **Step 3: Implement remediation queue and consistency report**

Write deterministic CSVs atomically after every state change. `remediation-requests.csv` includes only unresolved latest statuses and uses actual cohort/cell/round so operators can locate registration. `consistency-report.csv` retains both concealed assessments, agreement status, and final adjudication without adding biological decision rows.

```python
REMEDIATION_COLUMNS = (
    "cohort",
    "cell_id",
    "acquisition_round",
    "revision_id",
    "reasons",
    "notes",
    "reviewer",
    "reviewed_at",
)
CONSISTENCY_COLUMNS = (
    "base_opaque_id",
    "repeat_opaque_id",
    "base_status",
    "repeat_status",
    "agreement",
    "adjudicated_decision",
)
```

- [ ] **Step 4: Implement immutable revision attachment**

`attach_revision_root` loads a complete target-blind QC root, considers only cases whose latest status is `needs-registration-improvement`, requires different SHA-256 identities for at least one overlay or technical record, and rejects changed case identity or missing evidence. Derive a new revision ID from the evidence digest and a new opaque presentation ID from the existing seed, revision ID, and case token. Append the revision and presentation, then persist before returning.

- [ ] **Step 5: Implement strict finalization and atomic CSV export**

Finalization checks all 40 original cases, latest revision status, repeat reconciliation, reviewer metadata, current evidence identities, and the hard-invalid technical case. Use `csv.DictWriter` with exact column order and semicolon-joined reasons. Preserve an existing final CSV as `qc-decisions.previous.csv` before atomic promotion. Record output SHA-256 and timestamp in review state only after reopening and validating the written CSV.

- [ ] **Step 6: Run focused tests and quality gates**

```bash
uv run --script scripts/run_tests.py tests/test_review_exports.py tests/test_review_store.py tests/test_review_inventory.py -q
uvx ruff check scripts/cyclic_stability/review_inventory.py scripts/cyclic_stability/review_store.py scripts/cyclic_stability/review_exports.py tests/test_review_exports.py
uvx ruff format --check scripts/cyclic_stability/review_inventory.py scripts/cyclic_stability/review_store.py scripts/cyclic_stability/review_exports.py tests/test_review_exports.py
git diff --check
```

Expected: remediation, export, state, and inventory tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add scripts/cyclic_stability/review_inventory.py scripts/cyclic_stability/review_store.py scripts/cyclic_stability/review_exports.py tests/test_review_exports.py
git commit -m "feat: support blinded QC remediation"
```

---

### Task 4: Archival PDF and Markdown Review Record

**Files:**
- Create: `scripts/cyclic_stability/review_report.py`
- Create: `tests/test_review_report.py`

**Interfaces:**
- Consumes: finalized `ReviewState`, `ReviewManifest`, and allowlisted overlay evidence.
- Produces: `write_review_report(state: ReviewState, manifest: ReviewManifest, pdf_path: Path, summary_path: Path) -> tuple[Path, Path]`.

- [ ] **Step 1: Write failing report tests**

```python
def test_report_is_target_blind_and_uses_truetype_fonts(tmp_path: Path) -> None:
    store, manifest = completed_review_store(tmp_path)
    pdf, summary = write_review_report(
        store.state,
        manifest,
        tmp_path / "review/review-record.pdf",
        tmp_path / "review/review-summary.md",
    )
    assert pdf.read_bytes().startswith(b"%PDF")
    assert "target channel" not in summary.read_text(encoding="utf-8").lower()
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42
```

Add a spy image loader test proving that every opened basename matches the DAPI/tubulin allowlist and that biological mapping is rendered only in the final appendix.

- [ ] **Step 2: Run report tests to verify failure**

```bash
uv run --script scripts/run_tests.py tests/test_review_report.py -q
```

Expected: collection fails because `cyclic_stability.review_report` does not exist.

- [ ] **Step 3: Implement randomized evidence pages**

Use `matplotlib.backends.backend_pdf.PdfPages`. Render one page per visual presentation with two channel sections and the 3 by 3 stage/view arrangement. Page titles contain opaque ID, revision, provisional/final decision, reasons, and notes but no biological identifiers. Load only paths already validated in `OverlayEvidence`; do not scan directories.

- [ ] **Step 4: Implement reconciliation, consistency, and provenance pages**

After visual pages, render technical summaries and concealed-repeat agreement. The final appendix may map opaque IDs to cohort/cell/round because it is generated only after finalization. Clearly label it as unblinded provenance. Generate a Markdown summary with counts, exclusions, remediation history, final revision IDs, reviewer, timestamps, repository state, and consistency results.

- [ ] **Step 5: Set publication-safe fonts and atomic report publication**

```python
matplotlib.rcParams.update(
    {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
)
```

Write the PDF and Markdown to unique sibling temporary paths, validate the PDF header and nonempty Markdown, then atomically promote each. Remove staged files on failure without replacing prior valid reports.

- [ ] **Step 6: Run focused tests and quality gates**

```bash
uv run --script scripts/run_tests.py tests/test_review_report.py tests/test_review_exports.py -q
uvx ruff check scripts/cyclic_stability/review_report.py tests/test_review_report.py
uvx ruff format --check scripts/cyclic_stability/review_report.py tests/test_review_report.py
git diff --check
```

Expected: report and export tests pass, with clean lint and formatting.

- [ ] **Step 7: Commit Task 4**

```bash
git add scripts/cyclic_stability/review_report.py tests/test_review_report.py
git commit -m "feat: render blinded QC review records"
```

---

### Task 5: Thin Tkinter Interface and PEP 723 Entry Point

**Files:**
- Create: `scripts/cyclic_stability/review_gui.py`
- Create: `scripts/review_registration_qc.py`
- Modify: `scripts/run_tests.py`
- Create: `tests/test_review_gui.py`

**Interfaces:**
- Consumes: Task 1 manifest, Task 2 store, Task 3 exports, and Task 4 report.
- Produces: `ReviewViewModel.from_store(store: ReviewStore) -> ReviewViewModel`, whose public presentation values contain no `CaseKey` or filesystem paths.
- Produces: private `EvidenceResolver(manifest: ReviewManifest)` that resolves opaque evidence tokens to allowlisted paths only at image-load time.
- Produces: `ReviewApplication(root: tkinter.Tk, store: ReviewStore, manifest: ReviewManifest)`, the sole widget owner.
- Produces: `main(argv: list[str] | None = None) -> int` in the PEP 723 script.

- [ ] **Step 1: Add Pillow to PEP 723 test dependencies and write failing GUI tests**

Add `"Pillow>=10,<13"` to `scripts/run_tests.py`. Tests must prove view-model blinding, keyboard mappings, save-before-navigation, no-display failure, and CLI argument ownership:

```python
def test_visual_view_model_exposes_no_biological_identity(tmp_path: Path) -> None:
    store, manifest = fresh_review_store(tmp_path)
    view = ReviewViewModel.from_store(store, manifest)
    payload = json.dumps(asdict(view.presentations[0]), sort_keys=True)
    assert "Cell" not in payload
    assert "golgi" not in payload
    assert "lamin" not in payload


def test_navigation_persists_before_changing_case(
    fake_app: RecordingReviewApplication,
) -> None:
    fake_app.select_status("accept", ("dapi-aligned",), "reviewed")
    fake_app.next_case()
    assert fake_app.calls[:2] == ["persist", "render-next"]
```

- [ ] **Step 2: Run GUI tests to verify failure**

```bash
uv run --script scripts/run_tests.py tests/test_review_gui.py -q
```

Expected: collection fails because `cyclic_stability.review_gui` does not exist.

- [ ] **Step 3: Implement the opaque view model and image loader**

`ReviewViewModel` exposes opaque ID, progress position, revision label, opaque evidence tokens keyed only by channel/stage/view, current provisional state, reasons, and notes. It does not expose `CaseKey`, filesystem paths, automated flags, repeat status, or technical reasons during visual review. `EvidenceResolver` privately maps a token back through the immutable manifest, revalidates the exact basename and SHA-256 identity, then returns the path to the loader. Use Pillow to open RGB TIFFs, verify mode/shape, create aspect-preserving thumbnails, and retain `ImageTk.PhotoImage` references.

In `tests/test_review_gui.py`, define `RecordingReviewApplication` as a widget-free harness around the navigation controller: it records `"persist"` when `persist_current()` is invoked and `"render-next"` when `render_current()` advances. The `fake_app` fixture returns this concrete harness from `fresh_review_store`; no undefined fake GUI utility is assumed.

- [ ] **Step 4: Implement the main visual-review window**

Create DAPI/tubulin `ttk.Notebook` tabs, each with a 3 by 3 grid. Add click-to-open `Toplevel` image windows with scrollable canvas, zoom controls, and pan bindings. Add status radio buttons, reason checkboxes, notes, progress, save indicator, previous/next buttons, and exact keyboard shortcuts `A`, `R`, `E`, `U`, arrows, and reason-number keys.

Every status or text change marks the form dirty. Navigation calls `persist_current()` and blocks when status has no reason. Window close persists a valid edit, warns on an incomplete dirty edit, and never discards silently.

- [ ] **Step 5: Implement reconciliation, remediation, and finalization screens**

After all visual presentations have assessments, show target-blind automated flags, seam/interior summaries, the technical affine case, repeat agreement, and unresolved states. Provide actions to adjudicate, write remediation exports, attach a revision root chosen with `filedialog.askdirectory`, and finalize. Finalization calls Task 3 first, then Task 4; failures show a concise dialog and preserve the prior state.

- [ ] **Step 6: Implement the PEP 723 CLI**

Declare the pinned Cy-ExM dependency already used by repository scripts plus Pillow and current scientific runtime bounds. Parse:

```text
--qc-root PATH
--review-dir PATH
--reviewer TEXT
--seed INTEGER
--add-revision-root PATH
--validate-only
```

`--qc-root` is required only for first creation; resume derives it from the manifest. `--seed` is rejected on resume. `--add-revision-root` requires existing state. `--validate-only` performs all inventory and state checks without creating Tk widgets and is the supported headless preflight. Normal launch catches `tkinter.TclError` from `Tk()` and reports that a GUI-enabled session is required.

- [ ] **Step 7: Run focused tests and headless CLI checks**

```bash
uv run --script scripts/run_tests.py tests/test_review_gui.py tests/test_review_report.py tests/test_review_exports.py tests/test_review_store.py tests/test_review_inventory.py -q
uv run --script scripts/review_registration_qc.py --qc-root results/qc-only --review-dir /tmp/cyclic-qc-review-preflight --reviewer preflight --seed 8142026 --validate-only
env -u DISPLAY uv run --script scripts/review_registration_qc.py --qc-root results/qc-only --review-dir /tmp/cyclic-qc-no-display --reviewer preflight --seed 8142026
```

Expected: tests pass; validate-only reports 40 cases, 39 visual cases, and four repeats; normal launch exits nonzero with a concise GUI-display message.

- [ ] **Step 8: Run quality gates**

```bash
uvx ruff check scripts/review_registration_qc.py scripts/run_tests.py scripts/cyclic_stability/review_gui.py tests/test_review_gui.py
uvx ruff format --check scripts/review_registration_qc.py scripts/run_tests.py scripts/cyclic_stability/review_gui.py tests/test_review_gui.py
git diff --check
```

Expected: lint, formatting, and diff checks are clean.

- [ ] **Step 9: Commit Task 5 with API reuse record**

```bash
git add scripts/review_registration_qc.py scripts/run_tests.py scripts/cyclic_stability/review_gui.py tests/test_review_gui.py
git commit -m "feat: add target-blind QC reviewer" -m "Reuse analysis:
- Composed the published registration-QC evidence and strict final-decision schema.
- Kept interactive review separate from run_registration_qc evidence generation.
- Added no VNC, server, scheduler, or target-image lifecycle."
```

---

### Task 6: Documentation, Full Verification, and VNC Handoff

**Files:**
- Modify: `README.md`
- Test: all `tests/`

**Interfaces:**
- Consumes: canonical CLI from Task 5.
- Produces: documented operator workflow and verified production preflight.

- [ ] **Step 1: Write a failing README contract test**

Add to `tests/test_review_gui.py`:

```python
def test_readme_documents_review_remediation_and_finalization() -> None:
    text = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    assert "review_registration_qc.py" in text
    assert "Needs registration improvement" in text
    assert "--add-revision-root" in text
    assert "qc-decisions.csv" in text
    assert "VNC" in text and "does not configure" in text
```

- [ ] **Step 2: Run the README test to verify failure**

```bash
uv run --script scripts/run_tests.py tests/test_review_gui.py::test_readme_documents_review_remediation_and_finalization -q
```

Expected: fails because the README lacks the reviewer workflow.

- [ ] **Step 3: Document the canonical workflow**

Add a `Target-blind registration review` section to `README.md` with:

- the VNC-session launch command;
- explicit statement that the script does not configure VNC;
- opaque randomization and concealed repeats;
- provisional decision meanings and reason tokens;
- autosave/resume paths and pretty JSON;
- remediation export, improved QC generation under a new root, and `--add-revision-root`;
- reconciliation and explicit finalization;
- generated files and the separate deliberate promotion into `config/qc-decisions.csv`; and
- the prohibition on target images and target-derived evidence.

- [ ] **Step 4: Run focused and full portable test suites**

```bash
uv run --script scripts/run_tests.py tests/test_review_inventory.py tests/test_review_store.py tests/test_review_exports.py tests/test_review_report.py tests/test_review_gui.py -q
uv run --script scripts/run_tests.py -q
```

Expected: all focused and full tests pass with zero failures.

- [ ] **Step 5: Run repository quality gates**

```bash
uvx ruff check .
uvx ruff format --check .
git diff --check
```

Expected: all gates exit zero.

- [ ] **Step 6: Run production inventory preflight without opening a GUI**

```bash
uv run --script scripts/review_registration_qc.py \
  --qc-root results/qc-only \
  --review-dir results/qc-review-preflight \
  --reviewer "preflight" \
  --seed 8142026 \
  --validate-only
```

Expected: validates exactly 40 biological cases, 39 visual cases, 43 randomized visual presentations, one technical case, 702 RGB overlays, and zero target images.

- [ ] **Step 7: Inspect generated preflight provenance**

Check:

```bash
jq '.' results/qc-review-preflight/randomization-manifest.json >/dev/null
jq '.' results/qc-review-preflight/review-state.json >/dev/null
rg -n '647|target.*\.tif|golgi.*\.tif|lamin.*\.tif' results/qc-review-preflight/*.json
```

Expected: JSON parses; any `golgi`/`lamin` occurrences are biological metadata only; no consumed image path contains a target-channel basename.

- [ ] **Step 8: Commit documentation**

```bash
git add README.md tests/test_review_gui.py
git commit -m "docs: explain blinded QC review workflow"
```

- [ ] **Step 9: Perform final clean-tree verification**

```bash
git status --short
git log -6 --oneline
```

Expected: no uncommitted files and six scoped implementation/documentation commits after the design and plan commits.

- [ ] **Step 10: Hand off the VNC smoke test**

Provide the user the canonical launch command against a non-final review directory. Ask them to verify window creation, DAPI/tubulin tab switching, nine panels per tab, full-resolution zoom/pan, one provisional assessment, application close, and resume. Do not call the GUI portion fully verified until this VNC smoke check succeeds.
