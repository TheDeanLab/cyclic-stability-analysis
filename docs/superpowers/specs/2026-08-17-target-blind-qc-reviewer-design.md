# Target-blind registration QC reviewer design

Date: 2026-08-17
Status: Approved conversational design; pending written-spec review

## Purpose

Provide a standalone Tkinter application for reviewing the existing Golgi and
Lamin registration-QC overlays without exposing either target channel. The
application will randomize cases behind opaque identifiers, preserve provisional
decisions across interrupted GUI sessions, support a registration-remediation
loop, and export only explicitly finalized binary decisions.

The reviewer will be launched from a terminal inside an existing GUI-enabled HPC
session, such as the user's VNC desktop. The application will not configure,
launch, monitor, or otherwise integrate with VNC.

## Scientific boundary

The application may read only:

- `results/qc-only/qc/decision-requests.csv`;
- DAPI and tubulin RGB overlays below
  `results/qc-only/cells/<cohort>/<cell>/qc/overlays/`;
- target-blind registration and deformation summaries during the reconciliation
  phase; and
- repository and file-identity metadata needed to bind decisions to reviewed
  evidence.

It must never discover, open, hash for display, thumbnail, or otherwise consume a
Golgi, Lamin, LAMP1, or other target-channel image. Target-derived masks and
target-derived metrics are also forbidden. The initial visual pass hides automated
registration flags to prevent them from anchoring the reviewer.

The current inventory contains 40 non-reference cell-round cases. Thirty-nine
have reviewable image overlays. Lamin Cell1 acquisition round 1 has no overlays
because its affine transform is improper and contains a reflection; it is handled
as a separate target-blind technical case.

## Deliverable and launch contract

Add one PEP 723 entry point:

```text
scripts/review_registration_qc.py
```

Canonical launch:

```bash
uv run --script scripts/review_registration_qc.py \
  --qc-root results/qc-only \
  --review-dir results/qc-review \
  --reviewer "K Dean"
```

The command requires Python 3.12 and Tkinter 8.6. Python dependencies declared in
the PEP 723 header will include only packages needed to read the RGB TIFF overlays,
render the interface, and generate the archival PDF. A missing graphical display
will produce a concise error explaining that the command must be launched inside a
GUI-enabled session.

The program owns review state and review exports under `--review-dir`. It does not
modify QC inputs, registration outputs, scientific configuration, or
`config/qc-decisions.csv` without an explicit final export action.

## Randomization and blinding

On first launch, the application will:

1. Validate the complete decision-request inventory and expected overlay set.
2. Compute identities for every consumed input.
3. Generate a fixed randomization seed when one is not supplied.
4. Assign opaque review identifiers unrelated to cohort, cell, or round.
5. Randomize the 39 visual cases.
6. Add four concealed repeat cases sampled deterministically from reviewable cases.
7. Save the immutable case mapping before presenting the first image.

The seed and mapping are reused on resume; reopening cannot reshuffle cases.
Concealed repeats receive new opaque identifiers, and the interface gives no hint
that they are repeats. The mapping is retained in the review provenance but is not
shown during visual review.

The technical Lamin Cell1 round-1 case does not masquerade as an image-reviewable
case. It is presented during technical reconciliation with its improper-affine
evidence and proposed exclusion reason.

## Interface

### Main review window

The header shows only:

- opaque review identifier;
- completed and total visual-case counts;
- current save status; and
- navigation controls.

Biological cohort, cell, acquisition round, workflow round, automated validity,
and automated reason are hidden.

Two tabs separate DAPI and tubulin. Each tab contains a 3 by 3 grid:

| | Pre-registration | Affine | Nonlinear |
| --- | --- | --- | --- |
| XY | overlay | overlay | overlay |
| XZ | overlay | overlay | overlay |
| YZ | overlay | overlay | overlay |

Images retain their aspect ratio. Clicking a panel opens a full-resolution viewer
with zoom and pan. The application does not alter the saved source overlays.

### Provisional controls

The visual pass offers four provisional states:

- `Accept`;
- `Needs registration improvement`;
- `Exclude`; and
- `Needs adjudication`.

The reviewer must select at least one standardized reason for every state and may
add free-text notes. Initial reason tokens include:

- `dapi-aligned`;
- `tubulin-aligned`;
- `no-interior-artifact`;
- `persistent-interior-misalignment`;
- `nonlinear-worse-than-affine`;
- `seam-artifact`;
- `tearing-or-duplication`;
- `z-misalignment`;
- `identity-mismatch`;
- `insufficient-common-support`; and
- `invalid-affine-reflection`.

The reason list is fixed by code for this review and uses semicolon-separated tokens
in CSV exports. Free-text notes remain in review provenance and do not substitute
for a reason token.

Keyboard shortcuts are:

- `A`: accept;
- `R`: needs registration improvement;
- `E`: exclude;
- `U`: needs adjudication;
- left and right arrows: previous and next case; and
- number keys: toggle the displayed common reason tokens.

Navigation never discards an unsaved edit. Every decision change is persisted
before another case is displayed.

### Reconciliation and consistency

After the initial visual pass, the application unlocks a target-blind technical
reconciliation screen. It shows:

- automated registration flags;
- whether non-positive Jacobians are confined to the chunk-seam band;
- interior deformation summaries;
- the hard-invalid affine evidence; and
- concealed-repeat agreement or disagreement.

The reconciliation screen does not expose target images or target-derived data.
Repeat disagreements and `Needs adjudication` states must be resolved before final
export. The two assessments of a concealed repeat remain preserved even after an
adjudicated final decision is selected.

## Review state and pretty JSON

Review state is a human-readable JSON document with:

- two-space indentation;
- stable, sorted keys;
- ISO-8601 timestamps;
- explicit schema and format versions;
- reviewer identity;
- repository commit and dirty-state evidence;
- randomization seed and opaque mapping;
- identities of every consumed QC input;
- one structured history per biological case;
- separate entries for concealed repeat assessments;
- provisional and final decisions;
- reason tokens and free-text notes;
- remediation revisions; and
- export history.

Every update is staged in the review directory, flushed, atomically promoted, and
followed by a directory synchronization where supported. The previously valid JSON
is retained as a single readable backup. On reopening, both current and backup
records are parsed and validated; recovery never silently merges conflicting state.

Review state is append-preserving at the assessment-history level. Editing a
decision adds a timestamped assessment or adjudication event rather than erasing the
prior event.

## Registration-remediation loop

`Needs registration improvement` is non-final and blocks decision export. Selecting
it records the reviewed artifact identities, reasons, notes, and timestamp, then adds
the actual cohort, cell, and acquisition round to
`remediation-requests.csv`. This operational queue is intentionally unblinded so the
registration can be located and improved.

The reviewer application never changes or executes registration. Improved
registration and target-blind QC must be published under a new output root so the
original reviewed evidence remains immutable.

To attach revised QC to an existing review:

```bash
uv run --script scripts/review_registration_qc.py \
  --review-dir results/qc-review \
  --reviewer "K Dean" \
  --add-revision-root results/qc-revision-02
```

The application imports only cases currently awaiting registration improvement. A
revision must have different artifact identities, complete target-blind QC evidence,
and the same scientific case identity. It receives a new opaque review identifier
and re-enters the randomized visual queue. The original assessment stays immutable.

A case may pass through multiple remediation revisions. Its final state may be
`accept` or `exclude`; another improvement request remains provisional. Final export
binds the decision to the exact accepted or excluded artifact revision.

## Outputs

The review directory contains:

- `review-state.json`: current pretty, validated state;
- `review-state.previous.json`: last valid pretty state;
- `randomization-manifest.json`: pretty immutable initial mapping and seed;
- `remediation-requests.csv`: unresolved registration-improvement queue;
- `consistency-report.csv`: concealed-repeat assessments and agreement status;
- `qc-decisions.csv`: final configuration-compatible decisions after explicit
  finalization;
- `review-record.pdf`: archival visual and technical review record; and
- `review-summary.md`: concise provenance, counts, exclusions, remediation history,
  and consistency summary.

The PDF uses Matplotlib TrueType font settings:

```python
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
```

The archival PDF contains randomized visual evidence, final decisions, technical
reconciliation, and the consistency summary. A clearly separated provenance
appendix maps opaque identifiers to biological cases. No target-channel image is
included.

## Finalization rules

Finalization is a distinct, explicit action and is rejected when:

- any original case lacks a final `accept` or `exclude` state;
- a case still needs registration improvement or adjudication;
- a concealed repeat disagreement is unresolved;
- reviewer metadata is missing;
- any reviewed input identity has changed;
- a referenced remediation revision is incomplete; or
- the technical hard-invalid case has not been adjudicated.

The exported `qc-decisions.csv` has exactly the existing schema:

```text
cohort,cell_id,acquisition_round,decision,reasons,reviewer,reviewed_at
```

Concealed repeats are not exported as additional biological decisions. Export is
staged and atomically promoted. An existing export is preserved as a backup before
replacement. The application does not automatically overwrite
`config/qc-decisions.csv`; copying or promoting a reviewed export into scientific
configuration remains a separate deliberate step.

## Error handling

The application fails closed before displaying images when the inventory is
incomplete, input identities conflict, an overlay is unreadable or not RGB, the
review state is malformed, or a consumed image basename or declared image role falls
outside the exact DAPI/tubulin overlay allowlist. Cohort names may appear in parent
case directories and provenance metadata; they never authorize reading a target
image.

An unreadable image after the application starts marks that case unavailable and
blocks finalization; it does not convert the case to an exclusion. Unexpected GUI
exceptions leave the last atomic state usable. Error dialogs provide the affected
opaque identifier and a concise recovery action without revealing biological
identity during the blinded phase.

## Testing and verification

State and export logic will be independent of Tk widgets so most behavior can be
tested without a display. Tests will cover:

- complete inventory validation;
- deterministic randomization and fixed resume order;
- four concealed repeats and non-disclosure in the visual interface;
- target-channel and target-derived-input rejection;
- image readability, RGB shape, and expected 3 by 3 panel inventory;
- pretty JSON formatting and schema validation;
- atomic-save interruption and backup recovery;
- assessment-history preservation;
- provisional reason requirements;
- remediation queue generation;
- revision identity and completeness validation;
- repeat reconciliation;
- finalization refusal for every unresolved state;
- exact `qc-decisions.csv` schema and biological-case uniqueness;
- archival PDF generation and Type 42 font configuration; and
- clear failure without a graphical display.

A GUI smoke test will run under an available display or virtual framebuffer when the
environment provides one. Production verification will launch the application in
the user's existing VNC session, inspect representative DAPI and tubulin panels,
exercise autosave/resume, and perform a non-final test export before real review.

## Reuse analysis

- Closest existing components: `scripts/run_registration_qc.py`,
  `cyclic_stability.registration_qc.publish_qc_outputs`,
  `cyclic_stability.config.load_analysis_config`, and the existing
  `decision-requests.csv` and `qc-decisions.csv` schemas.
- Decision: compose the immutable QC outputs and existing strict decision schema;
  add a separate reviewer entry point and private review-state module because the
  current components do not own interactive review, randomization, autosave,
  remediation history, or finalization.
- Rejected alternative: extending `run_registration_qc.py` would mix target-blind
  evidence generation with subjective human review and would duplicate neither
  responsibility cleanly.
- Rejected alternative: a browser application would add a server and browser state
  despite Tkinter 8.6 being available in the selected VNC workflow.
- Documentation updates: README launch and review workflow, full Numpydoc for the
  user-callable review entry point/state owner, and this design plus its later
  implementation plan.

## Success criteria

The feature is complete when a reviewer can launch one PEP 723 Tkinter command in
the existing VNC session, review every eligible randomized DAPI/tubulin case, resume
after interruption without lost history, request and re-review improved
registrations, reconcile concealed repeats and technical evidence, and explicitly
export a validated binary decision set tied to exact artifact revisions. No target
image or target-derived evidence may be accessed at any point.
