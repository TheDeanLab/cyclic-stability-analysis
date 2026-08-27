# Cyclic stability analysis

This standalone, publication-local analysis evaluates cyclic imaging evidence without
writing to the source ND2/TIFF data or Cy-ExM registration artifacts. The frozen
scientific configuration is `config/analysis.yaml`; the audited Golgi and Lamin
ND2-to-TIFF correspondence is `config/correspondence.csv`. LAMP1 is intentionally
absent until image-based cross-round identities have been audited and recorded.

## Data and quality-control boundary

All image inputs, workflow manifests, execution profiles, and Cy-ExM outputs are
read-only inputs. `RoundSpec.workflow_round` is explicitly recorded because Golgi
acquisition rounds 1 and 2 map to workflow rounds 2 and 1, respectively. No analysis
code derives a workflow round from acquisition numbering.

`producer_source_z_start_offset` in the correspondence CSV attests to the number
of leading Z planes removed from the correspondence TIFF when the Cy-ExM producer
source was created. It is a nonnegative integer; the exact `unresolved` sentinel
keeps non-equivalent source copies fail-closed until an operator records the missing
provenance.

Registration review is target-blind: target-channel images must not be opened before
the reviewer records accept/exclude decisions for the non-reference rounds in
`config/qc-decisions.csv`. An empty decision file is the required initial state and
pauses target-specific analysis rather than inventing eligibility.

## Commands

Use the PEP 723 test bridge with its dependencies isolated outside this repository:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_config.py -q
```

The same two exports are required before every later `uv` command. The execution
profiles deliberately keep machine resources separate from frozen scientific inputs;
the local profile is sequential with conservative pre-pilot values and the optional
SLURM profile requests one process per array element.

For a validated SLURM profile, submit one immutable task index per array element;
the Python runner never submits jobs itself.  After determining the task count from
the validated configuration, use the profile's array concurrency and resources, for
example:

```bash
sbatch --array=0-24%3 --cpus-per-task=4 --mem=32G --time=12:00:00 \
  --wrap='uv run --script scripts/run_analysis.py \
    --analysis-config config/analysis.yaml --correspondence config/correspondence.csv \
    --qc-decisions config/qc-decisions.csv --execution-profile config/execution-slurm.yaml \
    --run-dir results/20260816-example --cohort golgi --task-index "$SLURM_ARRAY_TASK_ID"'
```

## Target-blind registration review

The registration review is a separate, target-blind operator workflow over the
read-only `results/qc-only` evidence.  It shows randomized opaque review IDs and
concealed repeat presentations; biological identity, target-channel images, and
target-derived evidence are prohibited before review finalization.  The interface
opens only the allowlisted DAPI and tubulin RGB overlay grid.  Do not add, open, or
use `647`/target images, Golgi/Lamin target images, or any target-derived diagnostic
as review evidence.

Use the headless preflight below before the manual review.  Its directory is
disposable and is deliberately different from the actual review directory:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/review_registration_qc.py \
  --qc-root results/qc-only \
  --review-dir results/qc-review-preflight \
  --reviewer "preflight" \
  --seed 8142026 \
  --validate-only
```

This validates the immutable inventory and durable state without importing the GUI.
It may create `randomization-manifest.json` and `review-state.json` in the
preflight directory; inspect either pretty JSON artifact with `jq '.' PATH` and
discard the directory when preflight is complete.  Do not reuse that state for the
actual review.

Inside the operator's already running GUI-enabled VNC session, start the actual
review in its own empty, non-final directory:

```bash
uv run --script scripts/review_registration_qc.py \
  --qc-root results/qc-only \
  --review-dir results/qc-review-actual \
  --reviewer "REVIEWER NAME" \
  --seed 8142026
```

The script does not configure VNC, a display server, browser, scheduler, or
registration.  It requires the existing GUI-enabled VNC session and Tkinter 8.6;
outside such a session it exits with concise guidance rather than opening a GUI.
After the first launch, resume without `--seed` (and normally without
`--qc-root`), retaining the same review directory and reviewer:

```bash
uv run --script scripts/review_registration_qc.py \
  --review-dir results/qc-review-actual \
  --reviewer "REVIEWER NAME"
```

Each provisional visual assessment needs one status and at least one controlled
reason token.  `Accept` and `Exclude` are provisional visual conclusions;
`Needs registration improvement` creates a non-final remediation request; and
`Needs adjudication` requires an explicit reconciliation decision.  The controlled
reason tokens are `dapi-aligned`, `tubulin-aligned`, `no-interior-artifact`,
`persistent-interior-misalignment`, `nonlinear-worse-than-affine`, `seam-artifact`,
`tearing-or-duplication`, `z-misalignment`, `identity-mismatch`,
`insufficient-common-support`, and `invalid-affine-reflection` (the technical
case's controlled exclusion reason).  The interface autosaves on navigation and
valid close; `review-state.json` (with `review-state.previous.json` recovery
backup) makes the review restart-safe, while `randomization-manifest.json` preserves
the opaque order and concealed repeats.

Automated-invalid cases that have a complete visual overlay grid are still reviewed.
Their automated flags and raw reasons stay hidden during the visual pass and appear
only in target-blind reconciliation, alongside the technical case, deformation
summaries, and repeat agreement.  Free-text reviewer notes remain persisted but do
not appear on blinded pages; they appear only in the final unblinded report
appendix.

After all visual presentations have an assessment, open reconciliation.  Write the
current operational exports: `remediation-requests.csv` identifies unresolved
`Needs registration improvement` cases and `consistency-report.csv` records
concealed-repeat agreement.  Regenerate improved *target-blind* QC evidence outside
this application under a new root (for example, `results/qc-only-revision-1`), then
attach it to the existing review state and continue reviewing the appended revision:

```bash
uv run --script scripts/review_registration_qc.py \
  --review-dir results/qc-review-actual \
  --reviewer "REVIEWER NAME" \
  --add-revision-root results/qc-only-revision-1
```

`--add-revision-root` is resume-only and requires a distinct, newly generated QC
root with changed allowlisted evidence for every unresolved remediation case.  It
never reruns registration or changes the original QC root.  Reconcile all repeat
disagreements, `Needs adjudication` cases, and the hard-invalid technical case with
explicit `Accept` or `Exclude` decisions before finalization.

Finalization atomically writes `qc-decisions.csv`, `review-record.pdf`, and
`review-summary.md` in the actual review directory (and retains prior final-decision
CSV state as `qc-decisions.previous.csv` when applicable).  The final CSV contains
one biological decision per original case, not blinded repeats.  Review its
unblinded provenance appendix and deliberately copy or otherwise promote it into
`config/qc-decisions.csv` only after the review is accepted; the application never
automatically promotes a final biological CSV into configuration.

## Run identity and outputs

Each run identity is derived from the digest of the validated scientific YAML,
correspondence rows, and QC decisions. Runtime products belong only under
`results/RUN_ID/` and will include the frozen input copies, validation record,
target-blind QC evidence, restart-safe cell tasks, compact tables, figures,
`report.md`, and `run.json`. Generated results, smoke crops, and partial writes are
ignored by Git so this repository retains the exact source code and scientific inputs
needed to reproduce an analysis identity.
