# Cyclic stability analysis

This standalone, publication-local analysis evaluates cyclic imaging evidence without
writing to the source ND2/TIFF data or Cy-ExM registration artifacts. The frozen
scientific configuration is `config/analysis.yaml`; the audited Golgi and Lamin
ND2-to-TIFF correspondence is `config/correspondence.csv`. LAMP1 analysis uses
the image-verified cell identities recorded in that same correspondence file.
The revised Lamin configuration and correspondence are described below.

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

## Setup and development

This repository uses a shared, non-packaged uv project. `pyproject.toml` declares
Python 3.12 and the runtime/development dependencies; `uv.lock` fixes their exact
versions. `.python-version` selects 3.12.11, matching the completed Lamin run.
The initial lock preserves all 60 packages in that run's test/runtime environment,
including the pinned Cy-ExM commit. Dependency upgrades should be deliberate and
validated separately from changes to analysis methods.

Clone the private repository using a GitHub account with access to both this
repository and `TheDeanLab/Cy-ExM`:

```bash
gh auth login
gh auth setup-git
gh repo clone TheDeanLab/cyclic-stability-analysis
cd cyclic-stability-analysis
```

Use uv 0.7.13 or newer (CI uses 0.7.13). On BioHPC, export these paths **before
any uv command** to keep environments and caches on `/project`, even when the
checkout is on `/archive`. Collaborators should select their own writable
`/project` locations rather than another user's environment.

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
export UV_PROJECT_ENVIRONMENT=/project/bioinformatics/Danuser_lab/Dean/dean/uv/environments/cyclic-stability-analysis
```

On a workstation, omit those BioHPC-specific exports to use a local `.venv`.
Install the locked environment and run the portable checks from the repo root:

```bash
uv sync --locked --group dev
uv run --locked pytest tests/test_config.py -q
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Tests use synthetic inputs and temporary directories; they do not launch
registration or require lab datasets. `scripts/run_tests.py` remains a compatible
entry point (`uv run --locked python scripts/run_tests.py tests/test_config.py -q`)
and respects explicit test selections. pytest's import paths are configured in
`pyproject.toml`; no manual `PYTHONPATH` is needed.

## Commands

Use `uv run --locked python scripts/NAME.py ...` for analysis and review commands.
The explicit `python` makes the shared project environment authoritative, including
for plotting scripts that also support standalone execution. For example:

```bash
uv run --locked python scripts/run_analysis.py --help
uv run --locked python scripts/plot_lamin_run.py --help
```

The three `plot_*` scripts also retain PEP 723 metadata and adjacent `.py.lock`
files for independent use: `uv run --locked --script scripts/plot_lamin_run.py ...`.
That mode uses the script lock rather than `uv.lock`. Their initial versions match
the project lock. Keep both execution modes validated when updating dependencies.
Other entry points use only the shared project environment.

The execution profiles keep machine resources separate from scientific inputs.
The local profile uses two process workers with four native threads each; the
optional SLURM profile requests one process per array element. Real analyses also
require the configured archive data, registration outputs, and QC decisions.

## Continuous integration

GitHub Actions runs Ruff independently of private dependencies and runs the full
synthetic suite plus CLI help checks in the locked Python 3.12 environment.
Because Cy-ExM is private, configure an Actions secret named `CYEXM_READ_TOKEN`
with read-only contents access to `TheDeanLab/Cy-ExM`. An organization secret must
be made available to this repository. Enter credentials in GitHub's secret settings,
never in this repository or a command pasted into chat. The test job fails with a
clear setup message if the secret is missing; it does not silently skip tests.

The workflow uses read-only repository permissions, disables persisted checkout
credentials, and supplies the dependency credential only during environment sync.
It does not contact BioHPC, launch jobs, open a GUI, or read archived images.

For a validated SLURM profile, submit one immutable task index per array element;
the Python runner never submits jobs itself.  After determining the task count from
the validated configuration, use the profile's array concurrency and resources.
This example matches the checked-in SLURM profile and five Golgi fields. The
64 GiB setting is a template, not a validated full-field Lamin allocation; the
completed Lamin workflow used the bounded local profile described above:

```bash
sbatch --array=0-4%2 --cpus-per-task=1 --mem=64G --time=12:00:00 \
  --wrap='uv run --locked python scripts/run_analysis.py \
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
uv run --locked python scripts/review_registration_qc.py \
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
uv run --locked python scripts/review_registration_qc.py \
  --qc-root results/qc-only \
  --review-dir results/qc-review-actual \
  --reviewer "REVIEWER NAME" \
  --seed 8142026
```

The script does not configure VNC, a display server, browser, scheduler, or
registration.  It requires the existing GUI-enabled VNC session and Tkinter 8.6;
outside such a session it exits with concise guidance rather than opening a GUI.
`launch_review.sh` uses the existing Tk-enabled conda interpreter with its own
locked environment on `/project`; it does not replace the headless project
environment. Set `CYCLIC_REVIEW_PYTHON` to select another Tk-enabled interpreter and
`CYCLIC_REVIEW_ENVIRONMENT` to override that GUI environment path.
After the first launch, resume without `--seed` (and normally without
`--qc-root`), retaining the same review directory and reviewer:

```bash
uv run --locked python scripts/review_registration_qc.py \
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
uv run --locked python scripts/review_registration_qc.py \
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

## Lamin watershed revision (2026-09-21)

`config/analysis-lamin-watershed.yaml` enables physical-distance watershed with
2 µm peak prominence on the original DAPI foreground. Distances use the expanded
image coordinates and recorded XYZ spacing; seed prominence is not minimum seed
separation. Omit `nuclear.watershed_seed_prominence_um` to retain legacy connected
components. The production Golgi and LAMP1 configurations are unchanged.

The revised Lamin configuration records one `lamin_nuclear_exclusions` entry for
the unresolved original Cell4/N003 object. Each entry identifies a cell and a
physical XYZ point, guards the match with an exact `expected_voxel_count`, and
records a reason. A missing, duplicate, or changed match stops that task. The
excluded object remains in nuclear QC and diagnostic evidence; ordinal IDs are
not used to select exclusions because watershed splitting can renumber nuclei.

Run all five fields with the revised correspondence and original registration QC,
a new result root, and the bounded local execution profile. The revised CSV changes
only Lamin ND2 paths to their current `raw_data/roundN` locations; all 25 files
were verified against their original SHA-256 identities:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --locked python scripts/run_analysis.py \
  --analysis-config config/analysis-lamin-watershed.yaml \
  --correspondence config/correspondence-lamin-watershed.csv \
  --qc-decisions results/full-lamin-20260827/20260829T004550Z-d44d439cb118/config/qc-decisions.csv \
  --execution-profile config/execution-local.yaml \
  --results-root results/full-lamin-watershed-20260921 --cohort lamin
```

Use the completed run directory printed by the runner as `--run-dir` for
`scripts/plot_lamin_run.py`, and choose a new `--output-stem` under `plots/`.
The plot reads terminal-selected attempts and excludes `aggregate` rows from
nucleus trajectories and bootstrap statistics. Earlier figures counted those
five field summaries as additional nuclei (19 entries versus 14 actual nuclei).
The bootstrap remains a descriptive nucleus-level interval, not an estimate of
independent biological replication across fields.

Each completed field now retains `diagnostics/consensus-nuclear-labels.npz` with
`labels_zyx`, `spacing_xyz_um`, and `nucleus_ids`: label value `i + 1` maps to
`nucleus_ids[i]`. This records exact accepted masks and supports parent-to-daughter
correspondence; rejected and explicitly excluded candidates remain in
`nuclear-component-qc.csv` and the component-inventory diagnostics.
