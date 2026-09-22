# Reproducible Python Project Implementation Plan

> Execute inline in the isolated `chore/python-project` worktree. The user
> approved the project-environment, fixture repair, CI, and documentation scope.

**Goal:** Provide one locked Python 3.12 analysis environment and portable checks.
**Architecture:** Keep the existing scripts/package layout; use a non-packaged uv
project with shared dependencies and pytest/Ruff configuration. Keep the three
standalone plotting scripts independently runnable with their own locks.
**Tech stack:** Python 3.12.11, uv 0.7.13 or newer, pytest, Ruff, GitHub Actions.
**Spec:** The accepted recommendation in this conversation; no scientific changes.

## Constraints and reuse analysis

- Reuse the existing CLI scripts, `scripts/cyclic_stability`, tests and test bridge.
  No new public API, CLI wrapper, source-layout migration, or build backend.
- Preserve Cy-ExM commit `f6e0de36cd7f6b5e85f093884c2cf224225f4ec2`.
- Seed dependency resolution from the Python environment used for the completed
  Lamin run; compare installed versions after syncing a new isolated environment.
- Keep active environments/caches under `/project`; never mutate the production
  cached script environment or large archived analysis artifacts.
- Synthetic tests and command help checks may run in CI. No scheduler contact or
  lab-image access. GUI operation still requires an existing Tk-enabled display.
- Private Cy-ExM access needs a separately supplied read-only Actions credential;
  do not copy the operator's GitHub token into repository secrets.

## Tasks

- [x] Reproduce the two known failures, then repair the CSV fixture by named
  columns and update the injected normal-failure callable to accept keyword args.
- [x] Add `pyproject.toml`, `.python-version`, `uv.lock`, and appropriate ignores.
  Define runtime dependencies, dev tools, pytest import/test paths, Ruff settings.
  Remove duplicated inline dependency metadata from project-only entry points.
  Make `run_tests.py` respect explicit test selections and pytest's testpaths.
- [x] Generate locks for the three PEP 723 plotting scripts from the same runtime
  baseline. Verify shared versions match the project lock.
- [x] Update README and GUI launcher to use the locked project environment,
  document installation/private dependency access and `/project` storage, and
  correct current cohort and execution-profile descriptions.
- [x] Add Linux Python 3.12 CI for locked installation, lint/format, synthetic tests,
  and CLI help checks; wire private dependency access to `CYEXM_READ_TOKEN`.
- [ ] Supply the read-only Actions secret and verify the cloud test job. No
  repository or accessible organization secret is currently configured; a
  clarification request is pending with the user.
- [x] Verify targeted fixture tests, full suite, Ruff, lock consistency, isolated
  environment versions, and command help; review the final diff.

Publication: commit the verified tree, fast-forward the main and existing analysis
branches, push, verify exact remote refs, and inspect the hosted CI status.
The credential-dependent cloud test gate remains explicitly pending above.

## Verification and implementation details

- Reproduced both baseline failures before repairing their fixtures; both pass.
- Migrated all eight project-only script headers to shared project dependencies;
  retained and locked the three standalone plot scripts.
- Seeded the first lock with temporary constraints from the completed Lamin
  environment, removed those bootstrap constraints, and relocked without upgrades.
  A fresh environment has exactly the same 60 installed package versions plus Ruff.
- All three plot locks match their corresponding project versions; their standalone
  help commands pass, as do all ten CLI help commands in the project environment.
- pytest collection exposed the missing repository-root import path; added `.`
  alongside `scripts` and `tests`. The entry-point subset passes (7 tests), and
  the compatibility bridge selects only the requested configuration tests (27).
- Made the existing GUI launcher regression portable with a temporary interpreter
  and uv executable; it now asserts the selected interpreter, isolated project
  environment, and checkout-relative working directory. The launcher retains its
  Tk-enabled lab default and adds optional `CYCLIC_REVIEW_PYTHON` selection.
- Targeted fixture/launcher/entry-point gate: 10 passed. Ruff lint/format and shell
  syntax checks pass. A separate dev-only environment passes the CI lint commands.
- AST comparison confirms no analysis, plotting, or review Python behavior changed;
  only the compatibility test bridge changed execution behavior. Existing import
  ordering and one expression were formatted to meet the pinned Ruff configuration.
- Scope Ruff discovery to Python source so adopting it does not rewrite historical
  Markdown plans. Preserve existing sys.path bootstrap and local mapper idioms.

## Final local acceptance

- Full suite: **448 passed in 384.16 seconds** in the new locked environment.
- Updated the existing ND2-pin regression to inspect both `pyproject.toml` and
  `uv.lock`, replacing its obsolete dependency-header lookup. Its runner subset
  passes (9 tests); the final full-suite result includes this repair.
- Final Ruff checks pass for all 63 Python files; `uv lock --check`, three script
  lock checks, ten project CLI help checks, three standalone plot help checks,
  and `bash -n launch_review.sh` pass.
- Verified installed Cy-ExM's direct-URL metadata names the preserved commit.
- Prepared the documented project environment at
  `/project/bioinformatics/Danuser_lab/Dean/dean/uv/environments/cyclic-stability-analysis`.
- No archived data, completed measurements, registration outputs, or production
  cached script environments were modified.
