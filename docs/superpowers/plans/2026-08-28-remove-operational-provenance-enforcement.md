# Remove Operational Provenance Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permit analysis and review to proceed across unrelated repository or input changes while retaining direct scientific data validation.

**Architecture:** Persist existing identity metadata for traceability but make it non-blocking. Task output publication continues to validate output location and readability; review state continues to validate its manifest and evidence files.

**Tech Stack:** Python 3.12, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-28-remove-operational-provenance-enforcement-design.md`

## Global Constraints

- Keep required-file, decoding, output-containment, and image-geometry checks.
- Do not alter raw datasets or existing result directories.
- Retain identity fields in JSON records as informational metadata.

---

### Task 1: Remove task-lifecycle identity enforcement

**Files:** `scripts/cyclic_stability/task.py`, `tests/test_task.py`

- [ ] Add a regression where a producer changes an admitted input and its result completes.
- [ ] Remove current-identity and retained-input revalidation at reuse, execution, and completion boundaries; retain admission and output validation.
- [ ] Run the focused task tests.

### Task 2: Remove repository-identity review lockout

**Files:** `scripts/cyclic_stability/review_store.py`, `tests/test_review_exports.py`, `tests/test_review_store.py`

- [ ] Add regressions proving repository-identity drift does not prevent reopening or finalizing a valid review.
- [ ] Remove repository-identity comparisons in `ReviewStore.open` and `ReviewStore.validate_current_bindings`.
- [ ] Run the focused review tests.

### Task 3: Verify the changed operational boundary

**Files:** documentation above and the modified sources/tests.

- [ ] Run `uv run pytest tests/test_task.py tests/test_review_store.py tests/test_review_exports.py -q`.
- [ ] Run Ruff format and lint on modified files.
- [ ] Commit only the source, tests, and documentation for this change.
