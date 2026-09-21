# U2OS Golgi Segmentation Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a fast, standalone Cell 1 segmentation comparison and viewer-friendly artifacts without rerunning registration or changing the production analysis.

**Architecture:** A PEP 723 script reads the immutable completed-attempt NPZ, constructs a target-independent pilot domain from the saved perinuclear ROI plus nuclear mask, and compares individually named local-background/hysteresis candidates on each round's XY maximum projection. It exports exact registered Golgi crops, masks, metrics, montage QC, and provenance under a new archive subdirectory; production integration is deferred until visual acceptance.

**Tech Stack:** Python 3.12, NumPy, SciPy, scikit-image, tifffile, matplotlib, pytest, uv.

**Spec:** Accepted inline plan in the 2026-08-30 Golgi segmentation review conversation.

## Global Constraints

- Keep active code and uv caches on `/project`; keep large image outputs on `/archive`.
- Do not modify registration artifacts or the dirty publication analysis checkout.
- Preserve source voxel spacing `(x, y, z) = (0.1083, 0.1083, 0.3000)` micrometers and crop origin.
- Do not use the nuclear mask as a hard Golgi exclusion in the pilot.
- Show one candidate mask at a time; never render a union across parameter settings.
- Use one fixed candidate definition across all acquisition rounds.
- Treat visual review as plausibility QC, not ground-truth accuracy.

---

### Task 1: Tested segmentation kernel

**Files:**
- Create: `scripts/golgi_segmentation_pilot.py`
- Create: `tests/test_golgi_segmentation_pilot.py`

**Interfaces:**
- Produces: `CandidateSpec`, `dog_hysteresis_mask(image, support, pixel_size_xy_um, spec)`, and `mask_statistics(image, mask, support, nuclear_mask)`.

- [x] Write tests proving local-background hysteresis retains an elongated bright object, rejects smooth background, and never leaves supplied support.
- [x] Run the targeted test and observe the expected import failure.
- [x] Implement the minimal kernel and statistics.
- [x] Run the targeted tests to green.

### Task 2: Standalone exports and QC report

**Files:**
- Modify: `scripts/golgi_segmentation_pilot.py`
- Modify: `tests/test_golgi_segmentation_pilot.py`

**Interfaces:**
- Consumes: one immutable `golgi-registered-overlays.npz`.
- Produces: compressed OME-TIFF inputs and masks, `candidate-metrics.csv`, `manifest.json`, and one PNG montage per candidate.

- [x] Write tests for deterministic output names, manifest geometry, one-mask-per-candidate diagnostics, and non-overwriting output behavior.
- [x] Run the targeted test and observe the expected failures.
- [x] Implement CLI export and rendering.
- [x] Run the targeted tests to green.

### Task 3: Cell 1 pilot execution and review

**Files:**
- Generate under: `/archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/u2os_golgi/segmentation-optimization/cell1`

**Interfaces:**
- Consumes: attempt `18c8fa50c0ec44beb3de79052e8e0b12`.
- Produces: an artifact-backed visual comparison of three fixed candidates across rounds 1--5.

- [x] Run the script into a new timestamped output directory.
- [x] Validate TIFF shapes, manifest identities, CSV row count, finite metrics, and expected montage inventory.
- [x] Inspect all candidate montages and report which candidate, if any, is suitable for a 3-D follow-up.

### Task 4: Verification and handoff

**Files:**
- Verify: all changed source, tests, and generated pilot artifacts.

- [x] Run the targeted pilot tests.
- [x] Run Ruff and formatting checks on the new Python files.
- [x] Re-run the broader suite and distinguish the three recorded baseline failures.
- [x] Report results and request visual acceptance before production integration.

## Execution Record

- Pilot output: `20260830-cell1-mip-pilot-v1`.
- Focused verification: 9 passed; Ruff check and format check passed.
- Full suite after implementation: 404 passed, with the same 3 failures present
  in the pre-implementation baseline (395 passed, 3 failed).
- Visual outcome: sensitive is oversegmented, balanced retains substantial
  punctate signal, and conservative is the suitable candidate for a 3-D follow-up
  subject to human visual acceptance.
