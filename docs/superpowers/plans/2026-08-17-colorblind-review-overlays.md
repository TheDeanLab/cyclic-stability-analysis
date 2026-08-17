# Colorblind-Friendly Review Overlays Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render immutable target-blind overlays with the reference round in magenta, the moving round in green, and equal overlap in white in both Tk and PDF output.

**Architecture:** Add one internal NumPy palette conversion that operates only after existing descriptor-relative evidence validation and RGB decoding. Compose that conversion from the existing GUI resolver and PDF visual-page renderer; never rewrite or cache source TIFF evidence.

**Tech Stack:** Python 3.12, NumPy, Pillow, tifffile, Tkinter 8.6, Matplotlib, pytest, Ruff.

## Global Constraints

- Preserve every source overlay byte, manifest identity, SHA-256 digest, and target-blind allowlist.
- The input scientific roles remain red = reference and green = moving round.
- The displayed mapping is exactly `(R, G, B) -> (R, G, R)`.
- Accept only nonempty `H x W x 3` `numpy.uint8` arrays and return a new array with the same shape and dtype.
- Apply the identical conversion to Tk thumbnails, the enlarged Tk viewer, and blinded PDF mosaics.
- Keep `matplotlib.rcParams['pdf.fonttype'] = 42` and `matplotlib.rcParams['ps.fonttype'] = 42`.
- Do not open, derive, display, or name Golgi, Lamin, or LAMP1 target-channel images in blinded output.
- Before every `uv` command export `UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache` and `UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python`.

---

### Task 1: Shared Palette Conversion

**Files:**
- Create: `scripts/cyclic_stability/review_palette.py`
- Create: `tests/test_review_palette.py`

**Interfaces:**
- Consumes: one decoded `numpy.ndarray`.
- Produces: `apply_review_palette(image: NDArray[np.uint8]) -> NDArray[np.uint8]`.

- [ ] **Step 1: Write failing exact-mapping and validation tests**

```python
def test_palette_maps_reference_to_magenta_and_moving_to_green() -> None:
    source = np.array([[[20, 0, 7], [0, 30, 8], [40, 40, 9]]], dtype=np.uint8)
    original = source.copy()

    displayed = apply_review_palette(source)

    assert np.array_equal(
        displayed,
        np.array([[[20, 0, 20], [0, 30, 0], [40, 40, 40]]], dtype=np.uint8),
    )
    assert displayed.dtype == np.uint8
    assert displayed.shape == source.shape
    assert not np.shares_memory(displayed, source)
    assert np.array_equal(source, original)


@pytest.mark.parametrize(
    "invalid",
    (
        np.zeros((0, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint16),
    ),
)
def test_palette_rejects_non_rgb_uint8_images(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="nonempty H x W x 3 uint8"):
        apply_review_palette(invalid)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run --script scripts/run_tests.py tests/test_review_palette.py -q
```

Expected: collection/import failure because `cyclic_stability.review_palette` does not exist.

- [ ] **Step 3: Implement the pure palette conversion**

```python
"""Colorblind-friendly display conversion for immutable review overlays."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_review_palette(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Map reference red to magenta while retaining moving-round green."""
    array = np.asarray(image)
    if (
        array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[0] < 1
        or array.shape[1] < 1
        or array.shape[2] != 3
    ):
        raise ValueError("review overlay must be a nonempty H x W x 3 uint8 image")
    displayed = np.empty_like(array)
    displayed[..., 0] = array[..., 0]
    displayed[..., 1] = array[..., 1]
    displayed[..., 2] = array[..., 0]
    return displayed


__all__ = ["apply_review_palette"]
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Task 1 command again. Expected: all palette tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add scripts/cyclic_stability/review_palette.py tests/test_review_palette.py
git commit -m "feat: add colorblind review palette"
```

---

### Task 2: Tk and PDF Rendering Integration

**Files:**
- Modify: `scripts/cyclic_stability/review_gui.py:18-24,229-244`
- Modify: `scripts/cyclic_stability/review_report.py:18-25,94-130`
- Modify: `tests/test_review_gui.py:195-205`
- Modify: `tests/test_review_report.py:88-118`

**Interfaces:**
- Consumes: `apply_review_palette(image: NDArray[np.uint8]) -> NDArray[np.uint8]` from Task 1 and existing verified overlay bytes.
- Produces: recolored `EvidenceResolver.load_image` / `load_thumbnail` results and recolored arrays passed to `_overlay_mosaic`.

- [ ] **Step 1: Extend the GUI test with hand-checked source/display channel assertions**

Load the source TIFF corresponding to the existing opaque token, then assert:

```python
displayed = np.asarray(resolver.load_image(token))
source = tifffile.imread(Path(overlay.path))
assert np.array_equal(displayed[..., 0], source[..., 0])
assert np.array_equal(displayed[..., 1], source[..., 1])
assert np.array_equal(displayed[..., 2], source[..., 0])
assert Path(overlay.path).read_bytes() == original_bytes
```

Retain the existing thumbnail mode and aspect-ratio assertions.

- [ ] **Step 2: Add a PDF decode-boundary test**

Add a private `_display_overlay(overlay: OverlayEvidence) -> NDArray[np.uint8]` composition in the report contract and write its test first:

```python
overlay = manifest.visual_presentations[0].overlays[0]
original_bytes = Path(overlay.path).read_bytes()
source = tifffile.imread(io.BytesIO(original_bytes))

displayed = review_report._display_overlay(overlay)

assert np.array_equal(displayed[..., 0], source[..., 0])
assert np.array_equal(displayed[..., 1], source[..., 1])
assert np.array_equal(displayed[..., 2], source[..., 0])
assert Path(overlay.path).read_bytes() == original_bytes
```

- [ ] **Step 3: Run GUI/report tests and verify RED**

Run:

```bash
uv run --script scripts/run_tests.py \
  tests/test_review_gui.py \
  tests/test_review_report.py -q
```

Expected: GUI pixels still use the saved red/green image and the report has no `_display_overlay` composition.

- [ ] **Step 4: Integrate the shared conversion into the GUI**

Import NumPy and `apply_review_palette`. After the existing Pillow RGB/nonempty checks in `EvidenceResolver.load_image`, convert without modifying the source object:

```python
displayed = apply_review_palette(np.asarray(image))
return Image.fromarray(displayed, mode="RGB")
```

Because thumbnails and enlarged viewers already consume `load_image`, both GUI paths inherit the same mapping without a second implementation.

- [ ] **Step 5: Integrate the shared conversion into the PDF**

Create the evidence-owning composition:

```python
def _display_overlay(overlay: OverlayEvidence) -> NDArray[np.uint8]:
    decoded = np.asarray(tifffile.imread(io.BytesIO(read_overlay_bytes(overlay))))
    return apply_review_palette(decoded)
```

Replace the inline tifffile decode in `_render_visual_page` with
`images.append(_display_overlay(overlay))`. Preserve the exact read-once,
allowlisted overlay traversal and the existing atomic report lifecycle.

- [ ] **Step 6: Run focused integration tests and verify GREEN**

Run the Task 2 command again. Expected: all GUI and report tests pass.

- [ ] **Step 7: Run format, lint, and complete test gates**

```bash
uvx ruff check .
uvx ruff format --check .
uv run --script scripts/run_tests.py -q
git diff --check
```

Expected: zero Ruff errors, formatting changes, test failures, or whitespace errors.

- [ ] **Step 8: Perform production and live-VNC acceptance**

Use the dynamic-Tk runtime already selected by `launch_review.sh`. Create a disposable review state with seed `8142026`, launch it on the active TurboVNC display, and confirm:

- preflight reports 40 cases, 39 visual cases, 4 repeats, 43 presentations, 1 technical case, and 702 overlays;
- the window title is `Target-blind registration QC review`;
- the rendered reference signal is magenta, moving signal is green, and overlap is white;
- no XCB error occurs;
- only the disposable probe process is terminated after capture.

- [ ] **Step 9: Commit Task 2**

```bash
git add \
  scripts/cyclic_stability/review_gui.py \
  scripts/cyclic_stability/review_report.py \
  tests/test_review_gui.py \
  tests/test_review_report.py
git commit -m "feat: render colorblind review overlays"
```

- [ ] **Step 10: Rebind the empty production review state**

Confirm `results/qc-review-actual/review-state.json` still has zero assessments,
move it to a dated recoverable backup, and recreate `results/qc-review-actual`
with reviewer `${USER:-Reviewer}`, seed `8142026`, `--validate-only`, and the
final clean repository revision. Never replace a state containing an assessment.
