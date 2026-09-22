# Uniform nuclear component inventories

Presentation-only redraw of the five terminal-selected Lamin attempts in
`results/full-lamin-watershed-20260921/20260921T213601Z`.

Every PDF has a 12 × 8 inch overview with fixed XY/XZ/YZ panel allocations,
physical aspect ratios, short IDs with leader lines, and all accepted/rejected
outlines. The full QC table follows on additional pages with wrapped reasons and
repeated headers. Component count does not shrink the images.

- [Cell1](Cell1/nuclear-component-inventory.pdf)
- [Cell2](Cell2/nuclear-component-inventory.pdf)
- [Cell3](Cell3/nuclear-component-inventory.pdf)
- [Cell4](Cell4/nuclear-component-inventory.pdf)
- [Cell5](Cell5/nuclear-component-inventory.pdf)

Original attempt directories, their PDFs, segmentation masks, measurements, and
completion records are preserved. Each cell has a `provenance.json` recording
its source attempt, input identities, outline reconstruction, rendering source,
and new PDF hash.

DAPI backgrounds use the original per-round normalization and voxelwise median
of the same recorded QC-accepted rounds. The median is evaluated in Z blocks to
bound temporary memory without changing its numerical definition. Cells with
only accepted components reuse their saved label masks. Cell4 needs DAPI-only
adjudication to recover the rejected outlines that were not saved as masks; its
QC rows and accepted masks must match the saved evidence exactly. No Lamin
profiles, biological measurements, registration, or QC decisions are recomputed
or changed.

To replay one field from this checkout with its existing project environment:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
export UV_PROJECT_ENVIRONMENT=/project/bioinformatics/Danuser_lab/Dean/dean/uv/environments/cyclic-stability-analysis
uv run --locked python plots/nuclear-component-inventories-watershed-20260921T213601Z-20260922/regenerate.py Cell4
```

The replay processes fields sequentially. Source DAPI files are checked against
recorded size and modification time; their originally recorded SHA-256 values
are retained in provenance. The source PDF, component QC table, and saved label
archive are freshly hash-verified before rendering.

## Verification

All five PDFs completed and passed the checks in [verification.json](verification.json).
All page sizes and image-panel widths match. The PDFs retain all 24 component
records (15 accepted, 9 rejected), every ID in each projection, and every full
rejection reason. Cell4 has three pages; the other fields have two each.
All five overview pages and both Cell4 table pages were visually inspected.

The nuclear and inventory test modules passed together: **44 passed**. The three
inventory cases also passed after adding explicit checks for label clipping and
duplicate table IDs. Modified Python files pass Ruff lint and formatting checks.

To repeat the PDF export audit after the environment exports above:

```bash
uv run --no-sync --with pymupdf==1.28.2 python plots/nuclear-component-inventories-watershed-20260921T213601Z-20260922/verify_pdfs.py
```
