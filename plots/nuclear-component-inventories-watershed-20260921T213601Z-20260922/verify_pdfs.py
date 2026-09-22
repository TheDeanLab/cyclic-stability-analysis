import ast
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(
    "/archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/cyclic-stability-analysis"
)
OUT = ROOT / "plots/nuclear-component-inventories-watershed-20260921T213601Z-20260922"
checks = []
image_horizontal_bounds = []
cells = sys.argv[1:] or ["Cell1", "Cell2", "Cell3", "Cell4", "Cell5"]
for cell in cells:
    provenance = json.loads((OUT / cell / "provenance.json").read_text())
    pdf = OUT / cell / "nuclear-component-inventory.pdf"
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == provenance["pdf_sha256"]
    attempt = Path(provenance["source_attempt"])
    assert (
        hashlib.sha256((attempt / "record.json").read_bytes()).hexdigest()
        == provenance["source_record_sha256"]
    )
    assert (
        hashlib.sha256(
            (attempt / "diagnostics/nuclear-component-inventory.pdf").read_bytes()
        ).hexdigest()
        == provenance["source_pdf_sha256"]
    )
    with (attempt / "nuclear-component-qc.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    with pymupdf.open(pdf) as doc:
        assert len(doc) >= 2
        assert all(tuple(page.rect) == (0.0, 0.0, 864.0, 576.0) for page in doc)
        overview_ids = re.findall(r"\b[NC]\d{3}\b", doc[0].get_text())
        table_ids = re.findall(
            r"\b[NC]\d{3}\b", "\n".join(page.get_text() for page in list(doc)[1:])
        )
        table_id_words = [
            word[4]
            for page in list(doc)[1:]
            for word in page.get_text("words")
            if re.fullmatch(r"[NC]\d{3}", word[4])
        ]
        assert all(overview_ids.count(row["nucleus_id"]) == 3 for row in rows)
        assert all(table_id_words.count(row["nucleus_id"]) == 1 for row in rows)
        joined = re.sub(
            r"\s+", "", "\n".join(page.get_text() for page in list(doc)[1:])
        )
        for row in rows:
            for reason in ast.literal_eval(row["rejection_reasons"]):
                assert re.sub(r"\s+", "", reason) in joined, (cell, reason)
            for name in ("expanded_volume_um3", "nominal_volume_um3"):
                assert f"{float(row[name]):.3f}" in joined
        images = doc[0].get_image_info()
        assert len(images) == 3
        assert all(image["width"] >= 300 for image in images)
        image_horizontal_bounds.append(
            [(image["bbox"][0], image["bbox"][2]) for image in images]
        )
        for page in list(doc)[1:]:
            assert "rejection code(s)" in page.get_text()
            assert "expanded volume" in page.get_text()
        # Check individual PDF text spans, including title and footer, for clipping.
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        x0, y0, x1, y1 = span["bbox"]
                        assert x0 >= 0 and y0 >= 0 and x1 <= 864 and y1 <= 576, (
                            cell,
                            span,
                        )
        checks.append(
            {
                "cell_id": cell,
                "pdf": str(pdf),
                "pages": len(doc),
                "components": len(rows),
                "accepted": sum(row["accepted"] == "True" for row in rows),
                "all_ids_on_each_projection": True,
                "full_qc_table_retained": True,
                "uniform_page_size_points": [864, 576],
                "image_sizes": [[image["width"], image["height"]] for image in images],
                "source_record_and_pdf_unchanged": True,
                "pdf_sha256": provenance["pdf_sha256"],
            }
        )
first = image_horizontal_bounds[0]
assert all(
    all(
        abs(a - b) < 0.01
        for interval, reference in zip(bounds, first, strict=True)
        for a, b in zip(interval, reference, strict=True)
    )
    for bounds in image_horizontal_bounds
)
result = {
    "status": "verified",
    "source_run": "full-lamin-watershed-20260921/20260921T213601Z",
    "all_five_inventory_pdfs_complete": len(cells) == 5,
    "fixed_projection_widths_verified": True,
    "checks": checks,
}
if len(cells) == 5:
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
