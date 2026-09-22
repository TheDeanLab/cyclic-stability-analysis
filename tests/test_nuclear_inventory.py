"""Inventory pages retain readable geometry and complete crowded QC tables."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cyclic_stability import nuclear
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages


def _consensus(count: int) -> nuclear.NuclearConsensus:
    rows = []
    diagnostics = []
    for index in range(count):
        accepted = index < 3
        identifier = f"{'N' if accepted else 'C'}{index + 1:03d}"
        reasons = (
            ()
            if accepted
            else (
                "explicit-qc:unresolved-merged-truncated-nuclei-original-Cell4-N003",
                "volume-below-minimum",
            )
        )
        centroid = (5.0 + index % 10, 12.0, 2.0)
        rows.append(
            nuclear.NuclearComponentQc(
                identifier, accepted, reasons, 1000.0, 100.0, 0.5, centroid, (0.0,) * 6
            )
        )
        projections = []
        for plane, dims in (("XY", (0, 1)), ("XZ", (0, 2)), ("YZ", (1, 2))):
            x, y = (centroid[d] for d in dims)
            outline = np.array(
                [
                    (x - 1, y - 1),
                    (x + 1, y - 1),
                    (x + 1, y + 1),
                    (x - 1, y + 1),
                    (x - 1, y - 1),
                ]
            )
            projections.append(
                nuclear.NuclearComponentProjection(plane, (outline,), 5, 5, False)
            )
        diagnostics.append(
            nuclear.NuclearComponentDiagnostic(
                identifier, accepted, reasons, tuple(projections)
            )
        )
    labels = np.zeros((6, 30, 30), dtype=np.uint16)
    return nuclear.NuclearConsensus(
        (), labels > 0, labels, tuple(rows), tuple(diagnostics), (1.0, 1.0, 1.0), 4.5
    )


@pytest.mark.parametrize("count", [3, 12, 45])
def test_inventory_pages_preserve_every_component_and_fit_text(
    tmp_path, monkeypatch, count
):
    pages = []
    original_save = PdfPages.savefig

    def capture(pdf, figure=None, **kwargs):
        figure.canvas.draw()
        pages.append(figure)
        return original_save(pdf, figure, **kwargs)

    monkeypatch.setattr(PdfPages, "savefig", capture)
    consensus = _consensus(count)
    dapi = np.random.default_rng(1).uniform(1, 20, size=consensus.labels_zyx.shape)
    path = nuclear.write_nuclear_component_diagnostic(
        tmp_path, "Cell4", {2: dapi}, consensus
    )
    assert path.is_file()
    assert len(pages) >= 2
    assert all(tuple(page.get_size_inches()) == (12.0, 8.0) for page in pages)
    overview = pages[0]
    FigureCanvasAgg(overview).draw()
    identifiers = {row.nucleus_id for row in consensus.component_qc}
    for axis in overview.axes:
        assert {text.get_text() for text in axis.texts} == identifiers
        assert axis.get_window_extent().width > 250
        assert axis.get_aspect() == 1.0
        assert axis.get_xlim() == (0.0, 29.0)
        for text in axis.texts:
            box = text.get_window_extent(overview.canvas.get_renderer())
            assert overview.bbox.contains(box.x0, box.y0)
            assert overview.bbox.contains(box.x1, box.y1)
    table_text = []
    for page in pages[1:]:
        FigureCanvasAgg(page).draw()
        renderer = page.canvas.get_renderer()
        for axis in page.axes:
            for table in axis.tables:
                for cell in table.get_celld().values():
                    text = cell.get_text()
                    table_text.append(text.get_text())
                    box = text.get_window_extent(renderer)
                    cell_box = cell.get_window_extent(renderer)
                    assert box.x0 >= cell_box.x0 - 1
                    assert box.x1 <= cell_box.x1 + 1
                    assert box.y0 >= cell_box.y0 - 1
                    assert box.y1 <= cell_box.y1 + 1
                    assert page.bbox.contains(box.x0, box.y0)
                    assert page.bbox.contains(box.x1, box.y1)
    assert {text for text in table_text if text in identifiers} == identifiers
    assert all(table_text.count(identifier) == 1 for identifier in identifiers)
    joined = "".join(table_text).replace("\n", "")
    if count > 3:
        assert (
            "explicit-qc:unresolved-merged-truncated-nuclei-original-Cell4-N003"
            in joined
        )
    assert not plt.get_fignums()
