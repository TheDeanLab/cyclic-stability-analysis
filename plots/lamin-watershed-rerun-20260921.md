# Revised Lamin measurement run

All five fields completed and the requested plotting script produced PNG/PDF output.
The automated audit verified 482 recorded content identities and all 240
plotted nucleus/round/endpoint measurements. The final overview PNG was visually inspected after completion; its panels and
labels render correctly.

- Source commit: `8910c3512c1d10d42f473816b3e8ddfe359fcdbb`.
- Run: `/endosome/archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/cyclic-stability-analysis/results/full-lamin-watershed-20260921/20260921T213601Z`.
- Accepted nuclei: Cell1 1; Cell2 4; Cell3 3; Cell4 3; Cell5 4; total 15.
- Physical EDT watershed uses 2 µm seed prominence in expanded-image coordinates,
  with 26-connectivity and the original foreground construction.
- Original Cell3/N002 and Cell4/N001 each become two nuclei. Original Cell4/N003
  remains an unresolved merged, truncated object and is explicitly excluded.
- All other accepted original objects retain their voxel masks. See
  [parent/daughter correspondence](lamin-watershed-parent-daughter-20260921.csv).
- The original plot counted five field aggregates as extra nuclei (19 entries
  versus 14 actual nuclei). The revised plot uses 15 individual nuclei only.
  [Measurement comparison](lamin-watershed-measurement-comparison-20260921.csv) separates that plotting
  correction from the watershed/exclusion changes.
- Means and 95% percentile bootstrap intervals remain descriptive nucleus-level
  summaries, not independent biological replication across fields.
- Ridge FWHM is the median width of the supported Lamin fluorescence profiles at
  half their peak intensity, measured across the nuclear surface along its normals.
  It describes apparent fluorescent-band width, not direct physical lamina thickness.

Outputs: `/endosome/archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/cyclic-stability-analysis/plots/lamin-results-mean-ci-watershed-20260921T213601Z.pdf`, `/endosome/archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/cyclic-stability-analysis/plots/lamin-results-mean-ci-watershed-20260921T213601Z.png`,
`/endosome/archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/cyclic-stability-analysis/plots/lamin-results-mean-ci-watershed-20260921T213601Z.provenance.json`. The original results and figures are preserved.

Validation before launch: real-data Cell3 smoke passed; full suite 445 passed with
2 documented pre-existing failures; final integration subset 21 passed; modified
Python lint/format and relevant configuration type checks passed. All 25 relocated
ND2 inputs match their original SHA-256 content identities.

Original nucleus IDs refer to the previous run. Revised Cell4/N003 is a daughter
of original Cell4/N001; it is not the excluded original Cell4/N003. The saved
parent/daughter correspondence resolves this renumbering explicitly.
