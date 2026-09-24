# Golgi rerun: completed and reviewed

All five fields completed under source commit `8b47049f917a5bd77937780d7e45d51eeb67144a`. The audit verified **530 saved output hashes**, all terminal-selected record/manifest hashes, and **50 masks** (25 registered, 25 native). Every retained component meets the 0.1 expanded µm³ cutoff (29 voxels), every mask stays inside its saved ROI, and **nuclear overlap is zero**. The minimum Otsu-to-processed-noise ratio was 13.58, above the admission limit of 3. The component filter removed 0.053–3.380% of threshold-positive volume across the 50 masks.

| Field | Accepted registered nuclei | Audited masks | Nuclear overlap (voxels) | Visual review |
|---|---:|---:|---:|---|
| Cell1 | 3 | 10 / 10 | 0 | [Golgi](../../results/20260924-golgi-otsu-v4/tasks/golgi-Cell1/attempts/a8abc1ef809c489e98e3f319cbc9652c/diagnostics/golgi-overlay-round-1.png), [nuclei](../../results/20260924-golgi-otsu-v4/review/Cell1-nuclear-review.png), [native](../../results/20260924-golgi-otsu-v4/review/Cell1-native-review.png) |
| Cell2 | 1 | 10 / 10 | 0 | [Golgi](../../results/20260924-golgi-otsu-v4/tasks/golgi-Cell2/attempts/37e96092cc3b4c4faf343d56b1946f37/diagnostics/golgi-overlay-round-1.png), [nuclei](../../results/20260924-golgi-otsu-v4/review/Cell2-nuclear-review.png), [native](../../results/20260924-golgi-otsu-v4/review/Cell2-native-review.png) |
| Cell3 | 2 | 10 / 10 | 0 | [Golgi](../../results/20260924-golgi-otsu-v4/tasks/golgi-Cell3/attempts/efdbaa95469c40fea05cb5119e9d0a24/diagnostics/golgi-overlay-round-1.png), [nuclei](../../results/20260924-golgi-otsu-v4/review/Cell3-nuclear-review.png), [native](../../results/20260924-golgi-otsu-v4/review/Cell3-native-review.png) |
| Cell4 | 1 | 10 / 10 | 0 | [Golgi](../../results/20260924-golgi-otsu-v4/tasks/golgi-Cell4/attempts/de1a7841924d474283328b28912b0c9f/diagnostics/golgi-overlay-round-1.png), [nuclei](../../results/20260924-golgi-otsu-v4/review/Cell4-nuclear-review.png), [native](../../results/20260924-golgi-otsu-v4/review/Cell4-native-review.png) |
| Cell5 | 2 | 10 / 10 | 0 | [Golgi](../../results/20260924-golgi-otsu-v4/tasks/golgi-Cell5/attempts/e40322a5cdc249e282ae8dcde0a0522b/diagnostics/golgi-overlay-round-1.png), [nuclei](../../results/20260924-golgi-otsu-v4/review/Cell5-nuclear-review.png), [native](../../results/20260924-golgi-otsu-v4/review/Cell5-native-review.png) |

The revised masks follow the bright Golgi structures without the widespread cytoplasmic foreground admitted by the old low-tail SBR thresholds. The old overlay displayed a union of nine masks; the new overlays display the actual measured mask, removed components, and same-Z contours. Cell1 and Cell3 now have three and two accepted nuclei, respectively, rather than one merged object each.

The method subtracts a per-plane exterior-ROI median, applies normalized Gaussian smoothing at sigma 0.1083 expanded µm, computes a 3D Otsu threshold, and removes 26-connected components smaller than 0.1 expanded µm³. The complete DAPI foreground is excluded. Nuclear preprocessing uses plane-background normalization and the existing Lamin physical watershed. Original unsmoothed intensities are integrated after subtraction of the recorded plane backgrounds and exposure normalization. Native and registered thresholds are estimated independently within their DAPI-defined domains.

## Review limits

Some native DAPI maximum projections retain broad basal foreground, including Cell1/Cell3 rounds 2/5 and Cell2 round 5. The registered multi-round consensus is cleaner. Native exclusion deliberately retains that foreground conservatively, so zero overlap proves exclusion from the saved DAPI mask, not ground-truth nuclear accuracy; some cytoplasmic signal may be excluded. The native 18 µm domain also differs from the former full-field domain. Old/new absolute integrated signals must not be interpreted as a biological change.

Small rejected puncta may include genuine Golgi fragments. The component cutoff is conservative and does not identify puncta biologically. Threshold sensitivity remains a limitation. These are technical image-stability measurements across five acquisition fields, several containing multiple nuclei; the field bootstrap is not a biological-replicate uncertainty estimate.

## Artifacts and reproduction

- [Summary figure](golgi-stability.png) and [PDF](golgi-stability.pdf).
- [Figure caption](../../results/20260924-golgi-otsu-v4/review/figure-caption.md), [methods](../../results/20260924-golgi-otsu-v4/review/methods.md), [audit](../../results/20260924-golgi-otsu-v4/review/audit.json), [all-mask table](../../results/20260924-golgi-otsu-v4/review/mask-audit.csv), and [visual-review notes](../../results/20260924-golgi-otsu-v4/review/visual-review.json).
- [Launch provenance](../../results/20260924-golgi-otsu-v4/review/launch-provenance.json); frozen inputs are in `config/`. Full task provenance is selected by each `tasks/golgi-CellN/terminal.json`.
- [Parameter study](../../results/golgi-otsu-component-study-20260923/README.md), final 25-volume registered pilot, nine-volume native pilot, and their input identities remain preserved.
- [Launch and verification scripts/logs](../../results/golgi-otsu-launch-20260924/README.md).

The original v2 results and failed v3 smoke were preserved. V3 correctly failed its native round-1 contrast check; v4 adds the validated bounded native ROI and minimal masked smoothing without weakening that gate. Validation included 471 tests before the final native-domain amendment, 156 focused integration tests after it, 21 final diagnostic/geometry tests, formatting/lint, and fresh code review.
