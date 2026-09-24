# Golgi segmentation revision, 24 September 2026

The old low-tail SBR threshold admitted extensive cytoplasmic background. The diagnostic projected a union of nine masks, making it even more permissive than any one measured mask. In Cell2 round 2, the middle old mask contained 12,672,070 voxels; the first ordinary Otsu pilot contained 108,731. The problem was excessive foreground, despite the bright visible Golgi.

## Prespecified method

Use `config/analysis-golgi-otsu.yaml`. For each target volume, subtract the median eligible exterior-ROI intensity separately in each Z plane and clip negative values to zero. Compute one Otsu threshold over the corrected ROI histogram, including zeros, using 1 intensity-unit bins. Retain 3D voxels strictly above the threshold, then retain 26-connected components with volume at least **0.1 expanded µm³**. With voxel spacing 0.1083 × 0.1083 × 0.3 µm, this means **29 voxels**. There is no smoothing, closing, filling, or largest-object selection. Otsu follows the [scikit-image implementation](https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu).

A nonconstant volume fails signal admission if its background noise scale is zero/nonfinite or its Otsu threshold is below three robust background noise scales. The scale is 1.4826 times the median absolute deviation from the individual plane medians over eligible voxels. This prevents pure noise from becoming a large connected measured structure; it does not raise or otherwise modify the Otsu threshold. Constant background gives an explicitly empty mask. A histogram range above 1,000,000 intensity units fails rather than allocating an unbounded histogram. These safeguards do not establish specificity for Golgi.

Native signal segmentation uses the full native extranuclear field. Registered morphology uses the existing common registration-valid support within 18 expanded µm of accepted nuclei. Both use the same Otsu and component rule, with independently estimated thresholds/backgrounds in their respective domains. Original, unsmoothed intensities minus the recorded plane background are integrated and divided by target exposure; exposure comparisons remain unchanged.

## DAPI and nuclear exclusion

The nuclear pipeline subtracts each DAPI plane's 10th percentile before positive 99th-percentile scaling. It then uses the existing consensus foreground and Lamin-style physical distance watershed with 2 µm seed prominence and 26-connectivity. Component quality criteria are unchanged. The five-field pilot accepted 3, 1, 2, 1, and 2 nuclei, respectively; Cell1 and Cell3 previously had merged nuclei and excess basal foreground.

Golgi masks exclude the **complete DAPI foreground**, including components rejected from nuclear morphology measurement. This is a voxelwise 3D exclusion. A Golgi maximum projection can overlap a projected nuclear outline because those foreground voxels lie at different Z positions. Same-plane panels and saved overlap counts provide the relevant exclusion check.

## Evidence and parameter choice

`results/golgi-otsu-component-study-20260923/` contains reproducible scripts, input provenance, two 675-condition sweeps, and five-field visual comparisons. Each sweep covers 25 volumes, threshold factors 0.8/1/1.2, and nine component cutoffs. Plain Otsu failed on Cell4/5 round 1 by selecting the acquisition background offset. Plane-median subtraction corrected that failure. The fixed 0.1 µm³ cutoff removed less than 0.5% of foreground volume in most rounds 2–5 and 2.7–4.7% in round 1. Larger cutoffs substantially removed visible signal; the smallest practical cutoff was selected without optimizing any between-round biological trend.

Exploratory sweeps reused the previous saved exterior ROI to isolate the threshold comparison. Production rebuilds the DAPI mask, ROI and registration validity from admitted artifacts. Therefore production values need not equal the study values exactly. The DAPI pilot and full-production smoke are separate validation stages.

## Outputs and interpretation

Each registered diagnostic archive contains one retained mask per round, removed small components, per-plane backgrounds, accepted nuclear labels, and the full nuclear exclusion mask. `golgi-segmentation-qc.json` records thresholds, size counts, background noise, and overlap counts. Each native round has a separate mask archive and JSON recording the exact backgrounds used for integrated signal. Diagnostic panels show raw signal, retained foreground, removed components, and a same-Z overlay rather than a union of thresholds.

Legacy endpoint names ending in `_median` are retained for table compatibility: morphology summaries now summarize **one prespecified mask**. Gradient similarity still summarizes its original three smoothing scales, and surface-agreement endpoints still use their configured tolerances. The one-mask `golgi-threshold-robustness.csv` is an endpoint export, not an uncertainty estimate. Full parameter sensitivity is in the separate study.

Puncta removed by size are not proven background; some may be genuine small Golgi fragments. Mask volume remains threshold-sensitive: changing the corrected Otsu threshold by ±20% changes measured volume appreciably. These outputs quantify technical image stability, not biological replication or demonstrated organelle specificity. Review the preserved removed components, nuclear inventories, and image support alongside summary plots.
