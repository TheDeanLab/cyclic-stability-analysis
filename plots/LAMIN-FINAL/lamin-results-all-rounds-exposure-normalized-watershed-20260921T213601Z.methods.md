# Image-analysis methods for the five-round lamin figure

This document describes the analysis underlying [the five-round figure](lamin-results-all-rounds-exposure-normalized-watershed-20260921T213601Z.pdf), its [caption](lamin-results-all-rounds-exposure-normalized-watershed-20260921T213601Z.caption.md) and [provenance](lamin-results-all-rounds-exposure-normalized-watershed-20260921T213601Z.provenance.json). It covers image registration through figure generation. Experimental staining, expansion chemistry and independent biological replication are outside the information established by these analysis records.

## Inputs and coordinate conventions

Five U2OS lamin image datasets, named Cell1–Cell5, were analyzed across five acquisition rounds. These identifiers denote image fields/datasets and must not be interpreted as five individual nuclei: several fields contain multiple nuclei. The analysis used three-dimensional DAPI, tubulin and lamin image stacks, with voxel spacing of 0.1083 × 0.1083 × 0.3000 µm in x, y and z. Arrays were stored in z, y, x order. Acquisition identities, channel assignments and exposure settings were taken from the run's saved correspondence table.

Physical distances in the segmentation and ridge measurements refer to the expanded-image coordinate system. The nominal linear expansion factor of 4.5 was used to convert segmented nuclear volumes to nominal pre-expansion volumes for component QC. The plotted ridge FWHM was not divided by 4.5.

Lamin exposure was 150 ms for acquisition round 1 and 200 ms for rounds 2–5; the recorded target laser setting was 40% in every round and field. DAPI and tubulin exposures were 200 ms throughout the lamin correspondence table. Acquisition round 2 was the fixed registration reference and remains the intensity-normalization reference. The figure retains original acquisition numbers 1–5.

## Registration and input validation

The Cy-ExM registration manifests specify tubulin as the registration channel. Moving acquisition rounds 1, 3, 4 and 5 were aligned to round 2 using high-accuracy TRSAA linear registration followed by high-accuracy SyNOnly nonlinear registration with the Mattes metric. The configured strategy was `serial_chunked`, with 512 × 512 × 512-voxel chunks, 15% overlap and cropping disabled. Tubulin-derived transformations were applied to tubulin, DAPI and lamin channels using linear interpolation. Thus lamin was measured in a shared reference geometry without using lamin itself as the registration channel.

The downstream analysis consumed completed registered-channel artifacts, including the reference-round volumes. Its input validation required workflow completion records and the relevant affine, warp, registration-metric and chunk-manifest artifacts for moving rounds. Acquisition correspondence and content identities were retained with the analysis run. The archived run README records verification of all 25 relocated ND2 input identities. The new figure was generated from existing measurements; registration and segmentation were not rerun when round 1 was added.

Registration manifests are retained under `Resubmission/u2os_lamin/20260815-u2os-lamin-cell{1..5}-dataset.yaml`. The measurement run records Cy-ExM commit `f6e0de36cd7f6b5e85f093884c2cf224225f4ec2`.

## Round-level quality control

Acquisition round 2 was included as the reference. Every other round required an explicit saved QC decision; only rounds marked `accept` entered lamin measurement and DAPI consensus construction. The saved decisions exclude round 1 for Cell1 and Cell4. Cell1 carries the reason `invalid-affine-reflection`. Cell4 is marked `exclude` but has the reason string `dapi-aligned`; that string does not explain the exclusion, and no more specific rationale is inferred here. These recorded decisions were preserved in the revised figure.

Round 1 had previously been omitted from plotting because its measurements were labeled `round-1-exposure-sensitivity`, whereas rounds 2–5 were labeled `matched-settings`. The revised plot includes both categories while retaining all existing QC filters. Including the exposure-sensitivity category does not override excluded registrations.

## DAPI consensus and nuclear segmentation

Nuclear boundaries were derived from DAPI independently of lamin intensity. For each QC-accepted registered DAPI volume, background was estimated from strictly positive voxels at or below their 30th percentile. The background location was their median, and the robust scale was 1.4826 times their median absolute deviation. Empty support and nonpositive or nonfinite scales were treated as invalid inputs.

Each DAPI volume was background-subtracted, clipped at zero and divided by the 99th percentile of its positive corrected intensities. The voxelwise median across accepted rounds was smoothed with a three-dimensional Gaussian of physical standard deviation 0.558 µm, using axis-specific voxel sigmas. Otsu thresholding produced foreground, followed by hole filling and binary closing with a physical spherical structuring element of radius `0.61 × 0.405 / 1.4 ≈ 0.1765 µm`. The closed mask was united with the filled mask to retain its foreground.

Touching nuclei were separated by marker-controlled watershed. Initial connected components used 26-connectivity. Within each component, an anisotropy-aware Euclidean distance transform was calculated using the physical voxel spacing. Markers were obtained from h-maxima with 2.0 µm prominence, also using 26-connectivity; components without a surviving marker received one at their maximum distance. Watershed was applied to the negative distance transform within the component using 26-connectivity. The implementation checks that the union of watershed labels exactly preserves the original DAPI foreground.

## Nuclear component QC and explicit exclusion

Each candidate was evaluated using volume, sphericity and image-boundary contact. Expanded volume was voxel count multiplied by voxel volume. Nominal volume was expanded volume divided by `4.5³`. Surface area was calculated from a marching-cubes mesh, and sphericity was calculated as `π^(1/3) (6V)^(2/3) / A`.

The saved configuration specifies:

| Criterion | Setting |
|---|---:|
| Nominal nuclear volume | 100–5,000 µm³, inclusive |
| Minimum sphericity | 0.15 |
| Maximum contact fraction on any image face | 0.10 |
| Accepted nuclear-component count per field | 1–6 |
| Watershed seed prominence | 2.0 µm in expanded coordinates |

For each of the six image faces, the contact fraction is the number of candidate foreground voxels on that face divided by the total number of pixels in the corresponding full-image face. It is not the fraction of the nucleus's surface touching the boundary. Rejected components retain diagnostic records and rejection reasons.

An additional DAPI-reviewed exclusion removed an unresolved merged, truncated object corresponding to original Cell4/N003. The saved rule identifies it by a point at approximately (200.2467, 83.7159, 15.6) µm and an expected voxel count of 7,248,164. The watershed revision split original Cell3/N002 and Cell4/N001 into two nuclei each. Nucleus numbering changed: revised Cell4/N003 is a daughter of original Cell4/N001 and is not the excluded object. The run's parent–daughter correspondence table documents this mapping.

Final accepted counts were 1, 4, 3, 3 and 4 nuclei for Cell1–Cell5, respectively, totaling 15. These are common consensus objects followed across accepted rounds, rather than independently segmented and rematched objects at each round.

## Surface construction and lamin profile sampling

Accepted nuclear masks were converted to triangular surfaces using marching cubes at level 0.5 with physical voxel spacing. Outward normals were derived from the physical gradient of a signed distance field, positive outside and negative inside the nucleus: `EDT(outside) − EDT(inside)`. The same consensus surface and normals were used for all accepted rounds of a field.

At each surface vertex, lamin intensity was sampled along the normal from −1.5 to +1.5 µm in 0.05 µm increments, yielding 61 samples per profile. Sampling used trilinear interpolation; coordinates outside the image returned NaN. Lamin background was estimated separately for each registered round using the same lower-positive-tail median and robust scale described above, over the full volume.

A profile was supported only if all sampled values were finite, its background-corrected peak was internal to the sampling window, its peak exceeded three times the robust background scale, and valid half-maximum crossings were found on both sides. Crossings were interpolated linearly. Unsupported profiles were not assigned zero widths or zero displacements. Profile support was assessed separately in each round, so the set of supported surface locations could change between rounds.

## Four plotted endpoints

1. **Surface coverage:** number of supported surface vertices divided by the total number of surface vertices. This is a vertex fraction, not an area-weighted mesh fraction.
2. **Continuity:** number of vertices in the largest connected supported region divided by the total number of surface vertices. Connectivity follows triangle edges between supported vertices. The denominator includes unsupported vertices; it is not restricted to the supported set.
3. **Ridge FWHM:** distance between interpolated half-peak crossings in each supported, background-corrected profile, summarized by the median across supported vertices within each nucleus and round. Units are expanded-image micrometers. This is apparent fluorescent-band width rather than physical lamina thickness.
4. **Integrated ridge intensity:** trapezoidal integral of the nonnegative, background-corrected profile across the full −1.5 to +1.5 µm sampling window, summarized by the median across supported profiles within each nucleus and round. Integration is over the full window, not only the FWHM interval. Raw units are arbitrary intensity units multiplied by micrometers.

The four plotted endpoints come from per-nucleus summary rows. Stored field-level aggregate rows are excluded, preventing aggregates from being counted as additional nuclei. Coverage and continuity are not multiplied by an exposure correction, and FWHM is not rescaled by exposure.

## Exposure normalization and reference scaling

For integrated ridge intensity only, the plotting script reads target exposure times from the measurement run's saved `config/correspondence.csv`, validates that every plotted field/round has a finite positive exposure, and checks that target laser settings match. It first divides the per-nucleus median intensity integral by exposure time, then divides by the same nucleus's exposure-normalized value in round 2:

```text
relative_intensity(nucleus, round) =
    [median_integral(nucleus, round) / exposure_ms(round)] /
    [median_integral(nucleus, 2) / exposure_ms(2)]
```

The round-2 value is therefore one for each nucleus. Relative to normalization by raw round-2 intensity alone, the exposure correction multiplies round-1 ratios by `200/150 = 4/3` and leaves rounds 2–5 unchanged. Correction assumes linear detector response. It does not equalize signal-to-noise ratio, recover undetected ridges, correct saturation, or establish equal staining efficiency. Consequently, exposure-dependent profile support can still affect all four measurements.

## Sample sizes, plotting and uncertainty

| Acquisition round | Fields contributing | Nuclei per endpoint |
|---|---|---:|
| 1 | Cell2, Cell3, Cell5 | 11 |
| 2 | Cell1–Cell5 | 15 |
| 3 | Cell1–Cell5 | 15 |
| 4 | Cell1–Cell5 | 15 |
| 5 | Cell1–Cell5 | 15 |

Colored points show per-nucleus values, and gray lines connect the same nucleus across available rounds. The blue line is the arithmetic mean of the available nucleus-level values at each round. Thus every nucleus receives equal weight, and fields with more included nuclei contribute more observations. For FWHM and intensity, the blue curve is a mean of per-nucleus medians, not a pooled profile median. The dashed intensity reference line denotes one.

Pointwise 95% confidence intervals were obtained by sampling nuclei with replacement independently within each round, calculating 10,000 resampled means and taking the 2.5th and 97.5th percentiles. The NumPy random generator uses seed 20260829, initialized separately for each endpoint. Bands are not simultaneous confidence bands, paired-change intervals or field-clustered intervals. No hypothesis test or significance annotation is included.

Nuclei within a field are not established independent biological replicates. These intervals describe the implemented nucleus-level resampling procedure and do not account for within-field dependence. Round 1 also uses a different field composition from later rounds: shifts in the mean can reflect both within-nucleus change and which nuclei contribute. The task-outcome panel reports computational completion of five field analyses; it does not certify biological preservation or five independent experiments.

## Additional run outputs

The measurement workflow also saves profile displacement relative to round 2, consensus nuclear geometry and lamin mask sensitivity measurements. The mask ensemble uses physical Gaussian sigmas of 0.279, 0.558 and 1.116 µm and background-scale thresholds of 3, 4 and 5. It records geometry, Jaccard overlap, directed boundary distances and surface Dice at tolerances of 0.5, 1.0 and 1.5 theoretical-resolution units. These auxiliary ensemble measurements are not the source of the four plotted endpoint curves or their bootstrap intervals. Saved component inventories, label volumes and profile diagnostics support QC review.

## Provenance and reproduction

The measurement source is [run 20260921T213601Z](../../results/full-lamin-watershed-20260921/20260921T213601Z/README.md). Its recorded analysis source commit is `8910c3512c1d10d42f473816b3e8ddfe359fcdbb`; its configuration digest is `878a2b49dc2f4c63ffc27e558d04be17b9394ba2f8a807adc8dfa1b85ee5d020`. The saved run configuration, rather than a subsequently edited working configuration, supplies the figure's acquisition settings.

For each task marked complete in `compact-tasks.csv`, the plotting script resolves the successful attempt through `tasks/<key>/terminal.json` and reads that attempt's `summary.csv`. It selects QC-included individual nucleus IDs matching `N[0-9]+`, the four requested endpoints and both comparison categories. The original measurement source predates the five-round plotting revision; the plot's provenance records the new normalization formula, exposures and endpoint-specific sample sizes.

From the analysis repository root, regenerate the figure with the existing project environment:

```bash
uv run --no-sync python scripts/plot_lamin_run.py \
  --run-dir results/full-lamin-watershed-20260921/20260921T213601Z \
  --output-stem plots/LAMIN-FINAL/lamin-results-all-rounds-exposure-normalized-watershed-20260921T213601Z
```

This command renders PDF and PNG and writes JSON provenance; it does not rerun image analysis. The original four-round figure is retained separately. The plotting revision was checked with six targeted tests, formatting and lint checks, direct verification of the 4/3 round-1 correction and unchanged later-round intensity ratios, and visual inspection of the output.

Implementation references: [nuclear analysis](../../scripts/cyclic_stability/nuclear.py), [background estimation](../../scripts/cyclic_stability/images.py), [artifact validation](../../scripts/cyclic_stability/cyexm_artifacts.py), [plot generation](../../scripts/plot_lamin_run.py), [saved analysis configuration](../../results/full-lamin-watershed-20260921/20260921T213601Z/config/analysis.yaml), [saved round QC decisions](../../results/full-lamin-watershed-20260921/20260921T213601Z/config/qc-decisions.csv), and [parent–daughter correspondence](../../results/full-lamin-watershed-20260921/20260921T213601Z/parent-daughter-correspondence.csv).
