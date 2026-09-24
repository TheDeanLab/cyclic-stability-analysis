# Golgi Otsu segmentation and rerun implementation plan

> Execute inline with `superpowers:executing-plans`; use a fresh code reviewer before launch.

**Goal:** Validate and install a simple Golgi segmentation, improve the DAPI exclusion mask, and rerun all five Golgi fields into a new immutable result directory.

**Authorization:** The user requested investigation of Otsu plus small-component removal and then explicitly requested the Golgi workflow rerun once a good solution is available. Continue without additional design approval.

**Architecture:** Add an opt-in plane-median-background-corrected 3D Otsu method with 26-connected components and a fixed physical volume cutoff. Reuse existing task/provenance/QC machinery and measurement paths. Keep legacy settings unchanged. Use the same segmentation method for registered morphology and native intensity measurement, with truthful method/threshold metadata. Save exact retained and rejected masks and same-plane QC images. Use DAPI-only plane-background normalization and the existing Lamin physical watershed for nuclear segmentation, conditional on real-data pilot verification.

**Tech stack:** Existing locked NumPy, SciPy, scikit-image, Matplotlib, pytest environment.

**Design evidence:** `results/golgi-otsu-component-study-20260923/` includes two 675-combination studies (25 volumes, three threshold factors, nine component-volume cutoffs). Plain Otsu fails on Cell4/5 round 1 at the acquisition offset. Subtracting the exterior ROI median per Z plane and clipping at zero corrects both; a 1-ADU histogram avoids a dtype-driven quantization change. Cutoff 0.1 expanded um3 = ceil(0.1/0.003518667) = 29 voxels removes at most 4.7% of Otsu volume. Preserve excluded puncta: size does not establish biological identity. Threshold sensitivity remains a measurement limitation, not proof of biological stability.

## Constraints

- No changes to original result directories; checkpoint before edits is 9355819, pushed and remote verified.
- All thresholds act on 3D voxels; no projected nuclear-column exclusion, morphological closing, or largest-component-only selection.
- Do not tune settings to flatten between-round measurements.
- Nuclear exclusion includes all DAPI foreground, including rejected nuclear components; component QC remains explicit.
- Raw intensity integration uses original intensities minus the recorded plane background, with exposure normalization unchanged.
- Optical/volume units are expanded image space; never silently use nominal native volume.
- Settings are optional with old defaults; no change to Lamin/LAMP1 when the new method is not enabled.
- Work in the existing dedicated analysis checkout, preserving its pre-existing work in the verified checkpoint. No additional worktree is needed for the requested in-place rerun.

## Tasks

- [x] Verify completed 25-volume studies and DAPI foreground pilot; save recommendation and visual comparisons.
- [x] Add failing regressions for offset invariance, 26-connectivity, physical size boundary, nuclear exclusion, empty masks, truthful metadata, and native intensity background correction.
- [x] Implement the small segmentation core, opt-in configuration, shared native/registered use, DAPI normalization, exclusion mask, and explicit diagnostics.
- [x] Run focused tests, formatting/lint/type checks where useful, broader relevant tests, and one fresh code review. Fix findings before launch.
- [x] Run a bounded full registered Cell2 segmentation smoke (including weak round 1), inspect it, and verify all five DAPI pilot cases.
- [x] Commit validated implementation/configuration. Launch the full five-field workflow with two workers/four native threads into a new result root.
- [x] Monitor terminal records to completion, audit required outputs and mask exclusion, regenerate plots, inspect final masks/figures, publish code/results, and distinguish artifact completion from scientific quality.

## Review focus

- Constant/empty or noise-only data must not become silently accepted empty agreement.
- Threshold metadata must never label Otsu as SBR or claim the original nine-mask ensemble.
- ROI membership and nuclear exclusion must survive component filtering and background correction.
- Native and registered segmentation paths must both use the chosen method.
- Rejected/truncated nuclear objects must not become eligible Golgi foreground.

## Execution ledger

- 2026-09-24: resumed after overnight permission prompt; GitHub access verified and checkpoint 9355819 pushed. Both exploratory sweeps finished. Production rerun not yet launched.

- 2026-09-24: all 25 saved volumes passed the production core contrast check (minimum Otsu/noise ratio 8.77). Five-field threshold and DAPI pilots visually inspected. Native masks/backgrounds now saved; normal and quantized noise controls fail admission. Fresh review findings addressed. Core focused tests: 63 passed; initial integration suite: 143 passed. Full suite in progress. New module static check: zero errors (third-party/NumPy typing warnings remain).

- Full suite: 471 passed in 395.60 s. Final focused tests after zero-MAD safeguard and plot-label changes: 63 passed; diagnostic integration: 4 passed. Formatting/lint and whitespace checks passed. Production code/config frozen for Cell2 smoke followed by same-run cohort completion.

- First full Cell2 smoke (v3, commit a78b12d) completed registered segmentation and visual QC (zero nuclear overlap), then failed native round 1 on the deliberate contrast gate: full-field T=5, noise=5.93. A native perinuclear ROI alone still failed. A distant bright outlier also made full-field Gaussian/Otsu unstable. The bounded 18 µm DAPI-only native ROI plus normalized Gaussian sigma 0.1083 µm succeeded (T=39, noise=1.887, 94,729 voxels), preserving the visible Golgi. Select the smallest successful smoothing, apply it consistently to both coordinate spaces, preserve original intensity integration, and retain the gate. Native domains keep every DAPI foreground fragment, without morphology/count QC. Save the actual nucleus separately from the ROI.
- Plan amendment: verify all five native Cell2 rounds and the four other weak native round-1 volumes, plus the 25 registered volumes under the final smoothing before retrying. Preserve v3 as failed; use a new v4 root. The first smoke already verified the unchanged registration and registered-morphology plumbing. Final full cohort still requires terminal completion, visual review, and exact mask audits.

- Final prelaunch gate: all 25 registered volumes and nine native volumes passed with zero nuclear overlap. All five registered review sheets and native field sheets were visually inspected. Focused integration suite: 156 passed; final geometry/diagnostic checks: 21 passed. Fresh reviewer found no actionable issues; formatting/lint passed and new-core static analysis had zero errors. Native pilot includes every Cell2 round and round 1 from the other four fields. Production will rebuild all nuclear masks; four registered pilot fields intentionally reused old ROIs to isolate segmentation behavior.

- Production v4 (8b47049) completed 5/5 fields with zero failed tasks at 2026-09-24 18:39 UTC. Registered nuclear counts: 3, 1, 2, 1, 2; all five nuclear inventories and registered rounds 1/2/5 visually inspected. Final native audit, plotting and review are completing separately. Native DAPI projections in some rounds retain basal foreground; conservative exclusion and its measurement implications are explicitly documented.

- Final acceptance: all 530 output hashes and terminal record/manifest identities verified; 50 masks passed component/ROI/exclusion audits with zero nuclear overlap. All five native rounds-1/2/5 review sheets inspected; final PNG and rasterized PDF inspected. Summary, caption, methods, audit and provenance published in `plots/GOLGI-OTSU-20260924/`; full report and per-field links in `results/20260924-golgi-otsu-v4/README.md`.
