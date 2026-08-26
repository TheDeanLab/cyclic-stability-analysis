# Multinuclear DAPI Consensus and Lamin Analysis Design

## Purpose

Replace the current largest-component-only DAPI consensus with a target-blind,
component-aware consensus that accepts one through six plausible nuclei. Preserve
individual nuclear identities for Lamin envelope measurements while retaining a
union nuclear mask for Golgi, LAMP1, and raw-signal region construction.

## Selected approach

Build one consensus DAPI intensity volume from the target-blind QC-accepted rounds,
as the existing workflow does. Threshold, fill, and close that consensus once, then
label every 26-connected component. Measure and adjudicate each component in physical
units. Accepted components become immutable individual `NuclearSurface` objects and
also contribute to an immutable union mask and integer label image.

This is preferred over retaining only the largest component because fields are
expected to contain multiple nuclei. It is also preferred over watershed splitting:
the current request is to accept independently segmented nuclei, while automatic
splitting of touching or merged nuclei would introduce a new segmentation model and
additional unvalidated parameters. A merged object is adjudicated by the same
geometry rules and is explicitly visible in diagnostics.

## Scientific configuration

The `nuclear` section of `config/analysis.yaml` will add these required settings:

- `accepted_component_count: [1, 6]`
- `nominal_volume_um3: [100.0, 5000.0]`
- `minimum_sphericity: 0.15`
- `maximum_face_contact_fraction: 0.10`

Nuclear volume bounds are stated in nominal biological-space cubic micrometers.
At runtime, both bounds are multiplied by
`nominal_expansion_factor ** 3` before comparison with voxel-derived expanded-space
volumes. Sphericity is dimensionless and is not expansion-adjusted. Face contact is
the fraction of foreground pixels on each of the six volume faces; any face above
the configured maximum rejects that component.

The strict configuration parser will reject missing or unknown keys, nonfinite
values, reversed bounds, a component-count lower bound below one, an upper bound
below the lower bound, sphericity outside `(0, 1]`, and face contact outside
`[0, 1]`. The analysis config digest will continue to bind the complete YAML, so the
new scientific parameters are automatically included in task identity and outputs.

## Component construction and adjudication

Consensus construction retains the existing target-blind inputs and preprocessing:
normalize each accepted DAPI round, take the voxelwise median, smooth in physical
units, and apply Otsu thresholding. The binary mask is hole-filled and physically
closed without first selecting a largest component. Components are then labeled with
26-connectivity.

For each component, calculate:

- expanded-space volume and nominal biological-space volume;
- mesh surface area and sphericity;
- centroid in expanded physical XYZ coordinates;
- contact fraction on every image face;
- acceptance status and an explicit tuple of rejection reasons.

Components passing every bound are assigned stable identifiers `N001` through
`N006`, ordered lexicographically by physical centroid `(x, y, z)`. This ordering is
deterministic because all rounds share the same consensus label image. Rejected
components retain diagnostic identifiers but never contribute to analysis masks.

Consensus creation fails closed when fewer than one or more than six components pass
all bounds. It also fails if an accepted component cannot form a nondegenerate
surface. It does not silently select a subset when more than six plausible nuclei
are present.

## Data model and compatibility

Introduce a `NuclearConsensus` value containing:

- the accepted `NuclearSurface` objects;
- the Boolean union mask;
- the integer accepted-component label image;
- component QC records for accepted and rejected candidates;
- registered voxel spacing and nominal expansion factor.

Each `NuclearSurface` gains a stable `nucleus_id`. The public consensus builder
returns `NuclearConsensus`, not a synthetic combined mesh. Compatibility properties
provide `mask_zyx` and `spacing_xyz_um` on the consensus so mask-based Golgi, LAMP1,
and signal consumers can use the accepted union without treating disconnected nuclei
as one geometric surface. Code requiring vertices, normals, curvature, or centroids
must iterate over `consensus.nuclei`.

`MetricRow`, `EnsembleRow`, and `LaminProfileRow` gain an explicit `nucleus_id`.
Per-nucleus rows use `N001` through `N006`; field-level union or organelle summaries
use `aggregate`. This avoids embedding an identifier in endpoint names and makes the
CSV schema directly groupable.

## Lamin analysis

Normal profiles, displacement, width, intensity, coverage, continuity, nuclear
geometry, and threshold-ensemble geometry are computed separately for each accepted
nucleus. A nucleus-specific ROI prevents Lamin signal belonging to a neighboring
nucleus from contributing to that nucleus's mask. The reference and moving masks are
compared only inside the same nucleus ROI.

The original threshold mask for a nucleus remains the direct threshold-ensemble mask
intersected with that nucleus's accepted DAPI ROI. Volume, Jaccard, and surface
distance use that unmodified mask.

Curvature calculation will tolerate isolated numerically degenerate vertices without
changing the threshold mask: retain only finite vertices with nonzero interpolated
normal length for curvature percentiles, record the valid-vertex coverage, and report
curvature endpoints as undefined with an explicit geometry status when coverage is
below `0.99`. Empty masks retain the existing explicit empty semantics. One invalid
curvature member must not abort the other ensemble members or the entire field.

Field-level Lamin summaries are robust medians over nucleus-level values and remain
separate rows with `nucleus_id=aggregate`. They do not pool vertices, because a large
nucleus must not dominate a smaller nucleus merely by contributing more mesh points.

## Golgi, LAMP1, and raw-signal behavior

Golgi perinuclear ROI construction uses the union of all accepted nuclear masks and
distances to the union of their surfaces. Golgi measurements remain field-level and
use `nucleus_id=aggregate`; assigning Golgi structures to individual nuclei is out of
scope.

LAMP1 extranuclear ROI construction excludes the union nuclear mask. LAMP1 object
measurements and raw-signal measurements remain field-level with
`nucleus_id=aggregate`. Nearest-nucleus distance may use the union surface, but this
change will not claim nucleus-specific lysosome ownership.

## Diagnostics and provenance

Add a target-blind DAPI diagnostic for every analyzed field containing orthogonal
maximum-intensity projections with:

- accepted components in distinct colorblind-friendly colors and labels;
- rejected components in gray with rejection codes;
- image-boundary contact highlighted;
- a compact table of component identifier, acceptance, expanded and nominal volume,
  sphericity, centroid, and maximum face-contact fraction.

Lamin profile and displacement plots are produced per nucleus and include the blinded
field identifier plus `N001`-style identifier. Matplotlib PDF/PS font types remain
42. Machine-readable component QC is published with the other per-task tables so the
scientific inclusion decision is auditable.

## Failure behavior

Configuration and consensus failures are explicit and fail closed. Rejected DAPI
components are not fatal unless the accepted count falls outside one through six.
Degenerate curvature for one Lamin threshold member produces missing curvature values
and a status, not a task failure. Failures during diagnostic publication retain the
existing atomic directory replacement behavior.

## Testing and acceptance

Synthetic tests will demonstrate:

1. one through six valid nuclei are retained with deterministic identifiers;
2. zero and seven valid nuclei fail closed;
3. debris, implausibly small or large objects, low-sphericity objects, and excessive
   face contact are rejected with exact reasons;
4. nominal volume bounds are converted by exactly `expansion_factor ** 3`;
5. per-nucleus Lamin profiles and ensemble rows never mix ROIs;
6. a fragmented Lamin mask with isolated degenerate vertices retains its volume and
   overlap measurements while curvature reports its status;
7. Golgi, LAMP1, and raw-signal consumers use the accepted union mask;
8. diagnostic labels and component-QC tables contain no target-derived segmentation;
9. existing single-nucleus behavior remains numerically consistent within existing
   tolerances; and
10. configuration parsing, CSV schemas, atomic publication, Ruff, formatting, and the
    full synthetic test suite remain green.

Production validation will first run the DAPI component inventory without reading
target channels and summarize accepted/rejected counts for every field. Any field
outside the one-to-six contract will be inspected before the target-specific analysis
is rerun.

## Scope exclusions

- No watershed or learned splitting of touching nuclei.
- No manual editing of nuclear masks in this change.
- No assignment of Golgi or lysosomes to a particular nucleus.
- No biologically favorable adjustment of thresholds based on Lamin, Golgi, or LAMP1.
- No rerun or modification of Cy-ExM registration artifacts.
