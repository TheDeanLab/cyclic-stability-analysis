# Colorblind-Friendly Review Overlays Design

**Date:** 2026-08-17

## Objective

Render target-blind registration overlays with the reference round in magenta,
the moving round in green, and matched signal in white. Apply the same palette
to the Tk reviewer and the blinded visual pages of the PDF review record.

The immutable RGB TIFF evidence remains byte-for-byte unchanged. Existing
manifest identities, SHA-256 digests, descriptor-relative reads, and target-blind
allowlisting remain the provenance authority.

## Reuse analysis

The GUI already decodes verified overlay bytes in `EvidenceResolver.load_image`,
and the report already decodes the same verified bytes before constructing its
mosaics. Neither path should gain an independent palette implementation. Add one
small internal array conversion in the review presentation layer and compose it
from both existing consumers. No user-callable API, evidence cache, alternate
manifest, or overlay lifecycle is needed.

Rejected alternatives:

- Regenerating the source TIFFs would invalidate evidence identities and durable
  review state.
- Publishing a parallel recolored TIFF cache would add unnecessary derived-file
  identity, invalidation, and cleanup rules.

## Palette contract

The saved overlay encodes reference intensity in red and moving-round intensity
in green. For every input pixel `(R, G, B)`, the displayed pixel is:

```text
(R, G, R)
```

Consequently:

- reference-only signal is magenta;
- moving-only signal is green;
- equally intense overlap is neutral white;
- unequal overlap retains a visible magenta or green residual.

The conversion accepts only a nonempty `H x W x 3` RGB array with unsigned
8-bit samples and returns a new array with the same shape and dtype. The source
array and source evidence bytes are never modified. The source blue component is
not displayed because the producing registration-QC contract reserves it as
zero; red and green remain the authoritative scientific roles.

## Data flow

1. Read the allowlisted TIFF once through the existing no-follow,
   descriptor-relative evidence reader.
2. Verify its persisted identity and decode it as RGB using the existing GUI or
   report boundary.
3. Apply the shared palette conversion in memory.
4. Resize the converted image for Tk thumbnails or assemble converted arrays
   into the PDF mosaic.
5. Retain all existing opaque identifiers, labels, ordering, and final unblinded
   provenance appendices.

The enlarged Tk viewer uses the same converted image path as the thumbnails.
The PDF remains editable with the existing TrueType font settings.

## Failure behavior

Malformed shape, zero-sized images, non-`uint8` samples, or nonfinite/unsupported
decoded data fail closed with a concise review-evidence error before rendering.
Palette conversion must not weaken symlink rejection, byte-identity validation,
target-channel exclusion, or finalization rollback behavior.

## Verification

Focused tests will prove:

- exact `(R, G, B) -> (R, G, R)` mapping on hand-derived pixels;
- preservation of shape and `uint8` dtype without source mutation;
- rejection of malformed inputs;
- GUI thumbnails and enlarged images use the converted palette;
- PDF visual mosaics use the identical conversion;
- evidence identities and source TIFF bytes remain unchanged;
- no target basename or target-channel image enters either rendering path.

The existing reviewer-focused suite, complete repository suite, Ruff checks,
format check, and live VNC launch remain the completion gates.
