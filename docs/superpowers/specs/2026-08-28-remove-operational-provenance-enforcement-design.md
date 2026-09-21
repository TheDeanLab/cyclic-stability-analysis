# Remove Operational Provenance Enforcement

## Goal

Allow cyclic-stability analysis and target-blind review to continue when the repository, configuration files, or raw inputs change after a task or review has started.

## Scope

Task lifecycle state may retain input identities and configuration digests as informational metadata, but it must not reject execution, reuse, completion, review reopening, or report finalization because those identities differ later. Direct validation remains: inputs must be readable, outputs must be regular files inside the attempt directory, and image/transform/geometry checks remain the analyzers' responsibility.

## Design

`task.py` will stop comparing captured `ContentIdentity` objects with their current paths after snapshot admission. Retained descriptors are still used to read an admitted producer input safely, but their identity mismatch is not a terminal failure. Existing result records retain their input identity fields.

`review_store.py` will retain the repository identity written at review creation as an informational record but will no longer compare it on open, state validation, or report finalization. Review manifests and evidence-file integrity remain validated.

## Verification

Regression tests will mutate an input during a task and change the synthetic repository identity after review creation. Both workflows must complete. Tests that cover unreadable evidence, malformed state, symlinks, output containment, and direct image validation remain unchanged.
