# TODO

## Decouple review resumption from harmless repository commits

The target-blind reviewer currently refuses to reopen a review when the
repository identity differs from the identity captured at review creation.
This is unnecessarily disruptive when the QC evidence itself is unchanged:
for example, a documentation edit, formatting-only commit, or a bug fix in a
different analysis path can force an otherwise complete review to be recreated.

### Proposed policy

Separate the immutable bindings into three categories:

- **Evidence identity:** QC tables, overlay bytes, manifests, and attached
  revision roots. Any change here must block resumption.
- **Reviewer/schema identity:** changes that could alter how saved decisions
  are interpreted. Incompatible changes must block resumption.
- **Repository source identity:** record the commit and working-tree digest for
  provenance, but do not automatically invalidate a review when evidence and
  schema remain compatible.

### Safe implementation

Add an explicit acknowledgement path, such as
`--acknowledge-repository-change`, that:

1. Revalidates every persisted evidence hash and manifest binding.
2. Confirms review-state/schema compatibility.
3. Records the old and new repository identities as a durable review event.
4. Allows resumption only when evidence is unchanged.
5. Includes both identities in the final report and export provenance.

Keep the current strict default until this is implemented and tested. The goal
is to avoid needless re-review while preserving a clear audit trail and never
allowing changed scientific evidence to enter an existing review silently.
