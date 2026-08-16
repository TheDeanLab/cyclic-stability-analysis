# Cyclic stability analysis

This standalone, publication-local analysis evaluates cyclic imaging evidence without
writing to the source ND2/TIFF data or Cy-ExM registration artifacts. The frozen
scientific configuration is `config/analysis.yaml`; the audited Golgi and Lamin
ND2-to-TIFF correspondence is `config/correspondence.csv`. LAMP1 is intentionally
absent until image-based cross-round identities have been audited and recorded.

## Data and quality-control boundary

All image inputs, workflow manifests, execution profiles, and Cy-ExM outputs are
read-only inputs. `RoundSpec.workflow_round` is explicitly recorded because Golgi
acquisition rounds 1 and 2 map to workflow rounds 2 and 1, respectively. No analysis
code derives a workflow round from acquisition numbering.

Registration review is target-blind: target-channel images must not be opened before
the reviewer records accept/exclude decisions for the non-reference rounds in
`config/qc-decisions.csv`. An empty decision file is the required initial state and
pauses target-specific analysis rather than inventing eligibility.

## Commands

Use the PEP 723 test bridge with its dependencies isolated outside this repository:

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_config.py -q
```

The same two exports are required before every later `uv` command. The execution
profiles deliberately keep machine resources separate from frozen scientific inputs;
the local profile is sequential with conservative pre-pilot values and the optional
SLURM profile requests one process per array element.

## Run identity and outputs

Each run identity is derived from the digest of the validated scientific YAML,
correspondence rows, and QC decisions. Runtime products belong only under
`results/RUN_ID/` and will include the frozen input copies, validation record,
target-blind QC evidence, restart-safe cell tasks, compact tables, figures,
`report.md`, and `run.json`. Generated results, smoke crops, and partial writes are
ignored by Git so this repository retains the exact source code and scientific inputs
needed to reproduce an analysis identity.
