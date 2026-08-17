cd /archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/cyclic-stability-analysis

export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python

reviewer_name="${REVIEWER_NAME:-${USER:-Reviewer}}"
review_args=(
  --qc-root results/qc-only
  --review-dir results/qc-review-actual
  --reviewer "$reviewer_name"
)

if [[ ! -f results/qc-review-actual/randomization-manifest.json ]]; then
  review_args+=(--seed 8142026)
fi

uv run --script scripts/review_registration_qc.py "${review_args[@]}"
