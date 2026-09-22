#!/usr/bin/env bash

cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1

export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python

review_python="${CYCLIC_REVIEW_PYTHON:-/project/bioinformatics/Danuser_lab/Dean/dean/conda-envs/cyclic-review-gui-py312/bin/python3.12}"
if [[ ! -x "$review_python" ]]; then
  echo "GUI Python is missing: $review_python" >&2
  exit 2
fi

reviewer_name="${REVIEWER_NAME:-${USER:-Reviewer}}"
review_args=(
  --qc-root results/qc-only
  --review-dir results/qc-review-actual
  --reviewer "$reviewer_name"
)

if [[ ! -f results/qc-review-actual/randomization-manifest.json ]]; then
  review_args+=(--seed 8142026)
fi

review_environment="${CYCLIC_REVIEW_ENVIRONMENT:-/project/bioinformatics/Danuser_lab/Dean/dean/uv/environments/cyclic-stability-review}"
UV_PROJECT_ENVIRONMENT="$review_environment" \
  uv run --locked --python "$review_python" python scripts/review_registration_qc.py "${review_args[@]}"
