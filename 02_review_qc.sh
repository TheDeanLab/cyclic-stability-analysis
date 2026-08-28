  cd /archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/cyclic-stability-analysis

  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python

  uv run --script scripts/review_registration_qc.py \
    --qc-root results/qc-only-with-lamp1 \
    --review-dir results/qc-review-lamp1 \
    --reviewer "$USER" \
    --seed 8142026
