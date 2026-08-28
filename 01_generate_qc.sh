  cd /archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/cyclic-stability-analysis

  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python

  uv run --script scripts/run_registration_qc.py \
    --analysis-config config/analysis.yaml \
    --correspondence config/correspondence.csv \
    --run-dir results/qc-only-with-lamp1 \
    --max-workers 2
