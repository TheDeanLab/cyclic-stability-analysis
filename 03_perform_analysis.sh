  cd /archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/cyclic-stability-analysis

  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python

  uv run --script scripts/run_analysis.py \
    --analysis-config config/analysis.yaml \
    --correspondence config/correspondence.csv \
    --qc-decisions config/qc-decisions.csv \
    --execution-profile config/execution-local.yaml \
    --results-root results \
    --progress-interval-s 60
