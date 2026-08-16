# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cy-exm[workflow] @ git+https://github.com/TheDeanLab/Cy-ExM.git@c8b6a9f4a2f5f5fab3296fdc81190a7324da2a52",
#   "nd2==0.11.3",
#   "numpy>=1.26,<3",
#   "scipy>=1.12,<2",
#   "scikit-image>=0.22,<1",
#   "tifffile>=2025.1.10",
#   "PyYAML>=6,<7",
#   "pandas>=2.2,<3",
#   "matplotlib>=3.8,<4",
#   "seaborn>=0.13,<1",
#   "Jinja2>=3.1,<4",
#   "psutil>=5.9,<8",
#   "pytest>=8,<9",
# ]
# ///
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
raise SystemExit(pytest.main([str(ROOT / "tests"), *sys.argv[1:]]))
