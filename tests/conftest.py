import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYSRC = ROOT / "pysrc"

os.environ.setdefault("MPLBACKEND", "Agg")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PYSRC) not in sys.path:
    sys.path.insert(0, str(PYSRC))

import numpy_compat  # noqa: E402,F401
