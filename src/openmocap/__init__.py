"""Geometry-first motion capture. Licensed model assets are supplied separately."""

import os

# Set before CLI/GUI imports NumPy; activation also covers callers importing NumPy first.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, os.environ.get("OPENMOCAP_NUM_THREADS", "1"))

__version__ = "0.1.0"
SCHEMA_VERSION = 1
