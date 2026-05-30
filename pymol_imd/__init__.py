"""pymol_imd: Multiscale Interactive Molecular Dynamics in PyMOL.

Importing this package does two things:

1. Side-effect: ``commands.py`` registers ``imd_setup_aa`` / ``imd_setup_cg`` /
   ``imd_setup_mlmm`` / ``imd_*`` with PyMOL's ``cmd.extend`` dispatcher, so
   they can be typed at the PyMOL prompt as bare commands.
2. Name binding: those same callables are re-exported here so a Python
   statement such as ``from pymol_imd import *; imd_setup_aa('x.pdb')`` (which
   is what the CLI sends via ``pymol -d ...``) resolves them in the
   Python namespace too.
"""

from __future__ import annotations

import os

# Enable PyTorch's MPS->CPU fallback for ops not yet implemented on Apple
# Silicon. MACE in particular hits float64 rebuild paths during model load /
# inference; without this fallback the user crashes the moment they invoke
# imd_setup_mlmm on a Mac. Set BEFORE importing torch (downstream of this).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"

from .commands import (  # noqa: F401
    imd_change,
    imd_minimize,
    imd_record,
    imd_run,
    imd_setup_aa,
    imd_setup_cg,
    imd_setup_mlmm,
    imd_stop,
)

__all__ = [
    "imd_setup_aa",
    "imd_setup_cg",
    "imd_setup_mlmm",
    "imd_record",
    "imd_minimize",
    "imd_run",
    "imd_stop",
    "imd_change",
]

print(f"pymol_imd {__version__} loaded")
