"""pymol_imd: Multiscale Interactive Molecular Dynamics in PyMOL.

Importing this package does two things:

1. Side-effect: ``commands.py`` registers ``setup_aa`` / ``setup_cg`` /
   ``setup_mlmm`` / ``imd_*`` with PyMOL's ``cmd.extend`` dispatcher, so
   they can be typed at the PyMOL prompt as bare commands.
2. Name binding: those same callables are re-exported here so a Python
   statement such as ``from pymol_imd import *; setup_aa('x.pdb')`` (which
   is what the CLI sends via ``pymol -d ...``) resolves them in the
   Python namespace too.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .commands import (  # noqa: F401
    imd_change,
    imd_minimize,
    imd_run,
    imd_stop,
    setup_aa,
    setup_cg,
    setup_mlmm,
)

__all__ = [
    "setup_aa",
    "setup_cg",
    "setup_mlmm",
    "imd_minimize",
    "imd_run",
    "imd_stop",
    "imd_change",
]

print(f"pymol_imd {__version__} loaded")
