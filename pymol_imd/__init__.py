"""pymol_imd: Multiscale Interactive Molecular Dynamics in PyMOL.

Importing this module from inside PyMOL registers the public commands
(setup_aa, setup_cg, setup_mlmm, imd_run, imd_minimize, imd_stop, imd_change).
"""

from __future__ import annotations

__version__ = "0.1.0"

# Side-effect import: registers commands via pymol.cmd.extend.
from . import commands as _commands  # noqa: F401

print(f"pymol_imd {__version__} loaded")
