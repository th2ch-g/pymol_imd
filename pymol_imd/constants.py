"""Single source of truth for tunable defaults and fixed physical constants.

This module is intentionally **CLI-safe**: it imports neither ``pymol`` nor
``openmm`` so that ``cli.py`` (which only assembles a ``pymol -d`` string and
must stay import-light) can read it freely. Values are bare floats / ints /
dicts in their documented unit; call sites attach OpenMM units (e.g.
``constants.AA_NONBONDED_CUTOFF_NM * unit.nanometer``).

Two kinds of values live here:

* **Fixed physical constants** (``EV_PER_KJMOL``, covalent radii, …). These
  encode physics, not preferences — they are centralized to remove
  duplication but are NOT meant to be user-tunable. Changing one will alter
  numerics and break the ML/MM regression smoke test.
* **Tunable defaults** (``DEFAULT_*`` global, ``AA_*`` / ``CG_*`` / ``MLMM_*``
  per-backend). These back the public command / CLI surface; a user override
  flows in via ``imd_setup_*`` kwargs or CLI flags and falls back here.

Changing any ``DEFAULT_*`` / per-backend value changes program behavior and
may require regenerating the smoke-test reference arrays.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Supported value enumerations (validation + CLI choices; single source)
# ---------------------------------------------------------------------------
SUPPORTED_LEVELS = ("aa", "cg", "mlmm")
SUPPORTED_INTEGRATORS = ("langevin", "langevin_middle")
SUPPORTED_ML_BACKENDS = ("uma", "orb", "mace", "aimnet2")
SUPPORTED_LINK_ATOM_METHODS = ("scaled", "fixed", "none")
SUPPORTED_DRAG_LEVELS = ("atom", "residue", "chain")

# ---------------------------------------------------------------------------
# Physical constants (FIXED — never user-tunable)
# ---------------------------------------------------------------------------
# MLIP backends report energy in eV and forces in eV/Å; OpenMM expects kJ/mol
# and kJ/(mol·nm). De-duplicated from imd.py and backends/mlmm.py, which each
# defined this independently. The "* 10.0" (1 Å = 0.1 nm) is kept as an
# explicit factor so the float result is bitwise-identical to the old literal.
EV_PER_KJMOL = 96.48533212331  # 1 eV = 96.485... kJ/mol
EV_PER_A_TO_KJ_PER_MOL_NM = EV_PER_KJMOL * 10.0

# IMDv3 wire-unit bridge (FIXED — IMDv3 mandates these units on the socket).
# IMDv3 lengths are angstrom and forces are kJ/(mol·angstrom); OpenMM uses nm
# and kJ/(mol·nm). Kept as explicit named factors so the server/client
# conversion at the socket boundary has a single source — the protocol module
# itself stays unit-agnostic. Mirrors the EV_PER_A_TO_KJ_PER_MOL_NM pattern.
NM_TO_ANGSTROM = 10.0  # nm -> angstrom (positions / box, server -> wire)
ANGSTROM_TO_NM = 0.1  # angstrom -> nm (positions in / drag delta, client side)
KJ_PER_NM_TO_KJ_PER_ANGSTROM = 0.1  # kJ/mol/nm -> kJ/mol/angstrom (forces out)
KJ_PER_ANGSTROM_TO_KJ_PER_NM = 10.0  # kJ/mol/angstrom -> kJ/mol/nm (MDComm in)

# Covalent radii in Angstrom for the Morokuma/Dapprich link-atom g-factor.
# Source: standard chemistry references (Cordero 2008, condensed). Only the
# elements seen at QM/MM boundaries in biomolecules are listed; unusual
# elements fall back to ``DEFAULT_COVALENT_RADIUS_ANG``.
COVALENT_RADIUS_ANG: dict[str, float] = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}
DEFAULT_COVALENT_RADIUS_ANG = 0.75

# Fixed link-atom bond lengths (Angstrom) for the "fixed" link-atom mode. Only
# the common QM-side anchors are tabulated — anything else raises so a silent
# wrong-distance never sneaks in.
FIXED_LINK_DISTANCE_ANG: dict[str, float] = {
    "C": 1.09,  # C-H
    "N": 1.01,  # N-H
    "O": 0.96,  # O-H
    "S": 1.34,  # S-H
}

# ---------------------------------------------------------------------------
# IMD runtime defaults (mirror IMDConfig field defaults; values unchanged)
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE_K = 300.0  # Langevin thermostat temperature (K)
DEFAULT_PRESSURE_BAR = 1.0  # MonteCarloBarostat pressure (bar; NPT only)
DEFAULT_FRICTION_PER_PS = 1.0  # Langevin friction (1/ps)
DEFAULT_TIMESTEP_PS = 0.002  # generic IMDConfig fallback (ps); backends override
DEFAULT_NSTEPS_PER_FRAME = 1  # MD steps advanced per GUI frame
DEFAULT_CONST_K = 10000.0  # pulling-force scale (kJ/mol/nm); backends override
DEFAULT_MOUSE_FORCE = True  # whether the user pull-force mechanism is active
DEFAULT_INTEGRATOR = "langevin"  # "langevin" | "langevin_middle"; AA overrides
DEFAULT_MAX_FORCE_KJ = 5000.0  # per-atom hard cap on pull-force magnitude (kJ/mol/nm)
DEFAULT_ENERGY_CUTOFF_KJ = 1e9  # |U| above this halts the loop as diverged (kJ/mol)
DEFAULT_FRAME_INTERVAL_MS = 1  # QTimer tick interval for the IMD / minimize loops (ms)
DEFAULT_DRAG_LEVEL = "atom"  # mouse-drag granularity: atom | residue | chain

# ---------------------------------------------------------------------------
# Per-backend defaults
# ---------------------------------------------------------------------------
# AA (Amber all-atom)
AA_RECOMMENDED_CONST_K = 50000.0
AA_INTEGRATOR = "langevin_middle"  # BAOAB; wider stability under pull forces
AA_TIMESTEP_PS = 0.002  # 2 fs (HBonds-constrained all-atom)
AA_NONBONDED_CUTOFF_NM = 1.0
AA_BOX_PADDING_NM = 1.0
AA_GAFF_VERSION = "gaff-2.2.20"
AA_PH = 7.4  # Modeller.addHydrogens protonation pH
AA_PARTIAL_CHARGE_METHOD = "am1bcc"  # OpenFF charge model for GAFF ligands

# CG (Martini)
CG_RECOMMENDED_CONST_K = 10000.0
CG_INTEGRATOR = "langevin"
CG_TIMESTEP_PS = 0.020  # 20 fs Martini 3 standard
CG_NONBONDED_CUTOFF_NM = 1.1  # Martini 3 requires >= 1.1 nm
CG_BOX_PADDING_NM = 1.0
CG_MARTINI_FF = "martini3001"  # martinize2 -ff label (Martini 3.0.0.1)
CG_MARTINIZE_MAXWARN = "99999"  # str: passed straight to the martinize2 CLI

# ML/MM (ONIOM)
MLMM_RECOMMENDED_CONST_K = 50000.0  # same as AA (ML region is atomistic)
MLMM_INTEGRATOR = "langevin"
MLMM_TIMESTEP_PS = 0.0005  # 0.5 fs (ML region carries no H-bond constraints)
MLMM_NONBONDED_CUTOFF_NM = 1.0
MLMM_BOX_PADDING_NM = 1.0  # bounding-box slack when pbc=True (mirrors AA/CG)
MLMM_DEFAULT_BACKEND = "aimnet2"  # smallest/fastest/login-free MLIP
MLMM_DEFAULT_LINK_ATOM_METHOD = "scaled"  # Morokuma/Dapprich g-factor
MLMM_DEFAULT_ML_SELECTION = "all"  # whole system (slow); users normally narrow it
MLMM_DEFAULT_DEVICE = "auto"  # torch device: auto -> cuda -> mps -> cpu
MLMM_DEFAULT_CHARGE = 0  # ML region net formal charge
MLMM_DEFAULT_MULT = 1  # ML region spin multiplicity (2S+1; 1 = singlet)
# Per-MLIP model identifiers (defaults for _build_mlip_calculator).
MLMM_MACE_MODEL = "MACE-OMOL-0"
MLMM_MACE_DTYPE = "float64"  # downgraded to float32 on MPS at runtime
MLMM_ORB_MODEL = "orb_v3_conservative_omol"
MLMM_AIMNET2_MODEL = "aimnet2"
MLMM_UMA_MODEL = "uma-s-1p1"
MLMM_UMA_TASK_NAME = "omol"
MLMM_MACE_COMPILE_MODE = "default"  # torch.compile mode passed to MACECalculator

# ---------------------------------------------------------------------------
# Minimize defaults
# ---------------------------------------------------------------------------
DEFAULT_MIN_MAX_ITER_PER_CHUNK = 50  # OpenMM minimizer iterations per Qt tick
DEFAULT_MIN_MAX_CHUNKS = 200  # hard cap on chunks before forced stop
DEFAULT_MIN_ENERGY_TOL_KJ = 1e-3  # consecutive-chunk |ΔU| convergence tolerance (kJ/mol)

# ---------------------------------------------------------------------------
# Recording defaults
# ---------------------------------------------------------------------------
DEFAULT_TRAJ_INTERVAL = 1  # write every N GUI frames to the trajectory
DEFAULT_FORCE_CSV_INTERVAL = 1  # write pull forces every N GUI frames

# ---------------------------------------------------------------------------
# Fixed numerical guard / viewer cosmetics (centralized; not user-tunable)
# ---------------------------------------------------------------------------
# Drag noise floor: prepared.pdb is written with %8.3f (1e-4 nm precision) and
# PyMOL stores float32, leaving a ~1e-4 nm ghost delta on every atom. 1e-3 nm
# (0.01 Å) sits above that floor while still catching realistic mouse drags.
# Tied to the PDB write format, so this is a fixed implementation detail.
DELTA_NOISE_NM = 1e-3

# CG viewer representation (cosmetic; users can still override interactively).
CG_SPHERE_SCALE = 0.4
CG_STICK_RADIUS = 0.25
VIEW_ZOOM_BUFFER = 2.0

# ---------------------------------------------------------------------------
# IMDv3 socket protocol (optional client/server mode; default path unaffected)
# ---------------------------------------------------------------------------
# These back the `pymol_imd serve` / `imd_connect` / `pymol_imd local` surface.
# Defaulting the host to loopback keeps the server local-only unless the user
# opts into a routable address. Port 8888 is the IMDv3 community default.
IMD_VERSION = 3  # protocol version announced in the handshake
IMD_DEFAULT_HOST = "127.0.0.1"  # loopback; server binds here unless overridden
IMD_DEFAULT_PORT = 8888  # IMDv3 default port
IMD_GO_TIMEOUT_S = 1.0  # engine waits this long for the client Go after handshake
IMD_CONNECT_TIMEOUT_S = 5.0  # client socket connect() timeout (s)
IMD_ACCEPT_TIMEOUT_S = None  # server accept() timeout (None = block forever)
IMD_DEFAULT_RATE = 1  # MD steps advanced per transmitted frame (>= 1)
IMD_SEND_VELOCITIES = False  # include a velocities packet in each frame
IMD_SEND_FORCES = False  # include a forces packet in each frame
IMD_RECV_BUFSIZE = 65536  # transport recv() chunk size (bytes)
IMD_CLIENT_FRAME_INTERVAL_MS = 8  # PyMOL client QTimer tick (ms); lower = smoother streaming
IMD_CLIENT_POLL_TIMEOUT_S = 0.01  # client per-tick wait for the first frame header
IMD_CLIENT_MAX_FRAMES_PER_TICK = 64  # cap frames drained per tick so the Qt loop never wedges
