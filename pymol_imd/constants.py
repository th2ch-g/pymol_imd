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
# Implicit-solvent (GB) knob for AA / ML-MM. "auto" = GB on in a non-periodic
# run (resolves to DEFAULT_IMPLICIT_SOLVENT_MODEL) and off under pbc; "none" =
# in-vacuo; the rest pick a specific OpenMM GB model. CG (Martini) has its own
# solvent treatment and does not expose this.
SUPPORTED_IMPLICIT_SOLVENT_MODELS = ("auto", "none", "gbn2", "obc2", "obc1", "gbn", "hct")
SUPPORTED_PLATFORMS = ("auto", "cuda", "opencl", "cpu", "reference")
SUPPORTED_ML_DEVICES = ("auto", "cuda", "mps", "cpu")
# Map the lowercase platform knob to OpenMM's case-sensitive Platform name.
OPENMM_PLATFORM_NAMES = {"cuda": "CUDA", "opencl": "OpenCL", "cpu": "CPU", "reference": "Reference"}

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

# Molar gas constant (kJ/mol/K) for the instantaneous-temperature estimate
# (T = 2*KE / (ndof * R)). Single source shared by the IMDSystem state-CSV
# recorder and the socket server's energies packet.
GAS_CONSTANT_KJ_PER_MOL_K = 0.00831446261815324

# Mass-density unit bridge (FIXED). 1 amu = 1.66053906660e-24 g and
# 1 nm^3 = 1e-21 cm^3, so (amu/nm^3) * 1.6605390666e-3 = g/cm^3. Used for the
# state-CSV density column (total system mass / periodic box volume).
AMU_PER_NM3_TO_G_PER_CM3 = 1.6605390666e-3
MINIMUM_BOX_MARGIN_NM = 0.01  # keep cutoff strictly below half the box length
# Allow solver convergence and single-precision coordinate roundoff in state validation.
CONSTRAINT_VALIDATION_TOLERANCE_FACTOR = 10.0
CONSTRAINT_POSITION_ROUNDOFF_ULPS = 8.0

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
DEFAULT_CONST_K = 10000.0  # pulling-force scale (kJ/mol/nm^2); backends override
DEFAULT_MOUSE_FORCE = True  # whether the user pull-force mechanism is active
DEFAULT_INTEGRATOR = "langevin"  # "langevin" | "langevin_middle"; AA overrides
DEFAULT_MAX_FORCE_KJ = 5000.0  # per-atom hard cap on pull-force magnitude (kJ/mol/nm)
DEFAULT_ENERGY_CUTOFF_KJ = 1e9  # |U| or kinetic energy above this rejects the frame (kJ/mol)
ENERGY_RECOVERY_RETRIES = 2  # Retry without pulling, then with zero velocities.
DEFAULT_FRAME_INTERVAL_MS = 1  # QTimer tick interval for the IMD / minimize loops (ms)
DEFAULT_DRAG_LEVEL = "atom"  # mouse-drag granularity: atom | residue | chain
DEFAULT_PLATFORM = "auto"  # OpenMM platform: auto (fastest) | cuda | opencl | cpu | reference

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
AA_IMPLICIT_SOLVENT = (
    "auto"  # GB model knob: auto -> gbn2 (non-periodic) / none (vacuo) / explicit model
)

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
MLMM_IMPLICIT_SOLVENT = "auto"  # GB model knob (same semantics as AA_IMPLICIT_SOLVENT)

# ---------------------------------------------------------------------------
# Implicit solvent (GB) mapping + resolver
# ---------------------------------------------------------------------------
# OpenMM selects the GB model via an implicit-solvent forcefield XML added to
# the ForceField file list (the `createSystem(implicitSolvent=...)` kwarg is the
# prmtop-only API and does NOT apply to the ForceField path). These five XMLs
# ship inside OpenMM (openmm/app/data/implicit/). GBn2 is the modern Amber
# recommendation, so it is what "auto" resolves to.
DEFAULT_IMPLICIT_SOLVENT_MODEL = "gbn2"  # what "auto" resolves to in a non-periodic run
IMPLICIT_SOLVENT_XML: dict[str, str] = {
    "gbn2": "implicit/gbn2.xml",
    "obc2": "implicit/obc2.xml",
    "obc1": "implicit/obc1.xml",
    "gbn": "implicit/gbn.xml",
    "hct": "implicit/hct.xml",
}


def resolve_platform_name(value: str) -> str | None:
    """Validate the platform knob; return OpenMM's case-sensitive Platform name.

    ``None`` means "let OpenMM pick the fastest available platform" (the
    historical default). ``value`` is case-insensitive. Kept here (not in
    imd.py) so the import-light ``cli.py`` can validate it pre-launch without
    importing OpenMM. Raises ``ValueError`` on an unknown name.
    """
    key = str(value).strip().lower()
    if key == "auto":
        return None
    if key not in OPENMM_PLATFORM_NAMES:
        raise ValueError(f"Unknown platform {value!r}. Choose from {SUPPORTED_PLATFORMS}.")
    return OPENMM_PLATFORM_NAMES[key]


def resolve_implicit_solvent(value: str, *, pbc: bool) -> str | None:
    """Resolve the ``implicit_solvent`` knob to an OpenMM implicit-solvent XML
    name, or ``None`` when no GB force should be added.

    ``value`` is case-insensitive, one of ``SUPPORTED_IMPLICIT_SOLVENT_MODELS``:

    * ``"auto"`` (default): GB on (``DEFAULT_IMPLICIT_SOLVENT_MODEL``) in a
      non-periodic run; off under ``pbc=True`` (GB yields to the explicit PME
      request rather than conflicting with it).
    * ``"none"``: GB off (in-vacuo), regardless of ``pbc``.
    * an explicit model (``"gbn2"`` / ``"obc2"`` / ...): that model, but raises
      ``ValueError`` under ``pbc=True`` since GB needs NoCutoff while PME is
      periodic — the two cannot coexist.

    Returns the XML resource name (e.g. ``"implicit/gbn2.xml"``) or ``None``.
    Kept here (not in ``prep.py``) so the import-light ``cli.py`` can reuse it
    for pre-launch validation without pulling in OpenMM. Raises ``ValueError``
    on an unknown model name or an explicit-GB-with-pbc conflict.
    """
    key = str(value).strip().lower()
    if key not in SUPPORTED_IMPLICIT_SOLVENT_MODELS:
        raise ValueError(
            f"Unknown implicit_solvent {value!r}. Choose from {SUPPORTED_IMPLICIT_SOLVENT_MODELS}."
        )
    if key == "none":
        return None
    if key == "auto":
        if pbc:
            return None  # GB yields to an explicit periodic (PME) request
        key = DEFAULT_IMPLICIT_SOLVENT_MODEL
    elif pbc:
        raise ValueError(
            f"implicit_solvent={value!r} (GB) is incompatible with pbc=True: GB implicit "
            "solvent requires a non-periodic NoCutoff system, while pbc=True uses PME. "
            "Use implicit_solvent='none' under pbc, or drop pbc to keep GB."
        )
    return IMPLICIT_SOLVENT_XML[key]


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
DEFAULT_STATE_CSV_INTERVAL = 1  # write the thermodynamic state CSV every N GUI frames
CSV_FLUSH_ROWS = 100  # stop/error paths always flush any remaining rows

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
# Defensive upper bound for one untrusted packet. The server additionally
# constrains MDComm counts to the simulation atom count before reading a body.
IMD_MAX_PACKET_BODY_BYTES = 256 * 1024 * 1024
IMD_CLIENT_FRAME_INTERVAL_MS = 8  # PyMOL client QTimer tick (ms); lower = smoother streaming
IMD_CLIENT_POLL_TIMEOUT_S = 0.01  # client per-tick wait for the first frame header
IMD_CLIENT_MAX_FRAMES_PER_TICK = 64  # cap frames drained per tick so the Qt loop never wedges
IMD_FORCE_LEASE_S = 0.25  # Release a pull when its client stops refreshing it.

# Experimental external CueMol3 client.
CUEMOL_FRAME_INTERVAL_MS = 33
CUEMOL_CDP_TIMEOUT_S = 5.0
CUEMOL_LAUNCH_TIMEOUT_S = 20.0
CUEMOL_INPUT_LEASE_MS = 2000
CUEMOL_MAX_FRAMES_PER_TICK = 1024
