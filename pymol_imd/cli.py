"""Command-line entry point for pymol_imd.

Wraps ``pymol`` so a single shell command sets up a simulation::

    pymol_imd local input.pdb --level cg
    pymol_imd launch input.pdb --level aa --ligand-sdf lig.sdf --minimize --run
    pymol_imd local input.pdb --level mlmm --ml-selection "resi 30-50"

The CLI itself does not import ``pymol``; it builds a Python statement and
hands it to PyMOL via the ``-d`` flag, so PyMOL evaluates exactly the same
``setup_*`` / ``imd_*`` commands an interactive user would type.
"""

from __future__ import annotations

import argparse
import math
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from . import constants

# Re-exported from constants.py (single source shared with the backends) so
# the CLI --choices stay in lockstep with backend validation.
ML_BACKENDS = constants.SUPPORTED_ML_BACKENDS
LINK_ATOM_METHODS = constants.SUPPORTED_LINK_ATOM_METHODS
LEVELS = constants.SUPPORTED_LEVELS
INTEGRATORS = constants.SUPPORTED_INTEGRATORS
DRAG_LEVELS = constants.SUPPORTED_DRAG_LEVELS
IMPLICIT_SOLVENT_MODELS = constants.SUPPORTED_IMPLICIT_SOLVENT_MODELS


def _build_parser(mode: str = "launch") -> argparse.ArgumentParser:
    if mode not in {"launch", "local", "serve"}:
        raise ValueError(f"Unsupported parser mode {mode!r}")
    frame_label = "server frame" if mode == "serve" else "GUI frame"
    record_start = "after each client's GO" if mode == "serve" else "on imd_run"
    p = argparse.ArgumentParser(
        prog=f"pymol_imd {mode}",
        description=(
            "Launch PyMOL with a pymol_imd backend already configured. "
            "Equivalent to typing `from pymol_imd import *; imd_setup_<level>(<pdb>, ...)` "
            "inside PyMOL."
        ),
    )
    # Keep helper functions mode-agnostic. Arguments omitted from a mode still
    # exist on the Namespace with a neutral value.
    p.set_defaults(
        nsteps_per_frame=None,
        const_k=None,
        frame_interval_ms=None,
        no_mouse_force=False,
        drag_level=None,
        run=False,
        pymol=None,
        print_only=False,
        host=constants.IMD_DEFAULT_HOST,
        port=None,
        rate=None,
        send_velocities=False,
        send_forces=False,
    )
    p.add_argument("pdb", help="Input PDB file (path is resolved to absolute).")
    p.add_argument(
        "--level",
        choices=LEVELS,
        default="aa",
        help="Simulation backend / resolution. Default: aa.",
    )

    # ---- common IMD options
    common = p.add_argument_group("common IMD options")
    common.add_argument(
        "--pbc",
        action="store_true",
        help="Enable periodic box. Default: GB implicit solvent (AA/MLMM) / auto-box (CG).",
    )
    common.add_argument(
        "--temperature", type=float, metavar="K", help="Langevin thermostat temperature (K)."
    )
    if mode != "serve":
        common.add_argument(
            "--nsteps-per-frame", type=int, metavar="N", help="MD steps executed per GUI frame."
        )
        common.add_argument(
            "--const-k", type=float, metavar="K", help="Pull-force scale (kJ/mol/nm^2)."
        )
    common.add_argument(
        "--max-force-kj",
        type=float,
        metavar="F",
        help="Optional per-atom pull cap (kJ/mol/nm), preserving direction. Default: no cap.",
    )
    common.add_argument(
        "--integrator",
        choices=INTEGRATORS,
        metavar="NAME",
        help=(
            f"Langevin integrator: {' / '.join(INTEGRATORS)}. Default: per-backend "
            "(AA langevin_middle; CG/MLMM langevin)."
        ),
    )
    common.add_argument(
        "--platform",
        choices=constants.SUPPORTED_PLATFORMS,
        metavar="P",
        help=(
            f"OpenMM compute platform for the MM engine (all levels): "
            f"{' / '.join(constants.SUPPORTED_PLATFORMS)}. Default auto (fastest "
            "available; CUDA > OpenCL > CPU). The ML/MM MLIP device is separate "
            "(--ml-device)."
        ),
    )
    common.add_argument(
        "--timestep-ps",
        type=float,
        metavar="PS",
        help="MD timestep (ps). Default: per-backend (AA 0.002 / CG 0.020 / MLMM 0.0005).",
    )
    common.add_argument(
        "--friction",
        type=float,
        metavar="PER_PS",
        help=f"Langevin friction (1/ps). Default {constants.DEFAULT_FRICTION_PER_PS:g}.",
    )
    common.add_argument(
        "--pressure",
        type=float,
        metavar="BAR",
        help=f"Barostat pressure (bar, NPT/--pbc only). Default {constants.DEFAULT_PRESSURE_BAR:g}.",
    )
    common.add_argument(
        "--energy-cutoff-kj",
        type=float,
        metavar="E",
        help="Optional |potential energy| / kinetic energy cutoff (kJ/mol). "
        "Exceeding it stops MD without recovery. Default: no cutoff.",
    )
    if mode != "serve":
        common.add_argument(
            "--frame-interval-ms",
            type=int,
            metavar="MS",
            help=(
                "QTimer tick interval for the IMD / minimize loops (ms). "
                f"Default {constants.DEFAULT_FRAME_INTERVAL_MS}."
            ),
        )
    common.add_argument(
        "--nonbonded-cutoff-nm",
        type=float,
        metavar="NM",
        help=(
            "Nonbonded cutoff (nm). Default: per-backend (AA/MLMM 1.0; CG 1.1, the "
            "Martini minimum)."
        ),
    )
    common.add_argument(
        "--implicit-solvent",
        choices=IMPLICIT_SOLVENT_MODELS,
        default=None,
        metavar="MODEL",
        help=(
            f"Implicit-solvent (GB) model for AA / ML-MM. One of: {', '.join(IMPLICIT_SOLVENT_MODELS)}. "
            "Default 'auto' = GBn2 in a non-periodic run, off under --pbc; 'none' = in-vacuo. "
            "Not valid for --level cg (Martini has its own solvent treatment)."
        ),
    )
    common.add_argument(
        "--box-padding-nm",
        type=float,
        metavar="NM",
        help=(
            "Bounding-box padding (nm) when a periodic box is built (AA/MLMM --pbc; CG always). "
            f"Default {constants.AA_BOX_PADDING_NM:g}."
        ),
    )
    if mode != "serve":
        common.add_argument(
            "--no-mouse-force", action="store_true", help="Disable the user-pull mechanism."
        )
        common.add_argument(
            "--drag-level",
            choices=DRAG_LEVELS,
            default=None,
            metavar="LEVEL",
            help=(
                f"Mouse-drag granularity for 3-Button Editing: {' / '.join(DRAG_LEVELS)}. "
                "Default: atom. residue / chain grab and pull a whole residue / chain at once."
            ),
        )
    common.add_argument(
        "--minimize", action="store_true", help="Call imd_minimize() right after setup."
    )
    common.add_argument(
        "--min-max-iter",
        type=int,
        metavar="N",
        help=(
            "Minimizer iterations per chunk (with --minimize). "
            f"Default {constants.DEFAULT_MIN_MAX_ITER_PER_CHUNK}."
        ),
    )
    common.add_argument(
        "--min-max-chunks",
        type=int,
        metavar="N",
        help=f"Max minimize chunks (with --minimize). Default {constants.DEFAULT_MIN_MAX_CHUNKS}.",
    )
    common.add_argument(
        "--min-tol-kj",
        type=float,
        metavar="E",
        help=(
            "Minimize convergence tolerance (kJ/mol, with --minimize). "
            f"Default {constants.DEFAULT_MIN_ENERGY_TOL_KJ:g}."
        ),
    )
    if mode == "launch":
        common.add_argument(
            "--run", action="store_true", help="Call imd_run() after setup (and minimize, if set)."
        )
    common.add_argument(
        "--traj-out",
        metavar="PATH",
        help=(
            "Record a trajectory. Format by extension: "
            ".xtc / .dcd / .pdb. Default: ./<pdb-stem>.xtc in the command cwd "
            "(recording is on by default; disable with --no-record). "
            f"Recording starts {record_start}."
        ),
    )
    common.add_argument(
        "--traj-interval",
        type=int,
        metavar="N",
        help=f"Write every N {frame_label}s to the trajectory (default 1).",
    )
    common.add_argument(
        "--force-csv",
        metavar="PATH",
        help=(
            "Record user pull forces to CSV (columns: "
            "frame,step,time_ps,atom_index,fx,fy,fz,fmag). atom_index is "
            "0-based (add 1 for a PyMOL selection). Default: "
            "./<pdb-stem>_forces.csv in the command cwd (on by default; "
            f"disable with --no-record). Recording starts {record_start}."
        ),
    )
    common.add_argument(
        "--force-csv-interval",
        type=int,
        metavar="N",
        help=f"Write forces every N {frame_label}s (default 1).",
    )
    common.add_argument(
        "--state-csv",
        metavar="PATH",
        help=(
            "Record thermodynamic state to CSV (columns: "
            "frame,step,time_ps,potential_energy_kj,kinetic_energy_kj,"
            "total_energy_kj,temperature_k; under --pbc also volume_nm3,"
            "density_g_cm3). Pressure is omitted (no instantaneous virial "
            "under IMD pulling). Default: ./<pdb-stem>_state.csv in the command "
            f"cwd (on by default; disable with --no-record). Starts {record_start}."
        ),
    )
    common.add_argument(
        "--state-csv-interval",
        type=int,
        metavar="N",
        help=f"Write the state CSV every N {frame_label}s (default 1).",
    )
    common.add_argument(
        "--no-record",
        action="store_true",
        help=(
            "Disable default recording. Without this a trajectory + force CSV "
            "+ state CSV are written by default. Conflicts with an explicit "
            "--traj-out / --force-csv / --state-csv."
        ),
    )
    common.add_argument(
        "--topology-dir",
        metavar="DIR",
        help=(
            "Directory for the IMD intermediate / topology files (AA/MLMM: "
            "<stem>_prepared.pdb; CG also cg.pdb / cg.top / *.itp). "
            "Default: ./<pdb-stem>_topology in the launch cwd."
        ),
    )

    # ---- AA-specific
    aa = p.add_argument_group("--level aa options")
    aa.add_argument(
        "--ligand-sdf", metavar="PATH", help="Ligand SDF for GAFF + AM1-BCC parameterization."
    )
    aa.add_argument(
        "--ph",
        type=float,
        metavar="PH",
        help=f"Modeller.addHydrogens protonation pH. Default {constants.AA_PH:g}.",
    )
    aa.add_argument(
        "--gaff-version",
        metavar="VER",
        help=f"GAFF version for ligand templates. Default {constants.AA_GAFF_VERSION}.",
    )

    # ---- CG-specific
    cg = p.add_argument_group("--level cg options")
    cg.add_argument(
        "--ff",
        metavar="NAME",
        help=f"Martini forcefield label (default: {constants.CG_MARTINI_FF}).",
    )
    cg.add_argument(
        "--no-elastic-network",
        action="store_true",
        help="Disable martinize2 elastic network (-elastic).",
    )
    cg.add_argument(
        "--martinize-maxwarn",
        dest="maxwarn",
        metavar="N",
        help=f"martinize2 -maxwarn value. Default {constants.CG_MARTINIZE_MAXWARN}.",
    )

    # ---- MLMM-specific
    mlmm = p.add_argument_group("--level mlmm options")
    mlmm.add_argument(
        "--ml-selection",
        metavar="SEL",
        help=(
            "ML region selector (REQUIRED for --level mlmm). Uses PyMOL's "
            "selection language. Examples: 'all', 'resi 30-50', "
            "'resi 1 and resn MET', 'chain A and not resn HOH', "
            "'byres (resn LIG) within 5 of chain A'."
        ),
    )
    mlmm.add_argument(
        "--ml-backend",
        choices=ML_BACKENDS,
        metavar="NAME",
        default=None,
        help=(
            f"MLIP backend. One of: {', '.join(ML_BACKENDS)}. "
            "If omitted, imd_setup_mlmm picks 'aimnet2' — smallest model "
            "(~10 MB), fastest CPU inference, no login required."
        ),
    )
    mlmm.add_argument(
        "--ml-device",
        choices=constants.SUPPORTED_ML_DEVICES,
        metavar="DEV",
        help="Torch device: auto / cuda / mps / cpu.",
    )
    mlmm.add_argument("--ml-charge", type=int, metavar="Z", help="ML region net charge.")
    mlmm.add_argument(
        "--ml-mult",
        type=int,
        metavar="M",
        help=(
            "ML region spin multiplicity (2S+1). Default: 1 (singlet). "
            "Passed directly to every MLIP backend as a multiplicity."
        ),
    )
    mlmm.add_argument(
        "--ml-compile",
        action="store_true",
        help=(
            "Opt-in torch.compile for the MLIP (AIMNet2). Numerically "
            "~identical, biggest speedup on CUDA, smaller on CPU. Default off."
        ),
    )
    mlmm.add_argument(
        "--enable-cueq",
        action="store_true",
        help=(
            "Opt-in cuEquivariance acceleration for MACE (CUDA only; needs "
            "`pixi run install-cueq`). Safe no-op elsewhere. Don't combine "
            "with --ml-compile."
        ),
    )
    mlmm.add_argument(
        "--link-atom-method",
        choices=LINK_ATOM_METHODS,
        metavar="MODE",
        default=None,
        help=(
            f"How to place / redistribute link atoms at the QM/MM boundary. "
            f"One of: {', '.join(LINK_ATOM_METHODS)}. "
            "Default: 'scaled' (Morokuma/Dapprich g-factor). "
            "'fixed' keeps Q-H at a tabulated bond length using the IMOMM "
            "projector Jacobian. 'none' disables link atoms — only safe for "
            "molecularly closed ML selections."
        ),
    )

    if mode in {"launch", "local"}:
        launcher = p.add_argument_group("launcher")
        launcher.add_argument(
            "--pymol",
            metavar="PATH",
            default=None,
            help="Path to the pymol executable (auto-detected on PATH if omitted).",
        )
        launcher.add_argument(
            "--print-only",
            action="store_true",
            help="Print the resolved PyMOL command and exit without launching.",
        )
    else:
        server = p.add_argument_group("server")
        server.add_argument("--host", default=constants.IMD_DEFAULT_HOST, metavar="ADDR")
        server.add_argument("--port", type=int, default=constants.IMD_DEFAULT_PORT, metavar="PORT")
        server.add_argument(
            "--rate",
            type=int,
            default=constants.IMD_DEFAULT_RATE,
            metavar="N",
            help="MD steps per transmitted frame.",
        )
        server.add_argument(
            "--send-velocities", action="store_true", help="Include velocities each frame."
        )
        server.add_argument("--send-forces", action="store_true", help="Include forces each frame.")
    return p


def _resolve_path(p: str | None) -> str | None:
    if p is None:
        return None
    return str(Path(p).expanduser().resolve())


def _collect_setup_kwargs(args: argparse.Namespace) -> dict:
    """Pull non-default values from argparse into a setup_*-compatible dict.

    Keys must match the ``imd_setup_*`` parameter names exactly, because
    ``_build_pymol_statement`` emits them verbatim as ``k=v``. Flags every
    backend accepts (incl. ``box_padding_nm`` — each builds a box under
    ``pbc``, and CG always) go in the common block; backend-specific ones
    live in the per-level branches below.
    """
    kwargs: dict = {}

    # Common: only include when the user explicitly set them. These are
    # accepted by all three imd_setup_* functions.
    if args.pbc:
        kwargs["pbc"] = True
    if args.no_mouse_force:
        kwargs["mouse_force"] = False
    if args.topology_dir is not None:
        kwargs["work_dir"] = _resolve_path(args.topology_dir)
    for src in (
        "temperature",
        "nsteps_per_frame",
        "const_k",
        "max_force_kj",
        "integrator",
        "timestep_ps",
        "friction",
        "pressure",
        "energy_cutoff_kj",
        "frame_interval_ms",
        "nonbonded_cutoff_nm",
        "box_padding_nm",
        "drag_level",
        "platform",
    ):
        val = getattr(args, src)
        if val is not None:
            kwargs[src] = val

    if args.level == "aa":
        if args.ligand_sdf:
            kwargs["ligand_sdf"] = _resolve_path(args.ligand_sdf)
        if args.ph is not None:
            kwargs["ph"] = args.ph
        if args.gaff_version is not None:
            kwargs["gaff_version"] = args.gaff_version
        if args.implicit_solvent is not None:
            kwargs["implicit_solvent"] = args.implicit_solvent
    elif args.level == "cg":
        if args.ff is not None:
            kwargs["ff"] = args.ff
        if args.no_elastic_network:
            kwargs["elastic_network"] = False
        if args.maxwarn is not None:
            kwargs["maxwarn"] = args.maxwarn
    elif args.level == "mlmm":
        for src, dst in [
            ("ml_selection", "ml_selection"),
            ("ml_backend", "ml_backend"),
            ("ml_device", "ml_device"),
            ("ml_charge", "ml_charge"),
            ("ml_mult", "ml_mult"),
            ("link_atom_method", "link_atom_method"),
        ]:
            val = getattr(args, src)
            if val is not None:
                kwargs[dst] = val
        if args.ml_compile:
            kwargs["ml_compile"] = True
        if args.enable_cueq:
            kwargs["enable_cueq"] = True
        if args.implicit_solvent is not None:
            kwargs["implicit_solvent"] = args.implicit_solvent
    return kwargs


def _collect_record_kwargs(args: argparse.Namespace) -> dict:
    """Pull recording flags into an ``imd_record(...)``-compatible dict.

    ``main`` fills ``traj_out`` / ``force_csv`` / ``state_csv`` with cwd
    defaults before this runs, so all are normally set. Returns an empty dict
    only when all are None (the ``--no-record`` path), which reads as "disable
    recording".
    """
    traj_out = _resolve_path(args.traj_out)
    force_csv = _resolve_path(args.force_csv)
    state_csv = _resolve_path(args.state_csv)
    if traj_out is None and force_csv is None and state_csv is None:
        return {}
    kwargs: dict = {}
    if traj_out is not None:
        kwargs["traj"] = traj_out
        if args.traj_interval is not None:
            kwargs["traj_interval"] = args.traj_interval
    if force_csv is not None:
        kwargs["force_csv"] = force_csv
        if args.force_csv_interval is not None:
            kwargs["force_csv_interval"] = args.force_csv_interval
    if state_csv is not None:
        kwargs["state_csv"] = state_csv
        if args.state_csv_interval is not None:
            kwargs["state_csv_interval"] = args.state_csv_interval
    return kwargs


def _collect_minimize_kwargs(args: argparse.Namespace) -> dict:
    """Pull minimize tuning flags into an ``imd_minimize(...)``-compatible dict.

    Only consulted when ``--minimize`` is set; each unset flag falls back to
    the ``imd_minimize`` / ``constants.py`` default.
    """
    kwargs: dict = {}
    if args.min_max_iter is not None:
        kwargs["max_iter_per_chunk"] = args.min_max_iter
    if args.min_max_chunks is not None:
        kwargs["max_chunks"] = args.min_max_chunks
    if args.min_tol_kj is not None:
        kwargs["tol_kj"] = args.min_tol_kj
    return kwargs


def _build_pymol_statement(
    level: str,
    pdb_path: str,
    kwargs: dict,
    *,
    minimize: bool,
    run: bool,
    record_kwargs: dict | None = None,
    minimize_kwargs: dict | None = None,
) -> str:
    """Assemble the ``-d`` payload PyMOL will eval at startup."""
    kw_parts = [f"{k}={v!r}" for k, v in kwargs.items()]
    call_args = ", ".join([repr(pdb_path), *kw_parts])
    stmts = [
        "from pymol_imd import *",
        f"imd_setup_{level}({call_args})",
    ]
    # Arm recording right after setup so it is in place before imd_run.
    if record_kwargs:
        rec_parts = ", ".join(f"{k}={v!r}" for k, v in record_kwargs.items())
        stmts.append(f"imd_record({rec_parts})")
    if minimize:
        min_options = dict(minimize_kwargs or {})
        if run:
            min_options["run_after"] = True
        min_parts = ", ".join(f"{k}={v!r}" for k, v in min_options.items())
        stmts.append(f"imd_minimize({min_parts})")
    if run and not minimize:
        stmts.append("imd_run()")
    return "; ".join(stmts)


# --------------------------------------------------------------- socket modes
# IMDConfig knobs that _collect_setup_kwargs may emit; everything else it
# returns is a backend build_system kwarg. Used to split the two for the
# headless serve / local path (build_system rejects IMDConfig-only knobs).
_IMD_KNOB_KEYS = frozenset(
    {
        "temperature",
        "nsteps_per_frame",
        "const_k",
        "mouse_force",
        "integrator",
        "timestep_ps",
        "friction",
        "pressure",
        "max_force_kj",
        "energy_cutoff_kj",
        "frame_interval_ms",
        "drag_level",
        "platform",
    }
)


def _backend_for_level(level: str):
    if level == "aa":
        from .backends.aa import AABackend

        return AABackend()
    if level == "cg":
        from .backends.cg import CGBackend

        return CGBackend()
    from .backends.mlmm import MLMMBackend

    return MLMMBackend()


def _build_connect_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pymol_imd connect",
        description="Launch PyMOL as an IMDv3 client of a running `pymol_imd serve`.",
    )
    p.add_argument("pdb", help="The server's PREPARED PDB (same atom order as the server).")
    p.add_argument("--host", default=constants.IMD_DEFAULT_HOST, metavar="ADDR")
    p.add_argument("--port", type=int, default=constants.IMD_DEFAULT_PORT, metavar="PORT")
    p.add_argument("--object", default=None, metavar="NAME", help="PyMOL object name.")
    p.add_argument("--const-k", type=float, default=None, metavar="K", help="Drag sensitivity.")
    p.add_argument(
        "--max-force-kj",
        type=float,
        default=None,
        metavar="F",
        help="Optional per-atom pull cap (kJ/mol/nm). Default: no cap.",
    )
    p.add_argument(
        "--frame-interval-ms",
        type=int,
        default=None,
        metavar="MS",
        help="PyMOL client timer interval.",
    )
    p.add_argument(
        "--drag-level",
        choices=constants.SUPPORTED_DRAG_LEVELS,
        default=None,
        metavar="LEVEL",
        help="Mouse-drag granularity: atom / residue / chain (default atom).",
    )
    p.add_argument("--pymol", default=None, metavar="PATH", help="Path to the pymol executable.")
    p.add_argument("--print-only", action="store_true", help="Print the PyMOL command and exit.")
    return p


def _validate_mlmm_args(args: argparse.Namespace) -> int | None:
    """Mirror the up-front MLMM required-flag check used by the launch path."""
    if args.level != "mlmm":
        return None
    missing = [
        flag
        for flag, val in (
            ("--ml-selection", args.ml_selection),
            ("--ml-charge", args.ml_charge),
            ("--ml-mult", args.ml_mult),
        )
        if val is None
    ]
    if missing:
        print(f"pymol_imd: --level mlmm requires {', '.join(missing)}.", file=sys.stderr)
        return 2
    return None


def _validate_implicit_solvent_args(args: argparse.Namespace) -> int | None:
    """Reject --implicit-solvent for CG and surface GB/pbc conflicts before launch.

    The authoritative check is ``constants.resolve_implicit_solvent`` (it runs
    again inside the backend at build time); this just turns it into a clean
    pre-launch error for the AA / ML-MM launch + serve paths instead of an
    in-PyMOL / in-server traceback.
    """
    if getattr(args, "implicit_solvent", None) is None:
        return None
    if args.level == "cg":
        print(
            "pymol_imd: --implicit-solvent is not valid for --level cg "
            "(Martini has its own solvent treatment).",
            file=sys.stderr,
        )
        return 2
    try:
        constants.resolve_implicit_solvent(args.implicit_solvent, pbc=bool(args.pbc))
    except ValueError as exc:
        print(f"pymol_imd: {exc}", file=sys.stderr)
        return 2
    return None


def _validate_level_args(args: argparse.Namespace) -> int | None:
    groups = {
        "aa": {
            "--ligand-sdf": args.ligand_sdf,
            "--ph": args.ph,
            "--gaff-version": args.gaff_version,
        },
        "cg": {
            "--ff": args.ff,
            "--no-elastic-network": args.no_elastic_network,
            "--martinize-maxwarn": args.maxwarn,
        },
        "mlmm": {
            "--ml-selection": args.ml_selection,
            "--ml-backend": args.ml_backend,
            "--ml-device": args.ml_device,
            "--ml-charge": args.ml_charge,
            "--ml-mult": args.ml_mult,
            "--ml-compile": args.ml_compile,
            "--enable-cueq": args.enable_cueq,
            "--link-atom-method": args.link_atom_method,
        },
    }
    invalid = [
        flag
        for level, values in groups.items()
        if level != args.level
        for flag, value in values.items()
        if value not in (None, False)
    ]
    if invalid:
        print(
            f"pymol_imd: {' / '.join(invalid)} not valid for --level {args.level}.",
            file=sys.stderr,
        )
        return 2
    return None


def _run_minimize(imd, args: argparse.Namespace) -> None:
    # Use `is not None`, not `or`: an explicit 0 (uncapped iterations / disabled
    # early-convergence / zero chunks) is falsy, so `or` would silently swap it
    # for the default. Mirrors _collect_minimize_kwargs / imd_minimize.
    max_iter = (
        args.min_max_iter
        if args.min_max_iter is not None
        else constants.DEFAULT_MIN_MAX_ITER_PER_CHUNK
    )
    chunks = (
        args.min_max_chunks if args.min_max_chunks is not None else constants.DEFAULT_MIN_MAX_CHUNKS
    )
    tol = args.min_tol_kj if args.min_tol_kj is not None else constants.DEFAULT_MIN_ENERGY_TOL_KJ
    imd.reset_minimization()
    for _ in range(chunks):
        if imd.minimize_step(max_iterations=max_iter, tol_kj=tol):
            break


def _build_imd_system_headless(args: argparse.Namespace, pdb_path: Path):
    """Build a backend SystemBundle + IMDSystem with no PyMOL (serve / local)."""
    from . import commands
    from .imd import IMDConfig, IMDSystem

    setup_kwargs = _collect_setup_kwargs(args)
    build_kwargs = {k: v for k, v in setup_kwargs.items() if k not in _IMD_KNOB_KEYS}
    backend = _backend_for_level(args.level)
    bundle = backend.build_system(pdb_path, **build_kwargs)
    imd_knobs = commands._build_imd_kwargs(
        backend,
        temperature=args.temperature,
        nsteps_per_frame=args.nsteps_per_frame,
        const_k=args.const_k,
        integrator=args.integrator,
        timestep_ps=args.timestep_ps,
        friction=args.friction,
        pressure=args.pressure,
        max_force_kj=args.max_force_kj,
        energy_cutoff_kj=args.energy_cutoff_kj,
        frame_interval_ms=args.frame_interval_ms,
        platform=args.platform,
    )
    return IMDSystem(bundle, IMDConfig(**imd_knobs)), bundle


def _resolve_topology_dir(args: argparse.Namespace, pdb_path: Path) -> None:
    if args.topology_dir is None:
        args.topology_dir = str(Path.cwd() / f"{pdb_path.stem}_topology")


def _resolve_record_defaults(args: argparse.Namespace, pdb_path: Path) -> int | None:
    explicit = any(value is not None for value in (args.traj_out, args.force_csv, args.state_csv))
    if args.no_record:
        if explicit:
            print(
                "pymol_imd: --no-record conflicts with an explicit "
                "--traj-out / --force-csv / --state-csv.",
                file=sys.stderr,
            )
            return 2
        return None
    stem = pdb_path.stem
    if args.traj_out is None:
        args.traj_out = str(Path.cwd() / f"{stem}.xtc")
    if args.force_csv is None:
        args.force_csv = str(Path.cwd() / f"{stem}_forces.csv")
    if args.state_csv is None:
        args.state_csv = str(Path.cwd() / f"{stem}_state.csv")
    return None


def _validate_numeric_args(args: argparse.Namespace, *, mode: str) -> int | None:
    positive = {
        "temperature": "--temperature",
        "nsteps_per_frame": "--nsteps-per-frame",
        "max_force_kj": "--max-force-kj",
        "timestep_ps": "--timestep-ps",
        "pressure": "--pressure",
        "energy_cutoff_kj": "--energy-cutoff-kj",
        "frame_interval_ms": "--frame-interval-ms",
        "nonbonded_cutoff_nm": "--nonbonded-cutoff-nm",
        "box_padding_nm": "--box-padding-nm",
        "min_max_chunks": "--min-max-chunks",
        "traj_interval": "--traj-interval",
        "force_csv_interval": "--force-csv-interval",
        "state_csv_interval": "--state-csv-interval",
        "ml_mult": "--ml-mult",
        "rate": "--rate",
    }
    nonnegative = {
        "const_k": "--const-k",
        "friction": "--friction",
        "min_max_iter": "--min-max-iter",
        "min_tol_kj": "--min-tol-kj",
    }
    for attr, flag in positive.items():
        value = getattr(args, attr, None)
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0.0):
            print(f"pymol_imd: {flag} must be finite and > 0.", file=sys.stderr)
            return 2
    for attr, flag in nonnegative.items():
        value = getattr(args, attr, None)
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
            print(f"pymol_imd: {flag} must be finite and >= 0.", file=sys.stderr)
            return 2
    if args.ph is not None and (not math.isfinite(args.ph) or not 0.0 <= args.ph <= 14.0):
        print("pymol_imd: --ph must be finite and between 0 and 14.", file=sys.stderr)
        return 2
    if args.pressure is not None and not args.pbc:
        print("pymol_imd: --pressure requires --pbc.", file=sys.stderr)
        return 2
    if mode == "serve" and not 0 <= args.port <= 65535:
        print("pymol_imd: --port must be between 0 and 65535.", file=sys.stderr)
        return 2
    if args.maxwarn is not None:
        try:
            maxwarn = int(args.maxwarn)
        except ValueError:
            print("pymol_imd: --martinize-maxwarn must be an integer.", file=sys.stderr)
            return 2
        if maxwarn < 0:
            print("pymol_imd: --martinize-maxwarn must be >= 0.", file=sys.stderr)
            return 2
    return None


def _connect_statement(prepared: str, host: str, port: int, obj: str, extra: dict) -> str:
    kw = {"host": host, "port": port, "object": obj, **extra}
    kw_str = ", ".join(f"{k}={v!r}" for k, v in kw.items())
    return (
        "from pymol_imd import *; cmd.set('retain_order', 1); "
        "from pymol_imd.pymol_io import load_prepared; "
        f"load_prepared(cmd, {prepared!r}, {obj!r}); imd_connect({kw_str})"
    )


def _main_serve(argv: list[str]) -> int:
    args = _build_parser("serve").parse_args(argv)
    pdb_path = Path(args.pdb).expanduser().resolve()
    if not pdb_path.exists():
        print(f"pymol_imd: PDB file not found: {pdb_path}", file=sys.stderr)
        return 2
    rc = _validate_mlmm_args(args)
    if rc:
        return rc
    rc = _validate_level_args(args)
    if rc:
        return rc
    rc = _validate_implicit_solvent_args(args)
    if rc:
        return rc
    rc = _validate_numeric_args(args, mode="serve")
    if rc:
        return rc
    _resolve_topology_dir(args, pdb_path)
    rc = _resolve_record_defaults(args, pdb_path)
    if rc:
        return rc
    from .server import IMDServer

    try:
        imd, _bundle = _build_imd_system_headless(args, pdb_path)
        if args.minimize:
            _run_minimize(imd, args)
    except Exception as exc:
        print(f"pymol_imd serve: setup failed: {exc}", file=sys.stderr)
        return 1
    record_settings = _collect_record_kwargs(args) or None
    server = IMDServer(
        imd,
        host=args.host,
        port=args.port,
        rate=args.rate,
        send_velocities=args.send_velocities,
        send_forces=args.send_forces,
        record_settings=record_settings,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\npymol_imd serve: stopping")
        server.request_stop()
        return 130
    return 1 if server.failure_reason is not None else 0


def _main_connect(argv: list[str]) -> int:
    args = _build_connect_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        print("pymol_imd connect: --port must be between 1 and 65535.", file=sys.stderr)
        return 2
    for value, flag, allow_zero in (
        (args.const_k, "--const-k", True),
        (args.max_force_kj, "--max-force-kj", False),
        (args.frame_interval_ms, "--frame-interval-ms", False),
    ):
        if value is None:
            continue
        numeric = float(value)
        invalid = not math.isfinite(numeric) or numeric < 0.0 or (not allow_zero and numeric == 0.0)
        if invalid:
            relation = ">= 0" if allow_zero else "> 0"
            print(f"pymol_imd connect: {flag} must be finite and {relation}.", file=sys.stderr)
            return 2
    pdb_path = Path(args.pdb).expanduser().resolve()
    if not pdb_path.exists():
        print(f"pymol_imd: PDB file not found: {pdb_path}", file=sys.stderr)
        return 2
    pymol_bin = args.pymol or shutil.which("pymol")
    if not pymol_bin:
        print("pymol_imd: pymol executable not found in PATH.", file=sys.stderr)
        return 127
    obj = args.object or f"{pdb_path.stem}_imd"
    extra: dict = {}
    if args.const_k is not None:
        extra["const_k"] = args.const_k
    if args.max_force_kj is not None:
        extra["max_force_kj"] = args.max_force_kj
    if args.frame_interval_ms is not None:
        extra["frame_interval_ms"] = args.frame_interval_ms
    if args.drag_level is not None:
        extra["drag_level"] = args.drag_level
    statement = _connect_statement(str(pdb_path), args.host, args.port, obj, extra)
    cmd = [pymol_bin, "-d", statement]
    if args.print_only:
        print(shlex.join(cmd))
        return 0
    return subprocess.run(cmd).returncode


def _main_local(argv: list[str]) -> int:
    # local == the classic SAME-PROCESS QTimer mode (NO socket): one command to
    # launch PyMOL, set up the backend, and run interactively with mouse-drag
    # pulling — exactly the mechanism that has always worked. The IMDv3 socket
    # transport is only for serve/connect (splitting compute across machines),
    # so `local` simply implies --run on the standard launch path. This keeps
    # the local experience free of any socket latency / backpressure / freeze.
    return _main_launch(argv, mode="local")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: pymol_imd {launch,local,serve,connect} ...\n\n"
            "commands:\n"
            "  launch   configure and open PyMOL\n"
            "  local    configure, open PyMOL, and start the local run\n"
            "  serve    run a headless IMDv3 server\n"
            "  connect  open PyMOL as an IMDv3 client"
        )
        return 0 if argv else 2
    if argv[0] == "--version":
        from . import __version__

        print(__version__)
        return 0
    if argv[0] == "launch":
        return _main_launch(argv[1:], mode="launch")
    if argv and argv[0] == "serve":
        return _main_serve(argv[1:])
    if argv and argv[0] == "connect":
        return _main_connect(argv[1:])
    if argv and argv[0] == "local":
        return _main_local(argv[1:])
    print(
        f"pymol_imd: unknown command {argv[0]!r}. "
        "Use `pymol_imd launch <PDB>` (the old `pymol_imd <PDB>` form was removed).",
        file=sys.stderr,
    )
    return 2


def _main_launch(argv: list[str] | None, *, mode: str = "launch") -> int:
    args = _build_parser(mode).parse_args(argv)

    pdb_path = Path(args.pdb).expanduser().resolve()
    if not pdb_path.exists():
        print(f"pymol_imd: PDB file not found: {pdb_path}", file=sys.stderr)
        return 2

    # Level-specific required-flag validation, done up front so we don't
    # launch PyMOL only to error out from inside setup_*.
    rc = _validate_mlmm_args(args)
    if rc:
        return rc

    rc = _validate_implicit_solvent_args(args)
    if rc:
        return rc
    rc = _validate_level_args(args)
    if rc:
        return rc
    rc = _validate_numeric_args(args, mode=mode)
    if rc:
        return rc

    pymol_bin = args.pymol or shutil.which("pymol")
    if not pymol_bin:
        print(
            "pymol_imd: pymol executable not found in PATH. "
            "Activate the pixi env first (`pixi shell`) or pass --pymol /path/to/pymol.",
            file=sys.stderr,
        )
        return 127

    # Resolve recording / topology defaults now that the PDB stem is known
    # (argparse defaults can't see it). All paths anchor at the launch cwd.
    traj_explicit = args.traj_out is not None
    force_explicit = args.force_csv is not None
    state_explicit = args.state_csv is not None
    rc = _resolve_record_defaults(args, pdb_path)
    if rc:
        return rc
    run = bool(args.run) if mode == "launch" else True
    stem = pdb_path.stem
    if args.topology_dir is None:
        args.topology_dir = str(Path.cwd() / f"{stem}_topology")

    kwargs = _collect_setup_kwargs(args)
    record_kwargs = _collect_record_kwargs(args)
    minimize_kwargs = _collect_minimize_kwargs(args)
    if (traj_explicit or force_explicit or state_explicit) and not run:
        print(
            "pymol_imd: --traj-out / --force-csv / --state-csv given without --run; "
            "recording is armed but only starts on imd_run.",
            file=sys.stderr,
        )
    if minimize_kwargs and not args.minimize:
        print(
            "pymol_imd: --min-* tuning flags given without --minimize; they have no effect.",
            file=sys.stderr,
        )
    statement = _build_pymol_statement(
        args.level,
        str(pdb_path),
        kwargs,
        minimize=args.minimize,
        run=run,
        record_kwargs=record_kwargs,
        minimize_kwargs=minimize_kwargs,
    )

    cmd = [pymol_bin, "-d", statement]
    if args.print_only:
        print(shlex.join(cmd))
        return 0

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
