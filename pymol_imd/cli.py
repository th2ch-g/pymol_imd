"""Command-line entry point for pymol_imd.

Wraps ``pymol`` so a single shell command sets up a simulation::

    pymol_imd input.pdb --level cg
    pymol_imd input.pdb --level aa --ligand-sdf lig.sdf --minimize --run
    pymol_imd input.pdb --level mlmm --ml-selection "resi 30-50" --ml-backend uma

The CLI itself does not import ``pymol``; it builds a Python statement and
hands it to PyMOL via the ``-d`` flag, so PyMOL evaluates exactly the same
``setup_*`` / ``imd_*`` commands an interactive user would type.
"""

from __future__ import annotations

import argparse
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pymol_imd",
        description=(
            "Launch PyMOL with a pymol_imd backend already configured. "
            "Equivalent to typing `from pymol_imd import *; imd_setup_<level>(<pdb>, ...)` "
            "inside PyMOL."
        ),
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
        help="Enable periodic box. Default: vacuum (AA/MLMM) / auto-box (CG).",
    )
    common.add_argument(
        "--temperature", type=float, metavar="K", help="Langevin thermostat temperature (K)."
    )
    common.add_argument(
        "--nsteps-per-frame", type=int, metavar="N", help="MD steps executed per GUI frame."
    )
    common.add_argument("--const-k", type=float, metavar="K", help="Pull-force scale (kJ/mol/nm).")
    common.add_argument(
        "--max-force-kj",
        type=float,
        metavar="F",
        help=(
            "Per-atom hard cap on the pulling force magnitude (kJ/mol/nm). "
            f"Default {constants.DEFAULT_MAX_FORCE_KJ:g}. "
            "Lower to make divergence-on-strong-drag harder."
        ),
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
        help=(
            "|potential energy| above this (kJ/mol) halts the loop as diverged. "
            f"Default {constants.DEFAULT_ENERGY_CUTOFF_KJ:g}."
        ),
    )
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
        "--box-padding-nm",
        type=float,
        metavar="NM",
        help=(
            "Bounding-box padding (nm) when a periodic box is built (AA --pbc / CG). "
            f"Default {constants.AA_BOX_PADDING_NM:g}."
        ),
    )
    common.add_argument(
        "--no-mouse-force", action="store_true", help="Disable the user-pull mechanism."
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
    common.add_argument(
        "--run", action="store_true", help="Call imd_run() after setup (and minimize, if set)."
    )
    common.add_argument(
        "--traj-out",
        metavar="PATH",
        help=(
            "Record a trajectory during imd_run. Format by extension: "
            ".xtc / .dcd / .pdb. Default: ./<pdb-stem>.xtc in the launch cwd "
            "(recording is on by default; disable with --no-record). "
            "Recording starts on --run."
        ),
    )
    common.add_argument(
        "--traj-interval",
        type=int,
        metavar="N",
        help="Write every N GUI frames to the trajectory (default 1).",
    )
    common.add_argument(
        "--force-csv",
        metavar="PATH",
        help=(
            "Record user pull forces to CSV during imd_run (columns: "
            "frame,step,time_ps,atom_index,fx,fy,fz,fmag). atom_index is "
            "0-based (add 1 for a PyMOL selection). Default: "
            "./<pdb-stem>_forces.csv in the launch cwd (on by default; "
            "disable with --no-record). Recording starts on --run."
        ),
    )
    common.add_argument(
        "--force-csv-interval",
        type=int,
        metavar="N",
        help="Write forces every N GUI frames (default 1).",
    )
    common.add_argument(
        "--no-record",
        action="store_true",
        help=(
            "Disable default recording. Without this a trajectory + force CSV "
            "are written by default. Conflicts with an explicit "
            "--traj-out / --force-csv."
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
    mlmm.add_argument("--ml-device", metavar="DEV", help="Torch device: auto / cuda / mps / cpu.")
    mlmm.add_argument("--ml-charge", type=int, metavar="Z", help="ML region net charge.")
    mlmm.add_argument(
        "--ml-mult",
        type=int,
        metavar="M",
        help=(
            "ML region spin multiplicity (2S+1). Default: 1 (singlet). "
            "The unpaired-electron count expected by ORB (atoms.info['spin']) "
            "is auto-derived as mult - 1."
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

    # ---- launcher controls
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
    return p


def _resolve_path(p: str | None) -> str | None:
    if p is None:
        return None
    return str(Path(p).expanduser().resolve())


def _collect_setup_kwargs(args: argparse.Namespace) -> dict:
    """Pull non-default values from argparse into a setup_*-compatible dict.

    Keys must match the ``imd_setup_*`` parameter names exactly, because
    ``_build_pymol_statement`` emits them verbatim as ``k=v``. Only flags every
    backend accepts go in the common block; ``box_padding_nm`` is AA/CG-only
    (MLMM never builds a box) so it lives in those branches.
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
    ):
        val = getattr(args, src)
        if val is not None:
            kwargs[src] = val

    if args.level == "aa":
        if args.ligand_sdf:
            kwargs["ligand_sdf"] = _resolve_path(args.ligand_sdf)
        if args.box_padding_nm is not None:
            kwargs["box_padding_nm"] = args.box_padding_nm
        if args.ph is not None:
            kwargs["ph"] = args.ph
        if args.gaff_version is not None:
            kwargs["gaff_version"] = args.gaff_version
    elif args.level == "cg":
        if args.ff is not None:
            kwargs["ff"] = args.ff
        if args.no_elastic_network:
            kwargs["elastic_network"] = False
        if args.box_padding_nm is not None:
            kwargs["box_padding_nm"] = args.box_padding_nm
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
    return kwargs


def _collect_record_kwargs(args: argparse.Namespace) -> dict:
    """Pull recording flags into an ``imd_record(...)``-compatible dict.

    ``main`` fills ``traj_out`` / ``force_csv`` with cwd defaults before this
    runs, so both are normally set. Returns an empty dict only when both are
    None (the ``--no-record`` path), which reads as "disable recording".
    """
    traj_out = _resolve_path(args.traj_out)
    force_csv = _resolve_path(args.force_csv)
    if traj_out is None and force_csv is None:
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
        min_parts = ", ".join(f"{k}={v!r}" for k, v in (minimize_kwargs or {}).items())
        stmts.append(f"imd_minimize({min_parts})")
    if run:
        stmts.append("imd_run()")
    return "; ".join(stmts)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    pdb_path = Path(args.pdb).expanduser().resolve()
    if not pdb_path.exists():
        print(f"pymol_imd: PDB file not found: {pdb_path}", file=sys.stderr)
        return 2

    # Level-specific required-flag validation, done up front so we don't
    # launch PyMOL only to error out from inside setup_*.
    if args.level == "mlmm":
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
            print(
                f"pymol_imd: --level mlmm requires {', '.join(missing)}.\n"
                "ml_selection uses PyMOL's selection language. Example:\n"
                "  pymol_imd input.pdb --level mlmm \\\n"
                "      --ml-selection 'resi 1 and resn MET' \\\n"
                "      --ml-charge 0 --ml-mult 1\n"
                "Multiplicity (2S+1): 1=singlet, 2=doublet/radical, 3=triplet.",
                file=sys.stderr,
            )
            return 2

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
    stem = pdb_path.stem
    traj_explicit = args.traj_out is not None
    force_explicit = args.force_csv is not None
    if args.no_record:
        if traj_explicit or force_explicit:
            print(
                "pymol_imd: --no-record conflicts with an explicit "
                "--traj-out / --force-csv. Drop one.",
                file=sys.stderr,
            )
            return 2
    else:
        if args.traj_out is None:
            args.traj_out = str(Path.cwd() / f"{stem}.xtc")
        if args.force_csv is None:
            args.force_csv = str(Path.cwd() / f"{stem}_forces.csv")
    if args.topology_dir is None:
        args.topology_dir = str(Path.cwd() / f"{stem}_topology")

    kwargs = _collect_setup_kwargs(args)
    record_kwargs = _collect_record_kwargs(args)
    minimize_kwargs = _collect_minimize_kwargs(args)
    if (traj_explicit or force_explicit) and not args.run:
        print(
            "pymol_imd: --traj-out / --force-csv given without --run; "
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
        run=args.run,
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
