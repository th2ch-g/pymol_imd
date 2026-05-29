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

ML_BACKENDS = ("uma", "orb", "mace", "aimnet2")
LINK_ATOM_METHODS = ("scaled", "fixed", "none")
LEVELS = ("aa", "cg", "mlmm")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pymol_imd",
        description=(
            "Launch PyMOL with a pymol_imd backend already configured. "
            "Equivalent to typing `from pymol_imd import *; setup_<level>(<pdb>, ...)` "
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
            "Default 5000. Lower to make divergence-on-strong-drag harder."
        ),
    )
    common.add_argument(
        "--no-mouse-force", action="store_true", help="Disable the user-pull mechanism."
    )
    common.add_argument(
        "--minimize", action="store_true", help="Call imd_minimize() right after setup."
    )
    common.add_argument(
        "--run", action="store_true", help="Call imd_run() after setup (and minimize, if set)."
    )
    common.add_argument(
        "--traj-out",
        metavar="PATH",
        help=(
            "Record a trajectory during imd_run. Format by extension: "
            ".xtc / .dcd / .pdb. Requires --run."
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
            "0-based (add 1 for a PyMOL selection). Requires --run."
        ),
    )
    common.add_argument(
        "--force-csv-interval",
        type=int,
        metavar="N",
        help="Write forces every N GUI frames (default 1).",
    )

    # ---- AA-specific
    aa = p.add_argument_group("--level aa options")
    aa.add_argument(
        "--ligand-sdf", metavar="PATH", help="Ligand SDF for GAFF + AM1-BCC parameterization."
    )

    # ---- CG-specific
    cg = p.add_argument_group("--level cg options")
    cg.add_argument("--ff", metavar="NAME", help="Martini forcefield label (default: martini3001).")
    cg.add_argument(
        "--no-elastic-network",
        action="store_true",
        help="Disable martinize2 elastic network (-elastic).",
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
            "If omitted, setup_mlmm picks 'aimnet2' — smallest model "
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
    """Pull non-default values from argparse into a setup_*-compatible dict."""
    kwargs: dict = {}

    # Common: only include when user explicitly set them.
    if args.pbc:
        kwargs["pbc"] = True
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.nsteps_per_frame is not None:
        kwargs["nsteps_per_frame"] = args.nsteps_per_frame
    if args.const_k is not None:
        kwargs["const_k"] = args.const_k
    if args.max_force_kj is not None:
        kwargs["max_force_kj"] = args.max_force_kj
    if args.no_mouse_force:
        kwargs["mouse_force"] = False

    if args.level == "aa":
        if args.ligand_sdf:
            kwargs["ligand_sdf"] = _resolve_path(args.ligand_sdf)
    elif args.level == "cg":
        if args.ff is not None:
            kwargs["ff"] = args.ff
        if args.no_elastic_network:
            kwargs["elastic_network"] = False
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
    return kwargs


def _collect_record_kwargs(args: argparse.Namespace) -> dict:
    """Pull recording flags into an ``imd_record(...)``-compatible dict.

    Returns an empty dict when neither output path is set: an interval alone
    is meaningless and would otherwise produce ``imd_record(traj_interval=N)``,
    which (no paths) reads as "disable recording".
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


def _build_pymol_statement(
    level: str,
    pdb_path: str,
    kwargs: dict,
    *,
    minimize: bool,
    run: bool,
    record_kwargs: dict | None = None,
) -> str:
    """Assemble the ``-d`` payload PyMOL will eval at startup."""
    kw_parts = [f"{k}={v!r}" for k, v in kwargs.items()]
    call_args = ", ".join([repr(pdb_path), *kw_parts])
    stmts = [
        "from pymol_imd import *",
        f"setup_{level}({call_args})",
    ]
    # Arm recording right after setup so it is in place before imd_run.
    if record_kwargs:
        rec_parts = ", ".join(f"{k}={v!r}" for k, v in record_kwargs.items())
        stmts.append(f"imd_record({rec_parts})")
    if minimize:
        stmts.append("imd_minimize()")
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

    kwargs = _collect_setup_kwargs(args)
    record_kwargs = _collect_record_kwargs(args)
    if record_kwargs and not args.run:
        print(
            "pymol_imd: --traj-out / --force-csv given without --run; no "
            "recording will happen (recording starts on imd_run).",
            file=sys.stderr,
        )
    statement = _build_pymol_statement(
        args.level,
        str(pdb_path),
        kwargs,
        minimize=args.minimize,
        run=args.run,
        record_kwargs=record_kwargs,
    )

    cmd = [pymol_bin, "-d", statement]
    if args.print_only:
        print(shlex.join(cmd))
        return 0

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
