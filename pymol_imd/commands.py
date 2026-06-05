"""PyMOL command surface for pymol_imd.

This module is imported for its side effects: each ``cmd.extend(...)`` call
registers a PyMOL command. The actual simulation work is delegated to backend
strategies + :class:`IMDSystem`.

PyMOL passes every argument as a string, so we centralize the coercion logic
in :func:`_as_bool` / :func:`_as_int` / :func:`_as_float` and let the backends
work with proper Python types.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path

from . import constants
from .backends.aa import AABackend
from .backends.cg import CGBackend
from .backends.mlmm import MLMMBackend
from .client import IMDClient, PyMOLIMDClient
from .imd import IMDConfig, IMDSystem

logger = logging.getLogger(__name__)

# Module-global IMD state. PyMOL commands are free functions, so the system is
# kept here and shared across them.
_imd_system: IMDSystem | None = None
_timer = None  # type: ignore[var-annotated]
_timer_minimize = None  # type: ignore[var-annotated]
# Socket client mode (imd_connect / imd_disconnect). Kept separate from
# _imd_system (the in-process / server-side simulation) so the two modes never
# alias each other.
_imd_client: PyMOLIMDClient | None = None


# --------------------------------------------------------------------- helpers
def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: object) -> int:
    return int(value) if not isinstance(value, int) else value


def _as_float(value: object) -> float:
    return float(value) if not isinstance(value, float) else value


def _make_imd(bundle, pymol_object: str, **kwargs) -> IMDSystem:
    cfg = IMDConfig(**kwargs)
    return IMDSystem(bundle, cfg, pymol_object=pymol_object)


def _resolve_const_k(backend, const_k: object) -> float:
    """Pick the right pulling-force scale.

    Returns the user-supplied value if explicit; otherwise the backend's
    own ``recommended_const_k`` so the AA/CG/MLMM defaults live next to
    each backend instead of being copy-pasted across the command layer.
    """
    if const_k is None:
        return float(backend.recommended_const_k)
    return _as_float(const_k)


def _resolve_integrator(backend, integrator: object) -> str:
    """User-supplied integrator if explicit, else the backend recommendation.

    Mirrors :func:`_resolve_const_k`: AA recommends ``langevin_middle``,
    CG / MLMM recommend ``langevin`` — all sourced from constants.py.
    """
    if integrator is None:
        return str(backend.recommended_integrator)
    return str(integrator)


def _resolve_timestep_ps(backend, timestep_ps: object) -> float:
    """User-supplied timestep if explicit, else the backend recommendation.

    The per-backend recommendations (AA 2 fs / CG 20 fs / MLMM 0.5 fs) live in
    constants.py instead of being scattered across the setup functions.
    """
    if timestep_ps is None:
        return float(backend.recommended_timestep_ps)
    return _as_float(timestep_ps)


def _build_imd_kwargs(
    backend,
    *,
    temperature: object = None,
    nsteps_per_frame: object = None,
    const_k: object = None,
    mouse_force: object = None,
    integrator: object = None,
    timestep_ps: object = None,
    friction: object = None,
    pressure: object = None,
    max_force_kj: object = None,
    energy_cutoff_kj: object = None,
    frame_interval_ms: object = None,
    drag_level: object = None,
) -> dict:
    """Coerce the common IMD knobs into an ``IMDConfig`` kwargs dict.

    ``const_k`` / ``integrator`` / ``timestep_ps`` always resolve to a value
    (the backend recommendation when the user did not pass one). Every other
    knob is included only when explicitly set, so ``IMDConfig``'s own
    constants-backed default applies otherwise — keeping a single source of
    truth for the defaults instead of repeating literals in each setup_*.
    """
    kwargs: dict = {
        "const_k": _resolve_const_k(backend, const_k),
        "integrator": _resolve_integrator(backend, integrator),
        "timestep_ps": _resolve_timestep_ps(backend, timestep_ps),
    }
    if temperature is not None:
        kwargs["temperature"] = _as_float(temperature)
    if nsteps_per_frame is not None:
        kwargs["nsteps_per_frame"] = _as_int(nsteps_per_frame)
    if mouse_force is not None:
        kwargs["mouse_force"] = _as_bool(mouse_force)
    if friction is not None:
        kwargs["friction"] = _as_float(friction)
    if pressure is not None:
        kwargs["pressure"] = _as_float(pressure)
    if max_force_kj is not None:
        kwargs["max_force_kj"] = _as_float(max_force_kj)
    if energy_cutoff_kj is not None:
        kwargs["energy_cutoff_kj"] = _as_float(energy_cutoff_kj)
    if frame_interval_ms is not None:
        kwargs["frame_interval_ms"] = _as_int(frame_interval_ms)
    if drag_level is not None:
        # Validate at the shared confluence so every imd_setup_* path rejects a
        # bad level with one clear error, instead of silently degrading to
        # chain/residue dragging inside IMDSystem (only "atom" is special-cased
        # there, so any other string expands groups).
        kwargs["drag_level"] = _validate_drag_level(str(drag_level))
    return kwargs


def _validate_drag_level(drag_level: str) -> str:
    """Validate a drag level (atom / residue / chain), returning it unchanged.

    The pull is expanded to the whole residue / chain inside ``IMDSystem`` each
    frame. PyMOL's own ``mouse_selection_mode`` does NOT affect 3-Button Editing
    drag (Ctrl+drag moves a covalent fragment regardless), so nothing is set on
    PyMOL here — this only guards the value flowing into ``IMDConfig`` and the
    live ``imd_change`` / ``imd_drag_level`` commands.
    """
    if drag_level not in constants.SUPPORTED_DRAG_LEVELS:
        raise ValueError(
            f"Unknown drag_level {drag_level!r}. Choose from {constants.SUPPORTED_DRAG_LEVELS}."
        )
    return drag_level


def _apply_imd_representation(cmd, obj: str, backend_name: str) -> None:
    """Backend-aware default representation + drop into 3-Button Editing.

    Shared by the in-process setup path (:func:`_load_prepared_into_pymol`) and
    the socket client (:func:`imd_connect`) so both render and drag the same:
    AA / ML-MM get cartoon + ligand sticks, CG gets colored bead spheres.
    """
    if backend_name == "cg":
        # Spheres for the beads + sticks for the bonds (CONECT comes from
        # martinize2's cg.pdb directly so only chemical bonds are drawn).
        cmd.show_as("spheres", obj)
        cmd.show("sticks", obj)
        cmd.set("sphere_scale", constants.CG_SPHERE_SCALE, obj)
        cmd.set("stick_radius", constants.CG_STICK_RADIUS, obj)
        # Martini bead types aren't standard elements, so PyMOL paints them
        # with arbitrary fallback colors. Override: backbone skyblue, the rest
        # (side-chain beads) yelloworange for a clean CG look.
        cmd.color("skyblue", f"{obj} and name BB")
        cmd.color("yelloworange", f"{obj} and not name BB")
    else:  # aa, mlmm
        cmd.show_as("cartoon", obj)
        # Render any non-polymer atoms (ligands, ions left in) as sticks too.
        cmd.show("sticks", f"{obj} and not polymer")
    cmd.zoom(obj, buffer=constants.VIEW_ZOOM_BUFFER)
    # Drop into Editing mode so the user can immediately grab atoms with
    # Ctrl+left-drag and feed the pull-force mechanism without first toggling
    # Mouse > 3-Button Editing from the menu.
    cmd.config_mouse("three_button_editing")


def _load_prepared_into_pymol(
    prepared_pdb_path: Path, base_name: str, *, backend_name: str = "aa"
) -> str:
    """Load the prepared PDB into PyMOL as a NEW object and return its name.

    The new object is named ``<base>_imd`` (or ``<base>_imd_N`` if an IMD
    object already exists), so any already-loaded original object the user
    might have isn't clobbered.

    Representation is chosen per backend because CG beads don't carry CONECT
    bonds via the default ``PDBFile.writeFile`` path and their atom names
    (``BB``/``SC1``) don't match PyMOL's cartoon detection, so the default
    ``lines`` representation leaves the viewport empty.
    """
    from pymol import cmd

    cmd.set("retain_order", 1)
    existing = set(cmd.get_object_list())
    candidate = f"{base_name}_imd"
    suffix = 1
    while candidate in existing:
        suffix += 1
        candidate = f"{base_name}_imd_{suffix}"
    cmd.load(str(prepared_pdb_path), candidate)

    _apply_imd_representation(cmd, candidate, backend_name)
    return candidate


def _make_qtimer():
    from pymol.Qt import QtCore

    return QtCore.QTimer()


def _highlight_ml_region(obj_name: str, ml_indices: tuple[int, ...]) -> str:
    """Highlight the ML-region atoms while leaving the MM environment alone.

    Only the ML atoms get recolored: carbons take on the cbam magenta and
    other elements keep their CPK defaults (N=blue, O=red, S=yellow, H=white)
    so element identity stays readable. The MM region is left untouched,
    preserving whatever coloring the user already had on the object.

    OpenMM atom indices are 0-based; PyMOL's per-object atom ``index`` is
    1-based and stays aligned with OpenMM because ``retain_order=1`` is set
    on import. We mark the ML atoms via a b-factor flag instead of building
    a huge ``index 1+2+3+...`` selection string, which would break down on
    ML regions larger than a few hundred atoms.

    Returns the name of the persistent named selection ("<obj>_ml") so the
    user can re-target it later (e.g. ``util.cbac <obj>_ml`` to switch to
    cyan carbons).
    """
    from pymol import cmd, util

    ml_set = {int(i) + 1 for i in ml_indices}
    cmd.alter(obj_name, "b = 1.0 if index in ml_set else 0.0", space={"ml_set": ml_set})
    sel_name = f"{obj_name}_ml"
    # enable=0 keeps the named selection available for later commands but
    # suppresses the pink "selection indicator" dots that would otherwise
    # clutter the viewport immediately after setup.
    cmd.select(sel_name, f"{obj_name} and b > 0.5", enable=0)
    # cbam: color-by-atom with magenta carbons. Heteroatoms keep CPK colors,
    # so the user can still read off element identity at a glance.
    util.cbam(sel_name)
    # Sticks on top of the existing cartoon makes the ML region pop without
    # hiding the surrounding fold.
    cmd.show("sticks", sel_name)
    return sel_name


# --------------------------------------------------------------- setup_* family
def imd_setup_aa(
    pdb_file: str,
    ligand_sdf: str | None = None,
    pbc: object = False,
    temperature: object = None,
    nsteps_per_frame: object = None,
    const_k: object = None,
    mouse_force: object = None,
    integrator: object = None,
    timestep_ps: object = None,
    friction: object = None,
    pressure: object = None,
    max_force_kj: object | None = None,
    energy_cutoff_kj: object | None = None,
    frame_interval_ms: object | None = None,
    nonbonded_cutoff_nm: object | None = None,
    box_padding_nm: object | None = None,
    ph: object | None = None,
    gaff_version: str | None = None,
    implicit_solvent: str | None = None,
    drag_level: str = constants.DEFAULT_DRAG_LEVEL,
    work_dir: str | None = None,
) -> None:
    """Set up an all-atom IMD simulation with GBn2 implicit solvent (NVT) by default.

    Every knob left unset falls back to its single-source default: the
    per-backend recommendations (``const_k`` / ``integrator`` /
    ``timestep_ps``) come from ``AABackend``; the rest come from ``IMDConfig``
    / ``AABackend.build_system`` (see ``constants.py``). ``integrator``
    defaults to ``langevin_middle`` (BAOAB) for a wider stability margin under
    user-injected pull forces.

    ``implicit_solvent`` defaults to ``"auto"`` = GBn2 generalized-Born solvent
    (NoCutoff). Pass ``implicit_solvent="none"`` for the old in-vacuo behavior;
    ``pbc=True`` switches to explicit PME/NPT and disables GB.
    """
    global _imd_system, _timer, _timer_minimize
    pdb_path = Path(pdb_file)
    ligand_path = Path(ligand_sdf) if ligand_sdf else None

    backend = AABackend()
    aa_kwargs: dict = {
        "ligand_sdf": ligand_path,
        "pbc": _as_bool(pbc),
        "work_dir": Path(work_dir) if work_dir else None,
    }
    if nonbonded_cutoff_nm is not None:
        aa_kwargs["nonbonded_cutoff_nm"] = _as_float(nonbonded_cutoff_nm)
    if box_padding_nm is not None:
        aa_kwargs["box_padding_nm"] = _as_float(box_padding_nm)
    if ph is not None:
        aa_kwargs["ph"] = _as_float(ph)
    if gaff_version is not None:
        aa_kwargs["gaff_version"] = gaff_version
    if implicit_solvent is not None:
        aa_kwargs["implicit_solvent"] = str(implicit_solvent)
    bundle = backend.build_system(pdb_path, **aa_kwargs)
    obj_name = _load_prepared_into_pymol(
        bundle.prepared_pdb_path, pdb_path.stem, backend_name=bundle.backend_name
    )
    imd_kwargs = _build_imd_kwargs(
        backend,
        temperature=temperature,
        nsteps_per_frame=nsteps_per_frame,
        const_k=const_k,
        mouse_force=mouse_force,
        integrator=integrator,
        timestep_ps=timestep_ps,
        friction=friction,
        pressure=pressure,
        max_force_kj=max_force_kj,
        energy_cutoff_kj=energy_cutoff_kj,
        frame_interval_ms=frame_interval_ms,
        drag_level=drag_level,
    )
    _imd_system = _make_imd(bundle, obj_name, **imd_kwargs)
    _timer = _make_qtimer()
    _timer_minimize = _make_qtimer()
    print(f"imd_setup_aa: ready (object={obj_name}, atoms={_imd_system.n_atoms}, pbc={bundle.pbc})")


def imd_setup_cg(
    pdb_file: str,
    pbc: object = False,
    ff: str | None = None,
    elastic_network: object = True,
    temperature: object = None,
    nsteps_per_frame: object = None,
    const_k: object = None,
    mouse_force: object = None,
    integrator: object = None,
    timestep_ps: object = None,
    friction: object = None,
    pressure: object = None,
    max_force_kj: object | None = None,
    energy_cutoff_kj: object | None = None,
    frame_interval_ms: object | None = None,
    nonbonded_cutoff_nm: object | None = None,
    box_padding_nm: object | None = None,
    maxwarn: object | None = None,
    drag_level: str = constants.DEFAULT_DRAG_LEVEL,
    work_dir: str | None = None,
) -> None:
    """Set up a Martini CG IMD simulation via vermouth + martini_openmm.

    Every unset knob falls back to its single-source default (``ff`` →
    ``CGBackend.MARTINI_FF``; ``const_k`` / ``integrator`` / ``timestep_ps``
    → ``CGBackend`` recommendations; the rest → ``IMDConfig`` /
    ``CGBackend.build_system``; all defined in ``constants.py``).

    ``timestep_ps`` defaults to the Martini 3 standard (20 fs). CG beads have
    no fast X-H modes, so the long timestep is safe and makes CG feel ~10x
    faster than AA per wall-clock second. Override if you observe instability.
    """
    global _imd_system, _timer, _timer_minimize
    pdb_path = Path(pdb_file)

    cg_kwargs: dict = {
        "pbc": _as_bool(pbc),
        "elastic_network": _as_bool(elastic_network),
    }
    if ff is not None:
        cg_kwargs["ff"] = ff
    if work_dir:
        cg_kwargs["work_dir"] = Path(work_dir)
    if nonbonded_cutoff_nm is not None:
        cg_kwargs["nonbonded_cutoff_nm"] = _as_float(nonbonded_cutoff_nm)
    if box_padding_nm is not None:
        cg_kwargs["box_padding_nm"] = _as_float(box_padding_nm)
    if maxwarn is not None:
        cg_kwargs["maxwarn"] = str(maxwarn)
    backend = CGBackend()
    bundle = backend.build_system(pdb_path, **cg_kwargs)
    obj_name = _load_prepared_into_pymol(
        bundle.prepared_pdb_path, pdb_path.stem, backend_name=bundle.backend_name
    )
    imd_kwargs = _build_imd_kwargs(
        backend,
        temperature=temperature,
        nsteps_per_frame=nsteps_per_frame,
        const_k=const_k,
        mouse_force=mouse_force,
        integrator=integrator,
        timestep_ps=timestep_ps,
        friction=friction,
        pressure=pressure,
        max_force_kj=max_force_kj,
        energy_cutoff_kj=energy_cutoff_kj,
        frame_interval_ms=frame_interval_ms,
        drag_level=drag_level,
    )
    _imd_system = _make_imd(bundle, obj_name, **imd_kwargs)
    _timer = _make_qtimer()
    _timer_minimize = _make_qtimer()
    print(
        f"imd_setup_cg: ready (object={obj_name}, atoms={_imd_system.n_atoms}, "
        f"dt={imd_kwargs['timestep_ps'] * 1000:.1f}fs, pbc={bundle.pbc})"
    )


def imd_setup_mlmm(
    pdb_file: str,
    ml_selection: str | None = None,
    ml_backend: str = constants.MLMM_DEFAULT_BACKEND,
    ml_device: str = constants.MLMM_DEFAULT_DEVICE,
    ml_charge: object = constants.MLMM_DEFAULT_CHARGE,
    ml_mult: object = constants.MLMM_DEFAULT_MULT,
    pbc: object = False,
    box_padding_nm: object | None = None,
    temperature: object = None,
    nsteps_per_frame: object = None,
    const_k: object = None,
    mouse_force: object = None,
    integrator: object = None,
    link_atom_method: str = constants.MLMM_DEFAULT_LINK_ATOM_METHOD,
    max_force_kj: object | None = None,
    energy_cutoff_kj: object | None = None,
    frame_interval_ms: object | None = None,
    nonbonded_cutoff_nm: object | None = None,
    friction: object | None = None,
    pressure: object | None = None,
    timestep_ps: object = None,
    ml_compile: object = False,
    enable_cueq: object = False,
    implicit_solvent: str | None = None,
    drag_level: str = constants.DEFAULT_DRAG_LEVEL,
    work_dir: str | None = None,
) -> None:
    """Set up an ONIOM-style ML/MM IMD simulation (GBn2 implicit solvent by default).

    Unset knobs fall back to single-source defaults: ``const_k`` /
    ``integrator`` / ``timestep_ps`` from ``MLMMBackend``; ``ml_backend`` /
    ``ml_device`` / ``link_atom_method`` and the rest from ``constants.py``.

    ``timestep_ps`` defaults to 0.5 fs because the ML region carries no
    constraints (the MLIP and OpenMM constraint framework can't share state),
    so C-H bond vibrations (~10 fs period) must be resolved directly. AA-like
    2 fs steps are not safe here and tend to drift / blow up under MLIP
    numerical noise. Override only if you know the ML region has no fast
    modes (e.g. all-heavy-atom selection).

    Parameters
    ----------
    ml_selection :
        REQUIRED. Selection of atoms to treat with the MLIP. Examples:
        ``"all"`` (entire system, slow on large proteins), ``"resi 30-50"``,
        ``"chain A"``, ``"resn LIG"``, ``"0,1,2,3"``. Omitting this raises
        an error to prevent accidental whole-system MLIP simulations.
    ml_backend :
        One of ``uma`` / ``orb`` / ``mace`` / ``aimnet2``. Defaults to
        AIMNet2 — smallest model (~10 MB, bundled), fastest CPU inference,
        works out of the box. UMA needs HF auth.
    ml_device :
        ``auto`` (cuda → mps → cpu), or explicit ``cuda``/``mps``/``cpu``.
    link_atom_method :
        ``"scaled"`` (default, Morokuma/Dapprich g-factor) places the link H
        at ``r_Q + g*(r_M - r_Q)`` so it stretches with the broken bond;
        ``"fixed"`` keeps the Q-H distance at a tabulated value (1.09 Å for
        C, 1.01 Å for N, …) using the IMOMM projector Jacobian; ``"none"``
        disables link-atom handling (only safe for molecularly closed
        selections).
    ml_compile :
        Opt-in ``torch.compile`` for the MLIP (AIMNet2 today). Numerically
        ~identical but not bitwise, so the default is False. Biggest speedup
        on CUDA; smaller on CPU.
    enable_cueq :
        Opt-in cuEquivariance acceleration for MACE (CUDA only; needs the
        cuequivariance packages via ``pixi run install-cueq``). Safe to leave
        off elsewhere — MACE logs it as disabled. Do not combine with
        ``ml_compile``.
    implicit_solvent :
        GB implicit-solvent model for the MM terms. Defaults to ``"auto"`` =
        GBn2 (NoCutoff); ``"none"`` runs in vacuo. GB is applied to BOTH the
        full system and the subtracted MM subset, so the ONIOM subtraction
        stays consistent. ``pbc=True`` switches to PME/NPT and disables GB.
    """
    global _imd_system, _timer, _timer_minimize
    if ml_selection is None:
        raise ValueError(
            "imd_setup_mlmm requires ml_selection. Selection language is PyMOL's "
            "(full syntax: and / or / not / within / byres / resi / resn / "
            "chain / name). Examples:\n"
            "  imd_setup_mlmm input.pdb ml_selection='resi 30-50'\n"
            "  imd_setup_mlmm input.pdb ml_selection='resi 1 and resn MET'\n"
            "  imd_setup_mlmm input.pdb ml_selection='chain A and not resn HOH'\n"
            "  imd_setup_mlmm input.pdb ml_selection='byres (resn LIG) within 5 of chain A'\n"
            "  imd_setup_mlmm input.pdb ml_selection='all'  # whole system (slow)"
        )
    pdb_path = Path(pdb_file)

    backend = MLMMBackend()
    mlmm_kwargs: dict = {
        "pbc": _as_bool(pbc),
        "ml_selection": ml_selection,
        "ml_backend": ml_backend,
        "ml_device": ml_device,
        "ml_charge": _as_int(ml_charge),
        "ml_mult": _as_int(ml_mult),
        "ml_compile": _as_bool(ml_compile),
        "enable_cueq": _as_bool(enable_cueq),
        "link_atom_method": link_atom_method,
        "work_dir": Path(work_dir) if work_dir else None,
    }
    if nonbonded_cutoff_nm is not None:
        mlmm_kwargs["nonbonded_cutoff_nm"] = _as_float(nonbonded_cutoff_nm)
    if box_padding_nm is not None:
        mlmm_kwargs["box_padding_nm"] = _as_float(box_padding_nm)
    if implicit_solvent is not None:
        mlmm_kwargs["implicit_solvent"] = str(implicit_solvent)
    bundle = backend.build_system(pdb_path, **mlmm_kwargs)
    obj_name = _load_prepared_into_pymol(
        bundle.prepared_pdb_path, pdb_path.stem, backend_name=bundle.backend_name
    )
    ml_indices = bundle.extra["ml_indices"] if bundle.extra else ()
    ml_sel_name = _highlight_ml_region(obj_name, ml_indices) if ml_indices else None
    imd_kwargs = _build_imd_kwargs(
        backend,
        temperature=temperature,
        nsteps_per_frame=nsteps_per_frame,
        const_k=const_k,
        mouse_force=mouse_force,
        integrator=integrator,
        timestep_ps=timestep_ps,
        friction=friction,
        pressure=pressure,
        max_force_kj=max_force_kj,
        energy_cutoff_kj=energy_cutoff_kj,
        frame_interval_ms=frame_interval_ms,
        drag_level=drag_level,
    )
    _imd_system = _make_imd(bundle, obj_name, **imd_kwargs)
    _timer = _make_qtimer()
    _timer_minimize = _make_qtimer()
    n_ml = len(ml_indices)
    n_link = len(bundle.extra.get("link_atoms", ())) if bundle.extra else 0
    sel_msg = f", ml_sel={ml_sel_name}" if ml_sel_name else ""
    link_msg = f", link_atoms={n_link}({link_atom_method})" if n_link else ""
    print(
        f"imd_setup_mlmm: ready (object={obj_name}, atoms={_imd_system.n_atoms}, "
        f"ml_atoms={n_ml}, ml_backend={ml_backend}, "
        f"dt={imd_kwargs['timestep_ps'] * 1000:.2f}fs, "
        f"pbc={bundle.pbc}{link_msg}{sel_msg})"
    )


# ----------------------------------------------------------------- imd_* family
def imd_record(
    traj: object | None = None,
    traj_interval: object = constants.DEFAULT_TRAJ_INTERVAL,
    force_csv: object | None = None,
    force_csv_interval: object = constants.DEFAULT_FORCE_CSV_INTERVAL,
    state_csv: object | None = None,
    state_csv_interval: object = constants.DEFAULT_STATE_CSV_INTERVAL,
) -> None:
    """Arm trajectory / pull-force / state recording for the next ``imd_run``.

    Recording does not start immediately: it is configured here and begins
    when ``imd_run`` is called, stopping (files flushed + closed) on
    ``imd_stop`` or on divergence / error. This deferral lets ``imd_stop`` ->
    ``imd_run`` re-record cleanly and lets the trajectory interval pick up the
    current ``nsteps_per_frame``.

    Parameters
    ----------
    traj:
        Output trajectory path. Format is chosen by extension: ``.xtc`` /
        ``.dcd`` / ``.pdb``. Pass ``"off"`` (or call with no paths) to disable
        all recording.
    traj_interval:
        Write every N GUI frames to the trajectory (default 1).
    force_csv:
        Output CSV path for the user pull forces. Columns:
        ``frame,step,time_ps,atom_index,fx,fy,fz,fmag``. Only dragged atoms
        are written (one row each per recorded frame). ``atom_index`` is
        0-based (OpenMM / trajectory convention); add 1 for a PyMOL selection.
    force_csv_interval:
        Write forces every N GUI frames (default 1).
    state_csv:
        Output CSV path for thermodynamic state. Columns:
        ``frame,step,time_ps,potential_energy_kj,kinetic_energy_kj,
        total_energy_kj,temperature_k`` plus ``volume_nm3,density_g_cm3`` under
        PBC. One row per recorded frame. Pressure is intentionally omitted
        (OpenMM has no instantaneous pressure; IMD pull forces contaminate the
        virial).
    state_csv_interval:
        Write the state CSV every N GUI frames (default 1).

    Examples::

        imd_record traj=traj.xtc, force_csv=forces.csv, state_csv=state.csv
        imd_record traj=run.pdb, traj_interval=10
        imd_record off                       # disable recording
    """
    if _imd_system is None:
        print("imd: no system loaded — call imd_setup_aa / imd_setup_cg / imd_setup_mlmm first")
        return
    # "off" (or no paths at all) clears recording.
    if traj == "off" or (traj is None and force_csv is None and state_csv is None):
        _imd_system.set_recording()
        print("imd_record: recording disabled")
        return
    _imd_system.set_recording(
        traj=str(traj) if traj is not None else None,
        traj_interval=_as_int(traj_interval),
        force_csv=str(force_csv) if force_csv is not None else None,
        force_csv_interval=_as_int(force_csv_interval),
        state_csv=str(state_csv) if state_csv is not None else None,
        state_csv_interval=_as_int(state_csv_interval),
    )
    parts = []
    if traj is not None:
        parts.append(f"traj={traj} (every {_as_int(traj_interval)} frame[s])")
    if force_csv is not None:
        parts.append(f"force_csv={force_csv} (every {_as_int(force_csv_interval)} frame[s])")
    if state_csv is not None:
        parts.append(f"state_csv={state_csv} (every {_as_int(state_csv_interval)} frame[s])")
    print(f"imd_record: armed — {', '.join(parts)}. Starts on imd_run, stops on imd_stop.")


def imd_minimize(
    max_iter_per_chunk: object | None = None,
    max_chunks: object | None = None,
    tol_kj: object | None = None,
) -> None:
    """Run energy minimization on the Qt event loop.

    Stops when either:
      * two consecutive chunks differ in potential energy by less than
        ``tol_kj`` (the convergence path), or
      * ``max_chunks`` chunks have run without converging (the cap path).

    Parameters
    ----------
    max_iter_per_chunk:
        OpenMM minimizer iterations per Qt tick. Larger values make each
        tick heavier but reduce Python overhead.
    max_chunks:
        Hard upper bound on the number of chunks before a forced stop.
    tol_kj:
        Energy-change convergence tolerance (kJ/mol).

    Each argument falls back to its ``constants.py`` default when omitted.
    """
    if _imd_system is None:
        print("imd: no system loaded — call imd_setup_aa / imd_setup_cg / imd_setup_mlmm first")
        return
    if _imd_system.running:
        print("imd: already running — call imd_stop first")
        return
    max_iter = (
        _as_int(max_iter_per_chunk)
        if max_iter_per_chunk is not None
        else constants.DEFAULT_MIN_MAX_ITER_PER_CHUNK
    )
    chunk_cap = _as_int(max_chunks) if max_chunks is not None else constants.DEFAULT_MIN_MAX_CHUNKS
    tol = _as_float(tol_kj) if tol_kj is not None else constants.DEFAULT_MIN_ENERGY_TOL_KJ
    _imd_system.running = True

    # Wrap counter in a list so the closure can rebind without `nonlocal`.
    chunks_done = [0]

    def _tick():
        chunks_done[0] += 1
        converged = _imd_system.minimize_step(max_iterations=max_iter, tol_kj=tol)
        if converged:
            _timer_minimize.stop()
            _imd_system.running = False
            total_iter = chunks_done[0] * max_iter
            print(f"imd_minimize: converged at chunk {chunks_done[0]} ({total_iter} iter)")
            return
        if chunks_done[0] >= chunk_cap:
            _timer_minimize.stop()
            _imd_system.running = False
            total_iter = chunks_done[0] * max_iter
            print(
                f"imd_minimize: hit max_chunks={chunk_cap} ({total_iter} iter) "
                "without convergence — raise max_chunks or relax tolerance if needed"
            )

    _reset_timer(_timer_minimize)
    _timer_minimize.timeout.connect(_tick)
    _timer_minimize.start(_imd_system.config.frame_interval_ms)
    print(f"imd_minimize: started (max_chunks={chunk_cap}, iter_per_chunk={max_iter})")


def imd_run() -> None:
    """Start the production IMD loop.

    If an ``imd_record`` config is armed, trajectory / force-CSV recording
    starts here and stops on ``imd_stop`` (or automatically on divergence /
    error). With no armed config this behaves exactly as before.
    """
    if _imd_system is None:
        print("imd: no system loaded — call imd_setup_aa / imd_setup_cg / imd_setup_mlmm first")
        return
    if _imd_system.running:
        print("imd: already running")
        return
    _imd_system.running = True
    try:
        _imd_system.start_recording()
    except Exception as exc:
        # Bad trajectory extension / unwritable path etc. — don't start the
        # loop with a half-initialized recorder.
        _imd_system.running = False
        print(f"imd_run: recording setup failed — {exc}")
        return

    def _tick() -> None:
        try:
            _imd_system.step()
        except Exception as exc:
            logger.exception("imd_run: step failed")
            _imd_system.running = False
            print(f"imd_run: stopped due to error — {exc}")
        if not _imd_system.running:
            _timer.stop()
            _imd_system.stop_recording()
            print("imd_run: stopped (divergence or error). Adjust const_k / max_force_kj.")

    _reset_timer(_timer)
    _timer.timeout.connect(_tick)
    _timer.start(_imd_system.config.frame_interval_ms)
    print("imd_run: started")


def _reset_timer(timer) -> None:
    """Stop the timer and drop every previous `timeout` connection.

    Without this, `imd_stop → setup_* → imd_run` accumulates `_tick`
    connections on the same QTimer, so each tick fires `step()` multiple
    times. Qt raises if there's nothing to disconnect, so we swallow that.
    """
    if timer is None:
        return
    timer.stop()
    # Qt raises if there's nothing to disconnect; we don't care.
    with contextlib.suppress(TypeError, RuntimeError):
        timer.timeout.disconnect()


def imd_stop() -> None:
    """Stop minimization and production loops, and close any recording."""
    if _imd_system is None:
        return
    _imd_system.running = False
    if _timer is not None:
        _timer.stop()
    if _timer_minimize is not None:
        _timer_minimize.stop()
    # Flush + close trajectory / CSV (idempotent; safe when not recording).
    _imd_system.stop_recording()
    print("imd_stop: stopped")


def imd_change(
    const_k: object | None = None,
    mouse_force: object | None = None,
    nsteps_per_frame: object | None = None,
    temperature: object | None = None,
    max_force_kj: object | None = None,
    energy_cutoff_kj: object | None = None,
    drag_level: object | None = None,
) -> None:
    """Hot-patch IMD parameters without rebuilding the system.

    Updatable in place: ``const_k``, ``mouse_force``, ``nsteps_per_frame``,
    ``temperature`` (live Langevin integrator update), ``max_force_kj``,
    ``energy_cutoff_kj``, ``drag_level`` (atom / residue / chain, applied live).
    Other ``IMDConfig`` fields (``friction``,
    ``timestep_ps``, ``pressure``, ``integrator``) need a fresh ``setup_*``
    because they require rebuilding the integrator or the barostat;
    ``frame_interval_ms`` needs a stop/run cycle (it is read at timer start).

    Note: changing ``nsteps_per_frame`` while a recording is active does NOT
    retune the trajectory reporter's interval (it is fixed at ``imd_run``
    time). Run ``imd_stop`` then ``imd_run`` again to resync the trajectory
    sampling cadence.
    """
    if _imd_system is None:
        print("imd: no system loaded")
        return
    cfg = _imd_system.config
    if const_k is not None:
        cfg.const_k = _as_float(const_k)
    if mouse_force is not None:
        cfg.mouse_force = _as_bool(mouse_force)
    if nsteps_per_frame is not None:
        cfg.nsteps_per_frame = _as_int(nsteps_per_frame)
    if temperature is not None:
        cfg.temperature = _as_float(temperature)
        # Langevin integrator temperature can be updated live (same API on
        # both LangevinIntegrator and LangevinMiddleIntegrator).
        _imd_system.simulation.integrator.setTemperature(cfg.temperature)
    if max_force_kj is not None:
        cfg.max_force_kj = _as_float(max_force_kj)
    if energy_cutoff_kj is not None:
        cfg.energy_cutoff_kj = _as_float(energy_cutoff_kj)
    if drag_level is not None:
        # Applied live: IMDSystem reads cfg.drag_level each frame and expands the
        # pull to the whole residue / chain. No PyMOL call needed — its editing
        # drag ignores mouse_selection_mode.
        level = str(drag_level)
        if level not in constants.SUPPORTED_DRAG_LEVELS:
            print(
                f"imd_change: unknown drag_level {level!r}; "
                f"choose from {constants.SUPPORTED_DRAG_LEVELS}"
            )
        else:
            cfg.drag_level = level
    print(f"imd_change: {cfg}")


def imd_drag_level(level: str = constants.DEFAULT_DRAG_LEVEL) -> None:
    """Switch the mouse-drag granularity live: atom | residue | chain.

    Takes effect on the next frame: ``IMDSystem`` (or the socket client) expands
    each pull to the whole residue / chain of the dragged atoms. PyMOL's editing
    drag only moves a covalent fragment and ignores ``mouse_selection_mode``, so
    the expansion is done plugin-side rather than via a PyMOL setting.
    """
    try:
        level = _validate_drag_level(level)
    except ValueError as exc:
        print(f"imd_drag_level: {exc}")
        return
    applied = False
    if _imd_system is not None:
        _imd_system.config.drag_level = level
        applied = True
    if _imd_client is not None:
        _imd_client.drag_level = level
        applied = True
    if not applied:
        print("imd_drag_level: no system loaded — call imd_setup_* or imd_connect first")
        return
    print(f"imd_drag_level: drag granularity set to {level} (applies next frame)")


# ------------------------------------------------------------- socket client
def _default_imd_object(cmd) -> str | None:
    """Most recently loaded object (the usual target after loading a PDB)."""
    objs = cmd.get_object_list()
    return objs[-1] if objs else None


def _await_first_frame(client, timeout: float = 5.0):
    """Block briefly for the first coords frame so :func:`imd_connect` can
    validate the atom count before starting the viewer timer."""
    end = time.time() + timeout
    while time.time() < end:
        frame = client.poll_frame()
        if frame is not None and frame.positions_ang is not None:
            return frame
        time.sleep(0.01)
    return None


def imd_connect(
    host: str = constants.IMD_DEFAULT_HOST,
    port: object = constants.IMD_DEFAULT_PORT,
    object: str | None = None,
    const_k: object | None = None,
    max_force_kj: object | None = None,
    frame_interval_ms: object | None = None,
    drag_level: str = constants.DEFAULT_DRAG_LEVEL,
) -> None:
    """Connect this PyMOL to a running IMDv3 server and stream interactively.

    The MD runs on the server (e.g. ``pymol_imd serve ...``); this PyMOL
    displays each frame and turns your mouse-drag into forces, so you grab
    atoms exactly like the in-process mode. Load the server's PREPARED PDB
    first (same atom order) — its path is printed by the server as
    ``IMD_READY ... prepared=<path>``.

    Parameters
    ----------
    host, port:
        Server address. Defaults to 127.0.0.1:8888.
    object:
        PyMOL object whose atoms mirror the server topology. Defaults to the
        most recently loaded object.
    const_k, max_force_kj:
        Client-side drag sensitivity (kJ/mol/nm) and per-atom |F| cap. When
        omitted, ``const_k`` defaults to the per-backend recommendation (CG
        10000, AA / ML-MM 50000 — CG is auto-detected from the ``BB`` bead
        naming) so the pull feels identical to the same-process ``imd_setup_*``
        path; ``max_force_kj`` defaults to the global cap. The force is computed
        entirely client-side, so the server's own ``const_k`` (e.g.
        ``serve --const-k``) has NO effect — set it here instead.
    """
    global _imd_client
    try:
        from pymol import cmd
    except ImportError:
        print("imd_connect: PyMOL not available")
        return
    if _imd_client is not None:
        imd_disconnect()
    # imd_connect bypasses _build_imd_kwargs, so validate drag_level here too
    # (same single explicit error as the imd_setup_* path).
    try:
        drag_level = _validate_drag_level(str(drag_level))
    except ValueError as exc:
        print(f"imd_connect: {exc}")
        return
    cmd.set("retain_order", 1)
    obj = object or _default_imd_object(cmd)
    if obj is None:
        print("imd_connect: no object loaded — load the server's prepared PDB first")
        return

    client = IMDClient(host, _as_int(port))
    try:
        client.connect()
    except OSError as exc:
        print(f"imd_connect: cannot reach {host}:{port} — {exc}")
        return
    frame = _await_first_frame(client)
    if frame is None:
        print(f"imd_connect: connected to {host}:{port} but no frame arrived")
        client.disconnect()
        return
    n_server = frame.positions_ang.shape[0]
    n_pymol = cmd.count_atoms(f"({obj})")
    if n_pymol != n_server:
        print(
            f"imd_connect: atom mismatch — PyMOL object '{obj}' has {n_pymol} "
            f"atoms but the server has {n_server}. Load the server's prepared "
            f"PDB (printed as 'IMD_READY ... prepared=<path>')."
        )
        client.disconnect()
        return

    # The server's backend isn't carried over the IMD wire, so detect CG from
    # its bead naming (BB backbone bead). Used for BOTH the default pull-force
    # scale and the representation, so a connect with no const_k matches the
    # same-process imd_setup_* default for that backend (CG beads want a smaller
    # const_k than AA atoms: 10000 vs 50000; AA and ML-MM both use 50000, and
    # ML-MM has no BB beads so it falls into the AA branch correctly).
    backend_name = "cg" if cmd.count_atoms(f"({obj}) and name BB") > 0 else "aa"
    default_const_k = (
        constants.CG_RECOMMENDED_CONST_K
        if backend_name == "cg"
        else constants.AA_RECOMMENDED_CONST_K
    )
    ck = _as_float(const_k) if const_k is not None else default_const_k
    mf = _as_float(max_force_kj) if max_force_kj is not None else constants.DEFAULT_MAX_FORCE_KJ
    fi = (
        _as_int(frame_interval_ms)
        if frame_interval_ms is not None
        else constants.IMD_CLIENT_FRAME_INTERVAL_MS
    )
    pclient = PyMOLIMDClient(
        client, obj, const_k=ck, max_force_kj=mf, frame_interval_ms=fi, drag_level=drag_level
    )
    # Show the first frame right away, then start the streaming timer.
    pclient._display(frame.positions_ang)
    pclient._last_server_ang = frame.positions_ang
    # Mirror the in-process setup view + 3-Button Editing (CG beads vs cartoon).
    _apply_imd_representation(cmd, obj, backend_name)
    pclient.start()
    _imd_client = pclient
    print(f"imd_connect: streaming from {host}:{port} on '{obj}' ({n_server} atoms)")


def imd_disconnect() -> None:
    """Stop the socket client viewer and disconnect from the server."""
    global _imd_client
    if _imd_client is None:
        print("imd: no active socket client")
        return
    _imd_client.stop()
    _imd_client = None
    print("imd_disconnect: stopped")


# ----------------------------------------------------------- pymol registration
def _register() -> None:
    try:
        from pymol import cmd
    except ImportError:
        logger.debug("pymol not importable; skipping cmd.extend registration")
        return
    cmd.extend("imd_setup_aa", imd_setup_aa)
    cmd.extend("imd_setup_cg", imd_setup_cg)
    cmd.extend("imd_setup_mlmm", imd_setup_mlmm)
    cmd.extend("imd_record", imd_record)
    cmd.extend("imd_minimize", imd_minimize)
    cmd.extend("imd_run", imd_run)
    cmd.extend("imd_stop", imd_stop)
    cmd.extend("imd_change", imd_change)
    cmd.extend("imd_drag_level", imd_drag_level)
    cmd.extend("imd_connect", imd_connect)
    cmd.extend("imd_disconnect", imd_disconnect)


_register()
