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
from pathlib import Path

from .backends.aa import AABackend
from .backends.cg import CGBackend
from .backends.mlmm import MLMMBackend
from .imd import IMDConfig, IMDSystem

logger = logging.getLogger(__name__)

# Module-global IMD state. PyMOL commands are free functions, so the system is
# kept here and shared across them.
_imd_system: IMDSystem | None = None
_timer = None  # type: ignore[var-annotated]
_timer_minimize = None  # type: ignore[var-annotated]


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

    # Backend-aware default representation. Users can always override later.
    if backend_name == "cg":
        # Spheres for the beads + sticks for the bonds (CONECT comes from
        # martinize2's cg.pdb directly so only chemical bonds are drawn).
        cmd.show_as("spheres", candidate)
        cmd.show("sticks", candidate)
        cmd.set("sphere_scale", 0.4, candidate)
        cmd.set("stick_radius", 0.25, candidate)
        # Martini bead types aren't standard elements, so PyMOL paints them
        # with arbitrary fallback colors. Override: backbone in skyblue, the
        # rest (side-chain beads) in yelloworange for a clean CG look.
        cmd.color("skyblue", f"{candidate} and name BB")
        cmd.color("yelloworange", f"{candidate} and not name BB")
    else:  # aa, mlmm
        cmd.show_as("cartoon", candidate)
        # Render any non-polymer atoms (ligands, ions left in) as sticks too.
        cmd.show("sticks", f"{candidate} and not polymer")
    cmd.zoom(candidate, buffer=2.0)
    # Drop into Editing mode so the user can immediately grab atoms with
    # Ctrl+left-drag and feed the pull-force mechanism without first
    # toggling Mouse > 3-Button Editing from the menu.
    cmd.config_mouse("three_button_editing")
    return candidate


def _make_qtimer():
    from pymol.Qt import QtCore

    return QtCore.QTimer()


# --------------------------------------------------------------- setup_* family
def setup_aa(
    pdb_file: str,
    ligand_sdf: str | None = None,
    pbc: object = False,
    temperature: object = 300.0,
    nsteps_per_frame: object = 1,
    const_k: object = None,
    mouse_force: object = True,
) -> None:
    """Set up an all-atom IMD simulation in vacuo (NVT) by default.

    ``const_k`` falls back to ``AABackend.recommended_const_k`` (50000) when
    omitted; the backend owns its own recommended scale.
    """
    global _imd_system, _timer, _timer_minimize
    pdb_path = Path(pdb_file)
    ligand_path = Path(ligand_sdf) if ligand_sdf else None

    backend = AABackend()
    bundle = backend.build_system(
        pdb_path,
        ligand_sdf=ligand_path,
        pbc=_as_bool(pbc),
    )
    obj_name = _load_prepared_into_pymol(
        bundle.prepared_pdb_path, pdb_path.stem, backend_name=bundle.backend_name
    )
    _imd_system = _make_imd(
        bundle,
        obj_name,
        temperature=_as_float(temperature),
        nsteps_per_frame=_as_int(nsteps_per_frame),
        const_k=_resolve_const_k(backend, const_k),
        mouse_force=_as_bool(mouse_force),
    )
    _timer = _make_qtimer()
    _timer_minimize = _make_qtimer()
    print(f"setup_aa: ready (object={obj_name}, atoms={_imd_system.n_atoms}, pbc={bundle.pbc})")


def setup_cg(
    pdb_file: str,
    pbc: object = False,
    ff: str | None = None,
    elastic_network: object = True,
    temperature: object = 300.0,
    nsteps_per_frame: object = 1,
    const_k: object = None,
    mouse_force: object = True,
    timestep_ps: object = 0.020,
) -> None:
    """Set up a Martini CG IMD simulation via vermouth + martini_openmm.

    Backend-owned defaults: ``ff=None`` → ``CGBackend.MARTINI_FF`` (martini3001),
    ``const_k=None`` → ``CGBackend.recommended_const_k`` (10000).

    ``timestep_ps`` defaults to 20 fs — the Martini 3 standard. CG beads
    have no fast X-H modes, so the long timestep is safe and makes CG
    actually feel ~10x faster than AA per wall-clock second. Override with
    a smaller value if you observe instability.
    """
    global _imd_system, _timer, _timer_minimize
    pdb_path = Path(pdb_file)

    cg_kwargs: dict = {
        "pbc": _as_bool(pbc),
        "elastic_network": _as_bool(elastic_network),
    }
    if ff is not None:
        cg_kwargs["ff"] = ff
    backend = CGBackend()
    bundle = backend.build_system(pdb_path, **cg_kwargs)
    obj_name = _load_prepared_into_pymol(
        bundle.prepared_pdb_path, pdb_path.stem, backend_name=bundle.backend_name
    )
    _imd_system = _make_imd(
        bundle,
        obj_name,
        temperature=_as_float(temperature),
        nsteps_per_frame=_as_int(nsteps_per_frame),
        const_k=_resolve_const_k(backend, const_k),
        mouse_force=_as_bool(mouse_force),
        timestep_ps=_as_float(timestep_ps),
    )
    _timer = _make_qtimer()
    _timer_minimize = _make_qtimer()
    print(
        f"setup_cg: ready (object={obj_name}, atoms={_imd_system.n_atoms}, "
        f"dt={_as_float(timestep_ps) * 1000:.1f}fs, pbc={bundle.pbc})"
    )


def setup_mlmm(
    pdb_file: str,
    ml_selection: str | None = None,
    ml_backend: str = "aimnet2",
    ml_device: str = "auto",
    ml_charge: object = 0,
    ml_mult: object = 1,
    pbc: object = False,
    temperature: object = 300.0,
    nsteps_per_frame: object = 1,
    const_k: object = None,
    mouse_force: object = True,
) -> None:
    """Set up an ONIOM-style ML/MM IMD simulation.

    ``const_k=None`` falls back to ``MLMMBackend.recommended_const_k``.

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
    """
    global _imd_system, _timer, _timer_minimize
    if ml_selection is None:
        raise ValueError(
            "setup_mlmm requires ml_selection. Selection language is PyMOL's "
            "(full syntax: and / or / not / within / byres / resi / resn / "
            "chain / name). Examples:\n"
            "  setup_mlmm input.pdb ml_selection='resi 30-50'\n"
            "  setup_mlmm input.pdb ml_selection='resi 1 and resn MET'\n"
            "  setup_mlmm input.pdb ml_selection='chain A and not resn HOH'\n"
            "  setup_mlmm input.pdb ml_selection='byres (resn LIG) within 5 of chain A'\n"
            "  setup_mlmm input.pdb ml_selection='all'  # whole system (slow)"
        )
    pdb_path = Path(pdb_file)

    backend = MLMMBackend()
    bundle = backend.build_system(
        pdb_path,
        pbc=_as_bool(pbc),
        ml_selection=ml_selection,
        ml_backend=ml_backend,
        ml_device=ml_device,
        ml_charge=_as_int(ml_charge),
        ml_mult=_as_int(ml_mult),
    )
    obj_name = _load_prepared_into_pymol(
        bundle.prepared_pdb_path, pdb_path.stem, backend_name=bundle.backend_name
    )
    _imd_system = _make_imd(
        bundle,
        obj_name,
        temperature=_as_float(temperature),
        nsteps_per_frame=_as_int(nsteps_per_frame),
        const_k=_resolve_const_k(backend, const_k),
        mouse_force=_as_bool(mouse_force),
    )
    _timer = _make_qtimer()
    _timer_minimize = _make_qtimer()
    n_ml = len(bundle.extra["ml_indices"]) if bundle.extra else 0
    print(
        f"setup_mlmm: ready (object={obj_name}, atoms={_imd_system.n_atoms}, "
        f"ml_atoms={n_ml}, ml_backend={ml_backend}, pbc={bundle.pbc})"
    )


# ----------------------------------------------------------------- imd_* family
def imd_minimize(
    max_iter_per_chunk: object = 50,
    max_chunks: object = 200,
) -> None:
    """Run energy minimization on the Qt event loop.

    Stops when either:
      * two consecutive chunks differ in potential energy by less than
        ``1e-3 kJ/mol`` (the convergence path), or
      * ``max_chunks`` chunks have run without converging (the cap path).

    Parameters
    ----------
    max_iter_per_chunk:
        OpenMM minimizer iterations per Qt tick. Larger values make each
        tick heavier but reduce Python overhead.
    max_chunks:
        Hard upper bound on the number of chunks. Default 200 → up to
        ``200 * 50 = 10000`` minimization iterations before forced stop.
    """
    if _imd_system is None:
        print("imd: no system loaded — call setup_aa / setup_cg / setup_mlmm first")
        return
    if _imd_system.running:
        print("imd: already running — call imd_stop first")
        return
    max_iter = _as_int(max_iter_per_chunk)
    chunk_cap = _as_int(max_chunks)
    _imd_system.running = True

    # Wrap counter in a list so the closure can rebind without `nonlocal`.
    chunks_done = [0]

    def _tick():
        chunks_done[0] += 1
        converged = _imd_system.minimize_step(max_iterations=max_iter)
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
    _timer_minimize.start(1)
    print(f"imd_minimize: started (max_chunks={chunk_cap}, iter_per_chunk={max_iter})")


def imd_run() -> None:
    """Start the production IMD loop."""
    if _imd_system is None:
        print("imd: no system loaded — call setup_aa / setup_cg / setup_mlmm first")
        return
    if _imd_system.running:
        print("imd: already running")
        return
    _imd_system.running = True

    def _tick() -> None:
        try:
            _imd_system.step()
        except Exception as exc:
            logger.exception("imd_run: step failed")
            _imd_system.running = False
            print(f"imd_run: stopped due to error — {exc}")
        if not _imd_system.running:
            _timer.stop()
            print("imd_run: stopped (divergence or error). Adjust const_k / max_pull_nm.")

    _reset_timer(_timer)
    _timer.timeout.connect(_tick)
    _timer.start(1)
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
    """Stop minimization and production loops."""
    if _imd_system is None:
        return
    _imd_system.running = False
    if _timer is not None:
        _timer.stop()
    if _timer_minimize is not None:
        _timer_minimize.stop()
    print("imd_stop: stopped")


def imd_change(
    const_k: object | None = None,
    mouse_force: object | None = None,
    nsteps_per_frame: object | None = None,
    temperature: object | None = None,
    max_pull_nm: object | None = None,
    energy_cutoff_kj: object | None = None,
) -> None:
    """Hot-patch IMD parameters without rebuilding the system.

    Updatable in place: ``const_k``, ``mouse_force``, ``nsteps_per_frame``,
    ``temperature`` (live Langevin integrator update), ``max_pull_nm``,
    ``energy_cutoff_kj``. Other ``IMDConfig`` fields (``friction``,
    ``timestep_ps``, ``pressure``) need a fresh ``setup_*`` because they
    require rebuilding the integrator or the barostat.
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
        # Langevin integrator temperature can be updated live.
        _imd_system.simulation.integrator.setTemperature(cfg.temperature)
    if max_pull_nm is not None:
        cfg.max_pull_nm = _as_float(max_pull_nm)
    if energy_cutoff_kj is not None:
        cfg.energy_cutoff_kj = _as_float(energy_cutoff_kj)
    print(f"imd_change: {cfg}")


# ----------------------------------------------------------- pymol registration
def _register() -> None:
    try:
        from pymol import cmd
    except ImportError:
        logger.debug("pymol not importable; skipping cmd.extend registration")
        return
    cmd.extend("setup_aa", setup_aa)
    cmd.extend("setup_cg", setup_cg)
    cmd.extend("setup_mlmm", setup_mlmm)
    cmd.extend("imd_minimize", imd_minimize)
    cmd.extend("imd_run", imd_run)
    cmd.extend("imd_stop", imd_stop)
    cmd.extend("imd_change", imd_change)


_register()
