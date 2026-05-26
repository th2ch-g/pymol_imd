"""PyMOL command surface for pymol_imd.

This module is imported for its side effects: each ``cmd.extend(...)`` call
registers a PyMOL command. The actual simulation work is delegated to backend
strategies + :class:`IMDSystem`.

PyMOL passes every argument as a string, so we centralize the coercion logic
in :func:`_as_bool` / :func:`_as_int` / :func:`_as_float` and let the backends
work with proper Python types.
"""

from __future__ import annotations

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


def _load_prepared_into_pymol(prepared_pdb_path: Path, base_name: str) -> str:
    """Load the prepared PDB into PyMOL as a NEW object and return its name.

    The original object (e.g. ``1ubq``) is left untouched so the user can
    diff-view the pre/post-prep structures. The new object is named
    ``<base>_imd`` plus a numeric suffix if a previous IMD object exists.

    Returns the actual PyMOL object name assigned.
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
    const_k: object = 5000.0,
    mouse_force: object = True,
) -> None:
    """Set up an all-atom IMD simulation in vacuo (NVT) by default."""
    global _imd_system, _timer, _timer_minimize
    pdb_path = Path(pdb_file)
    ligand_path = Path(ligand_sdf) if ligand_sdf else None

    bundle = AABackend().build_system(
        pdb_path,
        ligand_sdf=ligand_path,
        pbc=_as_bool(pbc),
    )
    obj_name = _load_prepared_into_pymol(bundle.prepared_pdb_path, pdb_path.stem)
    _imd_system = _make_imd(
        bundle,
        obj_name,
        temperature=_as_float(temperature),
        nsteps_per_frame=_as_int(nsteps_per_frame),
        const_k=_as_float(const_k),
        mouse_force=_as_bool(mouse_force),
    )
    _timer = _make_qtimer()
    _timer_minimize = _make_qtimer()
    print(f"setup_aa: ready (object={obj_name}, atoms={_imd_system.n_atoms}, pbc={bundle.pbc})")


def setup_cg(
    pdb_file: str,
    pbc: object = False,
    ff: str = "martini30",
    elastic_network: object = True,
    temperature: object = 300.0,
    nsteps_per_frame: object = 1,
    const_k: object = 5000.0,
    mouse_force: object = True,
) -> None:
    """Set up a Martini CG IMD simulation via vermouth + martini_openmm."""
    global _imd_system, _timer, _timer_minimize
    pdb_path = Path(pdb_file)

    bundle = CGBackend().build_system(
        pdb_path,
        pbc=_as_bool(pbc),
        ff=ff,
        elastic_network=_as_bool(elastic_network),
    )
    obj_name = _load_prepared_into_pymol(bundle.prepared_pdb_path, pdb_path.stem)
    _imd_system = _make_imd(
        bundle,
        obj_name,
        temperature=_as_float(temperature),
        nsteps_per_frame=_as_int(nsteps_per_frame),
        const_k=_as_float(const_k),
        mouse_force=_as_bool(mouse_force),
    )
    _timer = _make_qtimer()
    _timer_minimize = _make_qtimer()
    print(f"setup_cg: ready (object={obj_name}, atoms={_imd_system.n_atoms}, pbc={bundle.pbc})")


def setup_mlmm(
    pdb_file: str,
    ml_selection: str = "all",
    pbc: object = False,
    temperature: object = 300.0,
    nsteps_per_frame: object = 1,
    const_k: object = 5000.0,
    mouse_force: object = True,
) -> None:
    """Set up an ML/MM IMD simulation via mlmm_toolkit (Linux-only)."""
    global _imd_system, _timer, _timer_minimize
    pdb_path = Path(pdb_file)

    bundle = MLMMBackend().build_system(
        pdb_path,
        pbc=_as_bool(pbc),
        ml_selection=ml_selection,
    )
    obj_name = _load_prepared_into_pymol(bundle.prepared_pdb_path, pdb_path.stem)
    _imd_system = _make_imd(
        bundle,
        obj_name,
        temperature=_as_float(temperature),
        nsteps_per_frame=_as_int(nsteps_per_frame),
        const_k=_as_float(const_k),
        mouse_force=_as_bool(mouse_force),
    )
    _timer = _make_qtimer()
    _timer_minimize = _make_qtimer()
    print(f"setup_mlmm: ready (object={obj_name}, atoms={_imd_system.n_atoms}, pbc={bundle.pbc})")


# ----------------------------------------------------------------- imd_* family
def imd_minimize(max_iter_per_chunk: object = 50) -> None:
    """Run energy minimization on the Qt event loop until convergence."""
    if _imd_system is None:
        print("imd: no system loaded — call setup_aa / setup_cg / setup_mlmm first")
        return
    if _imd_system.running:
        print("imd: already running — call imd_stop first")
        return
    max_iter = _as_int(max_iter_per_chunk)
    _imd_system.running = True

    def _tick():
        converged = _imd_system.minimize_step(max_iterations=max_iter)
        if converged:
            _timer_minimize.stop()
            _imd_system.running = False
            print("imd_minimize: converged")

    _timer_minimize.timeout.connect(_tick)
    _timer_minimize.start(1)
    print("imd_minimize: started")


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

    _timer.timeout.connect(_tick)
    _timer.start(1)
    print("imd_run: started")


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
    """Hot-patch IMD parameters without rebuilding the system."""
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
