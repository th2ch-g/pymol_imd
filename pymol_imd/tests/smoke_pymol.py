"""Exercise real PyMOL commands and Qt timers, with optional GUI rendering.

Run with ``pixi run smoke-pymol`` or in a fresh viewer with
``pixi run pymol -k -r pymol_imd/tests/smoke_pymol.py``.
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
import time
import weakref
from pathlib import Path

import numpy as np
from pymol import cmd
from pymol.Qt import QtCore, QtWidgets

from pymol_imd import commands
from pymol_imd.client import PyMOLIMDClient

log = logging.getLogger("smoke-pymol")


def _write_peptide(path: Path) -> None:
    """Write a heavy-atom alanine tripeptide with both termini complete."""
    residues = (
        (
            (-0.677, -1.230, -0.491),
            (-0.001, 0.064, -0.491),
            (1.499, -0.110, -0.491),
            (2.030, -1.227, -0.502),
            (-0.509, 0.856, 0.727),
        ),
        (
            (2.250, 0.939, -0.479),
            (3.700, 0.770, -0.479),
            (4.400, 2.108, -0.464),
            (3.774, 3.175, -0.443),
            (4.071, -0.082, -1.704),
        ),
        (
            (5.689, 2.140, -0.474),
            (6.365, 3.434, -0.460),
            (7.865, 3.260, -0.474),
            (8.396, 2.143, -0.507),
            (5.867, 4.207, 0.773),
            (8.638, 4.226, -0.455),
        ),
    )
    lines = []
    for resi, positions in enumerate(residues, 1):
        for name, (x, y, z) in zip(("N", "CA", "C", "O", "CB", "OXT"), positions, strict=False):
            lines.append(
                f"ATOM  {len(lines) + 1:5d}  {name:<3} ALA A{resi:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {name[0]:>2}  "
            )
    path.write_text("\n".join([*lines, "END", ""]))


def _until(app, condition, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            raise AssertionError("PyMOL/Qt workflow timed out")
        app.processEvents()
        time.sleep(0.005)


def _exercise_backend(app, pdb: Path, level: str, output: Path | None) -> list:
    setup = getattr(commands, f"imd_setup_{level}")
    kwargs = {"platform": "cpu", "nsteps_per_frame": 1, "frame_interval_ms": 5}
    if level == "mlmm":
        kwargs.update(ml_selection="resi 1-2", ml_charge=0, ml_mult=1, ml_device="cpu")
    setup(str(pdb), work_dir=str(pdb.parent / level), **kwargs)
    system = commands._imd_system
    obj = system.pymol_object
    assert cmd.count_atoms(obj) == system.n_atoms
    assert cmd.get_setting_int("retain_order") == 1
    if level != "cg":
        assert cmd.count_atoms(f"{obj} and hydro") > 0
    if level == "mlmm":
        correction = system._mlmm_force
        for i in range(correction.getNumParticles()):
            assert tuple(correction.getParticleParameters(i)[1]) == (0.0, 0.0, 0.0)
        system.apply_mlmm_if_present()
        system.set_pull_forces(np.array([0]), np.ones((system.n_atoms, 3)))
        system.minimize_step(max_iterations=20)
        assert not system._active_pull_idx_prev
        for i in range(correction.getNumParticles()):
            assert tuple(correction.getParticleParameters(i)[1]) == (0.0, 0.0, 0.0)
    paths = [pdb.parent / f"{level}{suffix}" for suffix in (".xtc", "_forces.csv", "_state.csv")]
    commands.imd_record(traj=str(paths[0]), force_csv=str(paths[1]), state_csv=str(paths[2]))
    commands.imd_minimize(max_iter_per_chunk=20, max_chunks=3, run_after=True)
    _until(app, lambda: system.simulation.currentStep >= 3)
    assert commands._timer.isActive() and not commands._timer_minimize.isActive()
    commands.imd_stop()
    assert not system.running and not system._recording
    assert all(path.stat().st_size > 0 for path in paths)

    for level_name, groups in (
        ("atom", np.arange(system.n_atoms)),
        ("residue", system._atom_residue_idx),
        ("chain", system._atom_chain_idx),
    ):
        commands.imd_drag_level(level_name)
        system._sync_positions_to_pymol()
        cmd.alter_state(1, f"{obj} and index 1", "x += 0.1")
        system.step()
        expected = np.flatnonzero(groups == groups[0])
        np.testing.assert_array_equal(system._last_pull_active_idx, expected)
        assert np.all(np.linalg.norm(system._last_pull_forces[expected], axis=1) > 0)
    commands.imd_stop()
    commands.imd_record("off")
    assert not system._active_pull_idx_prev

    positions = system._read_openmm_positions_nm() * 10
    displayed = np.asarray(cmd.get_coords(obj, state=1))
    np.testing.assert_allclose(displayed, positions, atol=1e-5)
    _check_coordset_sync(system)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        other_objects = [name for name in cmd.get_names("objects", enabled_only=1) if name != obj]
        for name in other_objects:
            cmd.disable(name)
        try:
            cmd.png(str(output / f"{level}.png"), width=640, height=480, ray=1, quiet=1)
        finally:
            for name in other_objects:
                cmd.enable(name)
    log.info(
        "%s PyMOL/Qt OK: setup, minimize -> run, stop, all drag levels, coordinate sync", level
    )
    owned = [system, system.simulation.context, commands._timer, commands._timer_minimize]
    if level == "mlmm":
        owned.extend((system._mlip_atoms, system.bundle.extra["ml_calculator"]))
    return [weakref.ref(item) for item in owned]


def _check_coordset_sync(system) -> None:
    obj = system.pymol_object
    legacy = []
    cmd.iterate_state(1, obj, "coords.append((x, y, z))", space={"coords": legacy})
    np.testing.assert_array_equal(system._read_pymol_positions_nm(), np.asarray(legacy) * 0.1)
    other = cmd.get_unused_name("imd_sync_probe")
    cmd.create(other, obj)
    original_other = cmd.get_coordset(other, state=1)
    second_state = cmd.get_coordset(obj, state=1) + 1
    cmd.load_coordset(second_state, obj, state=2)
    try:
        system._sync_positions_to_pymol()
        np.testing.assert_array_equal(cmd.get_coordset(other, state=1), original_other)
        np.testing.assert_array_equal(cmd.get_coordset(obj, state=2), second_state)
        assert cmd.count_states(obj) == 2
    finally:
        cmd.delete(other)
        cmd.delete_states(obj, "2")


def _check_error_cleanup(app) -> None:
    system = commands._imd_system
    obj = system.pymol_object
    commands.imd_run()
    system.set_pull_forces(np.array([0]), np.ones((system.n_atoms, 3)))
    cmd.remove(f"{obj} and index {system.n_atoms}")
    _until(app, lambda: not system.running)
    assert not commands._timer.isActive()
    assert not system._active_pull_idx_prev and not system._recording
    log.info("local object edit stops Qt and releases all pull forces")


def _check_live_temperature(pdb: Path) -> None:
    from openmm import MonteCarloBarostat, unit

    commands.imd_setup_aa(str(pdb), pbc=True, platform="cpu")
    system = commands._imd_system
    commands.imd_change(temperature=350)
    assert system.simulation.integrator.getTemperature().value_in_unit(unit.kelvin) == 350
    assert system.simulation.context.getParameter(MonteCarloBarostat.Temperature()) == 350
    commands.imd_change(temperature=400, nsteps_per_frame=0)
    assert system.config.temperature == 350
    assert system.simulation.context.getParameter(MonteCarloBarostat.Temperature()) == 350
    log.info(
        "live NPT temperature updates both integrator and barostat; invalid updates are atomic"
    )


def _check_socket_cleanup() -> None:
    class Connection:
        closed = False

        def disconnect(self):
            self.closed = True

        def poll_frame(self):
            return None

    for delete_object in (False, True):
        obj = cmd.get_unused_name("imd_socket_probe")
        cmd.pseudoatom(obj, pos=[0, 0, 0])
        cmd.pseudoatom(obj, pos=[1, 0, 0])
        connection = Connection()
        viewer = PyMOLIMDClient(connection, obj, const_k=10000, max_force_kj=5000)
        viewer._last_server_ang = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        viewer._active_prev = {0}
        viewer.start()
        timer = viewer._timer
        if delete_object:
            cmd.delete(obj)
        else:
            cmd.remove(f"{obj} and index 2")
        viewer._tick()
        assert connection.closed and not timer.isActive()
        assert not viewer._active_prev
        cmd.delete(obj)
    log.info("socket object edits stop Qt and disconnect instead of retaining a pull")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    original_objects = set(cmd.get_object_list())
    output = os.environ.get("PYMOL_IMD_TEST_OUTPUT")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "test_peptide.pdb"
            _write_peptide(pdb)
            previous = []
            for level in ("aa", "cg", "mlmm"):
                current = _exercise_backend(app, pdb, level, Path(output) if output else None)
                QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
                gc.collect()
                assert all(ref() is None for ref in previous), (
                    "Previous setup is still retained",
                    [type(ref()).__name__ for ref in previous if ref() is not None],
                )
                previous = current
            _check_live_temperature(pdb)
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            gc.collect()
            assert all(ref() is None for ref in previous), "Previous ML/MM setup is still retained"
            log.info("setup replacement releases systems, Qt timers, contexts, and ML calculators")
            _check_error_cleanup(app)
            _check_socket_cleanup()
    finally:
        commands.imd_stop()
        commands.imd_disconnect()
        for obj in set(cmd.get_object_list()) - original_objects:
            cmd.delete(obj)
    log.info("PyMOL/Qt smoke test OK")
    return 0


if __name__ in {"__main__", "pymol"}:
    gui = isinstance(QtCore.QCoreApplication.instance(), QtWidgets.QApplication)
    if not gui:
        raise SystemExit(main())
    try:
        main()
    except Exception:
        log.exception("PyMOL/Qt smoke test failed")
        cmd.quit(1)
    else:
        cmd.quit(0)
