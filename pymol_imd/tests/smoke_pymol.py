"""Exercise real PyMOL commands and Qt timers, with optional GUI rendering.

Run with ``pixi run smoke-pymol`` or in a fresh viewer with
``pixi run pymol -k -r pymol_imd/tests/smoke_pymol.py``.
"""

from __future__ import annotations

import csv
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

from pymol_imd import commands, constants
from pymol_imd.client import Frame, PyMOLIMDClient

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


def _check_prepared_identifiers(pdb: Path) -> None:
    from openmm import Vec3, unit
    from openmm.app import PDBFile, element

    from pymol_imd.backends.mlmm import _resolve_ml_indices
    from pymol_imd.client import PyMOLIMDClient
    from pymol_imd.particle_masses import read_particle_masses, write_particle_masses
    from pymol_imd.prep import write_prepared_pdb
    from pymol_imd.pymol_io import load_prepared

    source = PDBFile(str(pdb))
    protein = next(source.topology.chains())
    protein.id = "Z"
    residues = list(protein.residues())
    for number, residue in enumerate(residues, 41):
        residue.id = str(number)
    residues[1].insertionCode = "A"
    water = source.topology.addResidue("HOH", source.topology.addChain("W"), "901")
    oxygen = source.topology.addAtom("O", element.oxygen, water)
    hydrogen1 = source.topology.addAtom("H1", element.hydrogen, water)
    hydrogen2 = source.topology.addAtom("H2", element.hydrogen, water)
    source.topology.addBond(oxygen, hydrogen1)
    source.topology.addBond(oxygen, hydrogen2)
    positions = list(source.positions.value_in_unit(unit.nanometer))
    positions.extend((Vec3(2, 0, 0), Vec3(2.1, 0, 0), Vec3(2, 0.1, 0)))
    prepared = write_prepared_pdb(
        source.topology, positions * unit.nanometer, pdb.parent / "identifiers.pdb"
    )
    reread = PDBFile(str(prepared))
    expected_ids = [
        (res.chain.id, res.id, res.insertionCode.strip()) for res in source.topology.residues()
    ]
    actual_ids = [
        (res.chain.id, res.id, res.insertionCode.strip()) for res in reread.topology.residues()
    ]
    assert actual_ids == expected_ids, (actual_ids, expected_ids)
    expected = tuple(
        atom.index for atom in source.topology.atoms() if atom.residue in (residues[1], water)
    )
    actual = _resolve_ml_indices(
        source.topology, "chain W or (chain Z and resi 42A)", prepared_pdb_path=prepared
    )
    assert actual == expected, (actual, expected)
    masses = np.linspace(1.25, 72, source.topology.getNumAtoms())
    write_particle_masses(prepared, masses)
    write_particle_masses(prepared, masses)
    np.testing.assert_array_equal(read_particle_masses(prepared), masses)
    obj = "mass_metadata_test"
    load_prepared(cmd, str(prepared), obj)
    viewer = PyMOLIMDClient(None, obj, const_k=50000, max_force_kj=5000, drag_level="chain")
    forces = np.zeros((len(masses), 3))
    forces[0, 0] = 10
    active, distributed, _ = viewer._expand_drag_groups(np.array([0]), forces, len(masses))
    np.testing.assert_array_equal(viewer._particle_masses, masses)
    assert len(active) == len(masses) - 3, "Chain pulling included the separate water chain"
    acceleration = distributed[active] / masses[active, None]
    np.testing.assert_allclose(acceleration, np.tile(acceleration[0], (len(active), 1)))
    cmd.delete(obj)
    water.chain.id = protein.id
    write_prepared_pdb(source.topology, positions * unit.nanometer, prepared)
    write_particle_masses(prepared, masses)
    load_prepared(cmd, str(prepared), obj)
    viewer = PyMOLIMDClient(None, obj, const_k=50000, max_force_kj=5000, drag_level="chain")
    active, _, _ = viewer._expand_drag_groups(np.array([0]), forces, len(masses))
    assert len(active) == len(masses) - 3, "TER-separated chains with equal labels were merged"
    cmd.delete(obj)
    log.info("prepared PDB preserves chain/residue/insertion identifiers and ML selection")


def _exercise_backend(app, pdb: Path, level: str, output: Path | None) -> list:
    from pymol_imd.client import PyMOLIMDClient
    from pymol_imd.particle_masses import read_particle_masses

    setup = getattr(commands, f"imd_setup_{level}")
    kwargs = {"platform": "cpu", "nsteps_per_frame": 1, "frame_interval_ms": 5}
    if level == "mlmm":
        kwargs.update(ml_selection="resi 1-2", ml_charge=0, ml_mult=1, ml_device="cpu")
    setup(str(pdb), work_dir=str(pdb.parent / level), **kwargs)
    system = commands._imd_system
    np.testing.assert_array_equal(
        read_particle_masses(system.bundle.prepared_pdb_path), system._particle_masses
    )
    obj = system.pymol_object
    viewer = PyMOLIMDClient(None, obj, const_k=50000, max_force_kj=5000)
    np.testing.assert_array_equal(
        viewer._group_of("residue", system.n_atoms), system._atom_residue_idx
    )
    np.testing.assert_array_equal(viewer._group_of("chain", system.n_atoms), system._atom_chain_idx)
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
    _until(app, lambda: system._frame_index >= 3)
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
    _check_async_drag(app, system, pdb.parent / f"{level}_async_forces.csv")

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


def _check_async_drag(app, system, force_csv: Path) -> None:
    commands.imd_drag_level("atom")
    system._sync_positions_to_pymol()
    cmd.alter_state(
        1,
        f"{system.pymol_object} and index 1",
        "x += offset",
        space={"offset": 2 * constants.DELTA_NOISE_NM * constants.NM_TO_ANGSTROM},
    )
    commands.imd_record(force_csv=str(force_csv))
    commands.imd_run()
    _until(app, lambda: system._frame_index >= 3)
    commands.imd_stop()
    commands.imd_record("off")
    with force_csv.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1, ("mouse offset was lost or applied more than once", rows)
    assert int(rows[0]["atom_index"]) == 0
    assert float(rows[0]["fx"]) > 0
    assert commands._local_loop is None and not system._active_pull_idx_prev
    log.info("worker consumes real PyMOL coordinate edits once and releases the pull")


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
    measurement = cmd.get_unused_name("imd_distance_probe")
    dynamic_measures = cmd.get_setting_int("dynamic_measures")
    cmd.set("dynamic_measures", 1)

    def measured_positions():
        # The session stores the coordinates used by the visible distance label.
        session = cmd.get_session(measurement)
        return np.asarray(session["names"][0][5][2][0][1]).reshape(2, 3)

    try:
        displaced = cmd.get_coordset(obj, state=1)
        displaced[0, 0] += 1
        cmd.load_coordset(displaced, obj, state=1)
        cmd.distance(measurement, f"{obj} and index 1", f"{obj} and index 2")
        system._sync_positions_to_pymol()
        expected = cmd.get_coordset(obj, state=1)
        np.testing.assert_allclose(measured_positions(), expected[:2], atol=1e-5)
        viewer = PyMOLIMDClient(None, obj, const_k=10000, max_force_kj=5000)
        streamed = expected.copy()
        streamed[0, 0] += 0.2
        viewer._display(streamed)
        np.testing.assert_allclose(measured_positions(), streamed[:2], atol=1e-5)
        cmd.set("dynamic_measures", 0)
        viewer._display(expected)
        np.testing.assert_allclose(measured_positions(), streamed[:2], atol=1e-5)
        np.testing.assert_array_equal(cmd.get_coordset(other, state=1), original_other)
        np.testing.assert_array_equal(cmd.get_coordset(obj, state=2), second_state)
        assert cmd.count_states(obj) == 2
    finally:
        cmd.set("dynamic_measures", dynamic_measures)
        cmd.delete(measurement)
        cmd.delete(other)
        cmd.delete_states(obj, "2")
        system._sync_positions_to_pymol()


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


def _check_socket_input_consumption() -> None:
    class Connection:
        def __init__(self):
            self.sent = []

        def send_forces(self, indices, forces):
            self.sent.append((indices.copy(), forces.copy()))

    obj = cmd.get_unused_name("imd_socket_input")
    cmd.pseudoatom(obj, pos=[0, 0, 0])
    connection = Connection()
    viewer = PyMOLIMDClient(connection, obj, const_k=10000, max_force_kj=5000)
    viewer._last_server_ang = np.zeros((1, 3))
    cmd.load_coordset(np.array([[0.2, 0, 0]]), obj, state=1)
    viewer._send_drag_forces()
    viewer._send_drag_forces()
    viewer._send_drag_forces()
    assert len(connection.sent) == 2
    assert connection.sent[0][0].tolist() == [0]
    assert connection.sent[1][0].size == 0, "An old edit was resent while waiting for a frame"
    viewer._particle_masses = np.array([12.0])
    viewer._group_ids = {"residue": np.array([0]), "chain": np.array([0])}
    for override in (None, 7500):
        viewer.max_force_kj = override
        for level in ("atom", "residue", "chain"):
            viewer.drag_level = level
            viewer._last_server_ang = np.zeros((1, 3))
            cmd.load_coordset(np.array([[100.0, 0, 0]]), obj, state=1)
            viewer._send_drag_forces()
            limit = 100000 if override is None else override
            np.testing.assert_allclose(connection.sent[-1][1], [[limit / 10, 0, 0]])
    cmd.delete(obj)


def _check_socket_feedback(app) -> None:
    class Connection:
        closed = False

        def __init__(self):
            self.sent = []

        def send_forces(self, indices, forces):
            self.sent.append((indices.copy(), forces.copy()))

        def disconnect(self):
            self.closed = True

    obj = cmd.get_unused_name("imd_socket_feedback")
    cmd.pseudoatom(obj, pos=[0, 0, 0])
    connection = Connection()
    viewer = PyMOLIMDClient(connection, obj, const_k=50000, max_force_kj=5000)
    viewer._last_server_ang = np.zeros((1, 3))
    frame_number = 0
    redrawing = False
    display = viewer._display

    def next_frame():
        nonlocal frame_number
        frame_number += 1
        return Frame(positions_ang=np.array([[0.2 * frame_number, 0, 0]]))

    def redraw(positions):
        nonlocal redrawing
        display(positions)
        if not redrawing:
            redrawing = True
            try:
                QtCore.QTimer.singleShot(0, viewer._tick)
                app.processEvents()
            finally:
                redrawing = False

    viewer._drain_to_latest = next_frame
    viewer._display = redraw
    try:
        for _ in range(100):
            viewer._tick()
        assert not connection.closed
        assert not connection.sent, "Server motion became a mouse force during a nested Qt tick"
        assert frame_number == 100, "A nested Qt tick advanced the display twice"
        np.testing.assert_allclose(cmd.get_coordset(obj, state=1), viewer._last_server_ang)
    finally:
        cmd.delete(obj)
    log.info("socket redraw consumes no MD motion as input, including nested Qt callbacks")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    original_objects = set(cmd.get_object_list())
    output = os.environ.get("PYMOL_IMD_TEST_OUTPUT")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "test_peptide.pdb"
            _write_peptide(pdb)
            _check_prepared_identifiers(pdb)
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
            _check_socket_input_consumption()
            _check_socket_feedback(app)
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
