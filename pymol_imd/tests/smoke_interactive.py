"""Check asynchronous local MD input, responsiveness and worker cleanup."""

from __future__ import annotations

import csv
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from openmm import unit
from pymol.Qt import QtCore

from pymol_imd import commands, constants
from pymol_imd.interactive import LocalIMDLoop
from pymol_imd.tests.smoke_recovery import make_system


class Viewer:
    """Represent GUI coordinates without requiring an OpenGL window."""

    def __init__(self, system):
        self.system = system
        self.thread = threading.get_ident()
        self.positions = system._read_openmm_positions_nm()
        self.frames = []
        system._read_pymol_positions_nm = self.read
        system._sync_positions_to_pymol = self.display

    def read(self):
        assert threading.get_ident() == self.thread
        return self.positions.copy()

    def display(self, state=None):
        assert threading.get_ident() == self.thread
        self.positions = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        self.frames.append(self.positions.copy())


def check_input_and_responsiveness(folder, app):
    system = make_system(folder)
    viewer = Viewer(system)
    advance = system.advance_openmm
    entered, release = threading.Event(), threading.Event()
    workers = set()

    def slow_frame(**kwargs):
        workers.add(threading.get_ident())
        entered.set()
        assert release.wait(5), "worker was never released"
        return advance(**kwargs)

    system.advance_openmm = slow_frame
    loop = LocalIMDLoop(system)

    def reentrant_display(state):
        loop.tick()
        viewer.display(state)

    system._sync_positions_to_pymol = reentrant_display
    heartbeats = []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: heartbeats.append(time.monotonic()))
    timer.start(1)
    try:
        viewer.positions[0, 0] += 0.02
        loop.tick()
        assert entered.wait(5)
        viewer.positions[0, 0] += 0.03
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            loop.tick()
            app.processEvents()
            time.sleep(0.001)
        assert len(heartbeats) >= 2, "MD blocked the Qt event loop"
        assert not viewer.frames, "an unfinished state reached the viewer"
        assert system._frame_index == 0, "more than one MD frame was queued"
        release.set()
        loop.wait()
        loop.tick()
        loop.wait()
        loop.tick()
        loop.wait()
        assert not system._active_pull_idx_prev, "release retained a pull force"
        system.config.max_force_kj = 500
        viewer.positions[0, 0] += 1.0
        loop.tick()
        loop.wait()
        assert system._force_exceeded_prev, "force cap was bypassed"
        assert system._active_pull_idx_prev == {0}
        system.running = False
        loop.tick()
    finally:
        release.set()
        timer.stop()
        loop.close()
        system.stop_recording()
    assert workers and viewer.thread not in workers and len(workers) == 1
    assert len(viewer.frames) == 4
    with (folder / "forces.csv").open() as stream:
        forces = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in forces] == [1, 2, 4]
    np.testing.assert_allclose([float(row["fx"]) for row in forces], [200, 300, 500])
    with (folder / "state.csv").open() as stream:
        assert [int(row["step"]) for row in csv.DictReader(stream)] == [1, 2, 3, 4]
    print("OK: Qt stays responsive, mouse deltas consumed once, release/cap/recording preserved")


def check_live_commands(folder):
    system = make_system(folder)
    viewer = Viewer(system)
    advance = system.advance_openmm
    entered, release = threading.Event(), threading.Event()
    temperature = system.config.temperature

    def slow_frame(**kwargs):
        entered.set()
        assert release.wait(5)
        assert system.config.temperature == temperature, "config changed during a pending frame"
        return advance(**kwargs)

    system.advance_openmm = slow_frame
    commands._imd_system = system
    commands._timer = QtCore.QTimer()
    commands._local_loop = LocalIMDLoop(system)
    loop = commands._local_loop
    release_timer = threading.Timer(0.05, release.set)
    try:
        viewer.positions[0, 0] += 0.02
        loop.tick()
        assert entered.wait(5)
        release_timer.start()
        commands.imd_change(temperature=350, mouse_force=False)
        assert release.is_set()
        assert system.simulation.integrator.getTemperature().value_in_unit(unit.kelvin) == 350
        assert not system._active_pull_idx_prev
        commands.imd_record("off")
        assert not system._recording
        commands.imd_stop()
        assert commands._local_loop is None
        assert not system.running and not commands._timer.isActive()
        assert len(viewer.frames) == 1
        loop.tick()
        assert system._frame_index == 1, "closed worker advanced another frame"
    finally:
        release.set()
        release_timer.join()
        commands.imd_stop()
        commands._imd_system = None
        commands._timer = None
    print("OK: live commands wait for the pending frame; stop joins and closes outputs")


def check_failure(folder, app):
    system = make_system(folder)
    Viewer(system)

    def fail():
        raise RuntimeError("injected ML failure")

    system.apply_mlmm_if_present = fail
    loop = LocalIMDLoop(system)
    try:
        loop.tick()
        try:
            loop.wait()
        except RuntimeError as exc:
            assert str(exc) == "injected ML failure"
        else:
            raise AssertionError("worker failure was hidden")
        try:
            loop.tick()
        except RuntimeError as exc:
            assert str(exc) == "injected ML failure"
        else:
            raise AssertionError("Qt did not receive the worker failure")
        assert system._frame_index == 0
    finally:
        loop.close()
        system.stop_recording()
    commands._imd_system = system
    commands._timer = QtCore.QTimer()
    system.running = False
    try:
        commands.imd_run()
        deadline = time.monotonic() + 5
        while system.running and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.001)
        assert not system.running and not commands._timer.isActive()
        assert commands._local_loop is None and not system._recording
        assert not system._active_pull_idx_prev
    finally:
        commands.imd_stop()
        commands._imd_system = None
        commands._timer = None
    print("OK: worker errors reach Qt and no failed frame is recorded")


def check_restart(folder):
    system = make_system(folder)
    Viewer(system)
    entered, release = threading.Event(), threading.Event()
    advance = system.advance_openmm

    def finishing_frame(**kwargs):
        system.running = False
        entered.set()
        assert release.wait(5)
        return advance(**kwargs)

    system.advance_openmm = finishing_frame
    commands._imd_system = system
    commands._timer = QtCore.QTimer()
    loop = commands._local_loop = LocalIMDLoop(system)
    release_timer = threading.Timer(0.05, release.set)
    try:
        loop.tick()
        assert entered.wait(5)
        release_timer.start()
        commands.imd_run()
        assert release.is_set() and loop._closed
        assert commands._local_loop is not loop
        assert system.running
        commands.imd_stop()
        assert commands._local_loop is None and not system._recording
    finally:
        release.set()
        if release_timer.ident is not None:
            release_timer.join()
        commands.imd_stop()
        commands._imd_system = None
        commands._timer = None
    print("OK: restarting a terminating worker joins it before creating the next loop")


def check_trajectory_parity(folder):
    system = make_system(folder)
    system.stop_recording()
    viewer = Viewer(system)
    context = system.simulation.context
    # Initialize the integrator's RNG before saving a replayable checkpoint.
    system.step()
    initial_step = system.simulation.currentStep
    checkpoint = context.createCheckpoint()
    initial = viewer.positions.copy()
    offsets = (0.01, 0.02, 0.0, -0.01, 0.0)
    expected = []
    for offset in offsets:
        viewer.positions[0, 0] += offset
        system.step()
        expected.append(viewer.positions.copy())
    system.clear_pull_forces()
    context.loadCheckpoint(checkpoint)
    viewer.positions = initial
    actual = []
    loop = LocalIMDLoop(system)
    try:
        for offset in offsets:
            viewer.positions[0, 0] += offset
            loop.tick()
            loop.wait()
            state = loop._future.result()
            actual.append(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        system.running = False
        loop.tick()
    finally:
        loop.close()
    np.testing.assert_array_equal(actual, expected)
    assert system.simulation.currentStep == initial_step + len(offsets)
    print("OK: synchronous and worker trajectories match exactly from the same checkpoint")


def check_feedback(folder):
    system = make_system(folder)
    system.config.const_k = 50000
    system.config.energy_cutoff_kj = constants.DEFAULT_ENERGY_CUTOFF_KJ
    viewer = Viewer(system)
    initial = viewer.positions.copy()
    loop = LocalIMDLoop(system)
    try:
        for frame in range(250):
            if frame == 50:
                viewer.positions[0, 0] += 1.0
            loop.tick()
            loop.wait()
        system.running = False
        loop.tick()
    finally:
        loop.close()
        system.stop_recording()
    assert len(viewer.frames) == 250
    assert np.max(np.abs(np.asarray(viewer.frames) - initial)) > 0.01
    assert not system._active_pull_idx_prev and system._recovery_count == 0
    with (folder / "forces.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1, "MD motion after release fed back into the pull force"
    assert int(rows[0]["step"]) == 51
    assert float(rows[0]["fmag"]) == system.config.max_force_kj
    print("OK: one capped edit produces one force sample across 250 moving MD frames")


def main():
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, check in (
            ("input", lambda folder: check_input_and_responsiveness(folder, app)),
            ("commands", check_live_commands),
            ("failure", lambda folder: check_failure(folder, app)),
            ("restart", check_restart),
            ("parity", check_trajectory_parity),
            ("feedback", check_feedback),
        ):
            folder = root / name
            folder.mkdir()
            check(folder)
    print("All interactive worker checks passed")


if __name__ == "__main__":
    main()
