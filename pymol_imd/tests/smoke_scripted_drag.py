"""Check code-driven dragging against actual OpenMM mouse-force integration."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import mdtraj
import numpy as np
import openmm
from openmm import unit
from openmm.app import Element, Topology

from pymol_imd.backends.base import SystemBundle
from pymol_imd.imd import IMDConfig, IMDSystem
from pymol_imd.scripted import run_drag


class CountingML:
    def __init__(self):
        self.calls = 0

    def get_forces(self, atoms):
        self.calls += 1
        return np.tile([0.01, 0, 0], (len(atoms), 1))


def make_system(root, *, mlmm=False):
    top = Topology()
    residue = top.addResidue("MOL", top.addChain())
    system = openmm.System()
    for _ in range(3):
        top.addAtom("C", Element.getBySymbol("C"), residue)
        system.addParticle(12)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.3, 100)
    bonds.addBond(1, 2, 0.3, 100)
    system.addForce(bonds)
    bundle = SystemBundle(
        system,
        top,
        np.array([[0, 0, 0], [0.3, 0, 0], [0.6, 0, 0]]) * unit.nanometer,
        "mlmm" if mlmm else "aa",
        root / "prepared.pdb",
        extra={
            "ml_indices": [0, 1, 2],
            "ml_symbols": ["C"] * 3,
            "ml_calculator": CountingML(),
            "ml_subset_system": openmm.XmlSerializer.deserialize(
                openmm.XmlSerializer.serialize(system)
            ),
        }
        if mlmm
        else None,
    )
    driver = IMDSystem(
        bundle,
        IMDConfig(
            platform="reference",
            temperature=1,
            friction=1,
            timestep_ps=0.001,
            const_k=1000,
            max_force_kj=50,
        ),
    )
    driver.simulation.integrator.setTemperature(0 * unit.kelvin)
    driver.simulation.integrator.setFriction(0 / unit.picosecond)
    driver.set_recording(
        traj=str(root / "trajectory.xtc"),
        force_csv=str(root / "forces.csv"),
        state_csv=str(root / "state.csv"),
    )
    return driver


def positions(driver):
    return (
        driver.simulation.context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(unit.nanometer)
    )


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def check_mouse_equivalence(root):
    scripted = make_system(root / "scripted")
    result = run_drag(
        scripted, [0, 2], [[0.2, 0, 0], [-0.2, 0, 0]], duration_ps=0.01, release_ps=0.006
    )
    mouse = make_system(root / "mouse")
    mouse._sync_positions_to_pymol = lambda state: None
    mouse.start_recording()
    mouse.running = True
    for step in range(16):
        target = np.array(positions(mouse))
        if step < 10:
            target[[0, 2], 0] += [0.02, -0.02]
        mouse._read_pymol_positions_nm = lambda target=target: target
        mouse.step()
    mouse.stop_recording()
    np.testing.assert_allclose(positions(mouse), positions(scripted), atol=1e-14, rtol=0)
    assert result.start_step == 0 and result.end_step == 16
    assert result.applied_frames == 10 and result.skipped_frames == result.recovery_frames == 0
    assert np.isclose(result.end_time_ps, 0.016)
    assert not scripted.running and not scripted._recording and not scripted._active_pull_idx_prev
    scripted_rows = rows(root / "scripted/forces.csv")
    assert scripted_rows == rows(root / "mouse/forces.csv")
    assert len(scripted_rows) == 20 and max(int(r["step"]) for r in scripted_rows) == 10
    trajectory = mdtraj.load(
        str(root / "scripted/trajectory.xtc"),
        top=mdtraj.Topology.from_openmm(scripted.bundle.topology),
    )
    np.testing.assert_allclose(trajectory.time, np.arange(1, 17) * 0.001, atol=1e-8)


def check_callback_cap_and_release(root):
    driver = make_system(root / "cap")
    samples = []

    def offset(elapsed, xyz):
        samples.append(elapsed)
        assert not xyz.flags.writeable and xyz.shape == (3, 3)
        return [[0.1, 0, 0], [1.0, 0, 0]] if elapsed < 0.003 else [[0.1, 0, 0], [-0.1, 0, 0]]

    result = run_drag(driver, [0, 2], offset, duration_ps=0.005, release_ps=0.002)
    np.testing.assert_allclose(samples, np.arange(5) * 0.001)
    assert result.skipped_frames == 0 and result.applied_frames == 5
    force_rows = rows(root / "cap/forces.csv")
    assert sorted(set(int(r["step"]) for r in force_rows)) == [1, 2, 3, 4, 5]
    assert max(float(r["fmag"]) for r in force_rows) == driver.config.max_force_kj
    assert [int(r["step"]) for r in rows(root / "cap/state.csv")] == list(range(1, 8))


def check_error_cleanup(root):
    driver = make_system(root / "error")

    def broken(elapsed, xyz):
        if elapsed >= 0.002:
            raise RuntimeError("intentional callback error")
        return [0.1, 0, 0]

    try:
        run_drag(driver, [1], broken, duration_ps=0.005)
    except RuntimeError as exc:
        assert "intentional callback error" in str(exc)
    else:
        raise AssertionError("The callback failure must propagate")
    assert driver.simulation.currentStep == 2
    assert not driver.running and not driver._recording and not driver._active_pull_idx_prev
    assert len(rows(root / "error/forces.csv")) == 2
    assert len(rows(root / "error/state.csv")) == 2


def check_recovery_and_mlmm(root):
    recovering = make_system(root / "recovery")
    recovering.config.energy_cutoff_kj = 0.001
    result = run_drag(recovering, [1], [0.1, 0, 0], duration_ps=0.002)
    assert result.recovery_frames == 1 and result.applied_frames == 0
    assert result.end_step == 2 and not rows(root / "recovery/forces.csv")

    driver = make_system(root / "mlmm", mlmm=True)
    calculator = driver.bundle.extra["ml_calculator"]
    calls_before = calculator.calls
    driver.config.drag_level = "residue"
    result = run_drag(driver, [1], [0.1, 0, 0], duration_ps=0.003, release_ps=0.002)
    assert calculator.calls - calls_before == 6  # Initial state and five accepted proposals.
    assert result.applied_frames == 3
    force_rows = rows(root / "mlmm/forces.csv")
    assert len(force_rows) == 9
    assert sorted(set(int(r["atom_index"]) for r in force_rows)) == [0, 1, 2]


def check_validation(root):
    driver = make_system(root / "invalid")
    driver.config.const_k = 0
    assert driver.apply_drag([0], [0.1, 0, 0]) == 0
    driver.clear_pull_forces()
    driver.config.const_k = 1000
    for indices, offsets in (
        ([True], [0, 0, 0]),
        ([0, 0], [0, 0, 0]),
        ([-1], [0, 0, 0]),
        ([3], [0, 0, 0]),
        ([0.5], [0, 0, 0]),
        ([], [0, 0, 0]),
        ([0], [float("nan"), 0, 0]),
        ([0], [0, 0]),
    ):
        try:
            run_drag(driver, indices, offsets, duration_ps=0.002)
        except ValueError:
            pass
        else:
            raise AssertionError((indices, offsets))
        assert not driver.running and not driver._recording
    try:
        run_drag(driver, [0], [0, 0, 0], duration_ps=0.0015)
    except ValueError as exc:
        assert "integer multiple" in str(exc)
    else:
        raise AssertionError("Partial integration steps must be rejected")
    driver.running = True
    try:
        run_drag(driver, [0], [0, 0, 0], duration_ps=0.002)
    except RuntimeError:
        assert driver.running
    else:
        raise AssertionError("Concurrent stepping must be rejected")


def main():
    with tempfile.TemporaryDirectory(prefix="imd-scripted-") as directory:
        root = Path(directory)
        check_mouse_equivalence(root)
        check_callback_cap_and_release(root)
        check_error_cleanup(root)
        check_recovery_and_mlmm(root)
        check_validation(root)
    print("smoke-scripted-drag: passed")


if __name__ == "__main__":
    main()
