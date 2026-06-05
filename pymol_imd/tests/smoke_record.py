"""Headless smoke test for trajectory + force-CSV + state-CSV recording.

Runs without PyMOL: builds an AA system, arms recording, monkeypatches the
PyMOL coordinate read so the first atoms are "dragged" (producing a non-empty
pull-force set), steps the IMD loop, then asserts the trajectory, force CSV and
thermodynamic state CSV were written correctly. Mirrors the monkeypatch trick
in ``smoke.py``'s ``_verify_force_limit``.

Usage::

    pixi run record-smoke
    pixi run record-smoke 1ubq      # override PDB id
"""

from __future__ import annotations

import csv
import logging
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("record-smoke")


def _make_dragging_imd(bundle, *, n_drag_atoms: int = 2):
    """Build a minimized IMDSystem whose PyMOL read drags the first atoms.

    The fake read shoves the first ``n_drag_atoms`` by 0.05 nm along x — well
    above the ``_DELTA_NOISE_NM`` (1e-3 nm) threshold — so they register as
    actively pulled and produce force-CSV rows.
    """
    from pymol_imd.imd import IMDConfig, IMDSystem

    cfg = IMDConfig(nsteps_per_frame=1, mouse_force=True, integrator="langevin_middle")
    imd = IMDSystem(bundle, cfg)
    imd.simulation.minimizeEnergy(maxIterations=200)

    def _fake_pymol_positions() -> np.ndarray:
        pos = imd._read_openmm_positions_nm().copy()
        pos[:n_drag_atoms] += np.array([0.05, 0.0, 0.0])
        return pos

    def _no_sync() -> None:
        # Headless: no PyMOL window to write back to. The recording path is
        # what we exercise here, so skip the coordinate sync entirely.
        return None

    imd._read_pymol_positions_nm = _fake_pymol_positions
    imd._sync_positions_to_pymol = _no_sync
    return imd


def _check_csv(csv_path: Path, *, max_force_kj: float) -> None:
    with open(csv_path, newline="") as fh:
        rows = list(csv.reader(fh))
    expected = ["frame", "step", "time_ps", "atom_index", "fx", "fy", "fz", "fmag"]
    if rows[0] != expected:
        raise AssertionError(f"CSV header mismatch: {rows[0]} != {expected}")
    data = rows[1:]
    if not data:
        raise AssertionError("CSV has no data rows (expected dragged-atom forces)")
    for r in data:
        atom_index = int(r[3])
        fmag = float(r[7])
        if atom_index < 0:
            raise AssertionError(f"negative atom_index {atom_index}")
        if fmag > max_force_kj + 1.0:
            raise AssertionError(f"|F|={fmag} exceeds limit {max_force_kj}")
    log.info("CSV OK: %d data rows, header matches, |F| within limit", len(data))


def _check_state_csv(csv_path: Path, *, pbc: bool, expected_rows: int) -> None:
    """The state CSV: header matches, one row per recorded frame, values sane.

    Unlike the force CSV (one row per dragged atom), the state CSV writes EVERY
    gated frame, so row count == recorded frames. PBC adds volume/density.
    """
    with open(csv_path, newline="") as fh:
        rows = list(csv.reader(fh))
    base = [
        "frame",
        "step",
        "time_ps",
        "potential_energy_kj",
        "kinetic_energy_kj",
        "total_energy_kj",
        "temperature_k",
    ]
    expected_header = base + (["volume_nm3", "density_g_cm3"] if pbc else [])
    if rows[0] != expected_header:
        raise AssertionError(f"state CSV header mismatch: {rows[0]} != {expected_header}")
    data = rows[1:]
    if len(data) != expected_rows:
        raise AssertionError(f"state CSV rows {len(data)} != expected {expected_rows}")
    for r in data:
        pe, ke, total, temp = float(r[3]), float(r[4]), float(r[5]), float(r[6])
        if not all(math.isfinite(v) for v in (pe, ke, total, temp)):
            raise AssertionError(f"non-finite state value in row {r}")
        if abs(total - (pe + ke)) > 1e-3:
            raise AssertionError(f"total_energy {total} != pe+ke {pe + ke}")
        if temp <= 0.0:
            raise AssertionError(f"non-positive temperature {temp}")
        if pbc:
            volume, density = float(r[7]), float(r[8])
            if volume <= 0.0 or density <= 0.0:
                raise AssertionError(f"non-positive volume/density: {volume}, {density}")
    log.info("state CSV OK: %d rows, header matches (pbc=%s), energies/T finite", len(data), pbc)


def _check_xtc(xtc_path: Path, bundle, *, expected_frames: int) -> None:
    import mdtraj

    top = mdtraj.Topology.from_openmm(bundle.topology)
    traj = mdtraj.load_xtc(str(xtc_path), top=top)
    if traj.n_frames != expected_frames:
        raise AssertionError(f"XTC frame count {traj.n_frames} != expected {expected_frames}")
    log.info("XTC OK: %d frames", traj.n_frames)


def _check_pdb_footer(pdb_path: Path) -> None:
    # PDBReporter writes its trailing END only in __del__ (via writeFooter), so
    # a standalone "END" line proves stop_recording's gc.collect() close path.
    lines = pdb_path.read_text().splitlines()
    if not any(line.strip() == "END" for line in lines):
        raise AssertionError("PDB trajectory missing END footer (close path failed)")
    log.info("PDB OK: END footer present")


def _check_csv_steps_match_traj(
    csv_path: Path, *, interval: int, nsteps_per_frame: int, n_frames: int
) -> None:
    """The CSV ``step`` values must equal the trajectory's frame steps.

    Both the force CSV and the trajectory reporter gate on
    ``currentStep % (interval * nsteps_per_frame) == 0``, so equal intervals
    MUST select identical steps. Guards the bug where the CSV gated on the
    per-run frame counter (reset by start_recording) while the trajectory gated
    on cumulative ``currentStep``, making ``interval > 1`` sample different
    frames.
    """
    with open(csv_path, newline="") as fh:
        rows = list(csv.reader(fh))[1:]
    csv_steps = sorted({int(r[1]) for r in rows})
    report_interval = interval * nsteps_per_frame
    expected = [report_interval * k for k in range(1, n_frames // interval + 1)]
    if csv_steps != expected:
        raise AssertionError(f"CSV steps {csv_steps} != expected trajectory steps {expected}")
    log.info("CSV/traj step alignment OK (interval=%d): steps=%s", interval, csv_steps)


def main(pdb_id: str = "1aki", n_steps: int = 6) -> int:
    from pymol_imd.backends.aa import AABackend
    from pymol_imd.prep import fetch_pdb

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdb_path = fetch_pdb(pdb_id, out_dir=tmp_path)
        bundle = AABackend().build_system(pdb_path)
        log.info("Built AA system: %d atoms", bundle.topology.getNumAtoms())

        # --- XTC + force CSV + state CSV, every frame ---
        xtc = tmp_path / "traj.xtc"
        csv_path = tmp_path / "forces.csv"
        state_path = tmp_path / "state.csv"
        imd = _make_dragging_imd(bundle)
        imd.set_recording(
            traj=str(xtc), traj_interval=1, force_csv=str(csv_path), state_csv=str(state_path)
        )
        imd.start_recording()
        for _ in range(n_steps):
            imd.step()
        imd.stop_recording()
        _check_xtc(xtc, bundle, expected_frames=n_steps)
        _check_csv(csv_path, max_force_kj=imd.config.max_force_kj)
        _check_state_csv(state_path, pbc=False, expected_rows=n_steps)

        # --- re-run safety: same paths again, frame count must not double ---
        imd.start_recording()
        for _ in range(n_steps):
            imd.step()
        imd.stop_recording()
        _check_xtc(xtc, bundle, expected_frames=n_steps)
        log.info("Re-run OK: trajectory overwritten, not doubled")

        # --- interval > 1: force CSV, state CSV and trajectory sample SAME steps ---
        xtc2 = tmp_path / "traj_i2.xtc"
        csv2 = tmp_path / "forces_i2.csv"
        state2 = tmp_path / "state_i2.csv"
        imd3 = _make_dragging_imd(bundle)
        imd3.set_recording(
            traj=str(xtc2),
            traj_interval=2,
            force_csv=str(csv2),
            force_csv_interval=2,
            state_csv=str(state2),
            state_csv_interval=2,
        )
        imd3.start_recording()
        for _ in range(n_steps):
            imd3.step()
        imd3.stop_recording()
        _check_xtc(xtc2, bundle, expected_frames=n_steps // 2)
        _check_csv_steps_match_traj(
            csv2, interval=2, nsteps_per_frame=imd3.config.nsteps_per_frame, n_frames=n_steps
        )
        _check_csv_steps_match_traj(
            state2, interval=2, nsteps_per_frame=imd3.config.nsteps_per_frame, n_frames=n_steps
        )

        # --- PDB footer close path ---
        pdb_traj = tmp_path / "traj.pdb"
        imd2 = _make_dragging_imd(bundle)
        imd2.set_recording(traj=str(pdb_traj), traj_interval=1)
        imd2.start_recording()
        for _ in range(n_steps):
            imd2.step()
        imd2.stop_recording()
        _check_pdb_footer(pdb_traj)

        # --- pbc=True: state CSV gains volume/density columns ---
        bundle_pbc = AABackend().build_system(pdb_path, pbc=True)
        state_pbc = tmp_path / "state_pbc.csv"
        imd_pbc = _make_dragging_imd(bundle_pbc)
        imd_pbc.set_recording(state_csv=str(state_pbc))
        imd_pbc.start_recording()
        for _ in range(n_steps):
            imd_pbc.step()
        imd_pbc.stop_recording()
        _check_state_csv(state_pbc, pbc=True, expected_rows=n_steps)

        log.info("Record smoke test OK")
    return 0


if __name__ == "__main__":
    pdb_id = sys.argv[1] if len(sys.argv) > 1 else "1aki"
    raise SystemExit(main(pdb_id))
