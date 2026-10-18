"""Headless smoke test: mouse-drag granularity (atom / residue / chain).

PyMOL's 3-Button Editing drag moves a covalent fragment and ignores
``mouse_selection_mode``, so residue / chain granularity is realized plugin-side
by expanding a per-atom pull to the whole residue / chain. This test verifies
that expansion WITHOUT a PyMOL window, on a tiny synthetic topology
(2 chains x 2 residues x 2 atoms = 8 atoms):

  * the drag-level constants are consistent and ``IMDConfig`` defaults to atom,
  * ``commands._validate_drag_level`` rejects unknown levels,
  * ``compute_pull_forces`` caps each atom's force without changing direction
    or discarding other atoms' sub-limit pulls,
  * ``IMDSystem._build_group_indices`` maps atoms to residue / chain correctly,
  * ``IMDSystem._expand_drag_groups`` expands a dragged atom to its whole
    residue / chain and copies the group's mean pull force to every member,
    including the multi-group (two chains dragged at once) case.

Usage::

    pixi run smoke-drag
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("smoke-drag")


def _toy_imd_system(drag_level: str):
    """Build an 8-atom IMDSystem (2 chains x 2 residues x 2 atoms), no PyMOL.

    Force field is empty (only the IMD pull force gets attached); we never step
    it — the test exercises the group-index build and the force expansion only.
    """
    import openmm
    from openmm import unit
    from openmm.app import Element, Topology

    from pymol_imd.backends.base import SystemBundle
    from pymol_imd.imd import IMDConfig, IMDSystem

    top = Topology()
    system = openmm.System()
    carbon = Element.getBySymbol("C")
    coords = []
    n = 0
    for _chain in range(2):
        chain = top.addChain()
        for _res in range(2):
            residue = top.addResidue("ALA", chain)
            for _atom in range(2):
                top.addAtom("C", carbon, residue)
                system.addParticle(12.0)
                coords.append([n * 0.3, 0.0, 0.0])
                n += 1
    positions = np.array(coords) * unit.nanometer
    bundle = SystemBundle(
        system=system,
        topology=top,
        positions=positions,
        backend_name="aa",
        prepared_pdb_path=Path("toy.pdb"),
    )
    return IMDSystem(bundle, IMDConfig(mouse_force=False, drag_level=drag_level))


def _check_constants() -> None:
    from pymol_imd import constants

    assert constants.SUPPORTED_DRAG_LEVELS == ("atom", "residue", "chain")
    assert constants.DEFAULT_DRAG_LEVEL == "atom"
    log.info("constants OK: levels=%s default=%s", constants.SUPPORTED_DRAG_LEVELS, "atom")


def _check_validate() -> None:
    from pymol_imd import commands

    for level in ("atom", "residue", "chain"):
        assert commands._validate_drag_level(level) == level
    try:
        commands._validate_drag_level("resi")
    except ValueError as exc:
        assert "Unknown drag_level" in str(exc), exc
        log.info("_validate_drag_level OK: good levels pass, bad level raises ValueError")
    else:
        raise AssertionError("expected ValueError for an invalid drag_level")


def _check_force_limit() -> None:
    """Check per-atom magnitude caps, directions, boundaries and invalid input."""
    from pymol_imd.imd import cap_pull_forces, compute_pull_forces

    const_k = 10000.0
    max_force_kj = 5000.0
    noise = 1e-3

    # (a) sub-limit drag applies normally: 0.1 nm * 10000 = 1000 < 5000.
    delta = np.zeros((3, 3))
    delta[1] = [0.1, 0.0, 0.0]
    idx, forces, n_exceeded = compute_pull_forces(delta, const_k, max_force_kj, noise)
    assert n_exceeded == 0, n_exceeded
    assert idx.tolist() == [1], idx
    assert np.allclose(forces[1], [1000.0, 0.0, 0.0]), forces[1]

    # (b) Cap only the over-limit atom; preserve its direction and other pulls.
    delta2 = np.zeros((3, 3))
    delta2[0] = [0.05, 0.0, 0.0]  # sub-limit (500), would normally be active
    delta2[2] = [0.6, -0.8, 0.0]
    idx2, forces2, n_exceeded2 = compute_pull_forces(delta2, const_k, max_force_kj, noise)
    assert n_exceeded2 == 1, n_exceeded2
    assert idx2.tolist() == [0, 2], idx2
    np.testing.assert_allclose(forces2, [[500, 0, 0], [0, 0, 0], [3000, -4000, 0]])
    _, smaller, count = compute_pull_forces(delta2, const_k, 1000, noise)
    assert count == 1
    np.testing.assert_allclose(smaller[[0, 2]], [[500, 0, 0], [600, -800, 0]])
    huge = np.array([[1e300, -1e300, 0]])
    bounded, count = cap_pull_forces(huge, max_force_kj)
    assert count == 1 and np.isfinite(bounded).all()
    np.testing.assert_allclose(bounded, np.array([[1, -1, 0]]) * (max_force_kj / np.sqrt(2)))

    # (c) exactly at the limit is NOT over (strict >): 0.5 nm * 10000 = 5000.
    delta3 = np.zeros((2, 3))
    delta3[1] = [0.5, 0.0, 0.0]
    idx3, forces3, n_exceeded3 = compute_pull_forces(delta3, const_k, max_force_kj, noise)
    assert n_exceeded3 == 0, n_exceeded3
    assert idx3.tolist() == [1], idx3
    assert np.allclose(forces3[1], [5000.0, 0.0, 0.0]), forces3[1]

    for invalid in (float("nan"), float("inf")):
        delta3[0, 0] = invalid
        indices, forces, rejected = compute_pull_forces(delta3, const_k, max_force_kj, noise)
        assert rejected == 1 and indices.size == 0 and not forces.any()

    log.info("force-limit OK: per-atom cap preserves directions, sub-limit pulls and boundaries")


def _check_group_indices() -> None:
    imd = _toy_imd_system("atom")
    assert imd._atom_residue_idx.tolist() == [0, 0, 1, 1, 2, 2, 3, 3], imd._atom_residue_idx
    assert imd._atom_chain_idx.tolist() == [0, 0, 0, 0, 1, 1, 1, 1], imd._atom_chain_idx
    assert imd.config.drag_level == "atom"
    log.info("group indices OK: residue=%s chain=%s", "[0,0,1,1,2,2,3,3]", "[0,0,0,0,1,1,1,1]")


def _check_expand_single() -> None:
    """Drag one atom; residue/chain expansion pulls the whole group."""
    imd = _toy_imd_system("chain")
    forces = np.zeros((8, 3))
    forces[0] = [10.0, 0.0, 0.0]  # drag atom 0 (chain 0, residue 0)
    active = np.array([0])

    # chain: atom 0 -> all of chain 0 (atoms 0..3), each gets the mean (=[10,0,0])
    idx, fout, _ = imd._expand_drag_groups(active, forces)
    assert idx.tolist() == [0, 1, 2, 3], idx
    assert np.allclose(fout[[0, 1, 2, 3]], [10.0, 0.0, 0.0]), fout[[0, 1, 2, 3]]
    assert np.allclose(fout[[4, 5, 6, 7]], 0.0)  # chain 1 untouched

    # residue: atom 0 -> only residue 0 (atoms 0,1)
    imd.config.drag_level = "residue"
    idx, fout, _ = imd._expand_drag_groups(active, forces)
    assert idx.tolist() == [0, 1], idx
    assert np.allclose(fout[[0, 1]], [10.0, 0.0, 0.0]), fout[[0, 1]]
    log.info("expand OK (single drag): chain->[0,1,2,3], residue->[0,1], force copied")


def _check_expand_multi() -> None:
    """Drag atoms in two different chains; each chain expands independently."""
    imd = _toy_imd_system("chain")
    forces = np.zeros((8, 3))
    forces[0] = [10.0, 0.0, 0.0]  # chain 0
    forces[4] = [0.0, 5.0, 0.0]  # chain 1
    active = np.array([0, 4])
    idx, fout, _ = imd._expand_drag_groups(active, forces)
    assert idx.tolist() == [0, 1, 2, 3, 4, 5, 6, 7], idx
    assert np.allclose(fout[[0, 1, 2, 3]], [10.0, 0.0, 0.0]), fout[[0, 1, 2, 3]]
    assert np.allclose(fout[[4, 5, 6, 7]], [0.0, 5.0, 0.0]), fout[[4, 5, 6, 7]]
    log.info("expand OK (multi drag): chain 0 -> Fx, chain 1 -> Fy, independent")


def _check_expand_mean() -> None:
    """Two atoms dragged in the SAME residue -> members get their mean force."""
    imd = _toy_imd_system("residue")
    forces = np.zeros((8, 3))
    forces[0] = [10.0, 0.0, 0.0]  # residue 0, atom 0
    forces[1] = [0.0, 0.0, 4.0]  # residue 0, atom 1
    active = np.array([0, 1])
    idx, fout, _ = imd._expand_drag_groups(active, forces)
    assert idx.tolist() == [0, 1], idx
    # mean of [10,0,0] and [0,0,4] = [5,0,2], applied to both members
    assert np.allclose(fout[[0, 1]], [5.0, 0.0, 2.0]), fout[[0, 1]]
    log.info("expand OK (mean): two dragged atoms in a residue -> mean force on all")


def _check_mass_distribution() -> None:
    from pymol_imd.client import PyMOLIMDClient
    from pymol_imd.cuemol.bridge import target_forces
    from pymol_imd.imd import expand_group_forces

    imd = _toy_imd_system("residue")
    imd._particle_masses = np.array([12, 1, 16, 0, 72, 54, 36, 72], dtype=float)
    client = PyMOLIMDClient(None, "unused", const_k=10000, max_force_kj=5000)
    client._particle_masses = imd._particle_masses
    client._group_ids = {"residue": imd._atom_residue_idx, "chain": imd._atom_chain_idx}
    keys = [[str(i // 4), str(i // 2), "MOL", str(i), ""] for i in range(8)]
    for level in ("residue", "chain"):
        imd.config.drag_level = client.drag_level = level
        groups = imd._atom_residue_idx if level == "residue" else imd._atom_chain_idx
        for magnitude in (100, 5000):
            forces = np.zeros((8, 3))
            forces[0] = [magnitude * 0.6, magnitude * 0.8, 0]
            active, actual, limited = imd._expand_drag_groups(np.array([0]), forces)
            members = np.flatnonzero(groups == groups[0])
            mobile = members[imd._particle_masses[members] > 0]
            np.testing.assert_array_equal(active, mobile)
            acceleration = actual[mobile] / imd._particle_masses[mobile, None]
            np.testing.assert_allclose(acceleration, np.tile(acceleration[0], (len(mobile), 1)))
            assert not actual[3].any(), "A massless particle received a pull"
            if magnitude == 100:
                np.testing.assert_allclose(actual.sum(axis=0), forces[0] * len(members))
                assert not limited
            else:
                assert limited and np.isclose(np.linalg.norm(actual, axis=1).max(), 5000)
            socket_active, socket_forces, socket_limited = client._expand_drag_groups(
                np.array([0]), forces, 8
            )
            np.testing.assert_array_equal(socket_active, active)
            np.testing.assert_allclose(socket_forces, actual)
            assert socket_limited == limited
            cue_active, cue_forces, cue_limited = target_forces(
                np.zeros((8, 3)),
                {"index": 0, "position": (forces[0] / 1000).tolist()},
                keys,
                const_k=10000,
                max_force_kj=5000,
                drag_level=level,
                particle_masses=imd._particle_masses,
            )
            np.testing.assert_array_equal(cue_active, active)
            np.testing.assert_allclose(cue_forces, actual)
            assert cue_limited == limited
        forces[4] = [0, 0, 20]
        active, result, _ = expand_group_forces(
            np.array([0, 4]), forces, groups, imd._particle_masses, 5000
        )
        other = np.flatnonzero(groups == groups[4])
        np.testing.assert_allclose(result[other].sum(axis=0), [0, 0, 20 * len(other)])
    log.info(
        "mass distribution OK: equal acceleration, total force, caps, massless sites and viewer parity"
    )


def main() -> int:
    _check_constants()
    _check_validate()
    _check_force_limit()
    _check_group_indices()
    _check_expand_single()
    _check_expand_multi()
    _check_expand_mean()
    _check_mass_distribution()
    log.info("Drag-level smoke test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
