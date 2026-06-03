"""Headless smoke test: mouse-drag granularity (atom / residue / chain).

PyMOL's 3-Button Editing drag moves a covalent fragment and ignores
``mouse_selection_mode``, so residue / chain granularity is realized plugin-side
by expanding a per-atom pull to the whole residue / chain. This test verifies
that expansion WITHOUT a PyMOL window, on a tiny synthetic topology
(2 chains x 2 residues x 2 atoms = 8 atoms):

  * the drag-level constants are consistent and ``IMDConfig`` defaults to atom,
  * ``commands._validate_drag_level`` rejects unknown levels,
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
    idx, fout = imd._expand_drag_groups(active, forces)
    assert idx.tolist() == [0, 1, 2, 3], idx
    assert np.allclose(fout[[0, 1, 2, 3]], [10.0, 0.0, 0.0]), fout[[0, 1, 2, 3]]
    assert np.allclose(fout[[4, 5, 6, 7]], 0.0)  # chain 1 untouched

    # residue: atom 0 -> only residue 0 (atoms 0,1)
    imd.config.drag_level = "residue"
    idx, fout = imd._expand_drag_groups(active, forces)
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
    idx, fout = imd._expand_drag_groups(active, forces)
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
    idx, fout = imd._expand_drag_groups(active, forces)
    assert idx.tolist() == [0, 1], idx
    # mean of [10,0,0] and [0,0,4] = [5,0,2], applied to both members
    assert np.allclose(fout[[0, 1]], [5.0, 0.0, 2.0]), fout[[0, 1]]
    log.info("expand OK (mean): two dragged atoms in a residue -> mean force on all")


def main() -> int:
    _check_constants()
    _check_validate()
    _check_group_indices()
    _check_expand_single()
    _check_expand_multi()
    _check_expand_mean()
    log.info("Drag-level smoke test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
