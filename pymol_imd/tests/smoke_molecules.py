"""Validate DNA, RNA, lipids, and AM1-BCC ligands with the AA backend."""

from __future__ import annotations

import argparse
import io
import logging
import tempfile
from pathlib import Path

import numpy as np
import openmm.app
from openmm import CustomGBForce, unit
from openmm.app import Modeller, PDBFile

from pymol_imd.backends.aa import AABackend
from pymol_imd.imd import IMDConfig, IMDSystem
from pymol_imd.tests.smoke_pymol import _write_peptide

log = logging.getLogger("smoke-molecules")


def write_inputs(root: Path) -> dict[str, Path]:
    """Generate small molecules from installed, versioned molecular templates."""
    from pymol import cmd
    from rdkit import Chem
    from rdkit.Chem import AllChem

    paths = {}
    for kind, sequence in (("dna", "ACGT"), ("rna", "ACGU")):
        obj = cmd.get_unused_name(f"imd_test_{kind}")
        try:
            cmd.fnab(sequence, name=obj, mode=kind.upper(), dbl_helix=0)
            cmd.remove(f"{obj} and hydro")
            # Use the standard hydroxyl-terminated strand, without a 5' phosphate.
            cmd.remove(f"{obj} and resi 1 and name P+O1P+O2P")
            paths[kind] = root / f"{kind}.pdb"
            cmd.save(str(paths[kind]), obj)
        finally:
            cmd.delete(obj)

    for lipid in ("POPC", "POPE"):
        source = Path(openmm.app.__file__).parent / "data" / f"{lipid}.pdb"
        pdb = PDBFile(str(source))
        modeller = Modeller(pdb.topology, pdb.positions)
        first = next(modeller.topology.residues())
        modeller.delete([res for res in modeller.topology.residues() if res is not first])
        modeller.topology.setPeriodicBoxVectors(None)
        for with_hydrogens in (True, False):
            name = lipid.lower() if with_hydrogens else f"{lipid.lower()}_no_h"
            if not with_hydrogens:
                modeller.delete(
                    [atom for atom in modeller.topology.atoms() if atom.element.symbol == "H"]
                )
            paths[name] = root / f"{name}.pdb"
            with paths[name].open("w") as stream:
                PDBFile.writeFile(modeller.topology, modeller.positions, stream)

    paths["peptide"] = root / "peptide.pdb"
    _write_peptide(paths["peptide"])
    ligand = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(ligand, randomSeed=42) == 0
    AllChem.UFFOptimizeMolecule(ligand)
    conf = ligand.GetConformer()
    for i in range(ligand.GetNumAtoms()):
        xyz = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (xyz.x + 20, xyz.y, xyz.z))
    ligand.SetProp("_Name", "ethanol")
    paths["sdf"] = root / "ligand.sdf"
    with Chem.SDWriter(str(paths["sdf"])) as writer:
        writer.write(ligand)
    peptide = PDBFile(str(paths["peptide"]))
    ligand_pdb = PDBFile(io.StringIO(Chem.MolToPDBBlock(ligand)))
    modeller = Modeller(peptide.topology, peptide.positions)
    modeller.add(ligand_pdb.topology, ligand_pdb.positions)
    paths["complex"] = root / "complex.pdb"
    with paths["complex"].open("w") as stream:
        PDBFile.writeFile(modeller.topology, modeller.positions, stream)
    return paths


def _heavy_positions(topology, positions) -> dict:
    coords = positions.value_in_unit(unit.nanometer)
    return {
        (atom.residue.index, atom.name): coords[atom.index]
        for atom in topology.atoms()
        if atom.element.symbol != "H"
    }


def _check_case(name: str, pdb: Path, root: Path, platform: str, **kwargs) -> None:
    source = PDBFile(str(pdb))
    bundle = AABackend().build_system(pdb, work_dir=root / name, **kwargs)
    original = _heavy_positions(source.topology, source.positions)
    prepared = _heavy_positions(bundle.topology, bundle.positions)
    for atom, xyz in original.items():
        np.testing.assert_allclose(prepared[atom], xyz, atol=1e-12)
    assert not bundle.system.usesPeriodicBoundaryConditions()
    assert any(isinstance(force, CustomGBForce) for force in bundle.system.getForces())
    counts = {"dna": 126, "rna": 127, "popc": 134, "pope": 125}
    expected = 42 if name.startswith("ligand_") else counts[name.removesuffix("_no_h")]
    assert bundle.topology.getNumAtoms() == expected, (name, bundle.topology.getNumAtoms())
    if name.startswith("ligand_"):
        assert bundle.topology.getNumResidues() == 4, "The bound ligand was duplicated"
    system = IMDSystem(bundle, IMDConfig(platform=platform, nsteps_per_frame=1, mouse_force=False))
    system.simulation.minimizeEnergy(maxIterations=20)
    system.running = True
    for _ in range(3):
        alive, state = system.advance_openmm(get_forces=True)
        assert alive
        assert np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        assert np.isfinite(forces).all()
    log.info("%s OK: %d atoms, GBn2, unchanged heavy atoms, finite MD", name, expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = write_inputs(root)
        for name in ("dna", "rna", "popc", "popc_no_h", "pope", "pope_no_h"):
            _check_case(name, paths[name], root, args.platform)
        for name, source in (("ligand_append", "peptide"), ("ligand_in_pdb", "complex")):
            _check_case(name, paths[source], root, args.platform, ligand_sdf=paths["sdf"])
    log.info("Molecule smoke test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
