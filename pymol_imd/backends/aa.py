"""All-atom Amber-family backend.

Uses the Amber ff14SB + nucleic-acid + lipid forcefields, GAFF 2 for ligands,
and AM1-BCC charges from OpenFF. Missing hydrogens are added with
``Modeller.addHydrogens`` only — no residue rebuilding.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from openmm import unit
from openmm.app import PME, ForceField, HBonds, Modeller, NoCutoff

from ..prep import load_and_complete_hydrogens, prepare_ligand_am1bcc, write_prepared_pdb
from .base import SystemBundle

logger = logging.getLogger(__name__)

# amber14-all.xml already bundles protein.ff14SB, DNA.OL15, RNA.OL3, lipid17.
# Adding them individually triggers "Residue template ALA already exists" in
# modern OpenMM (>=8.1). Water/ions are loaded conditionally if the input has them.
FORCEFIELD_FILES = (
    "amber14-all.xml",
    "amber14/tip3pfb.xml",
)


class AABackend:
    name = "aa"
    # |F|_max = max_pull_nm × const_k = 0.5 × 50000 = 25000 kJ/mol/nm, about
    # 10% of a typical bond stretching constant — strong drag, still safe.
    recommended_const_k = 50000.0

    def build_system(
        self,
        pdb_path: Path,
        *,
        ligand_sdf: Path | None = None,
        pbc: bool = False,
        box_padding_nm: float = 1.0,
        constraints: str | None = "HBonds",
        **_: object,
    ) -> SystemBundle:
        pdb_path = Path(pdb_path)
        forcefield = ForceField(*FORCEFIELD_FILES)

        if ligand_sdf is not None:
            ligand = prepare_ligand_am1bcc(ligand_sdf)
            from openmmforcefields.generators import GAFFTemplateGenerator

            generator = GAFFTemplateGenerator(molecules=ligand, forcefield="gaff-2.2.20")
            forcefield.registerTemplateGenerator(generator.generator)
            logger.info("Registered GAFF template for ligand %s", ligand.name)
        else:
            ligand = None

        modeller = load_and_complete_hydrogens(pdb_path, forcefield)
        if ligand is not None:
            ligand_top = ligand.to_topology()
            modeller.add(ligand_top.to_openmm(), ligand_top.get_positions().to_openmm())

        nonbonded_method = PME if pbc else NoCutoff
        if pbc:
            _set_bounding_box(modeller, padding_nm=box_padding_nm)

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=HBonds if constraints == "HBonds" else None,
        )

        work_dir = Path(tempfile.mkdtemp(prefix="pymol_imd_aa_"))
        prepared_pdb = write_prepared_pdb(
            modeller.topology,
            modeller.positions,
            work_dir / f"{pdb_path.stem}_prepared.pdb",
        )

        return SystemBundle(
            system=system,
            topology=modeller.topology,
            positions=modeller.positions,
            backend_name=self.name,
            prepared_pdb_path=prepared_pdb,
            pbc=pbc,
        )


def _set_bounding_box(modeller: Modeller, *, padding_nm: float) -> None:
    """Set a cubic periodic box that encloses the molecule with ``padding_nm`` slack."""
    positions = modeller.positions.value_in_unit(unit.nanometer)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    side = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) + 2.0 * padding_nm
    box = side * unit.nanometer
    modeller.topology.setUnitCellDimensions((box, box, box))
