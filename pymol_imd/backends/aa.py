"""All-atom Amber-family backend.

Uses the Amber ff14SB + nucleic-acid + lipid forcefields, GAFF 2 for ligands,
and AM1-BCC charges from OpenFF. Missing hydrogens are added with
``Modeller.addHydrogens`` only — no residue rebuilding.
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

from openmm import unit
from openmm.app import PME, ForceField, HBonds, Modeller, NoCutoff

from .. import constants
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
    # Per-backend recommendations (values centralized in constants.py). The
    # actual per-atom |F| is bounded by IMDConfig.max_force_kj regardless of
    # const_k, so const_k mostly controls how aggressively the pull responds
    # to small drags before it saturates at the cap.
    recommended_const_k = constants.AA_RECOMMENDED_CONST_K
    recommended_integrator = constants.AA_INTEGRATOR
    recommended_timestep_ps = constants.AA_TIMESTEP_PS

    def build_system(
        self,
        pdb_path: Path,
        *,
        ligand_sdf: Path | None = None,
        pbc: bool = False,
        box_padding_nm: float = constants.AA_BOX_PADDING_NM,
        constraints: str | None = "HBonds",
        nonbonded_cutoff_nm: float = constants.AA_NONBONDED_CUTOFF_NM,
        gaff_version: str = constants.AA_GAFF_VERSION,
        ph: float = constants.AA_PH,
        implicit_solvent: str = constants.AA_IMPLICIT_SOLVENT,
        work_dir: Path | None = None,
        **_: object,
    ) -> SystemBundle:
        pdb_path = Path(pdb_path)
        if not math.isfinite(nonbonded_cutoff_nm) or nonbonded_cutoff_nm <= 0.0:
            raise ValueError("nonbonded_cutoff_nm must be finite and > 0")
        if not math.isfinite(box_padding_nm) or box_padding_nm < 0.0:
            raise ValueError("box_padding_nm must be finite and >= 0")
        if not math.isfinite(ph) or not 0.0 <= ph <= 14.0:
            raise ValueError("ph must be finite and between 0 and 14")
        # GB implicit solvent (default) is selected by appending its XML to the
        # ForceField file list; the resolver yields None for vacuo / under pbc.
        implicit_xml = constants.resolve_implicit_solvent(implicit_solvent, pbc=pbc)
        ff_files = (*FORCEFIELD_FILES, implicit_xml) if implicit_xml else FORCEFIELD_FILES
        forcefield = ForceField(*ff_files)

        if ligand_sdf is not None:
            from copy import deepcopy
            from io import StringIO

            ligand = prepare_ligand_am1bcc(ligand_sdf)
            from openmmforcefields.generators import GAFFTemplateGenerator

            generator = GAFFTemplateGenerator(molecules=ligand, forcefield=gaff_version)
            # Expose the ligand's hydrogen template before Modeller runs.
            # GAFF generation changes conformers, so preserve the input copy.
            ffxml = generator.generate_residue_template(deepcopy(ligand))
            forcefield.loadFile(StringIO(ffxml))
            logger.info("Registered GAFF template for ligand %s", ligand.name)
        else:
            ligand = None

        modeller = load_and_complete_hydrogens(pdb_path, forcefield, ph=ph)
        if ligand is not None:
            ligand_template = ligand.to_smiles()
            existing = sum(
                template.name == ligand_template
                for template in forcefield.getMatchingTemplates(modeller.topology)
            )
            if existing:
                logger.info("Using %d existing ligand residue(s) from the input PDB", existing)
            else:
                ligand_top = ligand.to_topology()
                modeller.add(ligand_top.to_openmm(), ligand_top.get_positions().to_openmm())

        nonbonded_method = PME if pbc else NoCutoff
        if pbc:
            _set_bounding_box(
                modeller,
                padding_nm=box_padding_nm,
                nonbonded_cutoff_nm=nonbonded_cutoff_nm,
            )

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=nonbonded_cutoff_nm * unit.nanometer,
            constraints=HBonds if constraints == "HBonds" else None,
        )

        work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="pymol_imd_aa_"))
        work_dir.mkdir(parents=True, exist_ok=True)
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


def _set_bounding_box(
    modeller: Modeller,
    *,
    padding_nm: float,
    nonbonded_cutoff_nm: float = constants.AA_NONBONDED_CUTOFF_NM,
) -> None:
    """Set a cubic periodic box that encloses the molecule with ``padding_nm`` slack."""
    from openmm import Vec3

    positions = modeller.positions.value_in_unit(unit.nanometer)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    requested_side = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) + 2.0 * padding_nm
    minimum_side = 2.0 * nonbonded_cutoff_nm + constants.MINIMUM_BOX_MARGIN_NM
    side = max(requested_side, minimum_side)
    # The unit must wrap the WHOLE triple: ``(Vec3, Vec3, Vec3) * nanometer``
    # yields one Quantity(list-of-Vec3), which is what getPeriodicBoxVectors
    # returns and what computeLengthsAndAngles strips on PDB write. Uniting each
    # Vec3 separately (or passing scalar Quantities to setUnitCellDimensions)
    # leaves the CRYST1 lengths as Quantity and crashes PDBFile.writeHeader's
    # float formatting under pbc=True. (CG's _bounding_box_vectors unites each
    # Vec3 separately but dodges this by copying martinize2's PDB, never writing
    # one through PDBFile.)
    modeller.topology.setPeriodicBoxVectors(
        (
            Vec3(side, 0.0, 0.0),
            Vec3(0.0, side, 0.0),
            Vec3(0.0, 0.0, side),
        )
        * unit.nanometer
    )
