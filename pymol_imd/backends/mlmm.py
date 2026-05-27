"""ONIOM-style ML/MM backend.

Architecture (close to mlmm_toolkit's ONIOM but tailored for live IMD):

  E_total = E_MM(full system) + E_ML(ml_region) - E_MM(ml_region)

The full-system Amber MM force is what OpenMM integrates against. The ML/MM
delta correction ``F_ML - F_MM_subset`` is computed each frame on the
``ml_region`` atoms only and pushed into a dedicated ``CustomExternalForce``
in the driver layer.

MLIP backends are sourced from ``mlmm_toolkit`` so we don't reinvent UMA /
ORB / MACE / AIMNet2 wrappers. The runtime auto-selects CUDA → MPS → CPU
(Apple Silicon supported) via ``mlmm.mlmm_calc._auto_torch_device``.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import openmm
from openmm import unit
from openmm.app import HBonds, NoCutoff

from ..prep import load_and_complete_hydrogens, write_prepared_pdb
from .aa import FORCEFIELD_FILES
from .base import SystemBundle

logger = logging.getLogger(__name__)

SUPPORTED_ML_BACKENDS = ("uma", "orb", "mace", "aimnet2")

# Unit conversion: MLIP returns eV / (eV/Å), OpenMM wants kJ/mol / kJ/(mol·nm).
EV_PER_KJMOL = 96.48533212331  # 1 eV = 96.485... kJ/mol
# F[kJ/mol/nm] = F[eV/Å] * (kJ/mol per eV) * (Å/nm) = F[eV/Å] * 96.485 * 10
EV_PER_A_TO_KJ_PER_MOL_NM = EV_PER_KJMOL * 10.0


def _resolve_ml_indices(topology, ml_selection: str) -> tuple[int, ...]:
    """Resolve a PyMOL-style selection string to a tuple of atom indices.

    Accepted formats:
      * "all"            → every atom (purely-ML, no MM region)
      * "resi 30-50"     → residue id range (1-based, inclusive)
      * "chain A"        → entire chain
      * "resn LIG"       → residues by name
      * "<idx1>,<idx2>,..." → explicit 0-based atom indices
      * "ml_region.pdb"  → path to a PDB file whose atoms (by element + xyz
        match) name the ML region. Useful for the mlmm_toolkit style where
        a separate model PDB designates the ML region.

    The set is computed against the OpenMM ``Topology`` only; PyMOL is not
    consulted to keep this backend usable outside the GUI as well.
    """
    sel = ml_selection.strip()
    if sel == "all":
        return tuple(range(topology.getNumAtoms()))
    atoms = list(topology.atoms())

    if sel.startswith("resi "):
        spec = sel.removeprefix("resi ").strip()
        ranges: list[range] = []
        for piece in spec.split(","):
            piece = piece.strip()
            if "-" in piece:
                lo, hi = piece.split("-", 1)
                ranges.append(range(int(lo), int(hi) + 1))
            else:
                ranges.append(range(int(piece), int(piece) + 1))
        wanted = {i for r in ranges for i in r}
        return tuple(a.index for a in atoms if int(a.residue.id) in wanted)

    if sel.startswith("chain "):
        chain_id = sel.removeprefix("chain ").strip()
        return tuple(a.index for a in atoms if a.residue.chain.id == chain_id)

    if sel.startswith("resn "):
        names = {s.strip() for s in sel.removeprefix("resn ").strip().split(",")}
        return tuple(a.index for a in atoms if a.residue.name in names)

    # Fallback: comma-separated integer indices.
    if all(p.strip().isdigit() for p in sel.split(",")):
        return tuple(int(p) for p in sel.split(","))

    raise ValueError(
        f"Unsupported ml_selection {sel!r}. Use 'all', 'resi <range>', "
        f"'chain <id>', 'resn <name>', or comma-separated indices."
    )


def _build_subset_mm_system(
    full_topology,
    forcefield,
    ml_indices: tuple[int, ...],
):
    """Build an OpenMM System covering only the ML region for MM subtraction.

    Implemented by constructing a fresh ``Topology`` containing only the
    selected atoms (and any bonds whose both ends lie in the selection).
    The forcefield re-parameterizes it, so the returned System reports the
    same Amber MM forces on the ML region atoms as the full System does
    (modulo cross-region 1-2/1-3/1-4 terms, which is the link-atom problem
    we leave for a future revision).
    """
    from openmm.app import Topology

    keep = set(ml_indices)
    sub_top = Topology()
    chain_map: dict = {}
    residue_map: dict = {}
    atom_map: dict = {}

    for src_atom in full_topology.atoms():
        if src_atom.index not in keep:
            continue
        src_residue = src_atom.residue
        src_chain = src_residue.chain
        if src_chain not in chain_map:
            chain_map[src_chain] = sub_top.addChain(src_chain.id)
        if src_residue not in residue_map:
            residue_map[src_residue] = sub_top.addResidue(
                src_residue.name, chain_map[src_chain], src_residue.id
            )
        atom_map[src_atom] = sub_top.addAtom(
            src_atom.name, src_atom.element, residue_map[src_residue]
        )

    for a1, a2 in full_topology.bonds():
        if a1 in atom_map and a2 in atom_map:
            sub_top.addBond(atom_map[a1], atom_map[a2])

    sub_system = forcefield.createSystem(
        sub_top,
        nonbondedMethod=NoCutoff,
        constraints=None,
    )
    return sub_top, sub_system


def _atoms_from_topology(topology, indices: tuple[int, ...]):
    """Return ASE-style element symbols for the requested atom indices."""
    atoms = list(topology.atoms())
    return [atoms[i].element.symbol for i in indices]


class MLMMBackend:
    name = "mlmm"

    def build_system(
        self,
        pdb_path: Path,
        *,
        ligand_sdf: Path | None = None,
        pbc: bool = False,
        ml_selection: str = "all",
        ml_backend: str = "uma",
        ml_device: str = "auto",
        ml_charge: int = 0,
        ml_mult: int = 1,
        constraints: str | None = None,
        **_: object,
    ) -> SystemBundle:
        if ml_backend not in SUPPORTED_ML_BACKENDS:
            raise ValueError(
                f"Unknown ml_backend {ml_backend!r}. Choose from {SUPPORTED_ML_BACKENDS}."
            )

        # mlmm_toolkit is the source of MLIP wrappers + device picker.
        try:
            from mlmm.mlmm_calc import _auto_torch_device, _create_ml_backend
        except ImportError as exc:
            raise RuntimeError(
                "mlmm_toolkit is not installed in this environment. "
                "Run `pixi install` after cloning the fork at ../mlmm_toolkit."
            ) from exc

        from openmm.app import PME, ForceField

        forcefield = ForceField(*FORCEFIELD_FILES)

        modeller = load_and_complete_hydrogens(pdb_path, forcefield)
        # NOTE: ligand_sdf support delegated to AA backend; ML/MM with a
        # custom ligand needs the GAFF parameterisation path too. Keeping
        # it out of scope for the first ONIOM cut.
        if ligand_sdf is not None:
            raise NotImplementedError(
                "MLMM backend does not yet support ligand SDF; bake the "
                "ligand into the input PDB or open an issue."
            )

        nonbonded_method = PME if pbc else NoCutoff
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=HBonds if constraints == "HBonds" else None,
        )

        ml_indices = _resolve_ml_indices(modeller.topology, ml_selection)
        if not ml_indices:
            raise ValueError(
                f"ml_selection {ml_selection!r} matched zero atoms. "
                f"Check the selection or pre-load the PDB into PyMOL."
            )
        logger.info(
            "MLMM: %d / %d atoms selected as ML region (backend=%s)",
            len(ml_indices),
            modeller.topology.getNumAtoms(),
            ml_backend,
        )

        device_str, torch_device = _auto_torch_device(ml_device)
        logger.info("MLMM: ML backend running on %s", device_str)
        mlip = _create_ml_backend(
            ml_backend,
            ml_device=torch_device,
            model_charge=ml_charge,
            model_mult=ml_mult,
        )

        # Build the MM subset for the ML region so we can subtract its
        # Amber MM contribution every frame.
        sub_top, sub_system = _build_subset_mm_system(modeller.topology, forcefield, ml_indices)
        sub_integrator = openmm.VerletIntegrator(0.0 * unit.femtoseconds)
        sub_context = openmm.Context(sub_system, sub_integrator)

        work_dir = Path(tempfile.mkdtemp(prefix="pymol_imd_mlmm_"))
        prepared_pdb = write_prepared_pdb(
            modeller.topology,
            modeller.positions,
            work_dir / f"{pdb_path.stem}_prepared.pdb",
        )

        ml_symbols = _atoms_from_topology(modeller.topology, ml_indices)

        return SystemBundle(
            system=system,
            topology=modeller.topology,
            positions=modeller.positions,
            backend_name=self.name,
            prepared_pdb_path=prepared_pdb,
            pbc=pbc,
            extra={
                "ml_indices": ml_indices,
                "ml_symbols": ml_symbols,
                "ml_backend": mlip,
                "ml_backend_name": ml_backend,
                "ml_subset_context": sub_context,
                "ml_subset_n_atoms": sub_system.getNumParticles(),
            },
        )
