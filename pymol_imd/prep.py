"""Structure preparation utilities shared across backends.

Currently provides:
    * :func:`fetch_pdb`: download a PDB by ID from RCSB.
    * :func:`load_and_complete_hydrogens`: read a PDB into OpenMM and add missing
      hydrogens via :class:`openmm.app.Modeller`.
    * :func:`prepare_ligand`: load an SDF and assign AM1-BCC charges via OpenFF.

Hydrogen completion is intentionally kept to ``Modeller.addHydrogens`` only;
heavier rebuilding (missing residues, side-chain repair) is out of scope.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path

from openmm.app import ForceField, Modeller, PDBFile, Topology
from openmm.unit import Quantity

logger = logging.getLogger(__name__)

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def fetch_pdb(pdb_id: str, out_dir: Path | str = ".") -> Path:
    """Download ``pdb_id``.pdb from RCSB into ``out_dir`` and return the path.

    Idempotent: if the file already exists it is returned without re-download.
    Raises ``RuntimeError`` on network/HTTP failure with a clear message.
    """
    pdb_id = pdb_id.strip().lower()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise ValueError(f"PDB ID must be a 4-character alphanumeric code, got {pdb_id!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{pdb_id}.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Using cached PDB %s", dest)
        return dest

    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    logger.info("Downloading %s", url)
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    return dest


def load_and_complete_hydrogens(
    pdb_path: Path | str,
    forcefield: ForceField,
    *,
    ph: float = 7.4,
) -> Modeller:
    """Load ``pdb_path`` and add missing hydrogens using OpenMM's ``Modeller``.

    Returns a fully-constructed :class:`openmm.app.Modeller` whose topology and
    positions are ready to be fed into ``forcefield.createSystem``. If the input
    already has full hydrogens, ``addHydrogens`` is a no-op.
    """
    pdb = PDBFile(str(pdb_path))
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield=forcefield, pH=ph)
    return modeller


def _chemical_bond_only(atom_a, atom_b) -> bool:
    """Return True only for bonds that look like a true chemical connection.

    For Martini CG: chemical bonds are (a) intra-residue (BB-SC, SC-SC) and
    (b) sequential BB-BB peptide bonds (residue id differs by 1, same chain).
    Anything else — typically non-sequential BB-BB — is an elastic-network
    spring that we don't want in PyMOL's CONECT records.
    """
    ra = atom_a.residue
    rb = atom_b.residue
    if ra is rb:
        return True
    if ra.chain is not rb.chain:
        return False
    try:
        return abs(int(ra.id) - int(rb.id)) == 1
    except (ValueError, TypeError):
        return False


def _build_atom_to_pdb_serial(topology: Topology) -> dict:
    """Map each ``Atom`` to the 1-based serial number ``PDBFile.writeModel``
    will assign it.

    Must mirror ``PDBFile.writeModel``'s iteration exactly: each chain's
    final residue triggers a TER record that consumes one extra serial,
    shifting every subsequent atom's serial by +1 per chain. Failing to
    account for that makes CONECT records point at the wrong atoms — a
    visible disaster for waters / heteroatoms past the first chain.
    """
    mapping: dict = {}
    serial = 1
    for chain in topology.chains():
        residues = list(chain.residues())
        for res_idx, res in enumerate(residues):
            for atom in res.atoms():
                mapping[atom] = serial
                serial += 1
            if res_idx == len(residues) - 1:
                serial += 1  # TER record consumes one serial slot
    return mapping


def write_prepared_pdb(
    topology: Topology,
    positions: Quantity,
    out_path: Path | str,
    *,
    bond_filter=None,
) -> Path:
    """Write a PDB so PyMOL can be re-loaded against the OpenMM atom order.

    Required because ``Modeller.addHydrogens`` and ligand merging insert atoms
    that are absent from the input PDB; without re-loading PyMOL with this
    file, ``index - 1`` lookups in the IMD sync step land on the wrong atoms
    and the visible structure collapses.

    Emits ``CONECT`` records from ``topology.bonds()`` so PyMOL can draw
    bonds. Pass ``bond_filter=_chemical_bond_only`` (or a custom callable)
    to suppress Martini elastic-network springs etc.; the OpenMM ``System``
    still holds the full bond list, so MD physics is unaffected.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    atom_serial = _build_atom_to_pdb_serial(topology)

    with open(out_path, "w") as fh:
        PDBFile.writeHeader(topology, fh)
        PDBFile.writeModel(topology, positions, fh, modelIndex=1)
        for atom_a, atom_b in topology.bonds():
            if bond_filter is not None and not bond_filter(atom_a, atom_b):
                continue
            fh.write(f"CONECT{atom_serial[atom_a]:>5d}{atom_serial[atom_b]:>5d}\n")
        PDBFile.writeFooter(topology, fh)
    return out_path


def prepare_ligand_am1bcc(sdf_path: Path | str):
    """Load an SDF ligand and assign AM1-BCC partial charges via OpenFF.

    Returns the ``openff.toolkit.Molecule`` ready to be handed to
    :class:`openmmforcefields.generators.GAFFTemplateGenerator`.

    Imported lazily so that the AA backend stays importable even if the user is
    only running CG simulations.
    """
    from openff.toolkit import Molecule

    sdf_path = Path(sdf_path)
    if not sdf_path.exists():
        raise FileNotFoundError(f"Ligand SDF not found: {sdf_path}")
    mol = Molecule.from_file(str(sdf_path))
    if isinstance(mol, list):
        if len(mol) != 1:
            raise ValueError(f"Expected exactly one molecule in {sdf_path}, got {len(mol)}")
        mol = mol[0]
    mol.assign_partial_charges(partial_charge_method="am1bcc")
    return mol
