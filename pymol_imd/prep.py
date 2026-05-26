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


def write_prepared_pdb(
    topology: Topology,
    positions: Quantity,
    out_path: Path | str,
) -> Path:
    """Write a PDB so PyMOL can be re-loaded against the OpenMM atom order.

    Required because ``Modeller.addHydrogens`` and ligand merging insert atoms
    that are absent from the input PDB; without re-loading PyMOL with this
    file, ``index - 1`` lookups in the IMD sync step land on the wrong atoms
    and the visible structure collapses.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        PDBFile.writeFile(topology, positions, fh, keepIds=True)
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
