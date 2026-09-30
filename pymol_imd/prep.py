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
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from openmm.app import ForceField, Modeller, PDBFile, Topology, element
from openmm.unit import Quantity

from . import constants

logger = logging.getLogger(__name__)

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def _download_atomic(url: str, dest: Path, *, validate=None) -> None:
    """Download to a sibling temporary file, validate, then atomically replace."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        if validate is not None:
            validate(tmp_path)
        os.replace(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)


def _validate_pdb(path: Path) -> None:
    if path.stat().st_size == 0:
        raise ValueError("downloaded PDB is empty")
    try:
        pdb = PDBFile(str(path))
    except Exception as exc:
        raise ValueError(f"downloaded file is not a valid PDB: {exc}") from exc
    if pdb.topology.getNumAtoms() == 0:
        raise ValueError("downloaded PDB contains no atoms")


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
    if dest.exists():
        try:
            _validate_pdb(dest)
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring invalid cached PDB %s: %s", dest, exc)
        else:
            logger.info("Using cached PDB %s", dest)
            return dest

    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    logger.info("Downloading %s", url)
    try:
        _download_atomic(url, dest, validate=_validate_pdb)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    return dest


def load_and_complete_hydrogens(
    pdb_path: Path | str,
    forcefield: ForceField,
    *,
    ph: float = constants.AA_PH,
) -> Modeller:
    """Load ``pdb_path`` and add missing hydrogens using OpenMM's ``Modeller``.

    Returns a fully-constructed :class:`openmm.app.Modeller` whose topology and
    positions are ready to be fed into ``forcefield.createSystem``. If the input
    already has full hydrogens, ``addHydrogens`` is a no-op.
    """
    pdb = PDBFile(str(pdb_path))
    modeller = Modeller(pdb.topology, pdb.positions)
    variants = _template_hydrogen_variants(modeller.topology, forcefield)
    modeller.addHydrogens(forcefield=forcefield, pH=ph, variants=variants)
    return modeller


def _template_hydrogen_variants(topology: Topology, forcefield: ForceField):
    """Use unambiguous closed-residue templates for nonstandard hydrogens."""
    Modeller._loadStandardHydrogenDefinitions()
    residues = list(topology.residues())
    unknown = [res for res in residues if res.name not in Modeller._residueHydrogens]
    if not unknown:
        return None

    templates = {}
    for template in forcefield._templates.values():
        if template.externalBonds:
            continue
        heavy = frozenset(
            (atom.name, atom.element) for atom in template.atoms if atom.element != element.hydrogen
        )
        bonds = set()
        hydrogens = []
        for i, j in template.bonds:
            a, b = template.atoms[i], template.atoms[j]
            if a.element == element.hydrogen:
                hydrogens.append((a.name, b.name))
            elif b.element == element.hydrogen:
                hydrogens.append((b.name, a.name))
            else:
                bonds.add(frozenset((a.name, b.name)))
        if hydrogens:
            templates.setdefault(heavy, []).append((bonds, tuple(sorted(hydrogens))))

    residue_bonds = {res: set() for res in unknown}
    external = set()
    for a, b in topology.bonds():
        if a.residue is not b.residue:
            external.update((a.residue, b.residue))
        elif a.residue in residue_bonds and element.hydrogen not in (a.element, b.element):
            residue_bonds[a.residue].add(frozenset((a.name, b.name)))

    variants = [None] * len(residues)
    for residue in unknown:
        if residue in external:
            continue
        atoms = list(residue.atoms())
        heavy = frozenset(
            (atom.name, atom.element) for atom in atoms if atom.element != element.hydrogen
        )
        present = {atom.name for atom in atoms if atom.element == element.hydrogen}
        choices = {
            hydrogens
            for bonds, hydrogens in templates.get(heavy, ())
            if bonds == residue_bonds[residue] and present.issubset(name for name, _ in hydrogens)
        }
        if len(choices) > 1:
            raise ValueError(
                f"Hydrogen completion is ambiguous for residue {residue.id} ({residue.name}); "
                "provide its hydrogens explicitly."
            )
        if choices:
            hydrogens = choices.pop()
            if len(present) < len(hydrogens):
                variants[residue.index] = list(hydrogens)
    return variants


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


def prepare_ligand_am1bcc(
    sdf_path: Path | str,
    *,
    partial_charge_method: str = constants.AA_PARTIAL_CHARGE_METHOD,
):
    """Load an SDF ligand and assign partial charges via OpenFF.

    Returns the ``openff.toolkit.Molecule`` ready to be handed to
    :class:`openmmforcefields.generators.GAFFTemplateGenerator`.
    ``partial_charge_method`` defaults to AM1-BCC (the standard for GAFF); it
    is exposed for completeness but rarely needs changing.

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
    mol.assign_partial_charges(partial_charge_method=partial_charge_method)
    return mol
