"""Coarse-grained Martini backend.

The pipeline is two-step::

    AA PDB --(vermouth / martinize2)--> CG .top + .itp + .gro
           --(martini_openmm)----------> OpenMM System / Topology / Positions

Most of the heavy lifting is delegated to upstream tools; this module wires
them together and produces a :class:`SystemBundle` that the IMD driver can
consume identically to the AA path.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from openmm import unit
from openmm.app import PDBFile

from ..prep import write_prepared_pdb
from .base import SystemBundle

logger = logging.getLogger(__name__)

# Forcefield label fed to martinize2 (-ff). Run `martinize2 -list-ff` to see
# what's bundled; "martini3001" is Martini 3.0.0.1, the current standard.
MARTINI_FF = "martini3001"

# Martini 3 forcefield itp files. martinize2-generated .top references
# "martini.itp" by name, so we save the main forcefield under that exact
# filename so martini_openmm's #include lookup resolves.
_MARTINI3_FILES = {
    "martini.itp": (
        "https://raw.githubusercontent.com/marrink-lab/martini-forcefields/"
        "main/martini_forcefields/regular/v3.0.0/gmx_files/martini_v3.0.0.itp"
    ),
}


def _martini_cache_dir() -> Path:
    """Per-user cache for the Martini 3 itp set."""
    base = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(base) / "pymol_imd" / "martini3"


def _ensure_martini_ff(cache_dir: Path | None = None) -> Path:
    """Make sure Martini 3 itp files are present locally; return include dir.

    Downloads once into ``cache_dir`` (defaults to ``~/.cache/pymol_imd/martini3``)
    on first use. Subsequent calls reuse the cached files.
    """
    cache_dir = cache_dir or _martini_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, url in _MARTINI3_FILES.items():
        dest = cache_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        logger.info("Fetching Martini 3 forcefield: %s -> %s", url, dest)
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download Martini 3 forcefield {name} from {url}: {exc}. "
                f"Place the file at {dest} manually or pass include_dir=... to setup_cg."
            ) from exc
    return cache_dir


# Mapping from MDTraj DSSP one-letter codes to the Martini SS alphabet
# martinize2 expects via -ss. MDTraj's `simplified=True` already collapses
# things to H/E/C; we just translate to the lower-case martinize alphabet.
_DSSP_TO_MARTINI = {"H": "H", "E": "E", "C": "C", " ": "C"}


def _compute_ss_string(pdb_path: Path) -> str:
    """Return a per-protein-residue SS string ("CCHHH...") via MDTraj.

    We compute SS here so vermouth's external DSSP integration (which only
    understands DSSP 2.x/3.x CLI args) is bypassed. The input PDB itself is
    handed to martinize2 untouched — HOH / ligands trigger non-fatal
    "unknown-residue" warnings that we suppress with a large -maxwarn.
    """
    import mdtraj as md

    traj = md.load(str(pdb_path))
    protein_idx = traj.topology.select("protein")
    if protein_idx.size == 0:
        raise RuntimeError(f"No protein atoms found in {pdb_path}")
    protein = traj.atom_slice(protein_idx)
    ss = md.compute_dssp(protein, simplified=True)[0]
    return "".join(_DSSP_TO_MARTINI.get(c, "C") for c in ss)


class CGBackend:
    name = "cg"

    def build_system(
        self,
        pdb_path: Path,
        *,
        ligand_sdf: Path | None = None,
        pbc: bool = False,
        ff: str = MARTINI_FF,
        elastic_network: bool = True,
        work_dir: Path | None = None,
        include_dir: Path | None = None,
        **_: object,
    ) -> SystemBundle:
        if ligand_sdf is not None:
            raise NotImplementedError(
                "CG backend does not yet support custom ligand SDFs; "
                "the Martini ligand-mapping pipeline is not wired up. "
                "Open an issue with the ligand SMILES if needed."
            )

        if shutil.which("martinize2") is None:
            raise RuntimeError(
                "martinize2 not found in PATH. Activate the pixi env first "
                "(`pixi shell`) — vermouth provides this binary."
            )

        work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="pymol_imd_cg_"))
        work_dir.mkdir(parents=True, exist_ok=True)
        # martinize2 ignores the extension and always writes PDB coords;
        # name it accordingly so downstream readers don't get confused.
        cg_pdb = work_dir / "cg.pdb"
        cg_top = work_dir / "cg.top"

        ss_string = _compute_ss_string(pdb_path)
        # -maxwarn 99999: respect the input as-is. Non-protein residues (HOH,
        # ions, ligands) raise "unknown-residue" warnings but martinize2
        # itself does not abort on those, only on its warning-count gate.
        cmd = [
            "martinize2",
            "-f",
            str(pdb_path),
            "-x",
            str(cg_pdb),
            "-o",
            str(cg_top),
            "-ff",
            ff,
            "-ss",
            ss_string,
            "-maxwarn",
            "99999",
        ]
        if elastic_network:
            cmd += ["-elastic"]
        logger.info("Running %s", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=work_dir)

        coords = PDBFile(str(cg_pdb))
        # martini_openmm provides a drop-in replacement for GromacsTopFile that
        # understands Martini non-standard directives. The generated .top
        # references `martini.itp`, which must live in `include_dir`.
        from martini_openmm import MartiniTopFile  # type: ignore

        martini_include = include_dir or _ensure_martini_ff()

        # Martini uses a 1.1 nm cutoff and therefore requires periodic vectors.
        # If the user passed pbc=False (the AA default), we still need a box;
        # build one around the molecule + padding to keep things effectively
        # vacuum-like for short runs.
        box_vectors = coords.topology.getPeriodicBoxVectors()
        if box_vectors is None:
            box_vectors = _bounding_box_vectors(coords.positions, padding_nm=1.0)
            coords.topology.setPeriodicBoxVectors(box_vectors)
        top = MartiniTopFile(
            str(cg_top),
            periodicBoxVectors=box_vectors,
            includeDir=str(martini_include),
        )
        system = top.create_system(nonbonded_cutoff=1.1 * unit.nanometer)

        # PDBFile.writeFile chokes on Quantity-wrapped box vectors when writing
        # CRYST1 (openmm 8.x: Quantity[0].__float__ returns Quantity, not float).
        # Strip the box for the PDB output only; MD still uses the System's
        # box vectors which are set independently.
        saved_box = top.topology.getPeriodicBoxVectors()
        top.topology.setPeriodicBoxVectors(None)
        try:
            prepared_pdb = write_prepared_pdb(
                top.topology,
                coords.positions,
                work_dir / "cg_prepared.pdb",
            )
        finally:
            top.topology.setPeriodicBoxVectors(saved_box)

        return SystemBundle(
            system=system,
            topology=top.topology,
            positions=coords.positions,
            backend_name=self.name,
            prepared_pdb_path=prepared_pdb,
            pbc=pbc,
        )


def _bounding_box_vectors(positions, padding_nm: float = 1.0):
    """Cubic periodic vectors enclosing positions with padding (nm).

    Returns a tuple of three ``Vec3 * nanometer`` Quantities, the form
    ``Topology.setPeriodicBoxVectors`` and ``PDBFile.writeFile`` both expect.
    """
    from openmm import Vec3

    pos_nm = positions.value_in_unit(unit.nanometer)
    xs = [p[0] for p in pos_nm]
    ys = [p[1] for p in pos_nm]
    zs = [p[2] for p in pos_nm]
    side = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) + 2.0 * padding_nm
    return (
        Vec3(side, 0.0, 0.0) * unit.nanometer,
        Vec3(0.0, side, 0.0) * unit.nanometer,
        Vec3(0.0, 0.0, side) * unit.nanometer,
    )
