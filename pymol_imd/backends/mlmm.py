"""ONIOM-style ML/MM backend (mlmm_toolkit-free implementation).

Architecture (close to ONIOM but tailored for live IMD):

  E_total = E_MM(full system) + E_ML(ml_region) - E_MM(ml_region)

The full-system Amber MM force is what OpenMM integrates against. The ML/MM
delta correction ``F_ML - F_MM_subset`` is computed each frame on the
``ml_region`` atoms only and pushed into a dedicated ``CustomExternalForce``
in the driver layer.

MLIP backends (UMA / ORB / MACE / AIMNet2) are constructed directly here as
ASE ``Calculator`` instances. The IMD driver feeds each backend an
``ase.Atoms`` object every frame and reads back energy + forces — the same
API works for all four backends, no external dispatcher needed.
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

# Unit conversion: ASE returns eV / (eV/Å), OpenMM wants kJ/mol / kJ/(mol·nm).
EV_PER_KJMOL = 96.48533212331  # 1 eV = 96.485... kJ/mol
EV_PER_A_TO_KJ_PER_MOL_NM = EV_PER_KJMOL * 10.0


# ----------------------------------------------------------------------------
# Device selection
# ----------------------------------------------------------------------------
def _auto_torch_device(prefer: str = "auto", cuda_idx: int = 0):
    """Pick the best torch device, preferring CUDA, then MPS, then CPU.

    Returns ``(device_str, torch.device)``. ``device_str`` is the canonical
    short name ("cuda" / "mps" / "cpu") that ASE-side calculators accept.
    """
    import torch

    if prefer == "auto":
        if torch.cuda.is_available():
            return "cuda", torch.device(f"cuda:{cuda_idx}")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps", torch.device("mps")
        return "cpu", torch.device("cpu")
    if prefer == "cuda":
        return "cuda", torch.device(f"cuda:{cuda_idx}")
    if prefer == "mps":
        return "mps", torch.device("mps")
    return "cpu", torch.device("cpu")


# ----------------------------------------------------------------------------
# Per-backend ASE Calculator construction
# ----------------------------------------------------------------------------
def _build_mace_calculator(model: str, device_str: str, dtype: str):
    """Resolve a MACE model name and return an ASE Calculator.

    mace-torch ships several factories: ``mace_off`` (MACE-OFF23
    small/medium/large), ``mace_mp`` (MACE-MP materials), ``mace_omol``
    (MACE-OMOL chemistry), ``mace_anicc`` (ANI-1CC fit). We dispatch on a
    case-insensitive prefix of the model label.

    Note on MPS: Apple Silicon's MPS backend cannot allocate float64
    storage, but the MACE-OMOL model file is stored as float64 weights.
    Loading on MPS therefore crashes at ``torch.empty(dtype=float64,
    device='mps')`` inside torch's tensor rebuild path. We fall back to
    CPU automatically here — CPU MACE is still fast enough for IMD on
    small ML regions and is the only working option until upstream
    handles the dtype cast at load time.
    """
    from mace.calculators import mace_anicc, mace_mp, mace_off, mace_omol

    if device_str == "mps":
        logger.warning(
            "MACE on MPS would require float64 storage that MPS cannot "
            "allocate; running MACE on CPU instead."
        )
        device_str = "cpu"
        dtype = "float64"

    lower = model.lower()
    if lower.startswith("mp:") or lower.startswith("mace-mp"):
        name = model.split(":", 1)[-1] if ":" in model else model
        return mace_mp(model=name, device=device_str, default_dtype=dtype)
    if lower.startswith("anicc") or lower.startswith("mace-anicc"):
        return mace_anicc(device=device_str, default_dtype=dtype)
    if lower.startswith("omol") or lower.startswith("mace-omol"):
        # mace_omol only accepts None / "extra_large" — fold our common
        # public-facing labels onto the default.
        omol_aliases = {"mace-omol-0", "mace-omol", "omol", "extra_large"}
        omol_arg = None if lower in omol_aliases else model
        return mace_omol(model=omol_arg, device=device_str, default_dtype=dtype)
    if lower.startswith("off:") or lower.startswith("mace-off"):
        name = model.split(":", 1)[-1] if ":" in model else model
        return mace_off(model=name, device=device_str, default_dtype=dtype)
    # Treat as a local model file or a literal mace_off argument.
    return mace_off(model=model, device=device_str, default_dtype=dtype)


def _build_orb_calculator(model: str, device_str: str):
    try:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
    except ImportError as exc:
        raise RuntimeError(
            "orb-models is required for --ml-backend orb. "
            "Install with `pixi run pip install orb-models`."
        ) from exc
    orbff = getattr(pretrained, model)(device=device_str)
    return ORBCalculator(orbff, device=device_str)


def _build_aimnet2_calculator(model: str, device_str: str):
    try:
        from aimnet.calculators import AIMNet2Calculator
    except ImportError as exc:
        raise RuntimeError(
            "aimnet is required for --ml-backend aimnet2. "
            "Install with `pixi run pip install aimnet`."
        ) from exc
    return AIMNet2Calculator(model=model, device=device_str)


def _build_uma_calculator(model: str, device_str: str, task_name: str):
    try:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.pretrained_mlip import get_predict_unit
    except ImportError as exc:
        raise RuntimeError(
            "fairchem-core>=2.20.0 is required for --ml-backend uma. "
            "It should be pulled in automatically by `pixi run install-mlmm`."
        ) from exc
    predictor = get_predict_unit(model, device=device_str)
    return FAIRChemCalculator(predictor, task_name=task_name)


def _build_mlip_calculator(
    backend: str,
    *,
    device_str: str,
    mace_model: str = "MACE-OMOL-0",
    mace_dtype: str = "float64",
    orb_model: str = "orb_v3_conservative_omol",
    aimnet2_model: str = "aimnet2",
    uma_model: str = "uma-s-1p1",
    uma_task_name: str = "omol",
):
    """Build an ASE Calculator for the requested MLIP backend."""
    backend = backend.lower()
    if backend == "mace":
        return _build_mace_calculator(mace_model, device_str, mace_dtype)
    if backend == "orb":
        return _build_orb_calculator(orb_model, device_str)
    if backend == "aimnet2":
        return _build_aimnet2_calculator(aimnet2_model, device_str)
    if backend == "uma":
        return _build_uma_calculator(uma_model, device_str, uma_task_name)
    raise ValueError(f"Unknown ml_backend {backend!r}. Choose from {SUPPORTED_ML_BACKENDS}.")


# ----------------------------------------------------------------------------
# ML region resolution
# ----------------------------------------------------------------------------
def _resolve_ml_indices_via_pymol(prepared_pdb_path: Path, ml_selection: str) -> tuple[int, ...]:
    """Resolve ``ml_selection`` against PyMOL's full selection language.

    Spawns a private temporary object inside the active PyMOL session so we
    don't disturb any objects the user is interacting with, then runs
    ``cmd.iterate`` to harvest atom indices. The full PyMOL syntax is
    available: ``and`` / ``or`` / ``not`` / ``within`` / ``byres`` /
    ``resi`` / ``resn`` / ``chain`` / ``name`` / etc.
    """
    import os as _os

    from pymol import cmd

    obj = f"_pymol_imd_mlsel_{_os.getpid()}"
    if obj in cmd.get_object_list():
        cmd.delete(obj)
    cmd.set("retain_order", 1)
    cmd.load(str(prepared_pdb_path), obj)
    try:
        indices: list[int] = []
        cmd.iterate(
            f"({obj}) and ({ml_selection})",
            "indices.append(index - 1)",
            space={"indices": indices},
        )
        return tuple(indices)
    finally:
        if obj in cmd.get_object_list():
            cmd.delete(obj)


def _resolve_ml_indices_fallback(topology, ml_selection: str) -> tuple[int, ...]:
    """Mini-DSL resolver used when PyMOL isn't importable (headless tests).

    Supports a strict subset of the PyMOL grammar:
      * ``"all"`` / ``"resi 30-50"`` / ``"chain A"`` / ``"resn LIG"`` /
        comma-separated 0-based atom indices.
    No combiners (``and`` / ``or``); use the PyMOL path for those.
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

    if all(p.strip().isdigit() for p in sel.split(",")):
        return tuple(int(p) for p in sel.split(","))

    raise ValueError(
        f"Unsupported ml_selection {sel!r} (headless fallback). "
        f"Use 'all', 'resi <range>', 'chain <id>', 'resn <name>', or "
        f"comma-separated indices. For PyMOL grammar with 'and'/'or'/etc, "
        f"run inside an active PyMOL session."
    )


def _resolve_ml_indices(
    topology,
    ml_selection: str,
    *,
    prepared_pdb_path: Path | None = None,
) -> tuple[int, ...]:
    """Dispatch to PyMOL's selection engine when available, fall back to a
    mini-DSL otherwise.
    """
    sel = ml_selection.strip()
    if sel == "all":
        return tuple(range(topology.getNumAtoms()))

    try:
        from pymol import cmd  # noqa: F401
    except ImportError:
        return _resolve_ml_indices_fallback(topology, sel)

    if prepared_pdb_path is None:
        return _resolve_ml_indices_fallback(topology, sel)
    return _resolve_ml_indices_via_pymol(prepared_pdb_path, sel)


# ----------------------------------------------------------------------------
# MM subset for the ONIOM subtraction
# ----------------------------------------------------------------------------
def _build_subset_mm_system(full_topology, forcefield, ml_indices: tuple[int, ...]):
    """Build an OpenMM System covering only the ML region for MM subtraction.

    The returned System reports the Amber MM forces on the ML region atoms;
    each frame the IMD driver subtracts this from the MLIP forces to form
    the ONIOM correction. Bonds whose both ends lie in ``ml_indices`` are
    retained — cross-region 1-2/1-3/1-4 terms (the link-atom problem) are
    left as a future improvement.
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

    # `ignoreExternalBonds=True` lets the Amber forcefield parameterise the
    # boundary residues of the ML region whose peptide partner sits in the
    # MM region. Without it, every boundary residue is rejected with
    # "set of externally bonded atoms is missing N C atoms".
    sub_system = forcefield.createSystem(
        sub_top,
        nonbondedMethod=NoCutoff,
        constraints=None,
        ignoreExternalBonds=True,
    )
    return sub_top, sub_system


def _atoms_from_topology(topology, indices: tuple[int, ...]):
    """Return ASE-style element symbols for the requested atom indices."""
    atoms = list(topology.atoms())
    return [atoms[i].element.symbol for i in indices]


# ----------------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------------
class MLMMBackend:
    name = "mlmm"

    def build_system(
        self,
        pdb_path: Path,
        *,
        ligand_sdf: Path | None = None,
        pbc: bool = False,
        ml_selection: str = "all",
        ml_backend: str = "mace",
        ml_device: str = "auto",
        ml_charge: int = 0,
        ml_mult: int = 1,
        mace_model: str = "MACE-OMOL-0",
        constraints: str | None = None,
        **_: object,
    ) -> SystemBundle:
        if ml_backend not in SUPPORTED_ML_BACKENDS:
            raise ValueError(
                f"Unknown ml_backend {ml_backend!r}. Choose from {SUPPORTED_ML_BACKENDS}."
            )

        from openmm.app import PME, ForceField

        forcefield = ForceField(*FORCEFIELD_FILES)

        if ligand_sdf is not None:
            raise NotImplementedError(
                "MLMM backend does not yet support ligand SDF; bake the "
                "ligand into the input PDB or open an issue."
            )

        modeller = load_and_complete_hydrogens(pdb_path, forcefield)

        nonbonded_method = PME if pbc else NoCutoff
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=HBonds if constraints == "HBonds" else None,
        )

        # Write the prepared PDB up-front so PyMOL can use it to resolve the
        # ml_selection via its full selection language (`resi 1 and resn MET`
        # etc.). This is the same file commands.py later loads as the IMD
        # object — we just need it on disk earlier than usual.
        work_dir = Path(tempfile.mkdtemp(prefix="pymol_imd_mlmm_"))
        prepared_pdb = write_prepared_pdb(
            modeller.topology,
            modeller.positions,
            work_dir / f"{pdb_path.stem}_prepared.pdb",
        )

        ml_indices = _resolve_ml_indices(
            modeller.topology, ml_selection, prepared_pdb_path=prepared_pdb
        )
        if not ml_indices:
            raise ValueError(
                f"ml_selection {ml_selection!r} matched zero atoms. Check the selection syntax."
            )
        logger.info(
            "MLMM: %d / %d atoms selected as ML region (backend=%s)",
            len(ml_indices),
            modeller.topology.getNumAtoms(),
            ml_backend,
        )

        device_str, _ = _auto_torch_device(ml_device)
        logger.info("MLMM: ML backend running on %s", device_str)
        # MPS (Apple Silicon GPU) doesn't support float64; the MACE default
        # is float64 so we have to opt in to float32 when running on mps.
        mace_dtype = "float32" if device_str == "mps" else "float64"
        ml_calc = _build_mlip_calculator(
            ml_backend,
            device_str=device_str,
            mace_model=mace_model,
            mace_dtype=mace_dtype,
        )
        # Charge / multiplicity hints for backends that read them off
        # atoms.info (notably AIMNet2). MACE / ORB / UMA tolerate the extras.
        ml_calc_info = {"charge": int(ml_charge), "mult": int(ml_mult)}

        # MM subset for the ONIOM subtraction.
        _, sub_system = _build_subset_mm_system(modeller.topology, forcefield, ml_indices)
        sub_integrator = openmm.VerletIntegrator(0.0 * unit.femtoseconds)
        sub_context = openmm.Context(sub_system, sub_integrator)

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
                "ml_calculator": ml_calc,
                "ml_calc_info": ml_calc_info,
                "ml_backend_name": ml_backend,
                "ml_subset_context": sub_context,
                "ml_subset_n_atoms": sub_system.getNumParticles(),
            },
        )
