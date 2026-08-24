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
import math
import tempfile
from pathlib import Path

from openmm import unit
from openmm.app import PME, HBonds, NoCutoff

from .. import constants
from ..prep import load_and_complete_hydrogens, write_prepared_pdb
from .aa import FORCEFIELD_FILES, _set_bounding_box
from .base import SystemBundle

logger = logging.getLogger(__name__)

# Re-exported from constants.py (single source shared with cli.py) so error
# messages and validation here stay in lockstep with the CLI's --choices.
SUPPORTED_ML_BACKENDS = constants.SUPPORTED_ML_BACKENDS
SUPPORTED_LINK_ATOM_METHODS = constants.SUPPORTED_LINK_ATOM_METHODS


def _covalent_radius(elem: str) -> float:
    return constants.COVALENT_RADIUS_ANG.get(
        elem.capitalize(), constants.DEFAULT_COVALENT_RADIUS_ANG
    )


def _g_factor(qm_elem: str, mm_elem: str, link_elem: str = "H") -> float:
    """Morokuma/Dapprich g-factor for link-atom placement.

    g = (CR(QM) + CR(link)) / (CR(QM) + CR(MM)), with CR = covalent radius.
    The link atom sits at r_L = r_QM + g * (r_MM - r_QM). Unit cancels, so
    radii in Angstrom give the same g as radii in Bohr.
    """
    cr_q = _covalent_radius(qm_elem)
    cr_m = _covalent_radius(mm_elem)
    cr_l = _covalent_radius(link_elem)
    return (cr_q + cr_l) / (cr_q + cr_m)


def _detect_link_atoms(
    topology, ml_indices: tuple[int, ...], *, method: str = "scaled"
) -> tuple[tuple[int, int, float], ...]:
    """Enumerate ML/MM boundary bonds and emit link-atom records.

    Walks every bond in ``topology``; a bond with exactly one endpoint in
    ``ml_indices`` is a QM/MM boundary. For each boundary bond we emit a
    ``(qm_atom_index, mm_atom_index, param)`` triple, where ``param`` is the
    method-dependent placement parameter:

      * ``method="scaled"``: g-factor (Morokuma/Dapprich), unitless.
      * ``method="fixed"``:  fixed link bond length (Å) tabulated by QM
                             element.

    ``method="none"`` returns an empty tuple — useful for molecularly closed
    selections where link atoms aren't needed.
    """
    if method == "none":
        return ()
    if method not in SUPPORTED_LINK_ATOM_METHODS:
        raise ValueError(
            f"Unknown link_atom_method {method!r}. Choose from {SUPPORTED_LINK_ATOM_METHODS}."
        )
    keep = set(ml_indices)
    records: list[tuple[int, int, float]] = []
    for a1, a2 in topology.bonds():
        in1 = a1.index in keep
        in2 = a2.index in keep
        if in1 == in2:
            continue  # both QM or both MM — not a boundary
        q_atom, m_atom = (a1, a2) if in1 else (a2, a1)
        q_elem = q_atom.element.symbol
        m_elem = m_atom.element.symbol
        if method == "scaled":
            param = _g_factor(q_elem, m_elem)
        else:  # "fixed"
            if q_elem not in constants.FIXED_LINK_DISTANCE_ANG:
                raise ValueError(
                    f"Fixed link-atom mode does not support QM-side element "
                    f"{q_elem!r} (atom {q_atom.index}). Use link_atom_method="
                    f"'scaled' or extend constants.FIXED_LINK_DISTANCE_ANG."
                )
            param = constants.FIXED_LINK_DISTANCE_ANG[q_elem]
        records.append((q_atom.index, m_atom.index, float(param)))
    return tuple(records)


# ----------------------------------------------------------------------------
# Device selection
# ----------------------------------------------------------------------------
def _auto_torch_device(prefer: str = "auto", cuda_idx: int = 0):
    """Pick the best torch device, preferring CUDA, then MPS, then CPU.

    Returns ``(device_str, torch.device)``. ``device_str`` is the canonical
    short name ("cuda" / "mps" / "cpu") that ASE-side calculators accept.
    """
    import torch

    prefer = str(prefer).strip().lower()
    if prefer not in constants.SUPPORTED_ML_DEVICES:
        raise ValueError(
            f"Unknown ml_device {prefer!r}. Choose from {constants.SUPPORTED_ML_DEVICES}."
        )
    if prefer == "auto":
        if torch.cuda.is_available():
            return "cuda", torch.device(f"cuda:{cuda_idx}")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps", torch.device("mps")
        return "cpu", torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ml_device='cuda' requested, but CUDA is unavailable. Use the Pixi GPU "
                "environment or choose ml_device='cpu'."
            )
        return "cuda", torch.device(f"cuda:{cuda_idx}")
    if prefer == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError(
                "ml_device='mps' requested, but MPS is unavailable. Choose ml_device='cpu'."
            )
        return "mps", torch.device("mps")
    return "cpu", torch.device("cpu")


def _resolve_torch_device(prefer: str, backend: str) -> str:
    """Resolve availability and backend-specific MPS support once."""
    device_str, _device = _auto_torch_device(prefer)
    labels = {"aimnet2": "AIMNet2", "mace": "MACE", "orb": "ORB"}
    if device_str == "mps" and backend in labels:
        device_str = _mps_to_cpu_fallback(device_str, labels[backend])
    return device_str


def _configure_torch_threads(device_str: str, n_ml: int) -> None:
    """Optionally pin torch's CPU thread count for MLIP inference.

    Only acts on CPU. Controlled by the ``PYMOL_IMD_TORCH_THREADS`` env var:
    when set to a positive integer, ``torch.set_num_threads`` is called with
    it. We deliberately do NOT change the thread count automatically because
    the optimum is workload-dependent — small ML regions are often fastest
    with 1-2 threads (parallel-reduction overhead dominates the tiny forward
    pass), while large regions benefit from more — so silently forcing a value
    could slow some cases. The effective count is logged either way so users
    can tune it.

    Numerically lossless: at fixed coordinates the MLIP forces are
    bitwise-identical regardless of thread count (verified for AIMNet2).
    """
    if device_str != "cpu":
        return
    import os

    import torch

    requested = os.environ.get("PYMOL_IMD_TORCH_THREADS")
    if requested:
        try:
            n = int(requested)
        except ValueError:
            logger.warning("PYMOL_IMD_TORCH_THREADS=%r is not an integer; ignoring", requested)
        else:
            if n >= 1:
                torch.set_num_threads(n)
            else:
                logger.warning("PYMOL_IMD_TORCH_THREADS=%d must be >= 1; ignoring", n)
    logger.info(
        "MLMM: torch using %d CPU thread(s) for %d ML atoms (override with "
        "PYMOL_IMD_TORCH_THREADS)",
        torch.get_num_threads(),
        n_ml,
    )


def _log_torch_build(device_str: str) -> None:
    """Log the selected device + torch build so GPU use is verifiable.

    Surfaces ``torch.version.cuda`` (``None`` for CPU/MPS builds). If the user
    asked for CUDA but the active torch is a CPU build (the default env), warn
    with the concrete fix — otherwise the heavy ML inference silently runs on
    CPU and the slowdown looks mysterious.
    """
    import torch

    cuda_build = torch.version.cuda
    logger.info(
        "MLMM: ML backend on %s (torch %s, cuda build=%s)",
        device_str,
        torch.__version__,
        cuda_build,
    )
    if device_str == "cuda" and cuda_build is None:
        msg = (
            "ml_device=cuda requested but this torch is a CPU/MPS build "
            "(torch.version.cuda is None) — inference will fail or fall back. "
            "Use the GPU environment on linux-64: `pixi install -e gpu` then "
            "`pixi run -e gpu pymol_imd ... --level mlmm`."
        )
        logger.warning(msg)
        print(f"[pymol_imd] WARNING: {msg}")


# ----------------------------------------------------------------------------
# Per-backend ASE Calculator construction
# ----------------------------------------------------------------------------
# Per-backend reason why MPS (Apple Silicon GPU) isn't used. Each default
# MLIP hits a different wall, so the CPU-fallback notice names the real cause
# instead of a one-size-fits-all "float64" message.
_MPS_CPU_REASON: dict[str, str] = {
    "AIMNet2": (
        "AIMNet2's GPU path uses Warp CUDA kernels with no Metal backend, so "
        "MPS gives no speedup (and its weights can't load on MPS anyway)"
    ),
    "MACE": (
        "MACE foundation weights are float64 and MPS cannot allocate float64 "
        "storage; an MPS-float32 path is feasible but lowers precision (opt-in, "
        "not yet enabled)"
    ),
    "ORB": ("ORB's graph construction is CUDA-oriented and MPS is unsupported upstream"),
}


def _mps_to_cpu_fallback(device_str: str, backend_label: str) -> str:
    """Fall back to CPU for backends that can't (usefully) run on MPS.

    None of the default-bundled MLIPs benefit from MPS today — see
    ``_MPS_CPU_REASON`` for the backend-specific cause — so we transparently
    run on CPU and log the real reason. A MACE-only MPS-float32 path is
    feasible (load float32 on CPU, then ``.to('mps')``) but lowers precision
    vs MACE's native float64, so it is left as an opt-in future improvement.
    """
    if device_str != "mps":
        return device_str
    reason = _MPS_CPU_REASON.get(backend_label, f"{backend_label} cannot run on MPS")
    msg = f"{backend_label} on MPS → CPU: {reason}."
    logger.warning(msg)
    print(f"[pymol_imd] {msg}")
    return "cpu"


def _build_mace_calculator(
    model: str, device_str: str, dtype: str, *, enable_cueq: bool = False, ml_compile: bool = False
):
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

    ``enable_cueq`` turns on cuEquivariance acceleration. The foundation
    factories (mp/omol/off) forward ``**kwargs`` straight into
    ``MACECalculator``, so we pass ``enable_cueq=True`` through that channel.
    cuEq needs CUDA + the cuequivariance packages (`pixi run install-cueq` in
    the gpu env); on CPU/MPS or when those packages are absent MACE logs that
    cuEq is disabled and runs normally, so passing it is always safe. It is
    mutually exclusive with torch.compile — a known cuEq bug silently breaks
    MACE's backward pass when both are enabled.

    ``ml_compile`` sets MACECalculator's ``compile_mode='default'`` (torch.compile)
    through the same **kwargs channel. Since cuEq + compile is the broken combo,
    enable_cueq wins when both are requested (compile is skipped with a warning).
    """
    from mace.calculators import mace_anicc, mace_mp, mace_off, mace_omol

    if device_str == "mps":
        device_str = _mps_to_cpu_fallback(device_str, "MACE")
        dtype = constants.MLMM_MACE_DTYPE  # reset dtype forced to float32 for MPS

    # Forwarded to MACECalculator via the factory **kwargs passthrough. Only
    # the foundation factories (mp/omol/off) forward kwargs; anicc is left as-is.
    extra_kw: dict = {}
    if enable_cueq:
        extra_kw["enable_cueq"] = True
        if ml_compile:
            logger.warning(
                "MACE: enable_cueq and ml_compile both set; using cuEq and skipping "
                "compile (cuEq + torch.compile is a known broken combo)."
            )
    elif ml_compile:
        extra_kw["compile_mode"] = constants.MLMM_MACE_COMPILE_MODE
    lower = model.lower()
    if lower.startswith("mp:") or lower.startswith("mace-mp"):
        name = model.split(":", 1)[-1] if ":" in model else model
        return mace_mp(model=name, device=device_str, default_dtype=dtype, **extra_kw)
    if lower.startswith("anicc") or lower.startswith("mace-anicc"):
        return mace_anicc(device=device_str, default_dtype=dtype)
    if lower.startswith("omol") or lower.startswith("mace-omol"):
        # mace_omol's documented argument is `None` or `"extra_large"`; we
        # fold every common user-facing label (`MACE-OMOL-0`, `omol`, etc.)
        # AND `"extra_large"` itself onto `None`, which is the value that
        # actually triggers the download path. Any other string is treated
        # as a local file path or direct URL.
        omol_aliases = {"mace-omol-0", "mace-omol", "omol", "extra_large"}
        omol_arg = None if lower in omol_aliases else model
        return mace_omol(model=omol_arg, device=device_str, default_dtype=dtype, **extra_kw)
    if lower.startswith("off:") or lower.startswith("mace-off"):
        name = model.split(":", 1)[-1] if ":" in model else model
        return mace_off(model=name, device=device_str, default_dtype=dtype, **extra_kw)
    # Treat as a local model file or a literal mace_off argument.
    return mace_off(model=model, device=device_str, default_dtype=dtype, **extra_kw)


def _build_orb_calculator(model: str, device_str: str, *, ml_compile: bool = False):
    """Build an ORBCalculator for an orb-models pretrained model.

    orb-models 0.7.x: ``pretrained.<model>()`` returns a ``(model, adapter)``
    tuple and ``ORBCalculator`` lives under ``inference.calculator``. Older
    paths (``orb_models.forcefield.calculator``, single-return pretrained)
    are gone — this code targets 0.7.x onwards.

    ``ml_compile`` wraps the underlying model in ``torch.compile`` (ORB has no
    native compile flag). ORBCalculator keeps it as ``self.model`` and calls it
    every frame, so replacing that attribute with the compiled module is enough.
    """
    try:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
    except ImportError as exc:
        raise RuntimeError(
            "orb-models is required for --ml-backend orb. Install with `pixi run install-orb`."
        ) from exc

    if not hasattr(pretrained, model):
        available = sorted(
            n for n in dir(pretrained) if n.startswith("orb") and callable(getattr(pretrained, n))
        )
        raise ValueError(f"Unknown ORB model {model!r}. Available: {available}")
    device_str = _mps_to_cpu_fallback(device_str, "ORB")
    orb_model, adapter = getattr(pretrained, model)(device=device_str)
    calc = ORBCalculator(orb_model, adapter, device=device_str)
    if ml_compile:
        import torch

        calc.model = torch.compile(calc.model)
    return calc


def _build_aimnet2_calculator(
    model: str, device_str: str, charge: int, mult: int, *, compile_model: bool = False
):
    """Build an ASE-compatible AIMNet2 calculator.

    The plain ``AIMNet2Calculator`` in aimnet is a low-level torch wrapper
    and does **not** implement ASE's ``Calculator`` interface (no
    ``get_forces``). The ASE-facing class is ``AIMNet2ASE``, which wraps a
    ``base_calc`` and takes ``charge`` / ``mult`` in its constructor
    instead of via ``atoms.info``.

    ``compile_model`` forwards to ``AIMNet2Calculator``'s own ``compile_model``
    flag (``torch.compile``). The big win is on CUDA (~5x via Warp kernels); on
    CPU the gain is smaller. torch.compile is numerically ~identical but not
    bitwise, so it stays opt-in (never the silent default).
    """
    try:
        from aimnet.calculators.aimnet2ase import AIMNet2ASE
        from aimnet.calculators.calculator import AIMNet2Calculator
    except ImportError as exc:
        raise RuntimeError(
            "aimnet is required for --ml-backend aimnet2. Run `pixi install` to install it."
        ) from exc
    device_str = _mps_to_cpu_fallback(device_str, "AIMNet2")
    base = AIMNet2Calculator(model=model, device=device_str, compile_model=compile_model)
    return AIMNet2ASE(base_calc=base, charge=int(charge), mult=int(mult))


def _build_uma_calculator(model: str, device_str: str, task_name: str, *, ml_compile: bool = False):
    """Build a FAIRChem (UMA) ASE calculator.

    fairchem-core 2.20.x moved ``get_predict_unit`` to
    ``fairchem.core.calculate.pretrained_mlip`` (the old
    ``fairchem.core.pretrained_mlip`` path no longer imports). The predict unit
    holds the torch model at ``predictor.model`` whose core network is
    ``predictor.model.module``; ``ml_compile`` wraps that core in
    ``torch.compile`` (UMA has no native compile flag).
    """
    try:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.calculate.pretrained_mlip import get_predict_unit
    except ImportError as exc:
        raise RuntimeError(
            "fairchem-core>=2.20.0 is required for --ml-backend uma. "
            "Install with `pixi run install-uma` and `huggingface-cli login` "
            "(facebook/UMA is a gated HF repo)."
        ) from exc
    predictor = get_predict_unit(model, device=device_str)
    if ml_compile:
        import torch

        predictor.model.module = torch.compile(predictor.model.module)
    return FAIRChemCalculator(predictor, task_name=task_name)


def _build_mlip_calculator(
    backend: str,
    *,
    device_str: str,
    ml_charge: int = 0,
    ml_mult: int = 1,
    ml_compile: bool = False,
    enable_cueq: bool = False,
    mace_model: str = constants.MLMM_MACE_MODEL,
    mace_dtype: str = constants.MLMM_MACE_DTYPE,
    orb_model: str = constants.MLMM_ORB_MODEL,
    aimnet2_model: str = constants.MLMM_AIMNET2_MODEL,
    uma_model: str = constants.MLMM_UMA_MODEL,
    uma_task_name: str = constants.MLMM_UMA_TASK_NAME,
):
    """Build an ASE Calculator for the requested MLIP backend.

    ``ml_charge`` / ``ml_mult`` are only consumed by backends whose ASE
    wrapper takes them in the constructor (AIMNet2). ORB still reads them
    from ``atoms.info`` each frame.

    ``ml_compile`` requests ``torch.compile`` where the backend supports it
    (AIMNet2 today). It is numerically ~identical but not bitwise, hence
    opt-in; backends without a wired compile path ignore it.
    """
    backend = backend.lower()
    if backend == "mace":
        return _build_mace_calculator(
            mace_model, device_str, mace_dtype, enable_cueq=enable_cueq, ml_compile=ml_compile
        )
    if backend == "orb":
        return _build_orb_calculator(orb_model, device_str, ml_compile=ml_compile)
    if backend == "aimnet2":
        return _build_aimnet2_calculator(
            aimnet2_model, device_str, ml_charge, ml_mult, compile_model=ml_compile
        )
    if backend == "uma":
        return _build_uma_calculator(uma_model, device_str, uma_task_name, ml_compile=ml_compile)
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
        result = tuple(range(topology.getNumAtoms()))
        return _canonicalize_ml_indices(topology, result)

    try:
        from pymol import cmd  # noqa: F401
    except ImportError:
        result = _resolve_ml_indices_fallback(topology, sel)
        return _canonicalize_ml_indices(topology, result)

    if prepared_pdb_path is None:
        result = _resolve_ml_indices_fallback(topology, sel)
    else:
        result = _resolve_ml_indices_via_pymol(prepared_pdb_path, sel)
    return _canonicalize_ml_indices(topology, result)


def _canonicalize_ml_indices(topology, indices) -> tuple[int, ...]:
    """Return sorted unique valid indices in topology order."""
    values = tuple(int(index) for index in indices)
    n_atoms = topology.getNumAtoms()
    invalid = sorted({index for index in values if index < 0 or index >= n_atoms})
    if invalid:
        raise ValueError(
            f"ML atom indices outside [0, {n_atoms}): {invalid[:12]}"
            + (" ..." if len(invalid) > 12 else "")
        )
    canonical = tuple(sorted(set(values)))
    if len(canonical) != len(values):
        logger.warning("ML selection contained duplicate atom indices; duplicates were removed")
    return canonical


# ----------------------------------------------------------------------------
# MM subset for the ONIOM subtraction
# ----------------------------------------------------------------------------
def _build_subset_mm_system(
    full_topology,
    forcefield,
    ml_indices: tuple[int, ...],
    *,
    pbc: bool = False,
    box_vectors=None,
    nonbonded_cutoff_nm: float = constants.MLMM_NONBONDED_CUTOFF_NM,
):
    """Build an OpenMM System covering only the ML region for MM subtraction.

    The returned System reports the Amber MM forces on the ML region atoms;
    each frame the IMD driver subtracts this from the MLIP forces to form
    the ONIOM correction. Bonds whose both ends lie in ``ml_indices`` are
    retained — cross-region 1-2/1-3/1-4 terms (the link-atom problem) are
    left as a future improvement.

    The nonbonded treatment MUST match the full system so the ONIOM
    subtraction ``E_MM(full) - E_MM(region)`` cancels consistently: PME with
    the same ``box_vectors`` under PBC, NoCutoff in vacuo.
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
    #
    # PME on an isolated ML fragment is an approximation inherent to
    # ML/MM-under-PBC; the in-vacuo default (both terms NoCutoff) is the exact
    # path and stays bitwise-unchanged.
    if pbc:
        if box_vectors is not None:
            sub_top.setPeriodicBoxVectors(box_vectors)
        sub_system = forcefield.createSystem(
            sub_top,
            nonbondedMethod=PME,
            nonbondedCutoff=nonbonded_cutoff_nm * unit.nanometer,
            constraints=None,
            ignoreExternalBonds=True,
        )
    else:
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
    # Per-backend recommendations (values centralized in constants.py).
    recommended_const_k = constants.MLMM_RECOMMENDED_CONST_K
    recommended_integrator = constants.MLMM_INTEGRATOR
    recommended_timestep_ps = constants.MLMM_TIMESTEP_PS

    def build_system(
        self,
        pdb_path: Path,
        *,
        ligand_sdf: Path | None = None,
        pbc: bool = False,
        box_padding_nm: float = constants.MLMM_BOX_PADDING_NM,
        ml_selection: str = constants.MLMM_DEFAULT_ML_SELECTION,
        ml_backend: str = constants.MLMM_DEFAULT_BACKEND,
        ml_device: str = constants.MLMM_DEFAULT_DEVICE,
        ml_charge: int = constants.MLMM_DEFAULT_CHARGE,
        ml_mult: int = constants.MLMM_DEFAULT_MULT,
        ml_compile: bool = False,
        enable_cueq: bool = False,
        mace_model: str = constants.MLMM_MACE_MODEL,
        nonbonded_cutoff_nm: float = constants.MLMM_NONBONDED_CUTOFF_NM,
        constraints: str | None = None,
        link_atom_method: str = constants.MLMM_DEFAULT_LINK_ATOM_METHOD,
        implicit_solvent: str = constants.MLMM_IMPLICIT_SOLVENT,
        work_dir: Path | None = None,
        **_: object,
    ) -> SystemBundle:
        if ml_backend not in SUPPORTED_ML_BACKENDS:
            raise ValueError(
                f"Unknown ml_backend {ml_backend!r}. Choose from {SUPPORTED_ML_BACKENDS}."
            )
        if link_atom_method not in SUPPORTED_LINK_ATOM_METHODS:
            raise ValueError(
                f"Unknown link_atom_method {link_atom_method!r}. "
                f"Choose from {SUPPORTED_LINK_ATOM_METHODS}."
            )
        if not str(ml_selection).strip():
            raise ValueError("ml_selection must not be empty")
        for name, value in (("ml_charge", ml_charge), ("ml_mult", ml_mult)):
            if isinstance(value, bool) or int(value) != float(value):
                raise ValueError(f"{name} must be an integer")
        if int(ml_mult) < 1:
            raise ValueError("ml_mult must be >= 1")
        if not math.isfinite(nonbonded_cutoff_nm) or nonbonded_cutoff_nm <= 0.0:
            raise ValueError("nonbonded_cutoff_nm must be finite and > 0")
        if not math.isfinite(box_padding_nm) or box_padding_nm < 0.0:
            raise ValueError("box_padding_nm must be finite and >= 0")

        from openmm.app import ForceField

        # GB implicit solvent (default) is appended to the shared ForceField, so
        # BOTH the full system and the subtracted MM subset (which reuses this
        # same object) carry GB -- the ONIOM subtraction E_MM(full) - E_MM(subset)
        # stays consistent. Resolver yields None for vacuo / under pbc (PME path).
        implicit_xml = constants.resolve_implicit_solvent(implicit_solvent, pbc=pbc)
        ff_files = (*FORCEFIELD_FILES, implicit_xml) if implicit_xml else FORCEFIELD_FILES
        forcefield = ForceField(*ff_files)

        if ligand_sdf is not None:
            raise NotImplementedError(
                "MLMM backend does not yet support ligand SDF; bake the "
                "ligand into the input PDB or open an issue."
            )

        modeller = load_and_complete_hydrogens(pdb_path, forcefield)

        nonbonded_method = PME if pbc else NoCutoff
        if pbc:
            # Switch NVT(vacuo) -> NPT under PBC: enclose the molecule in a
            # bounding box so PME has periodic vectors (mirrors the AA backend).
            # The IMD driver adds a MonteCarloBarostat when bundle.pbc is set.
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

        # Write the prepared PDB up-front so PyMOL can use it to resolve the
        # ml_selection via its full selection language (`resi 1 and resn MET`
        # etc.). This is the same file commands.py later loads as the IMD
        # object — we just need it on disk earlier than usual.
        work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="pymol_imd_mlmm_"))
        work_dir.mkdir(parents=True, exist_ok=True)
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

        device_str = _resolve_torch_device(ml_device, ml_backend)
        _log_torch_build(device_str)
        _configure_torch_threads(device_str, len(ml_indices))
        # MPS (Apple Silicon GPU) doesn't support float64; the MACE default
        # is float64 so we have to opt in to float32 when running on mps.
        mace_dtype = "float32" if device_str == "mps" else constants.MLMM_MACE_DTYPE
        ml_calc = _build_mlip_calculator(
            ml_backend,
            device_str=device_str,
            ml_charge=int(ml_charge),
            ml_mult=int(ml_mult),
            ml_compile=ml_compile,
            enable_cueq=enable_cueq,
            mace_model=mace_model,
            mace_dtype=mace_dtype,
        )
        # `atoms.info` payload that satisfies every supported backend:
        # AIMNet2 reads `mult`, ORB reads `spin` (= unpaired electrons,
        # always `mult - 1` by definition), MACE / UMA ignore them.
        ml_calc_info = {
            "charge": int(ml_charge),
            "mult": int(ml_mult),
            "spin": int(ml_mult) - 1,
        }

        # MM subset for the ONIOM subtraction. Under PBC it reuses the
        # full-system box + PME so the subtracted E_MM(region) matches the
        # PME treatment of E_MM(full).
        box_vectors = modeller.topology.getPeriodicBoxVectors() if pbc else None
        _, sub_system = _build_subset_mm_system(
            modeller.topology,
            forcefield,
            ml_indices,
            pbc=pbc,
            box_vectors=box_vectors,
            nonbonded_cutoff_nm=nonbonded_cutoff_nm,
        )
        ml_symbols = _atoms_from_topology(modeller.topology, ml_indices)

        # Resolve QM/MM boundary bonds into link-atom records. Empty when the
        # ML region is molecularly closed (or method="none"). Each record is
        # (qm_atom_index, mm_atom_index, placement_param) — see
        # ``_detect_link_atoms`` for the meaning of placement_param.
        link_atoms = _detect_link_atoms(modeller.topology, ml_indices, method=link_atom_method)
        if link_atoms:
            logger.info(
                "MLMM: %d link atoms (method=%s) inserted at QM/MM boundary bonds",
                len(link_atoms),
                link_atom_method,
            )

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
                "ml_subset_system": sub_system,
                "ml_subset_n_atoms": sub_system.getNumParticles(),
                "link_atoms": link_atoms,
                "link_atom_method": link_atom_method,
            },
        )
