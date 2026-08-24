"""Interactive Molecular Dynamics (IMD) driver.

This module is backend-agnostic: it takes a :class:`SystemBundle` (built by an
AA / CG / ML-MM backend) and turns it into a runnable simulation that:

  * runs OpenMM steps on the PyMOL Qt event loop via :class:`QTimer`,
  * reads PyMOL atomic coordinates each frame and pushes a pulling force into
    a single :class:`CustomExternalForce` so the user can drag atoms in the
    PyMOL window during MD,
  * writes OpenMM positions back into PyMOL via ``cmd.alter_state``.
"""

from __future__ import annotations

import contextlib
import csv
import gc
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openmm
from openmm import (
    LangevinIntegrator,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    Platform,
    unit,
)
from openmm.app import DCDReporter, PDBReporter, Simulation, XTCReporter

from . import constants
from .backends.base import SystemBundle

logger = logging.getLogger(__name__)

# eV/Å -> kJ/(mol·nm) bridge. Single source of truth lives in constants.py;
# kept as a module alias so the per-frame call sites below stay terse.
_EV_PER_A_TO_KJ_PER_MOL_NM = constants.EV_PER_A_TO_KJ_PER_MOL_NM


def compute_pull_forces(
    delta_nm: np.ndarray,
    const_k: float,
    max_force_kj: float,
    noise_nm: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Turn a per-atom position delta (nm) into pull forces (kJ/mol/nm).

    Shared by the in-process IMD path
    (:meth:`IMDSystem._apply_pull_force_from_pymol`) and the socket client
    (:mod:`pymol_imd.client`) so dragging feels numerically identical in both.
    Pure / side-effect free.

    The per-atom force magnitude is checked against ``max_force_kj``. If ANY
    dragged atom would exceed the cap, the WHOLE frame's pull is skipped
    (no force applied) instead of being clamped, so a too-hard drag simply does
    nothing that frame rather than injecting a runaway force. The caller is
    expected to warn and keep the MD running when ``n_exceeded > 0``.

    Parameters
    ----------
    delta_nm:
        ``(n_atoms, 3)`` array of ``target - current`` positions in nm.
    const_k, max_force_kj, noise_nm:
        Pull-force scale (kJ/mol/nm), per-atom |F| limit (kJ/mol/nm), and the
        drag noise floor (nm) below which an atom counts as not dragged.

    Returns
    -------
    (active_idx, forces, n_exceeded):
        ``active_idx`` are the dragged atom indices (|delta| >= ``noise_nm``),
        EMPTY when the frame is skipped; ``forces`` is the full ``(n_atoms, 3)``
        array (kJ/mol/nm) — inactive atoms zeroed and active ones scaled by
        ``const_k`` (no clamp), or ALL-ZERO when the frame is skipped;
        ``n_exceeded`` is the number of dragged atoms whose |F| exceeded
        ``max_force_kj`` — nonzero means the whole frame's pull was skipped.
    """
    delta = np.array(delta_nm, dtype=float, copy=True)
    atom_max = np.max(np.abs(delta), axis=1)
    active_mask = atom_max >= noise_nm
    delta[~active_mask] = 0.0
    forces = delta * const_k
    fmag = np.linalg.norm(forces, axis=1)
    # Whole-frame skip: if ANY dragged atom's |F| exceeds the limit, drop the
    # entire frame's pull (empty active set, zero forces) so the MD keeps
    # running without a runaway force. Replaces the old per-atom magnitude clamp.
    n_exceeded = int(np.count_nonzero(fmag[active_mask] > max_force_kj))
    if n_exceeded:
        return np.empty(0, dtype=int), np.zeros_like(forces), n_exceeded
    active_idx = np.where(active_mask)[0]
    return active_idx, forces, 0


@dataclass
class IMDConfig:
    # Field defaults are sourced from constants.py (single source of truth);
    # the command layer overrides per-backend values via imd_setup_*.
    temperature: float = constants.DEFAULT_TEMPERATURE_K  # K
    pressure: float = constants.DEFAULT_PRESSURE_BAR  # bar (NPT only)
    friction: float = constants.DEFAULT_FRICTION_PER_PS  # 1/ps
    timestep_ps: float = constants.DEFAULT_TIMESTEP_PS  # ps
    nsteps_per_frame: int = constants.DEFAULT_NSTEPS_PER_FRAME
    const_k: float = constants.DEFAULT_CONST_K  # kJ/mol/nm scale for the pulling force
    mouse_force: bool = constants.DEFAULT_MOUSE_FORCE
    # Integrator selection: "langevin" (legacy BBK-like) or "langevin_middle"
    # (Leimkuhler-Matthews BAOAB). The middle scheme has 2nd-order accurate
    # configurational sampling and a wider stability window under external
    # forcing, so AA backends default to it; CG/MLMM keep "langevin" for now.
    integrator: str = constants.DEFAULT_INTEGRATOR
    # Per-atom |F| limit (kJ/mol/nm) on the pulling force. If ANY dragged atom
    # would exceed it, that whole frame's pull is skipped (no force applied) and
    # a throttled warning is logged — the MD keeps running. This prevents runaway
    # accelerations from a too-hard drag without stopping the simulation. (The
    # separate _diverged guard halts the loop on energy blow-up.)
    max_force_kj: float = constants.DEFAULT_MAX_FORCE_KJ
    # If |potential energy| exceeds this (kJ/mol), the simulation is treated
    # as diverged and the IMD loop halts. NaN energies trip the same guard.
    energy_cutoff_kj: float = constants.DEFAULT_ENERGY_CUTOFF_KJ
    # QTimer tick interval (ms) for both the production and minimize loops.
    # Read at timer start(); changing it mid-run needs a stop/run cycle.
    frame_interval_ms: int = constants.DEFAULT_FRAME_INTERVAL_MS
    # Mouse-drag granularity for 3-Button Editing (atom | residue | chain).
    # PyMOL moves only a covalent fragment, so the force path expands a dragged
    # atom's pull to its whole topology residue or chain when requested.
    drag_level: str = constants.DEFAULT_DRAG_LEVEL
    # OpenMM compute platform for the MM engine (all backends: AA / CG / ML-MM):
    # "auto" lets OpenMM pick the fastest available (CUDA > OpenCL > CPU), or an
    # explicit "cuda" / "opencl" / "cpu" / "reference". The ML/MM MLIP runs on a
    # SEPARATE device (see ml_device / _auto_torch_device) — this knob does not
    # affect it.
    platform: str = constants.DEFAULT_PLATFORM

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate and normalize all runtime controls."""
        self.integrator = str(self.integrator).strip().lower()
        self.drag_level = str(self.drag_level).strip().lower()
        self.platform = str(self.platform).strip().lower()
        if self.integrator not in constants.SUPPORTED_INTEGRATORS:
            raise ValueError(
                f"Unknown integrator {self.integrator!r}. "
                f"Choose from {constants.SUPPORTED_INTEGRATORS}."
            )
        if self.drag_level not in constants.SUPPORTED_DRAG_LEVELS:
            raise ValueError(
                f"Unknown drag_level {self.drag_level!r}. "
                f"Choose from {constants.SUPPORTED_DRAG_LEVELS}."
            )
        constants.resolve_platform_name(self.platform)
        positive_float_fields = (
            "temperature",
            "pressure",
            "timestep_ps",
            "max_force_kj",
            "energy_cutoff_kj",
        )
        nonnegative_float_fields = ("friction", "const_k")
        for name in (*positive_float_fields, *nonnegative_float_fields):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a number, got boolean {value!r}")
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
            if not math.isfinite(normalized):
                raise ValueError(f"{name} must be finite, got {value!r}")
            setattr(self, name, normalized)
        for name in positive_float_fields:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")
        for name in nonnegative_float_fields:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)!r}")
        for name in ("nsteps_per_frame", "frame_interval_ms"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be an integer > 0, got boolean {value!r}")
            try:
                numeric = float(value)
                normalized = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be an integer > 0, got {value!r}") from exc
            if not math.isfinite(numeric) or normalized != numeric or normalized <= 0:
                raise ValueError(f"{name} must be an integer > 0, got {value!r}")
            setattr(self, name, normalized)
        if not isinstance(self.mouse_force, bool):
            raise ValueError(f"mouse_force must be a boolean, got {self.mouse_force!r}")


class IMDSystem:
    """Runnable IMD wrapper built around a :class:`SystemBundle`.

    The pulling force is added to the bundle's ``System`` in-place; backends
    therefore stay free of any PyMOL / IMD coupling.
    """

    def __init__(
        self,
        bundle: SystemBundle,
        config: IMDConfig | None = None,
        *,
        pymol_object: str | None = None,
    ) -> None:
        """``pymol_object``: name of the PyMOL object whose atoms mirror the
        OpenMM topology. The IMD sync limits all PyMOL operations to this
        object so other objects in the scene are untouched.
        """
        self.bundle = bundle
        self.config = config or IMDConfig()
        self.pymol_object = pymol_object
        self.pymol_selector = f"({pymol_object})" if pymol_object else "all"

        self._pull_force = self._attach_pull_force(bundle.system)
        # Separate force for the ML/MM ONIOM correction so it stacks cleanly
        # with the user pull force without aliasing parameters.
        self._mlmm_force = self._attach_mlmm_correction_force(bundle)
        # Barostat attachment is also idempotent across IMDSystem rebuilds.
        if bundle.pbc and not self._has_barostat(bundle.system):
            bundle.system.addForce(
                MonteCarloBarostat(
                    self.config.pressure * unit.bar,
                    self.config.temperature * unit.kelvin,
                )
            )

        integrator_cls = (
            LangevinMiddleIntegrator
            if self.config.integrator == "langevin_middle"
            else LangevinIntegrator
        )
        integrator = integrator_cls(
            self.config.temperature * unit.kelvin,
            self.config.friction / unit.picosecond,
            self.config.timestep_ps * unit.picoseconds,
        )
        platform = self._resolve_platform()
        if platform is not None:
            self.simulation = Simulation(bundle.topology, bundle.system, integrator, platform)
        else:
            self.simulation = Simulation(bundle.topology, bundle.system, integrator)
        self.simulation.context.setPositions(bundle.positions)
        logger.info("OpenMM platform: %s", self.simulation.context.getPlatform().getName())
        self._ml_subset_integrator: openmm.Integrator | None = None
        self._ml_subset_context: openmm.Context | None = None
        self._init_ml_subset_context()
        self.n_atoms = bundle.topology.getNumAtoms()
        self.running = False
        self._last_min_energy: float | None = None
        # Tracks atoms whose pull force was non-zero on the previous frame
        # so we can zero them on release without scanning all n_atoms.
        self._active_pull_idx_prev: set[int] = set()
        # Edge-triggered guard for the pull-force limit warning: True while the
        # current drag exceeds max_force_kj (its frame pull is skipped). Warn
        # once on the rising edge, reset on the falling edge, so a sustained
        # over-limit drag doesn't flood the console (the QTimer ticks every few ms).
        self._force_exceeded_prev: bool = False

        # --- recording state (production run only; never during minimize) ---
        # set_recording() (imd_record) stores the pending config here;
        # start_recording() (imd_run) opens files / attaches the reporter;
        # stop_recording() (imd_stop) flushes + closes them. Kept on the
        # mutable IMDSystem because the frozen SystemBundle cannot hold it.
        self._record_settings: dict | None = None
        self._recording: bool = False
        self._traj_reporter: object | None = None
        self._force_csv_path: Path | None = None
        self._force_csv_file = None
        self._force_csv_writer = None
        self._force_csv_interval: int = 1
        self._force_csv_rows_since_flush: int = 0
        # Thermodynamic state CSV (energies / temperature, + volume/density under
        # PBC). Mirrors the force-CSV machinery; opened in start_recording.
        self._state_csv_path: Path | None = None
        self._state_csv_file = None
        self._state_csv_writer = None
        self._state_csv_interval: int = 1
        self._state_csv_rows_since_flush: int = 0
        self._state_csv_pbc: bool = False
        self._total_mass_amu: float | None = None  # cached lazily for density
        self._frame_index: int = 0  # per-run GUI-frame counter
        # Pull force cached each frame by _apply_pull_force_from_pymol so the
        # CSV writer reads it without recomputing.
        self._last_pull_active_idx: np.ndarray = np.empty(0, dtype=int)
        self._last_pull_forces: np.ndarray = np.empty((0, 3), dtype=float)

        # Tier-1 lossless ML/MM caches: per-frame-invariant index arrays / maps
        # plus a reusable ASE Atoms object so the hot path stops rebuilding them
        # every frame. All no-ops for non-MLMM backends.
        self._mlip_atoms = None
        self._ml_indices_list: list[int] = []
        self._n_ml = 0
        self._mlip_symbols: list[str] = []
        self._ml_local: dict[int, int] = {}
        self._bdy_local: dict[int, int] = {}
        self._init_mlmm_cache()
        self._warmup_mlmm()

        # Per-atom group indices for the residue / chain drag levels. PyMOL's
        # editing drag moves a covalent fragment (mouse_selection_mode does NOT
        # affect it), so to drag a whole residue / chain we expand the pull
        # ourselves using these maps. Built once — topology is fixed after setup.
        self._atom_residue_idx, self._atom_chain_idx = self._build_group_indices()

    # ------------------------------------------------------------------ setup
    _PULL_FORCE_EXPR = "- fx*x - fy*y - fz*z"
    _MLMM_FORCE_EXPR = "- gx*x - gy*y - gz*z"

    @staticmethod
    def _find_force(system: openmm.System, energy_expr: str) -> openmm.CustomExternalForce | None:
        for i in range(system.getNumForces()):
            f = system.getForce(i)
            if isinstance(f, openmm.CustomExternalForce) and f.getEnergyFunction() == energy_expr:
                return f
        return None

    @classmethod
    def _attach_pull_force(cls, system: openmm.System) -> openmm.CustomExternalForce:
        # Idempotent: if the same bundle is re-wrapped in a second IMDSystem
        # (e.g. user re-runs setup_*), reuse the already-attached force
        # instead of stacking another copy onto the System.
        existing = cls._find_force(system, cls._PULL_FORCE_EXPR)
        if existing is not None:
            return existing
        force = openmm.CustomExternalForce(cls._PULL_FORCE_EXPR)
        for name in ("fx", "fy", "fz"):
            force.addPerParticleParameter(name)
        zero = (0.0, 0.0, 0.0)
        for atom_idx in range(system.getNumParticles()):
            force.addParticle(atom_idx, zero)
        system.addForce(force)
        return force

    @classmethod
    def _attach_mlmm_correction_force(
        cls, bundle: SystemBundle
    ) -> openmm.CustomExternalForce | None:
        """Attach a CustomExternalForce that carries the ONIOM ML-vs-MM delta.

        ML region atoms get a particle in the canonical ``ml_indices`` order
        (local indices ``[0, n_ml)``). When link atoms are present, each
        unique MM-side boundary atom is appended as an additional particle
        (local indices ``[n_ml, n_ml + n_bdy)``) so the per-frame routine can
        push the redistributed link-atom force to those atoms in O(|ML|+|bdy|)
        without scanning the full system. Idempotent across IMDSystem
        rebuilds on the same bundle.
        """
        if not bundle.extra or "ml_indices" not in bundle.extra:
            return None
        existing = cls._find_force(bundle.system, cls._MLMM_FORCE_EXPR)
        if existing is not None:
            return existing
        force = openmm.CustomExternalForce(cls._MLMM_FORCE_EXPR)
        for name in ("gx", "gy", "gz"):
            force.addPerParticleParameter(name)
        for atom_idx in bundle.extra["ml_indices"]:
            force.addParticle(int(atom_idx), (0.0, 0.0, 0.0))

        # Boundary MM atoms (from link atoms) live outside ml_indices but
        # still need force slots so the redistributed F_M can land somewhere.
        ml_set = set(bundle.extra["ml_indices"])
        boundary_mm: list[int] = []
        seen: set[int] = set()
        for _q_idx, m_idx, _param in bundle.extra.get("link_atoms", ()):
            if m_idx in ml_set or m_idx in seen:
                continue
            seen.add(m_idx)
            boundary_mm.append(int(m_idx))
        for atom_idx in boundary_mm:
            force.addParticle(atom_idx, (0.0, 0.0, 0.0))
        bundle.extra["_mlmm_boundary_mm_indices"] = tuple(boundary_mm)
        bundle.system.addForce(force)
        return force

    @staticmethod
    def _has_barostat(system: openmm.System) -> bool:
        for i in range(system.getNumForces()):
            if isinstance(system.getForce(i), MonteCarloBarostat):
                return True
        return False

    def _resolve_platform(self) -> openmm.Platform | None:
        """Resolve the configured platform to an OpenMM ``Platform``, or None.

        None means "let OpenMM auto-select the fastest available platform" (the
        historical behavior). An explicit name OpenMM cannot provide raises a
        clear error (listing what IS available) rather than silently falling
        back, so a user's GPU request is never honored as CPU without notice.
        """
        name = constants.resolve_platform_name(self.config.platform)
        if name is None:
            return None
        try:
            return Platform.getPlatformByName(name)
        except Exception as exc:
            available = ", ".join(
                Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())
            )
            raise RuntimeError(
                f"OpenMM platform {name!r} is not available (have: {available}). "
                f"Install a {name}-enabled OpenMM or use platform='auto'/'cpu'."
            ) from exc

    def _init_ml_subset_context(self) -> None:
        """Create the ML-region MM Context on the main Context's platform."""
        extra = self.bundle.extra
        if extra is None or "ml_subset_system" not in extra:
            return
        self._ml_subset_integrator = openmm.VerletIntegrator(0.0 * unit.femtoseconds)
        platform = self.simulation.context.getPlatform()
        self._ml_subset_context = openmm.Context(
            extra["ml_subset_system"], self._ml_subset_integrator, platform
        )
        # Kept in ``extra`` for diagnostics and existing integrations.
        extra["ml_subset_context"] = self._ml_subset_context
        logger.info("MLMM subset OpenMM platform: %s", platform.getName())

    # ----------------------------------------------------------------- stepping
    def step(self) -> None:
        pre_state = None
        full_positions = None
        box_vectors = None
        if self.config.mouse_force or self._mlmm_force is not None:
            pre_state = self.simulation.context.getState(getPositions=True)
            full_positions = np.asarray(
                pre_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            )
            if self.bundle.pbc:
                box_vectors = pre_state.getPeriodicBoxVectors(asNumpy=True)
        if self.config.mouse_force:
            self._apply_pull_force_from_pymol(full_positions)
        elif self._active_pull_idx_prev:
            # mouse_force was just disabled mid-drag: clear the latched pull
            # force so no phantom constant force lingers in the OpenMM context.
            # Empty active_idx makes set_pull_forces zero exactly the
            # previously-active atoms, then clears _active_pull_idx_prev so this
            # runs only once per disable.
            self.set_pull_forces(np.empty(0, dtype=int), np.zeros((self.n_atoms, 3)))
        if self._mlmm_force is not None:
            self._apply_mlmm_correction_force(full_positions, box_vectors)
        self.simulation.step(self.config.nsteps_per_frame)  # trajectory reporter fires here
        state = self.simulation.context.getState(getPositions=True, getEnergy=True)
        if self._diverged(state):
            return
        if self._recording:
            self._record_force_csv()
            self._record_state_csv(state)
        self._frame_index += 1
        self._sync_positions_to_pymol(state)

    def apply_mlmm_if_present(self) -> None:
        """Apply the ONIOM ML/MM correction if this is an ML/MM bundle.

        OpenMM-only (no PyMOL). The socket server calls this each frame before
        stepping, mirroring the same-process ``step()`` order. No-op for the
        AA / CG backends (``_mlmm_force`` is ``None``).
        """
        if self._mlmm_force is not None:
            state = self.simulation.context.getState(getPositions=True)
            positions = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
            box_vectors = state.getPeriodicBoxVectors(asNumpy=True) if self.bundle.pbc else None
            self._apply_mlmm_correction_force(positions, box_vectors)

    def advance_openmm(
        self,
        *,
        get_positions: bool = False,
        get_velocities: bool = False,
        get_forces: bool = False,
    ) -> tuple[bool, object]:
        """Advance one frame without PyMOL and return ``(running, State)``.

        Server-side core of :meth:`step`: it steps ``nsteps_per_frame``, runs
        the divergence guard, and writes the force CSV when recording. It does
        NOT pull from PyMOL (the socket server injects MDComm forces into
        ``_pull_force`` via :meth:`set_pull_forces`) and does NOT sync positions
        to PyMOL (the socket client renders the received frame). ML/MM
        correction is applied separately via :meth:`apply_mlmm_if_present` so
        the server controls the inject -> correct -> step ordering.
        """
        self.simulation.step(self.config.nsteps_per_frame)
        state = self.simulation.context.getState(
            getPositions=get_positions,
            getEnergy=True,
            getVelocities=get_velocities,
            getForces=get_forces,
        )
        if self._diverged(state):
            return False, state
        if self._recording:
            self._record_force_csv()
            self._record_state_csv(state)
        self._frame_index += 1
        return self.running, state

    def _init_mlmm_cache(self) -> None:
        """Precompute per-frame-invariant ML/MM data once (Tier-1 lossless).

        The ML symbols, the atom-index list used to slice positions, and the
        QM/MM local-index maps never change after setup, so building them every
        frame in ``_apply_mlmm_correction_force`` was pure Python overhead. The
        ASE ``Atoms`` object is also built once here and only has its positions
        overwritten each frame: ASE invalidates its result cache on any
        position change (``tol=1e-15``), so reuse is numerically identical to
        rebuilding a fresh ``Atoms`` every frame.

        No-op for non-MLMM bundles (``_mlip_atoms`` stays ``None``).
        """
        extra = self.bundle.extra
        if extra is None or "ml_indices" not in extra:
            return
        from ase import Atoms

        ml_indices = extra["ml_indices"]
        ml_symbols = extra["ml_symbols"]
        link_atoms = extra.get("link_atoms", ())
        boundary_mm = extra.get("_mlmm_boundary_mm_indices", ())

        self._ml_indices_list = list(ml_indices)
        self._n_ml = len(ml_indices)
        # ML region symbols first, one "H" per link atom appended after — the
        # exact order _apply_mlmm_correction_force fills positions in.
        self._mlip_symbols = list(ml_symbols) + ["H"] * len(link_atoms)
        self._ml_local = {int(idx): i for i, idx in enumerate(ml_indices)}
        self._bdy_local = {int(idx): i for i, idx in enumerate(boundary_mm)}

        atoms = Atoms(symbols=self._mlip_symbols)
        atoms.calc = extra["ml_calculator"]
        # AIMNet2 reads charge / multiplicity from atoms.info; harmless for the
        # other backends. Set once — the payload never changes after setup.
        atoms.info.update(extra.get("ml_calc_info", {}))
        self._mlip_atoms = atoms

    def _warmup_mlmm(self) -> None:
        """Run one MLIP inference at setup so the first interactive frame is fast.

        The first MLIP call pays for model download, TorchScript/torch.compile
        tracing, and (AIMNet2) Warp kernel codegen — seconds of work that would
        otherwise freeze the viewer on the first dragged frame. Doing it here at
        setup moves that cost to where the user expects setup work, and primes
        the ML/MM correction force with the t0 delta. Numerically lossless: the
        throwaway forces are recomputed every frame anyway.

        No-op for non-MLMM backends. Failures are non-fatal (logged) — the real
        per-frame path surfaces any genuine error with full context.
        """
        if self._mlip_atoms is None:
            return
        try:
            self._apply_mlmm_correction_force()
        except Exception as exc:
            logger.warning("MLMM warmup inference failed (non-fatal): %s", exc)

    def _apply_mlmm_correction_force(
        self,
        full_positions: np.ndarray | None = None,
        box_vectors=None,
    ) -> None:
        """Compute ``F_ML - F_MM_subset`` on the ML region and push to OpenMM.

        Runs once per frame, before the MD step. The correction adds the MLIP
        energy gradient and subtracts the Amber MM contribution computed on
        the same atoms, yielding the ONIOM total force on the ML region.

        When ``link_atoms`` are present, link H atoms are appended to the
        ML-region ASE ``Atoms`` and their forces are redistributed to the
        QM-side (``ml_indices``) and MM-side (boundary) atoms using the
        Jacobian of the link-atom placement. The MM subset is **not** rebuilt
        with link atoms, matching the convention that the boundary bond's MM
        contribution is supplied solely by the full-system MM force.

        Every supported MLIP backend (UMA / ORB / MACE / AIMNet2) is wrapped
        as an ASE ``Calculator``, so a single ``atoms.get_forces()`` call
        works for all of them. The ASE ``Atoms`` object and the index maps are
        built once in :meth:`_init_mlmm_cache` and reused here.
        """
        extra = self.bundle.extra
        if extra is None:
            return
        ml_indices = extra["ml_indices"]
        link_atoms: tuple[tuple[int, int, float], ...] = extra.get("link_atoms", ())
        boundary_mm: tuple[int, ...] = extra.get("_mlmm_boundary_mm_indices", ())
        method: str = extra.get("link_atom_method", "scaled")
        sub_context = self._ml_subset_context
        if sub_context is None:
            raise RuntimeError("MLMM subset Context is not initialized")

        if full_positions is None:
            state = self.simulation.context.getState(getPositions=True)
            full_positions = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            )
            if self.bundle.pbc:
                box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        ml_positions_nm = full_positions[self._ml_indices_list]
        ml_positions_ang = ml_positions_nm * 10.0

        # 1) MLIP inference on the reused ASE Atoms. ML region first, link
        #    atoms appended after (order matches self._mlip_symbols).
        if link_atoms:
            link_positions_ang = self._link_atom_positions_ang(full_positions, link_atoms, method)
            mlip_positions_ang = np.vstack([ml_positions_ang, link_positions_ang])
        else:
            mlip_positions_ang = ml_positions_ang
        atoms = self._mlip_atoms
        atoms.set_positions(mlip_positions_ang)
        F_all_eV_per_A = np.asarray(atoms.get_forces(), dtype=np.float64)

        # 2) Partition MLIP forces. ML region atoms keep their forces; link
        #    atom forces are redistributed via the placement Jacobian to the
        #    QM-side (in ml_indices) and MM-side (in boundary_mm) atoms.
        n_ml = self._n_ml
        F_ml_eV_per_A = F_all_eV_per_A[:n_ml].copy()
        F_mm_boundary_eV_per_A = np.zeros((len(boundary_mm), 3), dtype=np.float64)

        if link_atoms:
            ml_local = self._ml_local
            bdy_local = self._bdy_local
            F_link_all = F_all_eV_per_A[n_ml:]
            for k, (q_idx, m_idx, param) in enumerate(link_atoms):
                F_link = F_link_all[k]
                f_q, f_m = self._redistribute_link_force(
                    F_link, full_positions[q_idx], full_positions[m_idx], param, method
                )
                F_ml_eV_per_A[ml_local[int(q_idx)]] += f_q
                # If m_idx happened to also be in the ML region (degenerate
                # but legal), the redistributed force lands on the ML slot
                # instead of a boundary slot.
                if int(m_idx) in bdy_local:
                    F_mm_boundary_eV_per_A[bdy_local[int(m_idx)]] += f_m
                else:
                    F_ml_eV_per_A[ml_local[int(m_idx)]] += f_m

        # 3) MM force on the ML-region subset (kJ/mol/nm).
        if self.bundle.pbc:
            if box_vectors is None:
                raise RuntimeError("Periodic MLMM correction requires box vectors")
            sub_context.setPeriodicBoxVectors(*box_vectors)
        sub_context.setPositions(ml_positions_nm * unit.nanometer)
        sub_state = sub_context.getState(getForces=True)
        F_mm_kj_per_nm = sub_state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )

        # 4) Delta in OpenMM units: F_ML[kJ/mol/nm] - F_MM[kJ/mol/nm] on the
        #    ML region, and pure redistributed F_M on the boundary MM atoms
        #    (subtractive term is absent there by construction).
        F_ml_kj_per_nm = F_ml_eV_per_A * _EV_PER_A_TO_KJ_PER_MOL_NM
        delta_ml = F_ml_kj_per_nm - np.asarray(F_mm_kj_per_nm)
        delta_bdy = F_mm_boundary_eV_per_A * _EV_PER_A_TO_KJ_PER_MOL_NM

        for local_idx, atom_idx in enumerate(ml_indices):
            gx, gy, gz = delta_ml[local_idx]
            self._mlmm_force.setParticleParameters(
                local_idx, int(atom_idx), (float(gx), float(gy), float(gz))
            )
        for k, atom_idx in enumerate(boundary_mm):
            gx, gy, gz = delta_bdy[k]
            self._mlmm_force.setParticleParameters(
                n_ml + k, int(atom_idx), (float(gx), float(gy), float(gz))
            )
        self._mlmm_force.updateParametersInContext(self.simulation.context)

    @staticmethod
    def _link_atom_positions_ang(
        full_positions_nm: np.ndarray,
        link_atoms: tuple[tuple[int, int, float], ...],
        method: str,
    ) -> np.ndarray:
        """Place each link H along the boundary bond, in Angstrom.

        scaled: r_L = r_Q + g * (r_M - r_Q),  param = g
        fixed : r_L = r_Q + d * u_QM,         param = d (Å), u_QM normalized
        """
        out = np.empty((len(link_atoms), 3), dtype=np.float64)
        for k, (q_idx, m_idx, param) in enumerate(link_atoms):
            r_q = full_positions_nm[q_idx] * 10.0
            r_m = full_positions_nm[m_idx] * 10.0
            vec = r_m - r_q
            if method == "scaled":
                out[k] = r_q + param * vec
            else:  # "fixed"
                norm = float(np.linalg.norm(vec))
                if norm < 1e-9:
                    # Degenerate Q≡M (numerical) — fall back to scaled
                    # placement so we never divide by zero.
                    out[k] = r_q
                else:
                    out[k] = r_q + (param / norm) * vec
        return out

    @staticmethod
    def _redistribute_link_force(
        f_link: np.ndarray,
        r_q_nm: np.ndarray,
        r_m_nm: np.ndarray,
        param: float,
        method: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Redistribute one link-atom force to (F_Q, F_M).

        scaled: J = [[(1-g)I], [gI]] — trivial linear partition.
        fixed : J carries an (I - u u^T) projector with a 1/R prefactor, so
                bond-stretching motion of (Q,M) does not feed forces into
                the link-atom position, only orientation changes do.
        """
        if method == "scaled":
            g = param
            return (1.0 - g) * f_link, g * f_link
        # fixed mode. param is the link-atom distance in Å; convert to nm so
        # the d/R ratio is unitless against r_q_nm / r_m_nm.
        vec_nm = r_m_nm - r_q_nm
        R_nm = float(np.linalg.norm(vec_nm))
        if R_nm < 1e-9:
            # Geometric degeneracy — return half-half as a safe fallback.
            return 0.5 * f_link, 0.5 * f_link
        u = vec_nm / R_nm
        d_over_R = (param * 0.1) / R_nm
        proj = np.eye(3) - np.outer(u, u)
        # Jacobian blocks (mlmm_toolkit / Morokuma):
        #   dR_dQ = I - d * (I - u u^T)/R    →  F_Q = (I - d_over_R * proj) f_L
        #   dR_dM =      d * (I - u u^T)/R   →  F_M =       d_over_R * proj  f_L
        proj_f = proj @ f_link
        f_q = f_link - d_over_R * proj_f
        f_m = d_over_R * proj_f
        return f_q, f_m

    def _diverged(self, state=None) -> bool:
        """Return True (and mark the system stopped) if energy went insane.

        Reads the potential energy after a step; NaN/inf or |U| > cutoff
        indicates the integrator blew up (typically from over-large pull
        forces or a too-large timestep). Stopping here keeps PyMOL from
        receiving garbage coordinates that would push atoms off-screen.
        """
        if state is None:
            state = self.simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        u_kj = energy.value_in_unit(unit.kilojoule_per_mole)
        if np.isfinite(u_kj) and abs(u_kj) <= self.config.energy_cutoff_kj:
            return False
        logger.error(
            "Diverged: U = %s kJ/mol (cutoff %.2e). Halting IMD loop. "
            "Lower const_k / max_force_kj or shorten timestep_ps.",
            u_kj,
            self.config.energy_cutoff_kj,
        )
        self.running = False
        return True

    def minimize_step(
        self,
        max_iterations: int = constants.DEFAULT_MIN_MAX_ITER_PER_CHUNK,
        *,
        tol_kj: float = constants.DEFAULT_MIN_ENERGY_TOL_KJ,
    ) -> bool:
        """Run a chunk of minimization; returns True when converged.

        ``tol_kj`` is the energy-change convergence tolerance: two consecutive
        chunks differing by less than this (kJ/mol) count as converged.
        """
        self.simulation.minimizeEnergy(maxIterations=max_iterations)
        self._sync_positions_to_pymol()
        energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
        energy_kj = energy.value_in_unit(unit.kilojoule_per_mole)
        converged = (
            self._last_min_energy is not None and abs(energy_kj - self._last_min_energy) < tol_kj
        )
        logger.info("minimize: U = %.4f kJ/mol%s", energy_kj, " (converged)" if converged else "")
        self._last_min_energy = energy_kj
        return converged

    def reset_minimization(self) -> None:
        """Clear cross-run convergence state before a new minimization."""
        self._last_min_energy = None

    def instantaneous_temperature(self, ke_kj: float | None = None) -> float:
        """Estimate T (K) from kinetic energy.

        Approximate: ``ndof = 3N - n_constraints`` (ignores the COM-motion
        remover's 3 dof). Shared by the state-CSV recorder and the socket
        server's energies packet so there is a single temperature formula.
        Reads KE from the context when ``ke_kj`` is None.
        """
        if ke_kj is None:
            state = self.simulation.context.getState(getEnergy=True)
            ke_kj = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        ndof = 3 * self.n_atoms - self.bundle.system.getNumConstraints()
        if ndof <= 0:
            return 0.0
        return 2.0 * ke_kj / (ndof * constants.GAS_CONSTANT_KJ_PER_MOL_K)

    # --------------------------------------------------------------- recording
    def set_recording(
        self,
        traj: str | None = None,
        traj_interval: int = constants.DEFAULT_TRAJ_INTERVAL,
        force_csv: str | None = None,
        force_csv_interval: int = constants.DEFAULT_FORCE_CSV_INTERVAL,
        state_csv: str | None = None,
        state_csv_interval: int = constants.DEFAULT_STATE_CSV_INTERVAL,
    ) -> None:
        """Store the pending recording config (does NOT open any file).

        Called by the ``imd_record`` command. The actual file open / reporter
        attach is deferred to :meth:`start_recording` (invoked by ``imd_run``)
        so that re-running (``imd_stop`` -> ``imd_run``) re-attaches cleanly
        and the reporter interval picks up the current ``nsteps_per_frame``.

        Passing ``traj``, ``force_csv`` and ``state_csv`` all as None clears
        recording.
        """
        if traj is None and force_csv is None and state_csv is None:
            self._record_settings = None
            return
        outputs = [
            Path(path).expanduser().resolve() for path in (traj, force_csv, state_csv) if path
        ]
        if len(outputs) != len(set(outputs)):
            raise ValueError("traj, force_csv, and state_csv must use different output paths")
        intervals = {
            "traj_interval": traj_interval,
            "force_csv_interval": force_csv_interval,
            "state_csv_interval": state_csv_interval,
        }
        for name, value in intervals.items():
            if isinstance(value, bool) or int(value) != float(value) or int(value) < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        self._record_settings = {
            "traj": traj,
            "traj_interval": int(traj_interval),
            "force_csv": force_csv,
            "force_csv_interval": int(force_csv_interval),
            "state_csv": state_csv,
            "state_csv_interval": int(state_csv_interval),
        }

    def start_recording(self) -> None:
        """Open the trajectory reporter / force CSV from the pending config.

        No-op when no recording was requested (so ``imd_run`` with no prior
        ``imd_record`` keeps its original behavior). If already recording, the
        previous outputs are closed first for a clean re-attach.
        """
        if self._record_settings is None:
            return
        if self._recording:
            self.stop_recording()
        self._frame_index = 0
        settings = self._record_settings
        traj = settings["traj"]
        force_csv = settings["force_csv"]
        state_csv = settings["state_csv"]

        try:
            if traj is not None:
                self._traj_reporter = self._make_traj_reporter(traj, settings["traj_interval"])
                self.simulation.reporters.append(self._traj_reporter)
            if force_csv is not None:
                self._force_csv_interval = settings["force_csv_interval"]
                path = Path(force_csv)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._force_csv_path = path
                self._force_csv_file = open(path, "w", newline="")  # noqa: SIM115
                self._force_csv_writer = csv.writer(self._force_csv_file)
                self._force_csv_writer.writerow(
                    ["frame", "step", "time_ps", "atom_index", "fx", "fy", "fz", "fmag"]
                )
                self._force_csv_file.flush()
                self._force_csv_rows_since_flush = 0
            if state_csv is not None:
                self._state_csv_interval = settings["state_csv_interval"]
                self._state_csv_pbc = bool(self.bundle.system.usesPeriodicBoundaryConditions())
                if self._total_mass_amu is None:
                    self._total_mass_amu = self._compute_total_mass_amu()
                path = Path(state_csv)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._state_csv_path = path
                self._state_csv_file = open(path, "w", newline="")  # noqa: SIM115
                self._state_csv_writer = csv.writer(self._state_csv_file)
                header = [
                    "frame",
                    "step",
                    "time_ps",
                    "potential_energy_kj",
                    "kinetic_energy_kj",
                    "total_energy_kj",
                    "temperature_k",
                ]
                if self._state_csv_pbc:
                    header += ["volume_nm3", "density_g_cm3"]
                self._state_csv_writer.writerow(header)
                self._state_csv_file.flush()
                self._state_csv_rows_since_flush = 0
        except Exception:
            # Roll back any partially-opened output (e.g. traj attached but
            # the CSV open failed) so we never leave a dangling reporter.
            self.stop_recording()
            raise

        self._recording = True
        logger.info(
            "recording: traj=%s (every %d frame[s]), force_csv=%s (every %d frame[s]), "
            "state_csv=%s (every %d frame[s])",
            traj,
            settings["traj_interval"],
            force_csv,
            settings["force_csv_interval"],
            state_csv,
            settings["state_csv_interval"],
        )

    def _make_traj_reporter(self, traj_path: str, traj_interval: int):
        """Build a format-appropriate OpenMM trajectory reporter.

        The file extension picks the format (.xtc / .dcd / .pdb). OpenMM's
        ``reportInterval`` is in MD steps; since IMD advances
        ``nsteps_per_frame`` steps per GUI frame, ``traj_interval`` (in GUI
        frames) maps to ``reportInterval = traj_interval * nsteps_per_frame``.
        Reporters fire on ``currentStep % reportInterval == 0``.

        Note: changing ``nsteps_per_frame`` mid-run via ``imd_change`` does NOT
        retune this interval (it is fixed at attach time); stop and re-run to
        resync the trajectory cadence.
        """
        ext = Path(traj_path).suffix.lower()
        reporter_classes = {
            ".xtc": XTCReporter,
            ".dcd": DCDReporter,
            ".pdb": PDBReporter,
        }
        reporter_class = reporter_classes.get(ext)
        if reporter_class is None:
            raise ValueError(
                f"Unsupported trajectory extension '{ext}' for {traj_path!r}. "
                "Use .xtc, .dcd, or .pdb."
            )
        report_interval = max(1, traj_interval * self.config.nsteps_per_frame)
        # Overwrite any existing file so a second imd_run to the same path is
        # consistent across formats (XTCReporter raises on a non-empty file).
        out = Path(traj_path)
        # Create the parent dir if missing — symmetry with the force-CSV path
        # in start_recording(); XTC/DCD/PDBReporter all raise on a nonexistent
        # directory. Cheap and idempotent.
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            logger.warning("recording: overwriting existing trajectory %s", traj_path)
            out.unlink()
        return reporter_class(traj_path, report_interval)

    def stop_recording(self) -> None:
        """Flush + close any active trajectory / CSV outputs. Idempotent.

        The trajectory reporter is removed from the simulation, its reference
        dropped, then ``gc.collect()`` is forced: PDBReporter writes its
        trailing footer and DCDReporter closes its file only in ``__del__``,
        so this is the format-uniform way to finalize the file. XTCReporter
        completes each frame at the C level and needs nothing extra.
        """
        if self._traj_reporter is not None:
            with contextlib.suppress(ValueError):
                self.simulation.reporters.remove(self._traj_reporter)
            self._traj_reporter = None
            gc.collect()
        if self._force_csv_file is not None:
            try:
                self._force_csv_file.flush()
                self._force_csv_file.close()
            finally:
                self._force_csv_file = None
                self._force_csv_writer = None
                self._force_csv_path = None
                self._force_csv_rows_since_flush = 0
        if self._state_csv_file is not None:
            try:
                self._state_csv_file.flush()
                self._state_csv_file.close()
            finally:
                self._state_csv_file = None
                self._state_csv_writer = None
                self._state_csv_path = None
                self._state_csv_rows_since_flush = 0
        self._recording = False

    def _record_force_csv(self) -> None:
        """Append the current frame's user pull forces to the CSV.

        One row per atom a pull force was applied to this frame: frame index,
        absolute MD step, time, the OpenMM 0-based atom index, the applied pull
        force vector (kJ/mol/nm) and its magnitude. With drag_level
        residue / chain this is the whole expanded group, every atom carrying the
        group's mean force (not just the grabbed atom). Frames with no applied
        force emit no rows (including over-limit frames whose pull was skipped).
        atom_index is 0-based (OpenMM / trajectory convention); add 1 for a
        PyMOL selection.
        """
        if self._force_csv_writer is None:
            return
        # Gate on the cumulative MD step (mirrors the trajectory reporter's
        # currentStep % (interval * nsteps_per_frame)), NOT the per-run GUI
        # frame counter: equal force_csv_interval / traj_interval then sample
        # the SAME frames, even across imd_stop -> imd_run cycles (currentStep
        # is never reset, _frame_index is). The CSV `step` column is the join
        # key to the trajectory; `frame` stays an advisory per-run ordinal.
        report_interval = max(1, self._force_csv_interval * self.config.nsteps_per_frame)
        step = self.simulation.currentStep
        if step % report_interval != 0:
            return
        active = self._last_pull_active_idx
        if active.size == 0:
            return
        time_ps = step * self.config.timestep_ps
        forces = self._last_pull_forces
        rows = []
        for idx in active:
            i = int(idx)
            fx, fy, fz = forces[i]
            fmag = float(np.sqrt(fx * fx + fy * fy + fz * fz))
            rows.append(
                (
                    self._frame_index,
                    step,
                    f"{time_ps:.6f}",
                    i,
                    f"{float(fx):.6f}",
                    f"{float(fy):.6f}",
                    f"{float(fz):.6f}",
                    f"{fmag:.6f}",
                )
            )
        self._force_csv_writer.writerows(rows)
        self._force_csv_rows_since_flush += len(rows)
        if self._force_csv_rows_since_flush >= constants.CSV_FLUSH_ROWS:
            self._force_csv_file.flush()
            self._force_csv_rows_since_flush = 0

    def _record_state_csv(self, state=None) -> None:
        """Append the current frame's thermodynamic quantities to the state CSV.

        GROMACS-style per-frame physical quantities: energies (kJ/mol),
        temperature (K), and — only under PBC — volume (nm^3) and density
        (g/cm^3). Gated on the cumulative MD step exactly like the force CSV /
        trajectory reporter, so equal intervals sample identical frames. Unlike
        the force CSV, EVERY gated frame is written (no dragged-atom dependency).
        Pressure is intentionally omitted: OpenMM reports no instantaneous
        pressure and IMD pull forces contaminate the virial.
        """
        if self._state_csv_writer is None:
            return
        report_interval = max(1, self._state_csv_interval * self.config.nsteps_per_frame)
        step = self.simulation.currentStep
        if step % report_interval != 0:
            return
        if state is None:
            state = self.simulation.context.getState(getEnergy=True)
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        total = pe + ke
        temp = self.instantaneous_temperature(ke)
        time_ps = step * self.config.timestep_ps
        row = [
            self._frame_index,
            step,
            f"{time_ps:.6f}",
            f"{pe:.6f}",
            f"{ke:.6f}",
            f"{total:.6f}",
            f"{temp:.6f}",
        ]
        if self._state_csv_pbc:
            volume_nm3 = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
            density = self._total_mass_amu / volume_nm3 * constants.AMU_PER_NM3_TO_G_PER_CM3
            row.append(f"{volume_nm3:.6f}")
            row.append(f"{density:.6f}")
        self._state_csv_writer.writerow(row)
        self._state_csv_rows_since_flush += 1
        if self._state_csv_rows_since_flush >= constants.CSV_FLUSH_ROWS:
            self._state_csv_file.flush()
            self._state_csv_rows_since_flush = 0

    def _compute_total_mass_amu(self) -> float:
        """Total system mass (amu) for the density column.

        Computed once and cached: particle masses are fixed after setup
        (``dalton`` == amu in OpenMM).
        """
        system = self.bundle.system
        return sum(
            system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(system.getNumParticles())
        )

    # ------------------------------------------------------------- pulling logic
    # Drag noise floor (nm). Rationale (PDB %8.3f precision + float32) lives
    # with the constant in constants.py.
    _DELTA_NOISE_NM = constants.DELTA_NOISE_NM

    def _apply_pull_force_from_pymol(self, openmm_xyz: np.ndarray | None = None) -> None:
        try:
            pymol_xyz = self._read_pymol_positions_nm()
        except ImportError:
            # No PyMOL → no user pull. Keep the cache honest so the CSV writer
            # doesn't re-emit a stale frame's active set, and re-arm the
            # over-limit warning so re-entering a live session warns cleanly.
            self._last_pull_active_idx = np.empty(0, dtype=int)
            self._force_exceeded_prev = False
            return
        # RuntimeError from atom-count mismatch etc. is intentionally NOT
        # caught here; that's a real fault and the user must hear about it.

        if openmm_xyz is None:
            openmm_xyz = self._read_openmm_positions_nm()
        delta = pymol_xyz - openmm_xyz
        active_idx, forces, n_exceeded = compute_pull_forces(
            delta, self.config.const_k, self.config.max_force_kj, self._DELTA_NOISE_NM
        )
        # Over-limit drag: the whole frame's pull was skipped. Warn once on the
        # rising edge (not every tick) and keep the MD running.
        if n_exceeded and not self._force_exceeded_prev:
            logger.warning(
                "Pull force on %d atom(s) exceeded max_force_kj=%.1f kJ/mol/nm; "
                "this frame's pull was skipped (no force applied), MD continues. "
                "Drag a shorter distance or raise max_force_kj to pull harder.",
                n_exceeded,
                self.config.max_force_kj,
            )
        self._force_exceeded_prev = bool(n_exceeded)
        # Drag a whole residue / chain when requested: PyMOL only moves a
        # covalent fragment under the mouse, so we propagate that pull to the rest
        # of the group here (no-op for the default atom level / a skipped frame).
        if self.config.drag_level != "atom" and active_idx.size:
            active_idx, forces = self._expand_drag_groups(active_idx, forces)
        self.set_pull_forces(active_idx, forces)

        if active_idx.size:
            # |F| the user requested — for the debug log only.
            requested = np.linalg.norm(delta[active_idx] * self.config.const_k, axis=1)
            logger.debug(
                "pull: %d atoms dragged, |F|max requested=%.1f / limit=%.1f kJ/mol/nm",
                int(active_idx.size),
                float(np.max(requested)),
                self.config.max_force_kj,
            )

    def set_pull_forces(self, active_idx: np.ndarray, forces: np.ndarray) -> None:
        """Write per-atom pull forces into the pull ``CustomExternalForce``.

        ``forces`` is the full ``(n_atoms, 3)`` array; only ``active_idx`` rows
        are applied. Atoms active last call but not now are zeroed (drag
        release), so the hot loop touches only the few dozen dragged atoms
        instead of all n_atoms. Calls ``updateParametersInContext`` once.
        Shared by the PyMOL drag path and the socket MDComm path.
        """
        to_zero = self._active_pull_idx_prev - set(active_idx.tolist())
        if active_idx.size or to_zero:
            for idx in active_idx:
                fx, fy, fz = forces[idx]
                self._pull_force.setParticleParameters(
                    int(idx), int(idx), (float(fx), float(fy), float(fz))
                )
            for idx in to_zero:
                self._pull_force.setParticleParameters(int(idx), int(idx), (0.0, 0.0, 0.0))
            self._pull_force.updateParametersInContext(self.simulation.context)
        self._active_pull_idx_prev = set(active_idx.tolist())
        # Cache for the force CSV writer (consumed by _record_force_csv).
        self._last_pull_active_idx = active_idx
        self._last_pull_forces = forces

    def clear_pull_forces(self) -> None:
        """Release all active pull forces and clear their recording cache."""
        self.set_pull_forces(np.empty(0, dtype=int), np.zeros((self.n_atoms, 3), dtype=float))
        self._force_exceeded_prev = False

    # --------------------------------------------------------- drag-level groups
    def _build_group_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Map each atom to its residue index and chain index (once at setup).

        Used to expand a per-atom pull to a whole residue / chain when
        ``drag_level`` is not ``"atom"``. OpenMM's ``residue.index`` and
        ``chain.index`` are globally unique within the topology, so equal
        indices reliably mark same-residue / same-chain membership.
        """
        residue_idx = np.empty(self.n_atoms, dtype=np.int64)
        chain_idx = np.empty(self.n_atoms, dtype=np.int64)
        for atom in self.bundle.topology.atoms():
            residue_idx[atom.index] = atom.residue.index
            chain_idx[atom.index] = atom.residue.chain.index
        return residue_idx, chain_idx

    def _expand_drag_groups(
        self, active_idx: np.ndarray, forces: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expand a per-atom pull to whole residues / chains.

        PyMOL's 3-Button Editing drags a covalent fragment, not a residue or
        chain (its ``mouse_selection_mode`` does not affect editing drag), so
        ``drag_level`` is realized here: every atom of a touched residue / chain
        receives the mean pull force of that group's dragged atoms, dragging the
        whole group along. Any over-limit frame is already skipped upstream in
        :func:`compute_pull_forces` (empty ``active_idx``), so this only runs on
        within-limit pulls; the group mean of within-limit vectors stays bounded.
        """
        group_of = (
            self._atom_residue_idx if self.config.drag_level == "residue" else self._atom_chain_idx
        )
        touched = np.unique(group_of[active_idx])
        expanded = np.where(np.isin(group_of, touched))[0]
        new_forces = forces.copy()
        for g in touched:
            members = expanded[group_of[expanded] == g]
            dragged = active_idx[group_of[active_idx] == g]
            new_forces[members] = forces[dragged].mean(axis=0)
        return expanded, new_forces

    # ------------------------------------------------------------ coord sync IO
    def _read_openmm_positions_nm(self) -> np.ndarray:
        state = self.simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        return np.asarray(pos)

    def _read_pymol_positions_nm(self) -> np.ndarray:
        from pymol import cmd

        coords: list[tuple[float, float, float]] = []
        cmd.iterate_state(
            1,
            self.pymol_selector,
            "coords.append((x, y, z))",
            space={"coords": coords},
        )
        if len(coords) != self.n_atoms:
            raise RuntimeError(
                f"PyMOL atom count {len(coords)} in selector '{self.pymol_selector}' "
                f"!= OpenMM atom count {self.n_atoms}"
            )
        # PyMOL stores coordinates in angstrom; OpenMM IMD logic works in nm.
        return np.asarray(coords) * 0.1

    def _sync_positions_to_pymol(self, state=None) -> None:
        try:
            from pymol import cmd
        except ImportError:  # pragma: no cover - headless smoke test
            return
        pymol_atom_count = cmd.count_atoms(self.pymol_selector)
        if pymol_atom_count == 0:
            # PyMOL is importable but no object is loaded against this
            # selector — typically a headless / scripted run. Skip silently
            # so backend smoke tests don't need a live PyMOL viewer.
            return
        if pymol_atom_count != self.n_atoms:
            raise RuntimeError(
                f"Refusing to push positions: PyMOL selector "
                f"'{self.pymol_selector}' has {pymol_atom_count} atoms but "
                f"OpenMM has {self.n_atoms}. Re-run setup_*."
            )
        if state is None:
            positions_nm = self._read_openmm_positions_nm()
        else:
            positions_nm = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            )
        positions_ang = positions_nm * 10.0
        space = {"positions": positions_ang}
        cmd.alter_state(
            1,
            self.pymol_selector,
            "x, y, z = positions[index - 1]",
            space=space,
        )
        cmd.refresh()
