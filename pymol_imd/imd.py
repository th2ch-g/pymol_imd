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

import logging
from dataclasses import dataclass

import numpy as np
import openmm
from openmm import LangevinIntegrator, LangevinMiddleIntegrator, MonteCarloBarostat, unit
from openmm.app import Simulation

from .backends.base import SystemBundle

logger = logging.getLogger(__name__)

# MLIP backends report energy in eV and forces in eV/Å; OpenMM expects
# kJ/mol and kJ/(mol·nm). 1 eV = 96.485 kJ/mol; 1 Å = 0.1 nm.
_EV_PER_A_TO_KJ_PER_MOL_NM = 96.48533212331 * 10.0


@dataclass
class IMDConfig:
    temperature: float = 300.0  # K
    pressure: float = 1.0  # bar (NPT only)
    friction: float = 1.0  # 1/ps
    timestep_ps: float = 0.002  # ps
    nsteps_per_frame: int = 1
    const_k: float = 10000.0  # kJ/mol/nm scale for the pulling force
    mouse_force: bool = True
    # Integrator selection: "langevin" (legacy BBK-like) or "langevin_middle"
    # (Leimkuhler-Matthews BAOAB). The middle scheme has 2nd-order accurate
    # configurational sampling and a wider stability window under external
    # forcing, so AA backends default to it; CG/MLMM keep "langevin" for now.
    integrator: str = "langevin"
    # Per-atom hard cap on the pulling force magnitude (kJ/mol/nm). Replaces
    # the old |delta| cap: a |F| cap is unit-meaningful (force) and prevents
    # runaway accelerations directly, regardless of how far the user drags
    # an atom or what const_k is set to.
    max_force_kj: float = 5000.0
    # If |potential energy| exceeds this (kJ/mol), the simulation is treated
    # as diverged and the IMD loop halts. NaN energies trip the same guard.
    energy_cutoff_kj: float = 1e9


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
        self.simulation = Simulation(bundle.topology, bundle.system, integrator)
        self.simulation.context.setPositions(bundle.positions)
        self.n_atoms = bundle.topology.getNumAtoms()
        self.running = False
        self._last_min_energy: float | None = None
        # Tracks atoms whose pull force was non-zero on the previous frame
        # so we can zero them on release without scanning all n_atoms.
        self._active_pull_idx_prev: set[int] = set()

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

    # ----------------------------------------------------------------- stepping
    def step(self) -> None:
        if self.config.mouse_force:
            self._apply_pull_force_from_pymol()
        if self._mlmm_force is not None:
            self._apply_mlmm_correction_force()
        self.simulation.step(self.config.nsteps_per_frame)
        if self._diverged():
            return
        self._sync_positions_to_pymol()

    def _apply_mlmm_correction_force(self) -> None:
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
        works for all of them.
        """
        from ase import Atoms

        extra = self.bundle.extra
        if extra is None:
            return
        ml_indices = extra["ml_indices"]
        ml_symbols = extra["ml_symbols"]
        link_atoms: tuple[tuple[int, int, float], ...] = extra.get("link_atoms", ())
        boundary_mm: tuple[int, ...] = extra.get("_mlmm_boundary_mm_indices", ())
        method: str = extra.get("link_atom_method", "scaled")
        sub_context: openmm.Context = extra["ml_subset_context"]
        ml_calc = extra["ml_calculator"]
        ml_info = extra.get("ml_calc_info", {})

        full_positions = self._read_openmm_positions_nm()
        ml_positions_nm = full_positions[list(ml_indices)]
        ml_positions_ang = ml_positions_nm * 10.0

        # 1) Build the ASE Atoms for MLIP inference. ML region first, link
        #    atoms appended after (order matters for force partitioning).
        if link_atoms:
            link_positions_ang = self._link_atom_positions_ang(full_positions, link_atoms, method)
            mlip_positions_ang = np.vstack([ml_positions_ang, link_positions_ang])
            mlip_symbols = list(ml_symbols) + ["H"] * len(link_atoms)
        else:
            mlip_positions_ang = ml_positions_ang
            mlip_symbols = list(ml_symbols)

        atoms = Atoms(symbols=mlip_symbols, positions=mlip_positions_ang)
        atoms.calc = ml_calc
        # AIMNet2 reads charge / multiplicity from atoms.info; harmless for
        # the other backends.
        atoms.info.update(ml_info)
        F_all_eV_per_A = np.asarray(atoms.get_forces(), dtype=np.float64)

        # 2) Partition MLIP forces. ML region atoms keep their forces; link
        #    atom forces are redistributed via the placement Jacobian to the
        #    QM-side (in ml_indices) and MM-side (in boundary_mm) atoms.
        n_ml = len(ml_indices)
        F_ml_eV_per_A = F_all_eV_per_A[:n_ml].copy()
        F_mm_boundary_eV_per_A = np.zeros((len(boundary_mm), 3), dtype=np.float64)

        if link_atoms:
            ml_local = {int(idx): i for i, idx in enumerate(ml_indices)}
            bdy_local = {int(idx): i for i, idx in enumerate(boundary_mm)}
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

    def _diverged(self) -> bool:
        """Return True (and mark the system stopped) if energy went insane.

        Reads the potential energy after a step; NaN/inf or |U| > cutoff
        indicates the integrator blew up (typically from over-large pull
        forces or a too-large timestep). Stopping here keeps PyMOL from
        receiving garbage coordinates that would push atoms off-screen.
        """
        energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
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

    def minimize_step(self, max_iterations: int = 50) -> bool:
        """Run a chunk of minimization; returns True when converged."""
        self.simulation.minimizeEnergy(maxIterations=max_iterations)
        self._sync_positions_to_pymol()
        energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
        energy_kj = energy.value_in_unit(unit.kilojoule_per_mole)
        converged = (
            self._last_min_energy is not None and abs(energy_kj - self._last_min_energy) < 1e-3
        )
        logger.info("minimize: U = %.4f kJ/mol%s", energy_kj, " (converged)" if converged else "")
        self._last_min_energy = energy_kj
        return converged

    # ------------------------------------------------------------- pulling logic
    # Noise threshold: prepared.pdb is written with %8.3f (0.001 A = 1e-4 nm
    # precision), and PyMOL stores coordinates as float32. The round-trip
    # leaves a ~1e-4 nm "ghost delta" on every atom, which used to produce a
    # uniform whole-molecule force. 1e-3 nm = 0.01 A sits comfortably above
    # that floor while still catching realistic mouse drags.
    _DELTA_NOISE_NM = 1e-3

    def _apply_pull_force_from_pymol(self) -> None:
        try:
            pymol_xyz = self._read_pymol_positions_nm()
        except ImportError:
            return  # No PyMOL → no user pull.
        # RuntimeError from atom-count mismatch etc. is intentionally NOT
        # caught here; that's a real fault and the user must hear about it.

        openmm_xyz = self._read_openmm_positions_nm()
        delta = pymol_xyz - openmm_xyz
        atom_max = np.max(np.abs(delta), axis=1)
        active_mask = atom_max >= self._DELTA_NOISE_NM
        delta[~active_mask] = 0.0

        # Apply the |F| cap directly on the per-atom force vector. Compared
        # to the previous |delta| cap, this makes the upper bound an explicit
        # function of force (in kJ/mol/nm) rather than an implicit one via
        # const_k * |delta|, so users tuning max_force_kj see a predictable
        # ceiling regardless of how const_k is set.
        forces = delta * self.config.const_k
        fmag = np.linalg.norm(forces, axis=1, keepdims=True)
        safe_fmag = np.where(fmag > 0.0, fmag, 1.0)
        forces = forces * (np.minimum(fmag, self.config.max_force_kj) / safe_fmag)
        active_idx = np.where(active_mask)[0]

        # Update only atoms that are active now OR were active last frame
        # (so we zero them on release). Skips the O(n_atoms) hot loop —
        # mouse drag normally touches ≤ a few dozen atoms.
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

        if active_idx.size:
            logger.debug(
                "pull: %d atoms dragged, |F|max requested=%.1f / cap=%.1f kJ/mol/nm",
                int(active_idx.size),
                float(np.max(fmag)),
                self.config.max_force_kj,
            )

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

    def _sync_positions_to_pymol(self) -> None:
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
        positions_ang = self._read_openmm_positions_nm() * 10.0
        space = {"positions": positions_ang}
        cmd.alter_state(
            1,
            self.pymol_selector,
            "x, y, z = positions[index - 1]",
            space=space,
        )
        cmd.refresh()
