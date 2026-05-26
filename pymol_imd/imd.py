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
from openmm import LangevinIntegrator, MonteCarloBarostat, unit
from openmm.app import Simulation

from .backends.base import SystemBundle

logger = logging.getLogger(__name__)


@dataclass
class IMDConfig:
    temperature: float = 300.0  # K
    pressure: float = 1.0  # bar (NPT only)
    friction: float = 1.0  # 1/ps
    timestep_ps: float = 0.002  # ps
    nsteps_per_frame: int = 1
    const_k: float = 5000.0  # kJ/mol/nm scale for the pulling force
    mouse_force: bool = True
    # Cap on |delta| (nm) per atom before multiplying by const_k. Prevents
    # runaway forces when the user drags an atom a huge distance: without
    # this, |F| = |delta| * const_k can dwarf bond stretching constants and
    # the structure explodes off-screen.
    max_pull_nm: float = 0.5
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
        if bundle.pbc:
            bundle.system.addForce(
                MonteCarloBarostat(
                    self.config.pressure * unit.bar,
                    self.config.temperature * unit.kelvin,
                )
            )

        integrator = LangevinIntegrator(
            self.config.temperature * unit.kelvin,
            self.config.friction / unit.picosecond,
            self.config.timestep_ps * unit.picoseconds,
        )
        self.simulation = Simulation(bundle.topology, bundle.system, integrator)
        self.simulation.context.setPositions(bundle.positions)
        self.n_atoms = bundle.topology.getNumAtoms()
        self.running = False
        self._last_min_energy: float | None = None

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _attach_pull_force(system: openmm.System) -> openmm.CustomExternalForce:
        force = openmm.CustomExternalForce("- fx*x - fy*y - fz*z")
        for name in ("fx", "fy", "fz"):
            force.addPerParticleParameter(name)
        zero = (0.0, 0.0, 0.0)
        for atom_idx in range(system.getNumParticles()):
            force.addParticle(atom_idx, zero)
        system.addForce(force)
        return force

    # ----------------------------------------------------------------- stepping
    def step(self) -> None:
        if self.config.mouse_force:
            self._apply_pull_force_from_pymol()
        self.simulation.step(self.config.nsteps_per_frame)
        if self._diverged():
            return
        self._sync_positions_to_pymol()

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
            "Lower const_k / max_pull_nm or shorten timestep_ps.",
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
        except Exception as exc:  # pragma: no cover - only happens without PyMOL
            logger.debug("Skipping pull force: PyMOL not available (%s)", exc)
            return
        openmm_xyz = self._read_openmm_positions_nm()
        delta = pymol_xyz - openmm_xyz
        # Mask per-atom: an atom counts as "dragged" only if its largest
        # component is above the noise floor. We zero the whole row so a
        # noisy +y on an otherwise-still atom doesn't leak a sideways force.
        atom_max = np.max(np.abs(delta), axis=1)
        delta[atom_max < self._DELTA_NOISE_NM] = 0.0

        # Cap per-atom |delta| at max_pull_nm so the force stays bounded even
        # if the user yanks an atom across the screen. Direction is kept;
        # only the magnitude is clipped.
        mag = np.linalg.norm(delta, axis=1, keepdims=True)
        safe_mag = np.where(mag > 0, mag, 1.0)
        scale = np.minimum(mag, self.config.max_pull_nm) / safe_mag
        delta = delta * scale

        n_dragged = int(np.count_nonzero(atom_max >= self._DELTA_NOISE_NM))
        if n_dragged:
            logger.debug(
                "pull: %d atoms dragged, max |delta| = %.4f nm (capped at %.3f)",
                n_dragged,
                float(np.max(mag)),
                self.config.max_pull_nm,
            )
        forces = delta * self.config.const_k
        for idx in range(self.n_atoms):
            fx, fy, fz = forces[idx]
            self._pull_force.setParticleParameters(idx, idx, (float(fx), float(fy), float(fz)))
        self._pull_force.updateParametersInContext(self.simulation.context)

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
