"""Continuous code-driven dragging through the normal IMD force path."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from openmm import unit

if TYPE_CHECKING:
    from .imd import IMDSystem

OffsetFunction = Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DragResult:
    start_step: int
    end_step: int
    start_time_ps: float
    end_time_ps: float
    drag_steps: int
    release_steps: int
    applied_frames: int
    skipped_frames: int
    recovery_frames: int  # Retained for compatibility; automatic recovery is disabled.
    wall_seconds: float


def _duration_steps(value: float, timestep_ps: float, name: str, *, allow_zero=False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite duration in ps")
    duration = float(value)
    if not math.isfinite(duration) or duration < 0 or (duration == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    steps = round(duration / timestep_ps)
    if not math.isclose(duration, steps * timestep_ps):
        raise ValueError(f"{name} must be an integer multiple of timestep_ps={timestep_ps}")
    return steps


def run_drag(
    system: IMDSystem,
    atom_indices: Sequence[int],
    offsets_angstrom: np.ndarray | Sequence[float] | OffsetFunction,
    *,
    duration_ps: float,
    release_ps: float = 0.0,
) -> DragResult:
    """Run a headless drag, then an optional unforced release segment.

    Select atoms using zero-based indices in ``system.bundle.topology``. A
    constant three-vector repeats the same virtual mouse offset on every atom
    at every integration step. An ``(n_selected, 3)`` array allows different
    offsets per atom. A callback ``(elapsed_ps, positions_angstrom) -> offsets``
    can change direction or targets over time; it receives read-only full-system
    coordinates and returns offsets in the selected-index order. An offset is
    converted to a force by the existing ``const_k`` scale. No coordinates are
    overwritten. Forces are uncapped unless a limit is explicitly configured.

    The system must be stopped, with ``nsteps_per_frame=1`` and no open recorder.
    This function owns the stepping loop and uses the system's configured
    trajectory/CSV outputs. ML/MM corrections are refreshed at every step.
    Forces and recorders are released on completion, integration failure, callback
    error or interruption. This is a synchronous Python API, not a Qt callback
    and not a physical mouse event.
    """
    if system.running or system._recording:
        raise RuntimeError("Stop the current MD loop and recording before run_drag")
    if system.config.nsteps_per_frame != 1:
        raise ValueError("run_drag requires nsteps_per_frame=1 for per-step force updates")
    timestep = system.config.timestep_ps
    drag_steps = _duration_steps(duration_ps, timestep, "duration_ps")
    release_steps = _duration_steps(release_ps, timestep, "release_ps", allow_zero=True)
    indices = np.asarray(atom_indices)
    if indices.ndim != 1 or not indices.size or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("atom_indices must be a nonempty one-dimensional integer sequence")
    if np.any(indices < 0) or np.any(indices >= system.n_atoms):
        raise ValueError("atom_indices contains an index outside the prepared topology")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("atom_indices must not contain duplicates")
    if not callable(offsets_angstrom):
        offsets_angstrom = np.array(offsets_angstrom, dtype=float, copy=True)
        if (
            offsets_angstrom.shape not in ((3,), (len(indices), 3))
            or not np.isfinite(offsets_angstrom).all()
        ):
            raise ValueError("offsets_angstrom must be finite with shape (3,) or (n_selected, 3)")
    simulation = system.simulation
    start_step = simulation.currentStep
    start_time = simulation.context.getState().getTime().value_in_unit(unit.picosecond)
    applied = skipped = 0
    started = time.monotonic()
    try:
        system.clear_pull_forces()
        system.start_recording()
        system.running = True
        for frame in range(drag_steps + release_steps):
            if frame < drag_steps:
                if callable(offsets_angstrom):
                    positions = (
                        simulation.context.getState(getPositions=True)
                        .getPositions(asNumpy=True)
                        .value_in_unit(unit.angstrom)
                    )
                    positions = np.array(positions, copy=True)
                    positions.setflags(write=False)
                    offsets = offsets_angstrom(frame * timestep, positions)
                else:
                    offsets = offsets_angstrom
                system.apply_drag(indices, offsets)
                skipped += int(
                    system._force_exceeded_prev and not system._last_pull_active_idx.size
                )
            else:
                system.clear_pull_forces()
            system.apply_mlmm_if_present()
            alive, _ = system.advance_openmm()
            if not alive:
                raise RuntimeError(f"MD stopped at step {simulation.currentStep}")
            if frame < drag_steps and np.any(system._last_pull_forces):
                applied += 1
    finally:
        system.running = False
        try:
            system.clear_pull_forces()
        finally:
            system.stop_recording()
    return DragResult(
        start_step=start_step,
        end_step=simulation.currentStep,
        start_time_ps=start_time,
        end_time_ps=simulation.context.getState().getTime().value_in_unit(unit.picosecond),
        drag_steps=drag_steps,
        release_steps=release_steps,
        applied_frames=applied,
        skipped_frames=skipped,
        recovery_frames=0,
        wall_seconds=time.monotonic() - started,
    )
