"""Keep local MD off the viewer thread, with at most one frame in flight."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
from openmm import unit


class LocalIMDLoop:
    """Exchange mouse deltas and accepted states between Qt and one worker.

    Only the Qt thread calls PyMOL. While a frame is running, only the worker
    accesses OpenMM or recording streams. Commands must call ``wait`` before
    changing simulation settings, or ``close`` before replacing the system.
    """

    def __init__(self, system) -> None:
        self.system = system
        self._input_reference = system._read_openmm_positions_nm()
        self._future: Future | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pymol-imd")
        self._closed = False
        self._in_tick = False

    def tick(self) -> None:
        """Consume a finished frame and submit new input without waiting."""
        if self._closed or self._in_tick:
            return
        self._in_tick = True
        try:
            self._exchange()
        finally:
            self._in_tick = False

    def _exchange(self) -> None:
        future = self._future
        if future is not None and not future.done():
            return
        state = None
        if future is not None:
            self._future = None
            state = future.result()

        # Compare against the last consumed mouse coordinates, not the newer
        # OpenMM state: simulation motion must never become a phantom drag.
        current = self.system._read_pymol_positions_nm()
        delta = current - self._input_reference
        if state is not None:
            self.system._sync_positions_to_pymol(state)
            self._input_reference = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            )
        else:
            self._input_reference = current
        if not self._closed and self.system.running:
            # Finish the redraw before MD can compete for the same GPU.
            self._future = self._executor.submit(self._advance, delta)

    def _advance(self, delta: np.ndarray):
        system = self.system
        if system.config.mouse_force:
            system._apply_drag_delta_nm(delta)
        elif system._active_pull_idx_prev:
            system.clear_pull_forces()
        system.apply_mlmm_if_present()
        _, state = system.advance_openmm(get_positions=True)
        return state

    def wait(self) -> None:
        """Finish the single pending frame before a command touches OpenMM."""
        if self._future is not None:
            self._future.result()

    def close(self):
        """Join the worker and return its final state for an optional redraw."""
        self._closed = True
        future, self._future = self._future, None
        self._executor.shutdown(wait=True)
        return future.result() if future is not None else None
