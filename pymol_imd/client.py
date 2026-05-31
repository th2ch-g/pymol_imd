"""IMDv3 client: a headless protocol receiver + a PyMOL viewer wrapper.

Two classes:

* :class:`IMDClient` — no PyMOL / OpenMM dependency. Connects to an IMDv3
  server, negotiates the session, reads frames, and sends MDComm forces /
  control packets. Used directly by the socket smoke test.
* :class:`PyMOLIMDClient` — drives an :class:`IMDClient` from a Qt timer:
  displays each received frame in PyMOL and turns mouse-drag deltas into
  MDComm force packets, reusing :func:`pymol_imd.imd.compute_pull_forces` so
  dragging over a socket feels identical to the same-process path.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass

import numpy as np

from . import constants, protocol, transport
from .imd import compute_pull_forces
from .protocol import IMDHeaderType, SessionInfo

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """One IMD frame's decoded payload (only the announced fields are set)."""

    step: int = 0
    time_ps: float = 0.0
    dt_ps: float = 0.0
    positions_ang: np.ndarray | None = None
    box_ang: np.ndarray | None = None
    energies: np.ndarray | None = None
    velocities_ang: np.ndarray | None = None
    forces_ang: np.ndarray | None = None


class IMDClient:
    """Headless IMDv3 receiver (no PyMOL / OpenMM)."""

    def __init__(
        self,
        host: str = constants.IMD_DEFAULT_HOST,
        port: int = constants.IMD_DEFAULT_PORT,
        *,
        connect_timeout: float = constants.IMD_CONNECT_TIMEOUT_S,
        poll_timeout: float = constants.IMD_CLIENT_POLL_TIMEOUT_S,
    ) -> None:
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self._poll_timeout = poll_timeout
        self.conn: transport.IMDConnection | None = None
        self.endian: protocol.Endian | None = None
        self.session: SessionInfo | None = None
        self._plan: list[IMDHeaderType] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def connect(self) -> SessionInfo:
        """Open the connection, parse handshake + session info, send Go."""
        self.conn = transport.connect(self.host, self.port, timeout=self.connect_timeout)
        # The handshake reveals the server's body endianness.
        self.endian = protocol.parse_handshake(
            transport.recv_exact(self.conn.sock, protocol.HEADER_SIZE)
        )
        self.conn.endian = self.endian
        # Session info packet.
        htype, slot = self.conn.recv_header()
        if htype is not IMDHeaderType.SESSIONINFO:
            raise protocol.ProtocolError(f"expected SESSIONINFO, got {htype.name}")
        self.session = protocol.parse_sessioninfo(self.conn.recv_body(slot))
        self._plan = self._frame_plan(self.session)
        # Go: tell the engine to start streaming frames.
        self.conn.send_bytes(protocol.pack_control(IMDHeaderType.GO))
        logger.info("connected to %s:%d; session=%s", self.host, self.port, self.session)
        return self.session

    @staticmethod
    def _frame_plan(si: SessionInfo) -> list[IMDHeaderType]:
        """The fixed per-frame packet order, filtered to the enabled ones."""
        plan: list[IMDHeaderType] = []
        if si.time:
            plan.append(IMDHeaderType.TIME)
        if si.energies:
            plan.append(IMDHeaderType.ENERGIES)
        if si.box:
            plan.append(IMDHeaderType.BOX)
        if si.coords:
            plan.append(IMDHeaderType.COORDS)
        if si.velocities:
            plan.append(IMDHeaderType.VELOCITIES)
        if si.forces:
            plan.append(IMDHeaderType.FORCES)
        return plan

    def poll_frame(self) -> Frame | None:
        """Return the next full frame, or None if none is ready this tick.

        Only the first header is read under a short timeout; once it arrives the
        rest of the frame is read blocking (frames stream contiguously, so an
        8-byte header splitting across the poll window is not a concern on a
        localhost / LAN socket).
        """
        if self._closed or not self._plan:
            return None
        self.conn.settimeout(self._poll_timeout)
        try:
            htype, slot = self.conn.recv_header()
        except (TimeoutError, BlockingIOError):
            return None
        except transport.ConnectionClosed:
            self._closed = True
            return None
        self.conn.settimeout(None)
        try:
            return self._assemble_frame(htype, slot)
        except transport.ConnectionClosed:
            self._closed = True
            return None

    def _assemble_frame(self, first_htype: IMDHeaderType, first_slot: int) -> Frame:
        frame = Frame()
        htype, slot = first_htype, first_slot
        for i, expected in enumerate(self._plan):
            if i > 0:
                htype, slot = self.conn.recv_header()
            if htype is not expected:
                raise protocol.ProtocolError(
                    f"frame desync: expected {expected.name}, got {htype.name}"
                )
            body = self.conn.recv_body(protocol.body_size(htype, slot))
            self._store(frame, htype, slot, body)
        return frame

    def _store(self, frame: Frame, htype: IMDHeaderType, slot: int, body: bytes) -> None:
        e = self.endian
        if htype is IMDHeaderType.TIME:
            frame.dt_ps, frame.time_ps, frame.step = protocol.parse_time(body, e)
        elif htype is IMDHeaderType.ENERGIES:
            _, frame.energies = protocol.parse_energies(body, e)
        elif htype is IMDHeaderType.BOX:
            frame.box_ang = protocol.parse_box(body, e)
        elif htype is IMDHeaderType.COORDS:
            frame.positions_ang = protocol.parse_array(body, slot, e)
        elif htype is IMDHeaderType.VELOCITIES:
            frame.velocities_ang = protocol.parse_array(body, slot, e)
        elif htype is IMDHeaderType.FORCES:
            frame.forces_ang = protocol.parse_array(body, slot, e)

    def send_forces(self, indices: np.ndarray, forces_kj_per_ang: np.ndarray) -> None:
        """Send an MDComm force packet (forces already in kJ/mol/angstrom)."""
        self.conn.send_bytes(protocol.pack_mdcomm(indices, forces_kj_per_ang, self.endian))

    def send_control(self, htype: IMDHeaderType, slot: int = 0) -> None:
        self.conn.send_bytes(protocol.pack_control(htype, slot))

    def disconnect(self) -> None:
        if self.conn is None:
            return
        with contextlib.suppress(OSError):
            self.conn.send_bytes(protocol.pack_control(IMDHeaderType.DISCONNECT))
        self.conn.close()
        self._closed = True


class PyMOLIMDClient:
    """Drive an :class:`IMDClient` from a PyMOL Qt timer (the interactive viewer).

    Each tick: read the user's drag (against the last server frame) and send it
    as forces, THEN display the newest received frame. The drag-read happens
    before the display overwrite so a frame push never erases an in-progress
    pull. Drag -> force uses the same :func:`compute_pull_forces` as the
    same-process path, so the feel is identical.
    """

    def __init__(
        self,
        client: IMDClient,
        pymol_object: str,
        *,
        const_k: float,
        max_force_kj: float,
        delta_noise_nm: float = constants.DELTA_NOISE_NM,
        frame_interval_ms: int = constants.IMD_CLIENT_FRAME_INTERVAL_MS,
    ) -> None:
        self.client = client
        self.pymol_object = pymol_object
        self.selector = f"({pymol_object})"
        self.const_k = const_k
        self.max_force_kj = max_force_kj
        self.delta_noise_nm = delta_noise_nm
        self.frame_interval_ms = frame_interval_ms
        self._timer = None
        self._last_server_ang: np.ndarray | None = None
        self._active_prev: set[int] = set()

    def start(self) -> None:
        from pymol.Qt import QtCore

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.frame_interval_ms)
        logger.info("PyMOL IMD client streaming on object '%s'", self.pymol_object)

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.client.disconnect()

    def _tick(self) -> None:
        if self.client.closed:
            self.stop()
            return
        # 1) Read the user's drag FIRST (vs the last displayed server coords),
        #    so the display overwrite below never wipes an in-progress pull.
        self._send_drag_forces()
        # 2) Drain to the newest available frame, then show it.
        latest = self._drain_to_latest()
        if latest is not None and latest.positions_ang is not None:
            self._display(latest.positions_ang)
            self._last_server_ang = latest.positions_ang

    def _drain_to_latest(self) -> Frame | None:
        """Read up to ``IMD_CLIENT_MAX_FRAMES_PER_TICK`` queued frames, newest wins.

        The cap is essential: the server streams a frame every MD step, so an
        unbounded ``while`` drain would never return when the server is faster
        than this tick — wedging the Qt event loop (the viewer-freeze bug). With
        the cap the tick always returns; any leftover frames are consumed on the
        next tick and the socket's own backpressure throttles the server.
        """
        latest = None
        for _ in range(constants.IMD_CLIENT_MAX_FRAMES_PER_TICK):
            frame = self.client.poll_frame()
            if frame is None:
                break
            latest = frame
        return latest

    def _display(self, positions_ang: np.ndarray) -> None:
        from pymol import cmd

        # Positions arrive in angstrom — already PyMOL's unit, so no conversion.
        cmd.alter_state(
            1, self.selector, "x, y, z = positions[index - 1]", space={"positions": positions_ang}
        )
        cmd.refresh()

    def _send_drag_forces(self) -> None:
        if self._last_server_ang is None:
            return
        from pymol import cmd

        coords: list[tuple[float, float, float]] = []
        cmd.iterate_state(1, self.selector, "coords.append((x, y, z))", space={"coords": coords})
        if len(coords) != self._last_server_ang.shape[0]:
            return  # atom-count mismatch (wrong object loaded); skip silently
        pymol_ang = np.asarray(coords)
        delta_nm = (pymol_ang - self._last_server_ang) * constants.ANGSTROM_TO_NM
        active_idx, forces_nm = compute_pull_forces(
            delta_nm, self.const_k, self.max_force_kj, self.delta_noise_nm
        )
        active_set = set(active_idx.tolist())
        if active_idx.size or self._active_prev:
            # n=0 MDComm (no active atoms) tells the server to release: it
            # zeroes any atom that was active on its previous frame.
            forces_ang = forces_nm[active_idx] * constants.KJ_PER_NM_TO_KJ_PER_ANGSTROM
            try:
                self.client.send_forces(active_idx, forces_ang)
            except OSError:
                self.stop()
                return
        self._active_prev = active_set
