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
import time
from dataclasses import dataclass

import numpy as np

from . import constants, protocol, transport
from .imd import compute_pull_forces
from .protocol import IMDHeaderType, SessionInfo
from .pymol_io import load_coordinates

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
        self._rx_buffer = bytearray()
        self._partial_frame: Frame | None = None
        self._frame_packet_index = 0
        self._n_atoms: int | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    def connect(self) -> SessionInfo:
        """Open the connection, parse handshake + session info, send Go."""
        self.conn = transport.connect(self.host, self.port, timeout=self.connect_timeout)
        try:
            # The handshake reveals the server's body endianness.
            self.endian = protocol.parse_handshake(
                transport.recv_exact(self.conn.sock, protocol.HEADER_SIZE)
            )
            self.conn.endian = self.endian
            htype, slot = self.conn.recv_header()
            if htype is not IMDHeaderType.SESSIONINFO:
                raise protocol.ProtocolError(f"expected SESSIONINFO, got {htype.name}")
            if slot != 7:
                raise protocol.ProtocolError(f"SESSIONINFO dataslot must be 7, got {slot}")
            self.session = protocol.parse_sessioninfo(self.conn.recv_body(slot))
            self._plan = self._frame_plan(self.session)
            if not self._plan:
                raise protocol.ProtocolError("SESSIONINFO enables no frame packets")
            self._rx_buffer.clear()
            self._partial_frame = None
            self._frame_packet_index = 0
            self._n_atoms = None
            self._closed = False
            # Go: tell the engine to start streaming frames.
            self.conn.send_bytes(protocol.pack_control(IMDHeaderType.GO))
        except Exception:
            self.conn.close()
            self.conn = None
            self._closed = True
            raise
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

        Bytes are accumulated in a persistent buffer. A timeout can therefore
        split either a header or body without discarding data or desynchronizing
        the next call. No part of a frame is read with an unbounded timeout.
        """
        if self._closed or not self._plan or self.conn is None:
            return None
        deadline = time.monotonic() + self._poll_timeout
        while True:
            frame = self._pop_buffered_frame()
            if frame is not None:
                return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            self.conn.settimeout(remaining)
            try:
                chunk = self.conn.sock.recv(constants.IMD_RECV_BUFSIZE)
            except (TimeoutError, BlockingIOError):
                return None
            except OSError:
                self._closed = True
                return None
            if not chunk:
                self._closed = True
                if self._rx_buffer or self._partial_frame is not None:
                    raise protocol.ProtocolError("connection closed in the middle of an IMD frame")
                return None
            self._rx_buffer.extend(chunk)

    def _pop_buffered_frame(self) -> Frame | None:
        """Decode complete packets from ``_rx_buffer`` without reading."""
        while True:
            if len(self._rx_buffer) < protocol.HEADER_SIZE:
                return None
            htype, slot = protocol.unpack_header(self._rx_buffer)
            expected = self._plan[self._frame_packet_index]
            if htype is not expected:
                raise protocol.ProtocolError(
                    f"frame desync: expected {expected.name}, got {htype.name}"
                )
            self._validate_frame_slot(htype, slot)
            body_nbytes = protocol.body_size(htype, slot)
            packet_nbytes = protocol.HEADER_SIZE + body_nbytes
            if len(self._rx_buffer) < packet_nbytes:
                return None
            body = bytes(self._rx_buffer[protocol.HEADER_SIZE : packet_nbytes])
            del self._rx_buffer[:packet_nbytes]
            if self._partial_frame is None:
                self._partial_frame = Frame()
            self._store(self._partial_frame, htype, slot, body)
            self._frame_packet_index += 1
            if self._frame_packet_index == len(self._plan):
                frame = self._partial_frame
                self._partial_frame = None
                self._frame_packet_index = 0
                return frame

    def _validate_frame_slot(self, htype: IMDHeaderType, slot: int) -> None:
        if htype in (IMDHeaderType.TIME, IMDHeaderType.ENERGIES, IMDHeaderType.BOX):
            if slot != 1:
                raise protocol.ProtocolError(f"{htype.name} dataslot must be 1, got {slot}")
            return
        if slot <= 0:
            raise protocol.ProtocolError(f"{htype.name} atom count must be positive, got {slot}")
        if self._n_atoms is None:
            self._n_atoms = slot
        elif slot != self._n_atoms:
            raise protocol.ProtocolError(
                f"{htype.name} atom count changed from {self._n_atoms} to {slot}"
            )

    def _store(self, frame: Frame, htype: IMDHeaderType, slot: int, body: bytes) -> None:
        e = self.endian
        if e is None:
            raise protocol.ProtocolError("body endianness is not negotiated")
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
        if self.conn is None or self.endian is None or self._closed:
            raise transport.ConnectionClosed("IMD client is not connected")
        self.conn.send_bytes(protocol.pack_mdcomm(indices, forces_kj_per_ang, self.endian))

    def send_control(self, htype: IMDHeaderType, slot: int = 0) -> None:
        if self.conn is None or self._closed:
            raise transport.ConnectionClosed("IMD client is not connected")
        self.conn.send_bytes(protocol.pack_control(htype, slot))

    def disconnect(self) -> None:
        if self.conn is None:
            return
        with contextlib.suppress(OSError):
            self.conn.send_bytes(protocol.pack_control(IMDHeaderType.DISCONNECT))
        self.conn.close()
        self.conn = None
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
        drag_level: str = constants.DEFAULT_DRAG_LEVEL,
    ) -> None:
        self.client = client
        self.pymol_object = pymol_object
        self.selector = f"({pymol_object})"
        self.const_k = const_k
        self.max_force_kj = max_force_kj
        self.delta_noise_nm = delta_noise_nm
        self.frame_interval_ms = frame_interval_ms
        self.drag_level = drag_level
        self._timer = None
        self._last_server_ang: np.ndarray | None = None
        self._active_prev: set[int] = set()
        # Edge-triggered guard for the pull-force limit warning (mirrors
        # IMDSystem._force_exceeded_prev): warn once when the drag goes over
        # max_force_kj, reset when it clears, so the Qt tick can't flood logs.
        self._force_exceeded_prev: bool = False
        # Per-atom residue / chain group ids read from the PyMOL object (lazily,
        # cached per level), used to expand a per-atom pull to the whole residue
        # / chain. Mirrors IMDSystem's in-process expansion so socket dragging
        # feels the same. The client has no OpenMM topology, hence the readout.
        self._group_ids: dict[str, np.ndarray] = {}

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
        self._active_prev.clear()
        self._last_server_ang = None

    def _tick(self) -> None:
        if self.client.closed:
            self.stop()
            return
        try:
            # Read drag before displaying a new frame can overwrite it.
            self._send_drag_forces()
            if self.client.closed:
                return
            latest = self._drain_to_latest()
            if latest is not None and latest.positions_ang is not None:
                self._display(latest.positions_ang)
                self._last_server_ang = latest.positions_ang
        except Exception as exc:
            logger.error("IMD client stream stopped: %s", exc)
            self.stop()

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

        self._validate_object(cmd, len(positions_ang))
        # Positions arrive in angstrom — already PyMOL's unit, so no conversion.
        load_coordinates(cmd, self.pymol_object, positions_ang)
        cmd.refresh()

    def _send_drag_forces(self) -> None:
        if self._last_server_ang is None:
            return
        from pymol import cmd

        self._validate_object(cmd, len(self._last_server_ang))
        coords = cmd.get_coordset(self.pymol_object, state=1)
        if coords is None or len(coords) != self._last_server_ang.shape[0]:
            raise RuntimeError("PyMOL coordinates changed; reload the prepared PDB and reconnect")
        pymol_ang = np.asarray(coords, dtype=float)
        delta_nm = (pymol_ang - self._last_server_ang) * constants.ANGSTROM_TO_NM
        active_idx, forces_nm, n_exceeded = compute_pull_forces(
            delta_nm, self.const_k, self.max_force_kj, self.delta_noise_nm
        )
        # Over-limit drag: the whole frame's pull was skipped. Warn once on the
        # rising edge (not every tick) and keep streaming.
        if n_exceeded and not self._force_exceeded_prev:
            logger.warning(
                "Pull force on %d atom(s) exceeded max_force_kj=%.1f kJ/mol/nm; "
                "this frame's pull was skipped (no force sent), MD continues. "
                "Drag a shorter distance or raise max_force_kj to pull harder.",
                n_exceeded,
                self.max_force_kj,
            )
        self._force_exceeded_prev = bool(n_exceeded)
        # Expand to the whole residue / chain (same behavior as the in-process
        # IMDSystem); no-op for the default atom level / a skipped frame.
        if self.drag_level != "atom" and active_idx.size:
            active_idx, forces_nm = self._expand_drag_groups(active_idx, forces_nm, len(coords))
        active_set = set(active_idx.tolist())
        if active_idx.size or self._active_prev:
            # n=0 MDComm (no active atoms) tells the server to release: it
            # zeroes any atom that was active on its previous frame.
            forces_ang = forces_nm[active_idx] * constants.KJ_PER_NM_TO_KJ_PER_ANGSTROM
            try:
                self.client.send_forces(active_idx, forces_ang)
            except (transport.ConnectionClosed, OSError):
                self.stop()
                return
        self._active_prev = active_set

    def _validate_object(self, cmd, n_atoms: int) -> None:
        if self.pymol_object not in cmd.get_object_list():
            raise RuntimeError(
                f"PyMOL object '{self.pymol_object}' was removed; reload the prepared PDB and reconnect"
            )
        actual = cmd.count_atoms(self.selector)
        if actual != n_atoms:
            raise RuntimeError(
                f"PyMOL object '{self.pymol_object}' has {actual} atoms, expected {n_atoms}; "
                "reload the prepared PDB and reconnect"
            )

    def _group_of(self, level: str, n_atoms: int) -> np.ndarray | None:
        """Per-atom integer group id for residue / chain, read from PyMOL once.

        Residues are keyed by ``(chain, resi)`` so identically-numbered residues
        in different chains don't merge; chains by chain id. Cached per level.
        Returns ``None`` if the readout count doesn't match (object mismatch),
        in which case the caller falls back to the unexpanded per-atom pull.
        """
        cached = self._group_ids.get(level)
        if cached is not None:
            return cached
        from pymol import cmd

        rows: list[tuple[str, str]] = []
        cmd.iterate(self.selector, "rows.append((chain, resi))", space={"rows": rows})
        if len(rows) != n_atoms:
            return None
        keys = [c for c, _r in rows] if level == "chain" else [f"{c}/{r}" for c, r in rows]
        lookup: dict[str, int] = {}
        ids = np.empty(n_atoms, dtype=np.int64)
        for i, key in enumerate(keys):
            ids[i] = lookup.setdefault(key, len(lookup))
        self._group_ids[level] = ids
        return ids

    def _expand_drag_groups(
        self, active_idx: np.ndarray, forces_nm: np.ndarray, n_atoms: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expand a per-atom pull to whole residues / chains (client side).

        Mirrors :meth:`pymol_imd.imd.IMDSystem._expand_drag_groups` but reads
        group membership from the PyMOL object (the client has no OpenMM
        topology). Every atom of a touched residue / chain gets the mean pull
        force of that group's dragged atoms. Falls back to the unexpanded pull
        when the group readout is unavailable.
        """
        group_of = self._group_of(self.drag_level, n_atoms)
        if group_of is None:
            return active_idx, forces_nm
        touched = np.unique(group_of[active_idx])
        expanded = np.where(np.isin(group_of, touched))[0]
        new_forces = forces_nm.copy()
        for g in touched:
            members = expanded[group_of[expanded] == g]
            dragged = active_idx[group_of[active_idx] == g]
            new_forces[members] = forces_nm[dragged].mean(axis=0)
        return expanded, new_forces
