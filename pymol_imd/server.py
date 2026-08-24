"""Headless IMDv3 server: drives an OpenMM :class:`IMDSystem` over a TCP socket.

The server owns the simulation and the connection lifecycle::

    bind/listen -> announce READY -> accept one client
      -> send Handshake -> send SessionInfo -> wait for Go (timeout)
      -> [receiver thread: MDComm forces + Pause/Resume/Disconnect/Kill]
      -> [main step thread: inject forces -> ML/MM correct -> step -> send frame]

Only the step thread touches OpenMM (a ``Context`` is not thread-safe); the
receiver thread stages incoming forces into a small locked buffer that the step
thread drains at the top of each frame. Wire units are IMDv3-fixed (angstrom,
kJ/mol/angstrom); the nm<->angstrom conversion happens here at the OpenMM
boundary via the named factors in :mod:`pymol_imd.constants`.
"""

from __future__ import annotations

import contextlib
import logging
import shlex
import socket
import threading
import time
from pathlib import Path

import numpy as np
from openmm import unit

from . import constants, protocol, transport
from .imd import IMDSystem
from .protocol import IMDHeaderType, SessionInfo

logger = logging.getLogger(__name__)


class IMDServer:
    """Serve one :class:`IMDSystem` to IMDv3 clients over TCP."""

    def __init__(
        self,
        imd: IMDSystem,
        *,
        host: str = constants.IMD_DEFAULT_HOST,
        port: int = constants.IMD_DEFAULT_PORT,
        rate: int | None = None,
        send_velocities: bool = constants.IMD_SEND_VELOCITIES,
        send_forces: bool = constants.IMD_SEND_FORCES,
        accept_timeout: float | None = constants.IMD_ACCEPT_TIMEOUT_S,
        go_timeout: float = constants.IMD_GO_TIMEOUT_S,
        record_settings: dict | None = None,
    ) -> None:
        self.imd = imd
        self.host = host
        self.port = port
        self._default_rate = max(
            1, int(rate if rate is not None else self.imd.config.nsteps_per_frame)
        )
        self.imd.config.nsteps_per_frame = self._default_rate
        self.accept_timeout = accept_timeout
        self.go_timeout = go_timeout
        self._prepared_pdb = str(imd.bundle.prepared_pdb_path)
        self.session = SessionInfo(
            time=True,
            energies=True,
            box=bool(imd.bundle.system.usesPeriodicBoundaryConditions()),
            coords=True,
            wrapped=False,
            velocities=bool(send_velocities),
            forces=bool(send_forces),
        )
        self._srv_sock: socket.socket | None = None
        self._conn: transport.IMDConnection | None = None
        self._endian: protocol.Endian = protocol.NATIVE
        self._recv_thread: threading.Thread | None = None
        # Shared MDComm force buffer (receiver thread writes, step thread drains).
        self._force_lock = threading.Lock()
        self._pending_idx: np.ndarray | None = None
        self._pending_forces_nm: np.ndarray | None = None
        self._pending_dirty = False
        # Control flags.
        self._stop = threading.Event()  # end the CURRENT session
        self._shutdown = threading.Event()  # end serve_forever's accept loop
        self._pause = threading.Event()
        self._kill = False
        self._force_exceeded_prev = False
        self._record_settings = dict(record_settings) if record_settings else None
        self._next_session_id = 1
        self.failure_reason: str | None = None

    # --------------------------------------------------------------- lifecycle
    def _ensure_listening(self) -> None:
        if self._srv_sock is None:
            self._srv_sock, self.port = transport.bind_listen(self.host, self.port)
            self._announce_ready()

    def _announce_ready(self) -> None:
        # Machine-readable line consumed by `pymol_imd local` / `connect` to
        # learn the port + prepared-PDB path (atom order must match on the
        # client). On stdout (not the logger) so a parent process can parse it.
        print(f"IMD_READY port={self.port} prepared={self._prepared_pdb}", flush=True)
        logger.info(
            "IMD server listening on %s:%d (atoms=%d, session=%s)",
            self.host,
            self.port,
            self.imd.n_atoms,
            self.session,
        )

    def serve_once(self) -> None:
        """Accept a single client and run until it disconnects / diverges."""
        try:
            self._ensure_listening()
            try:
                self._conn = transport.accept_one(self._srv_sock, timeout=self.accept_timeout)
            except TimeoutError:
                logger.error("no client connected within %ss", self.accept_timeout)
                return
            self._reset_session_state()
            try:
                self._run_session()
            finally:
                self._teardown_session()
        finally:
            self._close_listener()

    def serve_forever(self) -> None:
        """Accept clients until :meth:`request_stop` (one client at a time)."""
        try:
            self._ensure_listening()
            while not self._shutdown.is_set():
                try:
                    self._conn = transport.accept_one(self._srv_sock, timeout=1.0)
                except TimeoutError:
                    continue
                except OSError:
                    break
                self._reset_session_state()
                try:
                    self._run_session()
                finally:
                    self._teardown_session()
                if self._kill or self.failure_reason is not None:
                    break
        finally:
            self._teardown_session()
            self._close_listener()

    def request_stop(self) -> None:
        """Signal both the current session and the accept loop to end."""
        self._shutdown.set()
        self._stop.set()
        if self._conn is not None:
            self._conn.close()
        self._close_listener()

    def _close_listener(self) -> None:
        if self._srv_sock is not None:
            with contextlib.suppress(OSError):
                self._srv_sock.shutdown(socket.SHUT_RDWR)
            self._srv_sock.close()
            self._srv_sock = None

    def _reset_session_state(self) -> None:
        self._stop.clear()
        self._pause.clear()
        self.imd.running = False
        self.imd.stop_recording()
        self.imd.clear_pull_forces()
        self._force_exceeded_prev = False
        with self._force_lock:
            self._pending_dirty = False
            self._pending_idx = None
            self._pending_forces_nm = None

    def _teardown_session(self) -> None:
        self._stop.set()
        self.imd.running = False
        if self._conn is not None:
            self._conn.close()
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=1.0)
            self._recv_thread = None
        self._conn = None
        self.imd.clear_pull_forces()
        self.imd.stop_recording()

    # ----------------------------------------------------------------- session
    def _run_session(self) -> None:
        conn = self._conn
        if conn is None:
            raise RuntimeError("accepted IMD connection is missing")
        # The server packs all bodies in its own native endianness; the client
        # detects it from the handshake version int and parses accordingly.
        self._endian = protocol.NATIVE
        conn.endian = self._endian
        try:
            conn.send_bytes(protocol.pack_handshake())
            conn.send_bytes(protocol.pack_sessioninfo(self.session))
        except (transport.ConnectionClosed, OSError) as exc:
            logger.info("client disconnected during handshake (%s)", exc)
            return
        # Wait for the client's Go (bounded — an unresponsive peer must not
        # wedge the server here).
        conn.settimeout(self.go_timeout)
        try:
            htype, slot = conn.recv_header()
        except TimeoutError:
            logger.warning("client did not send GO within %ss", self.go_timeout)
            return
        except (protocol.ProtocolError, transport.ConnectionClosed, OSError) as exc:
            logger.info("client disconnected before GO (%s)", exc)
            return
        if htype is not IMDHeaderType.GO:
            logger.error("expected GO, got %s/%d; aborting session", htype.name, slot)
            return
        conn.settimeout(None)
        try:
            self._start_session_recording()
        except Exception as exc:
            self.failure_reason = f"recording setup failed: {exc}"
            self._shutdown.set()
            logger.exception("%s", self.failure_reason)
            with contextlib.suppress(OSError):
                conn.send_bytes(protocol.pack_control(IMDHeaderType.IOERROR))
            return
        logger.info("client connected; streaming frames")
        # advance_openmm() returns self.running, so the engine must be marked
        # running for the frame loop to proceed (mirrors imd_run()).
        self.imd.running = True
        self._recv_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._recv_thread.start()
        self._frame_loop()

    @staticmethod
    def _session_path(path: str, session_id: int) -> str:
        target = Path(path)
        return str(target.with_name(f"{target.stem}.session-{session_id:04d}{target.suffix}"))

    def _start_session_recording(self) -> None:
        """Resolve collision-free per-connection paths and start recording."""
        session_id = self._next_session_id
        settings = None
        if self._record_settings is not None:
            while True:
                candidate = dict(self._record_settings)
                paths = []
                for key in ("traj", "force_csv", "state_csv"):
                    value = candidate.get(key)
                    if value is not None:
                        candidate[key] = self._session_path(value, session_id)
                        paths.append(Path(candidate[key]))
                if not any(path.exists() for path in paths):
                    settings = candidate
                    break
                session_id += 1
            self.imd.set_recording(**settings)
            self.imd.start_recording()
        else:
            self.imd.set_recording()
        self._next_session_id = session_id + 1
        fields = [f"id={session_id:04d}"]
        if settings is not None:
            for key in ("traj", "force_csv", "state_csv"):
                if settings.get(key) is not None:
                    fields.append(f"{key}={shlex.quote(str(settings[key]))}")
        print(f"IMD_SESSION {' '.join(fields)}", flush=True)

    def _receiver_loop(self) -> None:
        conn = self._conn  # bound once; survives self._conn being cleared on teardown
        try:
            while not self._stop.is_set():
                htype, slot = conn.recv_header()
                self._validate_incoming_header(htype, slot)
                nbytes = protocol.body_size(htype, slot)
                body = conn.recv_body(nbytes) if nbytes else b""
                if not self._handle_incoming(htype, slot, body):
                    break
        except (protocol.ProtocolError, transport.ConnectionClosed, OSError) as exc:
            logger.info("receiver: connection closed (%s)", exc)
        finally:
            self._stop.set()

    def _validate_incoming_header(self, htype: IMDHeaderType, slot: int) -> None:
        allowed = {
            IMDHeaderType.MDCOMM,
            IMDHeaderType.PAUSE,
            IMDHeaderType.RESUME,
            IMDHeaderType.TRATE,
            IMDHeaderType.DISCONNECT,
            IMDHeaderType.KILL,
            IMDHeaderType.WAIT,
        }
        if htype not in allowed:
            raise protocol.ProtocolError(f"unexpected client packet {htype.name}")
        if htype is IMDHeaderType.MDCOMM and (slot < 0 or slot > self.imd.n_atoms):
            raise protocol.ProtocolError(f"MDComm count {slot} outside [0, {self.imd.n_atoms}]")

    def _handle_incoming(self, htype: IMDHeaderType, slot: int, body: bytes) -> bool:
        """Process one receiver-sent packet. Returns False to end the session."""
        if htype is IMDHeaderType.MDCOMM:
            idx, forces_ang = protocol.parse_mdcomm(body, slot, self._endian)
            forces_nm = forces_ang * constants.KJ_PER_ANGSTROM_TO_KJ_PER_NM
            if idx.size and ((idx < 0).any() or (idx >= self.imd.n_atoms).any()):
                raise protocol.ProtocolError(f"MDComm atom index outside [0, {self.imd.n_atoms})")
            magnitudes = np.linalg.norm(forces_nm, axis=1)
            exceeded = int(np.count_nonzero(magnitudes > self.imd.config.max_force_kj))
            if exceeded:
                if not self._force_exceeded_prev:
                    logger.warning(
                        "MDComm force on %d atom(s) exceeded max_force_kj=%.1f; "
                        "the packet was skipped and active pulls were released",
                        exceeded,
                        self.imd.config.max_force_kj,
                    )
                self._force_exceeded_prev = True
                idx = np.empty(0, dtype=np.int64)
                forces_nm = np.empty((0, 3), dtype=float)
            else:
                self._force_exceeded_prev = False
            with self._force_lock:
                self._pending_idx = idx
                self._pending_forces_nm = forces_nm
                self._pending_dirty = True
            return True
        if htype is IMDHeaderType.PAUSE:
            self._pause.set()
            return True
        if htype is IMDHeaderType.RESUME:
            self._pause.clear()
            return True
        if htype is IMDHeaderType.TRATE:
            if slot >= 1:
                self.imd.config.nsteps_per_frame = int(slot)
            else:
                self.imd.config.nsteps_per_frame = self._default_rate
            return True
        if htype is IMDHeaderType.DISCONNECT:
            logger.info("client requested disconnect")
            return False
        if htype is IMDHeaderType.KILL:
            logger.info("client requested kill")
            self._kill = True
            self._shutdown.set()
            return False
        # WAIT / others: ignored in v1.
        return True

    # -------------------------------------------------------------- frame loop
    def _frame_loop(self) -> None:
        dt_ps = self.imd.config.timestep_ps
        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.005)
                continue
            try:
                self._apply_pending_forces()
                self.imd.apply_mlmm_if_present()
                running, state = self.imd.advance_openmm(
                    get_positions=self.session.coords,
                    get_velocities=self.session.velocities,
                    get_forces=self.session.forces,
                )
            except Exception as exc:
                self.failure_reason = f"simulation step failed: {exc}"
                self._shutdown.set()
                logger.exception("%s", self.failure_reason)
                break
            if not running:
                logger.error("simulation diverged; ending session")
                self.failure_reason = "simulation diverged"
                self._shutdown.set()
                break
            try:
                self._send_frame(state, dt_ps)
            except protocol.ProtocolError as exc:
                self.failure_reason = f"frame encoding failed: {exc}"
                self._shutdown.set()
                logger.error("%s", self.failure_reason)
                break
            except (transport.ConnectionClosed, OSError) as exc:
                logger.info("send failed; client gone (%s)", exc)
                break

    def _apply_pending_forces(self) -> None:
        with self._force_lock:
            if not self._pending_dirty:
                return  # no new MDComm -> keep last frame's pull force in place
            idx = None if self._pending_idx is None else self._pending_idx.copy()
            forces_nm = None if self._pending_forces_nm is None else self._pending_forces_nm.copy()
            self._pending_dirty = False
        active = np.asarray(idx, dtype=int) if idx is not None else np.empty(0, dtype=int)
        full = np.zeros((self.imd.n_atoms, 3), dtype=float)
        if active.size:
            full[active] = forces_nm
        # set_pull_forces zeroes atoms active last frame but absent now, so a
        # client releasing a drag (n=0 MDComm) clears the force.
        self.imd.set_pull_forces(active, full)

    def _send_frame(self, state, dt_ps: float) -> None:
        conn = self._conn
        if conn is None:
            raise transport.ConnectionClosed("IMD connection was closed")
        e = self._endian
        si = self.session
        step = self.imd.simulation.currentStep
        # Fixed IMDv3 order: Time -> Energy -> Box -> Coords -> Velocities -> Forces.
        if si.time:
            conn.send_bytes(protocol.pack_time(dt_ps, step * dt_ps, step, e))
        if si.energies:
            ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            # (T, Etot, Epot, Evdw, Ecoul, Ebond, Eangle, Edihe, Eimpr); per-term
            # decomposition is a future enhancement (needs force groups).
            e9 = (self.imd.instantaneous_temperature(ke), pe + ke, pe, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            conn.send_bytes(protocol.pack_energies(step, e9, e))
        if si.box:
            vecs = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            conn.send_bytes(protocol.pack_box(np.asarray(vecs) * constants.NM_TO_ANGSTROM, e))
        if si.coords:
            pos_nm = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
            conn.send_bytes(
                protocol.pack_array(IMDHeaderType.COORDS, pos_nm * constants.NM_TO_ANGSTROM, e)
            )
        if si.velocities:
            vel = np.asarray(
                state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
            )
            conn.send_bytes(
                protocol.pack_array(IMDHeaderType.VELOCITIES, vel * constants.NM_TO_ANGSTROM, e)
            )
        if si.forces:
            frc = np.asarray(
                state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer
                )
            )
            conn.send_bytes(
                protocol.pack_array(
                    IMDHeaderType.FORCES, frc * constants.KJ_PER_NM_TO_KJ_PER_ANGSTROM, e
                )
            )
