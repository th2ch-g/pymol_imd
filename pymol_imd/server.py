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

import logging
import socket
import threading
import time

import numpy as np
from openmm import unit

from . import constants, protocol, transport
from .imd import IMDSystem
from .protocol import IMDHeaderType, SessionInfo

logger = logging.getLogger(__name__)

# Molar gas constant (kJ/mol/K) for the instantaneous-temperature estimate.
_GAS_CONSTANT_KJ = 0.00831446261815324


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
    ) -> None:
        self.imd = imd
        self.host = host
        self.port = port
        if rate is not None:
            # The transmitted MD steps per frame == IMDSystem's nsteps_per_frame,
            # so advance_openmm() picks it up directly (and TRATE can retune it).
            self.imd.config.nsteps_per_frame = max(1, int(rate))
        self.accept_timeout = accept_timeout
        self.go_timeout = go_timeout
        self._prepared_pdb = str(imd.bundle.prepared_pdb_path)
        self.session = SessionInfo(
            time=True,
            energies=True,
            box=bool(imd.bundle.pbc),
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

    def serve_forever(self) -> None:
        """Accept clients until :meth:`request_stop` (one client at a time)."""
        self._ensure_listening()
        while not self._shutdown.is_set():
            try:
                self._conn = transport.accept_one(self._srv_sock, timeout=1.0)
            except TimeoutError:
                continue  # re-check _shutdown periodically
            except OSError:
                break
            self._reset_session_state()
            try:
                self._run_session()
            finally:
                self._teardown_session()
            if self._kill:
                break
        if self._srv_sock is not None:
            self._srv_sock.close()
            self._srv_sock = None

    def request_stop(self) -> None:
        """Signal both the current session and the accept loop to end."""
        self._shutdown.set()
        self._stop.set()

    def _reset_session_state(self) -> None:
        self._stop.clear()
        self._pause.clear()
        with self._force_lock:
            self._pending_dirty = False
            self._pending_idx = None
            self._pending_forces_nm = None

    def _teardown_session(self) -> None:
        self._stop.set()
        if self._conn is not None:
            self._conn.close()
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=1.0)
            self._recv_thread = None
        self._conn = None

    # ----------------------------------------------------------------- session
    def _run_session(self) -> None:
        conn = self._conn
        # The server packs all bodies in its own native endianness; the client
        # detects it from the handshake version int and parses accordingly.
        self._endian = protocol.NATIVE
        conn.endian = self._endian
        conn.send_bytes(protocol.pack_handshake())
        conn.send_bytes(protocol.pack_sessioninfo(self.session))
        # Wait for the client's Go (bounded — an unresponsive peer must not
        # wedge the server here).
        conn.settimeout(self.go_timeout)
        try:
            htype, _ = conn.recv_header()
        except (transport.ConnectionClosed, OSError) as exc:
            logger.error("no Go from client within %ss (%s)", self.go_timeout, exc)
            return
        if htype is not IMDHeaderType.GO:
            logger.error("expected GO, got %s; aborting session", htype.name)
            return
        conn.settimeout(None)
        logger.info("client connected; streaming frames")
        # advance_openmm() returns self.running, so the engine must be marked
        # running for the frame loop to proceed (mirrors imd_run()).
        self.imd.running = True
        self._recv_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._recv_thread.start()
        self._frame_loop()

    def _receiver_loop(self) -> None:
        conn = self._conn  # bound once; survives self._conn being cleared on teardown
        try:
            while not self._stop.is_set():
                htype, slot = conn.recv_header()
                nbytes = protocol.body_size(htype, slot)
                body = conn.recv_body(nbytes) if nbytes else b""
                if not self._handle_incoming(htype, slot, body):
                    break
        except (transport.ConnectionClosed, OSError) as exc:
            logger.info("receiver: connection closed (%s)", exc)
        finally:
            self._stop.set()

    def _handle_incoming(self, htype: IMDHeaderType, slot: int, body: bytes) -> bool:
        """Process one receiver-sent packet. Returns False to end the session."""
        if htype is IMDHeaderType.MDCOMM:
            idx, forces_ang = protocol.parse_mdcomm(body, slot, self._endian)
            forces_nm = forces_ang * constants.KJ_PER_ANGSTROM_TO_KJ_PER_NM
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
            self._apply_pending_forces()
            self.imd.apply_mlmm_if_present()
            if not self.imd.advance_openmm():
                logger.error("simulation diverged; ending session")
                break
            try:
                self._send_frame(dt_ps)
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
        # MDComm indices come from an untrusted socket; drop any outside
        # [0, n_atoms) before scattering. An index >= n_atoms would raise an
        # uncaught IndexError that unwinds the step thread and kills the whole
        # server; a negative index would silently apply the force to the wrong
        # atom via numpy's negative indexing.
        if active.size:
            valid = (active >= 0) & (active < self.imd.n_atoms)
            if not valid.all():
                logger.warning(
                    "dropping %d out-of-range MDComm atom index(es) (n_atoms=%d)",
                    int((~valid).sum()),
                    self.imd.n_atoms,
                )
                active = active[valid]
                forces_nm = forces_nm[valid]
        full = np.zeros((self.imd.n_atoms, 3), dtype=float)
        if active.size:
            full[active] = forces_nm
        # set_pull_forces zeroes atoms active last frame but absent now, so a
        # client releasing a drag (n=0 MDComm) clears the force.
        self.imd.set_pull_forces(active, full)

    def _send_frame(self, dt_ps: float) -> None:
        conn = self._conn
        e = self._endian
        sim = self.imd.simulation
        si = self.session
        state = sim.context.getState(
            getPositions=si.coords,
            getEnergy=si.energies,
            getVelocities=si.velocities,
            getForces=si.forces,
        )
        step = sim.currentStep
        # Fixed IMDv3 order: Time -> Energy -> Box -> Coords -> Velocities -> Forces.
        if si.time:
            conn.send_bytes(protocol.pack_time(dt_ps, step * dt_ps, step, e))
        if si.energies:
            ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            # (T, Etot, Epot, Evdw, Ecoul, Ebond, Eangle, Edihe, Eimpr); per-term
            # decomposition is a future enhancement (needs force groups).
            e9 = (self._instantaneous_temperature(ke), pe + ke, pe, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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

    def _instantaneous_temperature(self, ke_kj: float) -> float:
        """Estimate T (K) from kinetic energy.

        Approximate: ``ndof = 3N - n_constraints`` (ignores the COM-motion
        remover's 3 dof). Good enough for a display value in the energies packet.
        """
        ndof = 3 * self.imd.n_atoms - self.imd.bundle.system.getNumConstraints()
        if ndof <= 0:
            return 0.0
        return 2.0 * ke_kj / (ndof * _GAS_CONSTANT_KJ)
