"""Headless IMDv3 socket round-trip smoke test (no PyMOL, no VMD).

Runs an :class:`IMDServer` (real OpenMM, tiny PDB) in a background thread and a
headless :class:`IMDClient` in the main thread on localhost. Asserts:

  * protocol pack/unpack round-trips for every header type and both endians,
  * the handshake endianness oracle (+ ProtocolError on garbage),
  * frame coordinates equal OpenMM positions x10 (nm -> angstrom),
  * an MDComm force round-trips: client kJ/mol/angstrom, /10 on the wire,
    server x10 back into the pull CustomExternalForce,
  * client/server atom-count agreement.

Usage::

    pixi run smoke-socket
    pixi run smoke-socket 1ubq
"""

from __future__ import annotations

import logging
import socket
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("smoke-socket")


def _test_protocol_roundtrip() -> None:
    """Pure pack/unpack checks — no socket, both endians."""
    from pymol_imd import protocol as p
    from pymol_imd.protocol import Endian
    from pymol_imd.protocol import IMDHeaderType as T

    for t in T:
        tt, slot = p.unpack_header(p.pack_header(t, 42))
        assert tt is t and slot == 42, (t, tt, slot)
    assert p.parse_handshake(p.pack_handshake()) is p.NATIVE

    for e in (Endian.BIG, Endian.LITTLE):
        large_step = 2**40 + 750
        packed_time = p.pack_time(0.002, 1.5, large_step, e)
        assert len(packed_time) == 32
        dt, tm, st = p.parse_time(packed_time[8:], e)
        assert abs(dt - 0.002) < 1e-9 and abs(tm - 1.5) < 1e-6 and st == large_step
        st2, e9 = p.parse_energies(p.pack_energies(750, range(9), e)[8:], e)
        assert st2 == 750 and np.allclose(e9, np.arange(9))
        box = np.arange(9).reshape(3, 3).astype(float)
        assert np.allclose(p.parse_box(p.pack_box(box, e)[8:], e), box, atol=1e-4)
        arr = np.random.RandomState(0).rand(5, 3)
        cb = p.pack_array(T.COORDS, arr, e)
        assert np.allclose(p.parse_array(cb[8:], 5, e), arr, atol=1e-6)
        idx = np.array([3, 1, 4])
        frc = np.random.RandomState(1).rand(3, 3)
        i2, f2 = p.parse_mdcomm(p.pack_mdcomm(idx, frc, e)[8:], 3, e)
        assert np.array_equal(i2, idx) and np.allclose(f2, frc, atol=1e-6)
        si = p.SessionInfo(
            time=True,
            energies=True,
            box=True,
            coords=True,
            wrapped=False,
            velocities=True,
            forces=False,
        )
        assert p.parse_sessioninfo(p.pack_sessioninfo(si)[8:]) == si

    golden = p.pack_header(T.TIME, 1) + struct.pack(">ddq", 0.002, 1.5, 2**40)
    assert p.pack_time(0.002, 1.5, 2**40, Endian.BIG) == golden
    try:
        p.pack_mdcomm(np.array([1, 1]), np.zeros((2, 3)), p.NATIVE)
    except p.ProtocolError:
        pass
    else:
        raise AssertionError("duplicate MDComm indices were accepted")
    nonbinary_flags = p.parse_sessioninfo(bytes((2, 1, 0, 1, 0, 0, 0)))
    assert nonbinary_flags.time and nonbinary_flags.energies
    assert p.body_size(T.TRATE, -1) == 0
    try:
        p.body_size(T.MDCOMM, -1)
    except p.ProtocolError:
        pass
    else:
        raise AssertionError("negative MDComm atom count was accepted")
    for invalid_packet in (
        lambda: p.pack_time(float("nan"), 1.0, 1, p.NATIVE),
        lambda: p.parse_array(np.full((1, 3), np.nan, dtype=np.float32).tobytes(), 1, p.NATIVE),
    ):
        try:
            invalid_packet()
        except p.ProtocolError:
            pass
        else:
            raise AssertionError("malformed or non-finite protocol payload was accepted")

    opp = "<i" if p.NATIVE is Endian.BIG else ">i"
    assert p.parse_handshake(struct.pack("!i", 4) + struct.pack(opp, 3)) is not p.NATIVE
    try:
        p.parse_handshake(struct.pack("!i", 4) + struct.pack("!i", 999))
    except p.ProtocolError:
        pass
    else:
        raise AssertionError("expected ProtocolError for bad handshake version")
    log.info("protocol round-trip OK (all header types, both endians, handshake oracle)")


def _test_fragmented_client_decoder() -> None:
    """Timeouts between arbitrary bytes must not lose framing state."""
    from pymol_imd import protocol as p
    from pymol_imd import transport
    from pymol_imd.client import IMDClient
    from pymol_imd.protocol import IMDHeaderType as T

    packet = b"".join(
        (
            p.pack_time(0.002, 0.002, 1, p.NATIVE),
            p.pack_energies(1, range(9), p.NATIVE),
            p.pack_array(T.COORDS, np.arange(9, dtype=float).reshape(3, 3), p.NATIVE),
        )
    )

    class FragmentSocket:
        def __init__(self, events):
            self.events = list(events)

        def settimeout(self, _timeout):
            return None

        def recv(self, nbytes):
            if not self.events:
                raise TimeoutError
            event = self.events.pop(0)
            if event is None:
                raise TimeoutError
            if len(event) > nbytes:
                self.events.insert(0, event[nbytes:])
                event = event[:nbytes]
            return event

        def sendall(self, _data):
            return None

        def shutdown(self, _how):
            return None

        def close(self):
            return None

    events = [packet[:3], None, packet[3:11], None, packet[11:47], None, packet[47:]]
    client = IMDClient(poll_timeout=0.001)
    client.conn = transport.IMDConnection(FragmentSocket(events), p.NATIVE)
    client.endian = p.NATIVE
    client._plan = [T.TIME, T.ENERGIES, T.COORDS]
    assert client.poll_frame() is None
    assert client.poll_frame() is None
    assert client.poll_frame() is None
    frame = client.poll_frame()
    assert frame is not None and frame.step == 1
    assert frame.positions_ang.shape == (3, 3)
    log.info("fragmented client decoder OK (header/body timeouts preserve bytes)")


def _wait_for_frame(client, timeout: float = 5.0):
    end = time.time() + timeout
    while time.time() < end:
        frame = client.poll_frame()
        if frame is not None and frame.positions_ang is not None:
            return frame
        time.sleep(0.005)
    raise AssertionError("no frame received within timeout")


def main(pdb_id: str = "1aki") -> int:
    _test_protocol_roundtrip()
    _test_fragmented_client_decoder()

    from pymol_imd.backends.aa import AABackend
    from pymol_imd.client import IMDClient
    from pymol_imd.imd import IMDConfig, IMDSystem
    from pymol_imd.prep import fetch_pdb
    from pymol_imd.protocol import IMDHeaderType
    from pymol_imd.server import IMDServer

    with tempfile.TemporaryDirectory() as tmp:
        pdb = fetch_pdb(pdb_id, out_dir=Path(tmp))
        bundle = AABackend().build_system(pdb)
        imd = IMDSystem(bundle, IMDConfig(nsteps_per_frame=1, mouse_force=False))
        imd.simulation.minimizeEnergy(maxIterations=200)
        n_atoms = bundle.topology.getNumAtoms()
        log.info("AA system: %d atoms", n_atoms)

        # Bind first (port assigned + READY printed), then serve on a thread so
        # the client knows the ephemeral port before connecting.
        server = IMDServer(
            imd,
            host="127.0.0.1",
            port=0,
            rate=1,
            accept_timeout=10.0,
            record_settings={
                "traj": str(Path(tmp) / "socket.xtc"),
                "force_csv": str(Path(tmp) / "socket_forces.csv"),
                "state_csv": str(Path(tmp) / "socket_state.csv"),
            },
        )
        server._ensure_listening()
        port = server.port
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        # A peer that vanishes during the handshake must not terminate the
        # long-running accept loop or poison the next session.
        probe = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        probe.close()
        time.sleep(0.05)
        assert t.is_alive(), "server stopped after an aborted handshake"

        client = IMDClient("127.0.0.1", port)
        si = client.connect()
        assert si.coords and si.time and si.energies, si
        log.info("connected; session=%s", si)

        frame = _wait_for_frame(client)
        assert frame.positions_ang.shape == (n_atoms, 3), frame.positions_ang.shape
        log.info("first frame OK: %d atoms, step=%d", frame.positions_ang.shape[0], frame.step)

        # Unit check: wire is angstrom, OpenMM is nm. Generous atol absorbs the
        # few MD steps the server thread advances between frame send and this read.
        omm_ang = imd._read_openmm_positions_nm() * 10.0
        diff = float(np.max(np.abs(frame.positions_ang - omm_ang)))
        assert diff < 5.0, f"coord scale mismatch (max |frame - openmm*10| = {diff} A)"
        log.info("coord unit OK: max|frame - openmm*10| = %.4f A", diff)

        # Force round-trip: 100 kJ/mol/angstrom -> 1000 kJ/mol/nm in the pull force.
        # Keep draining frames so the server's step thread keeps looping (and so
        # applies the MDComm) instead of blocking on a full TCP send buffer.
        client.send_forces(np.array([0, 1]), np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]))
        fx0 = 0.0
        for _ in range(200):
            client.poll_frame()
            fx0 = float(imd._pull_force.getParticleParameters(0)[1][0])
            if abs(fx0 - 1000.0) < 1.0:
                break
            time.sleep(0.005)
        assert abs(fx0 - 1000.0) < 1.0, f"force round-trip: expected 1000, got {fx0}"
        log.info("force round-trip OK: 100 kJ/mol/A on the wire -> %.1f kJ/mol/nm pulled", fx0)

        # Server-side safety applies even to a non-PyMOL client that bypasses
        # the client force cap. An over-limit packet releases the active pull.
        client.send_forces(np.array([0]), np.array([[1000.0, 0.0, 0.0]]))
        for _ in range(200):
            client.poll_frame()
            fx0 = float(imd._pull_force.getParticleParameters(0)[1][0])
            if abs(fx0) < 1e-12:
                break
            time.sleep(0.005)
        assert abs(fx0) < 1e-12, f"over-limit MDComm force remained active: {fx0}"

        client.send_control(IMDHeaderType.TRATE, 3)
        for _ in range(100):
            if imd.config.nsteps_per_frame == 3:
                break
            time.sleep(0.005)
        assert imd.config.nsteps_per_frame == 3
        client.send_control(IMDHeaderType.TRATE, -7)
        for _ in range(100):
            if imd.config.nsteps_per_frame == 1:
                break
            time.sleep(0.005)
        assert imd.config.nsteps_per_frame == 1

        # Freeze regression: the PyMOL viewer tick MUST return promptly even
        # while the server streams a frame every step. Mock the two PyMOL
        # touchpoints so _tick runs headless; an unbounded drain would hang here.
        from pymol_imd.client import PyMOLIMDClient

        viewer = PyMOLIMDClient(client, "dummy", const_k=50000.0, max_force_kj=5000.0)
        viewer._display = lambda positions_ang: None
        viewer._send_drag_forces = lambda: None
        time.sleep(0.2)  # let frames pile up in the socket buffer
        t0 = time.time()
        viewer._tick()
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"_tick took {elapsed:.2f}s — drain unbounded (would freeze the GUI)"
        log.info("freeze guard OK: _tick returned in %.3fs under continuous streaming", elapsed)

        for _ in range(5):
            _wait_for_frame(client)
        log.info("streamed additional frames OK")

        client.disconnect()
        end = time.time() + 3.0
        while time.time() < end and imd.running:
            time.sleep(0.01)
        assert not imd.running
        assert not imd._active_pull_idx_prev

        # A reconnect resumes the same simulation but gets new recording files
        # and no force inherited from the previous client.
        client2 = IMDClient("127.0.0.1", port)
        client2.connect()
        frame2 = _wait_for_frame(client2)
        assert frame2.step > frame.step
        assert float(imd._pull_force.getParticleParameters(0)[1][0]) == 0.0
        client2.disconnect()
        time.sleep(0.1)
        server.request_stop()
        t.join(timeout=3.0)
        assert not t.is_alive(), "server thread did not stop"
        for session_id in (1, 2):
            for stem, suffix in (
                ("socket", ".xtc"),
                ("socket_forces", ".csv"),
                ("socket_state", ".csv"),
            ):
                output = Path(tmp) / f"{stem}.session-{session_id:04d}{suffix}"
                assert output.exists() and output.stat().st_size > 0, output
        log.info("Socket smoke test OK")
    return 0


if __name__ == "__main__":
    pdb_id = sys.argv[1] if len(sys.argv) > 1 else "1aki"
    raise SystemExit(main(pdb_id))
