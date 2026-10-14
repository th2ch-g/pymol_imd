"""Headless regression checks for the external CueMol3 adapter."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import plistlib
import struct
import tempfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from pymol_imd.cuemol.__main__ import main
from pymol_imd.cuemol.bridge import CueMolBridge, CueMolIMD, read_prepared_pdb, target_forces
from pymol_imd.cuemol.cdp import CDPError, WebSocket, _local_url, _NoRedirects
from pymol_imd.cuemol.launch import close_launched_app, launch
from pymol_imd.protocol import IMDHeaderType


def _raises(error, operation):
    try:
        operation()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def _atom(serial, name, chain, residue, insertion=" ", alt=" "):
    return (
        f"ATOM  {serial:5d} {name:>4}{alt}ALA {chain}{residue:4d}{insertion}   "
        f"{1.0:8.3f}{2.0:8.3f}{3.0:8.3f}{1.0:6.2f}{0.0:6.2f}           C\n"
    )


def _test_mapping_and_forces(root):
    prepared = root / "prepared.pdb"
    text = _atom(8, "CA", "A", 1) + _atom(2, "N", "A", 1) + _atom(9, "CA", "B", 1)
    text += _atom(5, "CA", "A", 1, insertion="A", alt="B")
    prepared.write_text(text + "END\n")
    keys, positions = read_prepared_pdb(prepared)
    assert keys == [
        ["A", "1", "ALA", "CA", ""],
        ["A", "1", "ALA", "N", ""],
        ["B", "1", "ALA", "CA", ""],
        ["A", "1A", "ALA", "CA", "B"],
    ]
    np.testing.assert_allclose(positions, [[1, 2, 3]] * 4)
    for level, expected in (("atom", [0]), ("residue", [0, 1]), ("chain", [0, 1, 3])):
        indices, forces, exceeded = target_forces(
            positions,
            {"index": 0, "position": [1.1, 2, 3]},
            keys,
            const_k=50000,
            max_force_kj=5000,
            drag_level=level,
        )
        assert indices.tolist() == expected and not exceeded
        np.testing.assert_allclose(forces[indices], [[500, 0, 0]] * len(expected))
        assert not np.any(forces[2])
    for target, over_cap in ((None, False), ({"index": 0, "position": [3, 2, 3]}, True)):
        indices, forces, exceeded = target_forces(
            positions,
            target,
            keys,
            const_k=50000,
            max_force_kj=5000,
        )
        assert not len(indices) and not np.any(forces) and bool(exceeded) == over_cap
    _raises(
        ValueError,
        lambda: target_forces(
            positions,
            {"index": 99, "position": [0, 0, 0]},
            keys,
            const_k=50000,
            max_force_kj=5000,
        ),
    )
    for invalid in (
        text + _atom(10, "CA", "A", 1),
        "MODEL        1\n" + text + "MODEL        2\n",
        "END\n",
    ):
        prepared.write_text(invalid)
        _raises(ValueError, lambda: read_prepared_pdb(prepared))
    prepared.write_text(text)
    with contextlib.redirect_stderr(io.StringIO()):
        for flag, value in (("--duration", "nan"), ("--port", "0"), ("--frame-interval-ms", "0")):
            _raises(
                SystemExit,
                lambda flag=flag, value=value: main(
                    [
                        "connect",
                        str(prepared),
                        "--endpoint",
                        "http://localhost:9000",
                        flag,
                        value,
                    ]
                ),
            )


def _frame(opcode, payload, final=True):
    prefix = bytes([(0x80 if final else 0) | opcode])
    if len(payload) < 126:
        return prefix + bytes([len(payload)]) + payload
    return prefix + b"\x7e" + struct.pack("!H", len(payload)) + payload


class _Socket:
    def __init__(self):
        self.chunks = deque()
        self.sent = []
        self.closed = False

    def sendall(self, data):
        self.sent.append(data)
        if data.startswith(b"GET "):
            key = next(
                line.split(b": ", 1)[1]
                for line in data.split(b"\r\n")
                if line.startswith(b"Sec-WebSocket-Key:")
            )
            accept = base64.b64encode(
                hashlib.sha1(key + b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11").digest()
            )
            header = (
                b"HTTP/1.1 101 Switching Protocols\r\nSec-WebSocket-Accept: " + accept + b"\r\n\r\n"
            )
            self.chunks.extend([header[:7], header[7:]])

    def recv(self, size):
        data = self.chunks.popleft()
        if data is None:
            raise TimeoutError
        if len(data) > size:
            self.chunks.appendleft(data[size:])
        return data[:size]

    def settimeout(self, _timeout):
        pass

    def shutdown(self, _how):
        pass

    def close(self):
        self.closed = True


def _test_websocket():
    for url in (
        "http://example.com",
        "http://user:password@localhost",
        "https://localhost",
        "http://127.0.0.1.example.com",
    ):
        _raises(ValueError, lambda url=url: _local_url(url, "http"))
    _raises(
        CDPError,
        lambda: _NoRedirects().redirect_request(None, None, 302, "", {}, "https://example.com"),
    )
    sock = _Socket()
    with patch("pymol_imd.cuemol.cdp.socket.create_connection", return_value=sock):
        ws = WebSocket("ws://localhost:9000/devtools/browser/test")
    message = '{"value":"π"}'.encode()
    cut = message.index(b"\xcf") + 1
    first = _frame(1, message[:cut], final=False) + _frame(9, b"ping")
    tail = _frame(0, message[cut:])
    sock.chunks.extend([first[:1], first[1:], tail[:1], None, tail[1:]])
    _raises(TimeoutError, lambda: ws.receive_text(1))
    assert ws.receive_text(1) == message.decode()
    pong = sock.sent[-1]
    assert pong[0] == 0x8A and pong[1] == 0x84
    assert bytes(c ^ pong[2 + i % 4] for i, c in enumerate(pong[6:])) == b"ping"
    ws.send(b"x" * 130)
    sent = sock.sent[-1]
    assert sent[:4] == b"\x81\xfe\x00\x82"
    assert bytes(c ^ sent[4 + i % 4] for i, c in enumerate(sent[8:])) == b"x" * 130
    for bad in (b"\x81\x80", _frame(9, b"x", final=False), _frame(9, b"x" * 126)):
        ws.buffer = bytearray(bad)
        _raises(CDPError, ws._pop_frame)
    ws.buffer = bytearray(_frame(1, b"\xff"))
    _raises(CDPError, lambda: ws.receive_text(1))
    ws.close()
    assert sock.closed


class _Client:
    def __init__(self):
        self.conn = None
        self.closed = False
        self.session = SimpleNamespace(coords=True)
        self.frames = deque()
        self.forces = []
        self.controls = []

    def connect(self):
        self.conn = object()

    def poll_frame(self):
        return self.frames.popleft() if self.frames else None

    def send_forces(self, indices, forces):
        self.forces.append((indices.copy(), forces.copy()))

    def send_control(self, control):
        self.controls.append(control)

    def disconnect(self):
        self.conn = None
        self.closed = True


class _Bridge:
    atom_keys = [["A", "1", "ALA", "CA", ""]]
    initial = {"sceneId": 1, "objectId": 2}

    def __init__(self, client, script):
        self.client = client
        self.script = deque(script)
        self.target = None
        self.closed = False
        self.step = 0
        self.limit_warnings = []
        self.configured_drag_level = None

    def configure_drag_level(self, level):
        self.configured_drag_level = level

    def poll_input(self):
        row = self.script.popleft()
        if isinstance(row, BaseException):
            raise row
        self.target = row.get("target")
        self.step += 1
        self.client.frames.append(
            SimpleNamespace(positions_ang=np.zeros((len(self.atom_keys), 3)), step=self.step)
        )
        return {"events": row.get("events", []), "controls": row.get("controls", []), "counts": {}}

    def tick(self, positions, step, events):
        for event in events:
            if event["type"] == "release":
                self.target = None
            elif event["type"] == "down":
                self.target = event["target"]
        return {"target": self.target}

    def snapshot(self):
        return {"positions": [[0, 0, 0]] * len(self.atom_keys)}

    def status(self, step, force, *, over_limit=False):
        self.limit_warnings.append(over_limit)

    def close(self):
        self.closed = True


def _test_loop_cleanup(root):
    pull = {"target": {"index": 0, "position": [0.1, 0, 0]}}
    script = [
        pull,
        {"target": {"index": 0, "position": [10, 0, 0]}},
        {**pull, "controls": ["pause"]},
        {**pull, "controls": ["resume"]},
        {},
        {"controls": ["disconnect"]},
    ]
    client = _Client()
    bridge = _Bridge(client, script)
    runner = CueMolIMD(bridge, client, frame_interval_ms=1)
    with contextlib.redirect_stdout(io.StringIO()):
        stats = runner.run(evidence=root / "loop.json")
    assert stats["nonzero_force_packets"] == 2 and stats["force_cap_skips"] == 1
    assert stats["coordinate_max_error_ang"] == 0
    assert bridge.limit_warnings == [False, True, False, False, False]
    assert client.controls == [IMDHeaderType.PAUSE, IMDHeaderType.RESUME]
    np.testing.assert_allclose(client.forces[0][1], [[50, 0, 0]])
    assert not client.forces[-1][0].size and client.closed and bridge.closed
    assert json.loads((root / "loop.json").read_text())["frames_displayed"] == 5

    for error in (CDPError("viewport closed"), KeyboardInterrupt()):
        client = _Client()
        bridge = _Bridge(client, [pull, error])
        runner = CueMolIMD(bridge, client, frame_interval_ms=1)
        with contextlib.redirect_stdout(io.StringIO()):
            if isinstance(error, KeyboardInterrupt):
                assert runner.run()["interrupted"]
            else:
                _raises(CDPError, runner.run)
        assert client.forces[0][0].size and not client.forces[-1][0].size
        assert client.closed and bridge.closed

    output = root / "unwritable.json"
    output.mkdir()
    client = _Client()
    bridge = _Bridge(client, [{"controls": ["disconnect"]}])
    with contextlib.redirect_stdout(io.StringIO()):
        _raises(OSError, lambda: CueMolIMD(bridge, client).run(evidence=output))
    assert client.closed and bridge.closed


def _test_live_drag_level(root):
    target = {"index": 0, "position": [0.1, 0, 0]}
    script = [
        {"target": target},
        {"target": target, "controls": ["drag-level:residue"]},
        {"events": [{"type": "down", "target": target}]},
        {
            "target": target,
            "controls": ["drag-level:atom"],
            "events": [{"type": "release"}, {"type": "down", "target": target}],
        },
        {"target": target, "controls": ["pause", "drag-level:chain"]},
        {"controls": ["resume"]},
        {"events": [{"type": "down", "target": target}]},
        {"controls": ["disconnect"]},
    ]
    client = _Client()
    bridge = _Bridge(client, script)
    bridge.atom_keys = [
        ["A", "1", "ALA", "CA", ""],
        ["A", "1", "ALA", "N", ""],
        ["A", "2", "ALA", "CA", ""],
        ["B", "1", "ALA", "CA", ""],
    ]
    runner = CueMolIMD(bridge, client, drag_level="chain", frame_interval_ms=1)
    with contextlib.redirect_stdout(io.StringIO()):
        stats = runner.run(evidence=root / "drag-level.json")
    assert bridge.configured_drag_level == "chain"
    assert stats["drag_level"] == "chain" and stats["drag_level_changes"] == 3
    assert [indices.tolist() for indices, _ in client.forces] == [
        [0, 1, 2],
        [],
        [0, 1],
        [],
        [0],
        [],
        [],
        [0, 1, 2],
        [],
    ]
    for indices, forces in client.forces:
        if len(indices):
            np.testing.assert_allclose(forces, [[50, 0, 0]] * len(indices))
    records = [
        json.loads(line) for line in (root / "drag-level.events.jsonl").read_text().splitlines()
    ]
    assert [row["drag_level"] for row in records] == [
        "chain",
        "residue",
        "residue",
        "atom",
        "chain",
        "chain",
    ]
    assert records[1]["target"] is None and records[4]["target"] is None
    assert records[3]["target"] == target
    assert client.controls == [IMDHeaderType.PAUSE, IMDHeaderType.RESUME]
    assert client.closed and bridge.closed

    client = _Client()
    bridge = _Bridge(client, [{"target": target}, {"controls": ["drag-level:invalid"]}])
    with contextlib.redirect_stdout(io.StringIO()):
        _raises(ValueError, CueMolIMD(bridge, client, frame_interval_ms=1).run)
    assert not client.forces[-1][0].size and client.closed and bridge.closed


def _test_launcher(root):
    app = root / "CueMol3.app"
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    (contents / "MacOS" / "CueMol3").touch()
    (contents / "Info.plist").write_bytes(plistlib.dumps({"CFBundleExecutable": "CueMol3"}))
    profile = root / "profile"
    process = Mock()
    process.poll.return_value = None

    def started(*args, **kwargs):
        (profile / "DevToolsActivePort").write_text("9000\n/devtools/browser/test\n")
        return process

    with (
        patch("pymol_imd.cuemol.launch.sys.platform", "darwin"),
        patch("pymol_imd.cuemol.launch.subprocess.Popen", side_effect=started) as popen,
    ):
        child, endpoint = launch(app, profile, root / "app.log")
    assert child is process and endpoint == "http://127.0.0.1:9000"
    argv = popen.call_args.args[0]
    assert argv[:5] == ["open", "-n", "-W", "-a", str(app.resolve())]
    assert "--remote-debugging-port=0" in argv and "--remote-debugging-address=127.0.0.1" in argv
    _raises(RuntimeError, lambda: launch(app, profile, root / "app.log"))
    with patch("pymol_imd.cuemol.launch.CDP") as connection:
        close_launched_app(process, endpoint)
    connection.return_value.request.assert_called_once_with("Browser.close")
    connection.return_value.close.assert_called_once()
    process.terminate.assert_not_called()


def _test_cold_attach(root):
    prepared = root / "cold-start.pdb"
    prepared.write_text(_atom(1, "CA", "A", 1))
    connection = Mock()
    connection.events = [
        {
            "method": "Target.attachedToTarget",
            "sessionId": "page-session",
            "params": {"sessionId": "worker-session", "targetInfo": {"targetId": "worker"}},
        }
    ]
    connection.attach.return_value = "page-session"
    connection.request.return_value = {
        "targetInfos": [
            {"targetId": "page", "type": "page", "url": "file:///renderer/index.html"},
            {"targetId": "worker", "type": "worker", "url": ""},
        ]
    }
    polls = []

    def evaluate(expression, session):
        if 'internal.hasClass("SceneManager")' in expression:
            polls.append(True)
            return len(polls) > 1
        if 'const manager = internal.getService("SceneManager")' in expression:
            assert len(polls) >= 2, "adapter injected before native initialization"
        if expression.startswith("__cuemolIMD.load("):
            return {"positions": [[1, 2, 3]], "sceneId": 1, "objectId": 2}
        return {}

    connection.evaluate.side_effect = evaluate
    with patch("pymol_imd.cuemol.bridge.CDP", return_value=connection):
        bridge = CueMolBridge("http://localhost:9000", prepared)
    assert len(polls) == 2 and bridge.worker == "worker-session"
    bridge.close()
    bridge.close()
    connection.close.assert_called_once()
    expressions = [call.args[0] for call in connection.evaluate.call_args_list]
    assert expressions.count("globalThis.__cuemolIMDUI?.detach()") == 1
    assert expressions.count("globalThis.__cuemolIMD?.detach()") == 1


def main_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _test_mapping_and_forces(root)
        _test_websocket()
        _test_loop_cleanup(root)
        _test_live_drag_level(root)
        _test_launcher(root)
        _test_cold_attach(root)
    print(
        "CueMol smoke test OK (mapping, forces, live drag level, transport, cleanup, launcher, cold attach)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
