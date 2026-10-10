"""Small, synchronous, loopback-only CDP transport using the Python standard library."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import socket
import struct
import time
import urllib.parse
import urllib.request
from collections import deque


class CDPError(RuntimeError):
    """A debug connection or evaluated CueMol operation failed."""


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CDPError("The loopback CueMol debug endpoint must not redirect")


def _local_url(url: str, scheme: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != scheme
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(f"Expected a loopback {scheme} URL without credentials")
    return parsed


class WebSocket:
    """A bounded RFC 6455 text transport; fragmented messages and ping are supported."""

    MAX_MESSAGE = 32 * 1024 * 1024

    def __init__(self, url: str, timeout: float = 10.0):
        parsed = _local_url(url, "ws")
        self.sock = socket.create_connection((parsed.hostname, parsed.port or 80), timeout)
        self.buffer = bytearray()
        self.fragments = bytearray()
        self.fragment_opcode: int | None = None
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        path = urllib.parse.urlunparse(
            ("", "", parsed.path or "/", parsed.params, parsed.query, "")
        )
        request = (
            f"GET {path} HTTP/1.1\r\nHost: {parsed.netloc}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        try:
            self.sock.sendall(request.encode("ascii"))
            while b"\r\n\r\n" not in self.buffer:
                self._receive()
                if len(self.buffer) > 65536:
                    raise CDPError("WebSocket handshake header is too large")
            header, rest = self.buffer.split(b"\r\n\r\n", 1)
            self.buffer = bytearray(rest)
            lines = header.decode("latin1").split("\r\n")
            headers = {
                name.lower(): value.strip()
                for name, value in (line.split(":", 1) for line in lines[1:] if ":" in line)
            }
            expected = base64.b64encode(
                hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
            ).decode()
            if " 101 " not in lines[0] or headers.get("sec-websocket-accept") != expected:
                raise CDPError("CueMol rejected the WebSocket upgrade")
        except BaseException:
            self.sock.close()
            raise

    def _receive(self) -> None:
        chunk = self.sock.recv(65536)
        if not chunk:
            raise CDPError("CueMol closed the debug connection")
        self.buffer.extend(chunk)

    def send(self, payload: bytes, opcode: int = 1) -> None:
        if len(payload) > self.MAX_MESSAGE:
            raise CDPError("WebSocket message is too large")
        mask = os.urandom(4)
        size = len(payload)
        header = (
            bytes((0x80 | opcode, 0x80 | size))
            if size < 126
            else (
                bytes((0x80 | opcode, 0xFE)) + struct.pack("!H", size)
                if size <= 65535
                else bytes((0x80 | opcode, 0xFF)) + struct.pack("!Q", size)
            )
        )
        masked = bytes(value ^ mask[i % 4] for i, value in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def receive_text(self, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while True:
            self.sock.settimeout(max(0.001, deadline - time.monotonic()))
            frame = self._pop_frame()
            if frame is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for CueMol")
                self._receive()
                continue
            opcode, final, data = frame
            if opcode == 8:
                raise CDPError("CueMol closed the WebSocket")
            if opcode == 9:
                self.send(data, opcode=10)
                continue
            if opcode == 10:
                continue
            if opcode in (1, 2):
                if self.fragment_opcode is not None:
                    raise CDPError("Unexpected nested WebSocket message")
                self.fragment_opcode = opcode
            elif opcode != 0 or self.fragment_opcode is None:
                raise CDPError("Unexpected WebSocket opcode")
            self.fragments.extend(data)
            if len(self.fragments) > self.MAX_MESSAGE:
                raise CDPError("WebSocket message is too large")
            if final:
                message = bytes(self.fragments)
                kind = self.fragment_opcode
                self.fragments.clear()
                self.fragment_opcode = None
                if kind != 1:
                    raise CDPError("Expected a CDP text message")
                try:
                    return message.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise CDPError("CueMol sent invalid UTF-8") from exc

    def _pop_frame(self) -> tuple[int, bool, bytes] | None:
        data = self.buffer
        if len(data) < 2:
            return None
        first, second = data[:2]
        if first & 0x70 or second & 0x80:
            raise CDPError("Unexpected WebSocket extensions or masked server frame")
        length = second & 0x7F
        offset = 2
        if length in (126, 127):
            width = 2 if length == 126 else 8
            if len(data) < offset + width:
                return None
            length = int.from_bytes(data[offset : offset + width], "big")
            offset += width
        if length > self.MAX_MESSAGE:
            raise CDPError("WebSocket frame is too large")
        if first & 0x0F >= 8 and (not first & 0x80 or length > 125):
            raise CDPError("Invalid WebSocket control frame")
        if len(data) < offset + length:
            return None
        payload = bytes(data[offset : offset + length])
        del data[: offset + length]
        return first & 0x0F, bool(first & 0x80), payload

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()


class CDP:
    """Single-request CDP client with flattened target sessions."""

    def __init__(self, endpoint: str, timeout: float = 10.0):
        _local_url(endpoint, "http")
        self.timeout = timeout
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirects())
        with opener.open(endpoint.rstrip("/") + "/json/version", timeout=timeout) as res:
            version = json.load(res)
        self.ws = WebSocket(version["webSocketDebuggerUrl"], timeout)
        self.events: deque[dict] = deque(maxlen=1024)
        self.sequence = 0

    def request(self, method: str, params: dict | None = None, session: str | None = None) -> dict:
        self.sequence += 1
        request = {"id": self.sequence, "method": method, "params": params or {}}
        if session:
            request["sessionId"] = session
        self.ws.send(json.dumps(request, allow_nan=False).encode())
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                response = json.loads(self.ws.receive_text(max(0.001, deadline - time.monotonic())))
            except TimeoutError as exc:
                raise TimeoutError(f"{method} timed out") from exc
            if response.get("id") == self.sequence:
                if "error" in response:
                    raise CDPError(f"{method}: {response['error']}")
                return response.get("result", {})
            self.events.append(response)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{method} timed out")

    def evaluate(self, expression: str, session: str) -> object:
        result = self.request(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
            session,
        )
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            raise CDPError(details.get("exception", {}).get("description", str(details)))
        return result.get("result", {}).get("value")

    def attach(self, target_id: str) -> str:
        return self.request("Target.attachToTarget", {"targetId": target_id, "flatten": True})[
            "sessionId"
        ]

    def close(self) -> None:
        self.ws.close()
