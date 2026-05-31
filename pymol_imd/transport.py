"""TCP socket framing for IMDv3 (stdlib ``socket`` only).

Turns a connected socket into "read / write exactly one IMDv3 packet" and
hides TCP's partial-recv reality behind :func:`recv_exact`. Imports
:mod:`pymol_imd.protocol` (for header parsing) and ``constants`` (recv buffer
size) — no ``openmm`` / ``pymol``, so it stays headless-testable.
"""

from __future__ import annotations

import contextlib
import socket

from . import protocol
from .constants import IMD_RECV_BUFSIZE


class ConnectionClosed(Exception):
    """Raised when the peer closes the socket before ``n`` bytes arrive."""


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly ``n`` bytes, looping over partial ``recv`` calls."""
    if n <= 0:
        return b""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), IMD_RECV_BUFSIZE))
        if not chunk:
            raise ConnectionClosed(f"peer closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)


class IMDConnection:
    """One connected socket + the negotiated body endianness.

    ``endian`` is ``None`` until the handshake is parsed (client) or chosen
    (server), then pinned for the lifetime of the connection and used for all
    body (un)packing.
    """

    def __init__(self, sock: socket.socket, endian: protocol.Endian | None = None) -> None:
        self.sock = sock
        self.endian = endian

    def send_bytes(self, data: bytes) -> None:
        self.sock.sendall(data)

    def recv_header(self) -> tuple[protocol.IMDHeaderType, int]:
        return protocol.unpack_header(recv_exact(self.sock, protocol.HEADER_SIZE))

    def recv_body(self, nbytes: int) -> bytes:
        return recv_exact(self.sock, nbytes)

    def settimeout(self, timeout: float | None) -> None:
        self.sock.settimeout(timeout)

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.close()


def _tune(sock: socket.socket) -> None:
    # Disable Nagle: IMD frames are small and latency-sensitive, so we want
    # each packet on the wire immediately rather than coalesced with the next.
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def bind_listen(host: str, port: int = 0, *, backlog: int = 1) -> tuple[socket.socket, int]:
    """Bind + ``listen``. Returns ``(server_socket, actual_port)``.

    ``port=0`` asks the OS for an ephemeral port (tests use this to avoid
    collisions); the chosen port is returned so the caller can advertise it.
    Split from :func:`accept_one` so a server can print a READY line carrying
    the real port *before* it blocks waiting for a client.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(backlog)
    return srv, srv.getsockname()[1]


def accept_one(srv_sock: socket.socket, *, timeout: float | None = None) -> IMDConnection:
    """Accept a single client on an already-bound/listening socket."""
    if timeout is not None:
        srv_sock.settimeout(timeout)
    client_sock, _addr = srv_sock.accept()
    _tune(client_sock)
    return IMDConnection(client_sock)


def connect(host: str, port: int, *, timeout: float) -> IMDConnection:
    """Connect to a listening IMDv3 server. Returns an open :class:`IMDConnection`."""
    sock = socket.create_connection((host, port), timeout=timeout)
    _tune(sock)
    return IMDConnection(sock)
