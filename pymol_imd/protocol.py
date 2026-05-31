"""IMDv3 wire-format pack/unpack — the byte-level mirror of the protocol.

Pure functions over ``bytes`` / numpy arrays. Imports only stdlib + numpy +
``constants.IMD_VERSION``; deliberately free of ``socket`` / ``openmm`` /
``pymol`` so it is unit-testable headless and shared by both the server
(:mod:`pymol_imd.server`) and the client (:mod:`pymol_imd.client`).

Endianness rules (IMDv3 spec):
  * Every 8-byte header is big-endian: ``int32 type`` + ``int32 dataslot``.
  * Body packets are in the *native* endianness of the sending engine; the
    receiver learns that endianness from the handshake version int (which is
    sent unswapped) and uses it for all subsequent body (un)packing.

Units are NOT handled here — this module packs whatever floats it is given.
The angstrom<->nm and kJ/mol/angstrom<->kJ/mol/nm conversions live at the
OpenMM boundary in server.py / client.py (see ``constants.NM_TO_ANGSTROM`` etc.).
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np

from .constants import IMD_VERSION


class IMDHeaderType(IntEnum):
    """IMDv3 header type enum (the int32 in bytes 0-3 of every header)."""

    DISCONNECT = 0
    ENERGIES = 1
    COORDS = 2
    GO = 3
    HANDSHAKE = 4
    KILL = 5
    MDCOMM = 6  # forces, receiver -> engine
    PAUSE = 7
    TRATE = 8
    IOERROR = 9
    SESSIONINFO = 10
    RESUME = 11
    TIME = 12
    BOX = 13
    VELOCITIES = 14
    FORCES = 15
    WAIT = 16


class Endian(Enum):
    """Byte order, expressed as a ``struct`` format prefix."""

    BIG = ">"
    LITTLE = "<"


NATIVE: Endian = Endian.LITTLE if sys.byteorder == "little" else Endian.BIG

# Header is always 8 bytes (int32 type + int32 dataslot), always big-endian.
HEADER_SIZE = 8
_HEADER_FMT = "!ii"

# Fixed body byte sizes for the constant-width packets.
_SESSIONINFO_BODY = 7  # 7 int8 flags
_ENERGIES_BODY = 40  # int32 step + 9 float32
_TIME_BODY = 20  # float64 dt + float64 time + int32 step
_BOX_BODY = 36  # 9 float32


class ProtocolError(Exception):
    """Raised on a malformed handshake, truncated body, or unknown header."""


def _f4(endian: Endian) -> np.dtype:
    return np.dtype("<f4" if endian is Endian.LITTLE else ">f4")


def _i4(endian: Endian) -> np.dtype:
    return np.dtype("<i4" if endian is Endian.LITTLE else ">i4")


# --------------------------------------------------------------------- header
def pack_header(htype: IMDHeaderType, dataslot: int = 0) -> bytes:
    """Pack an 8-byte IMDv3 header (always big-endian)."""
    return struct.pack(_HEADER_FMT, int(htype), int(dataslot))


def unpack_header(buf: bytes) -> tuple[IMDHeaderType, int]:
    """Unpack an 8-byte header into ``(type, dataslot)`` (big-endian)."""
    if len(buf) < HEADER_SIZE:
        raise ProtocolError(f"header needs {HEADER_SIZE} bytes, got {len(buf)}")
    raw_type, slot = struct.unpack(_HEADER_FMT, buf[:HEADER_SIZE])
    try:
        return IMDHeaderType(raw_type), slot
    except ValueError as exc:
        raise ProtocolError(f"unknown IMD header type {raw_type}") from exc


def body_size(htype: IMDHeaderType, dataslot: int) -> int:
    """Byte length of the body that follows a header of this ``(type, slot)``.

    Control / handshake packets carry no body and return 0, so a generic recv
    loop can do ``recv_body(body_size(t, slot))`` for every packet uniformly.
    """
    if htype is IMDHeaderType.SESSIONINFO:
        return _SESSIONINFO_BODY
    if htype is IMDHeaderType.ENERGIES:
        return _ENERGIES_BODY
    if htype is IMDHeaderType.TIME:
        return _TIME_BODY
    if htype is IMDHeaderType.BOX:
        return _BOX_BODY
    if htype in (IMDHeaderType.COORDS, IMDHeaderType.VELOCITIES, IMDHeaderType.FORCES):
        return dataslot * 12  # n_atoms * 3 float32
    if htype is IMDHeaderType.MDCOMM:
        return dataslot * (4 + 12)  # n * (int32 index + 3 float32 force)
    return 0


# ------------------------------------------------------------------ handshake
def pack_handshake() -> bytes:
    """Pack the handshake: type(4) big-endian + version int in NATIVE order.

    The version int is written unswapped (``"="``) so the receiver can detect
    this engine's endianness by reading bytes 4-7 both ways and matching
    ``IMD_VERSION``.
    """
    return struct.pack("!i", int(IMDHeaderType.HANDSHAKE)) + struct.pack("=i", IMD_VERSION)


def parse_handshake(buf: bytes) -> Endian:
    """Validate a handshake packet and return the *peer's* body endianness."""
    if len(buf) < HEADER_SIZE:
        raise ProtocolError(f"handshake needs {HEADER_SIZE} bytes, got {len(buf)}")
    raw_type = struct.unpack("!i", buf[:4])[0]
    if raw_type != int(IMDHeaderType.HANDSHAKE):
        raise ProtocolError(f"expected HANDSHAKE (4), got header type {raw_type}")
    # The version int was sent unswapped. Try both interpretations; whichever
    # reads IMD_VERSION reveals the sender's byte order.
    if struct.unpack(">i", buf[4:8])[0] == IMD_VERSION:
        return Endian.BIG
    if struct.unpack("<i", buf[4:8])[0] == IMD_VERSION:
        return Endian.LITTLE
    raise ProtocolError(f"handshake version mismatch: neither endian reads {IMD_VERSION}")


# --------------------------------------------------------------- session info
@dataclass(frozen=True)
class SessionInfo:
    """Which packets the engine sends each frame (IMDv3 session-info flags).

    The fixed per-frame order is Time -> Energies -> Box -> Coords ->
    Velocities -> Forces; only the enabled ones are sent, but always in that
    order and always every frame.
    """

    time: bool = True
    energies: bool = True
    box: bool = False
    coords: bool = True
    wrapped: bool = False
    velocities: bool = False
    forces: bool = False


def pack_sessioninfo(si: SessionInfo) -> bytes:
    header = pack_header(IMDHeaderType.SESSIONINFO, _SESSIONINFO_BODY)
    body = struct.pack(
        "7b",
        int(si.time),
        int(si.energies),
        int(si.box),
        int(si.coords),
        int(si.wrapped),
        int(si.velocities),
        int(si.forces),
    )
    return header + body


def parse_sessioninfo(body: bytes) -> SessionInfo:
    if len(body) < _SESSIONINFO_BODY:
        raise ProtocolError(f"session info needs {_SESSIONINFO_BODY} bytes, got {len(body)}")
    t, e, b, c, w, v, f = struct.unpack("7b", body[:_SESSIONINFO_BODY])
    return SessionInfo(
        time=bool(t),
        energies=bool(e),
        box=bool(b),
        coords=bool(c),
        wrapped=bool(w),
        velocities=bool(v),
        forces=bool(f),
    )


# ----------------------------------------------------------------------- time
def pack_time(dt_ps: float, time_ps: float, step: int, endian: Endian) -> bytes:
    header = pack_header(IMDHeaderType.TIME, 1)
    # NOTE: step packed as int32 -> 20-byte body, matching the IMDv3 spec's
    # stated body length. If interop with a GROMACS/LAMMPS build that uses an
    # int64 step is ever needed, widen this one format string ("ddq" = 24 B);
    # nothing else depends on the step width.
    body = struct.pack(f"{endian.value}ddi", float(dt_ps), float(time_ps), int(step))
    return header + body


def parse_time(body: bytes, endian: Endian) -> tuple[float, float, int]:
    if len(body) < _TIME_BODY:
        raise ProtocolError(f"time needs {_TIME_BODY} bytes, got {len(body)}")
    dt_ps, time_ps, step = struct.unpack(f"{endian.value}ddi", body[:_TIME_BODY])
    return dt_ps, time_ps, step


# ------------------------------------------------------------------- energies
def pack_energies(step: int, energies9, endian: Endian) -> bytes:
    """Pack the energies block. ``energies9`` order:
    (T, Etot, Epot, Evdw, Ecoul, Ebond, Eangle, Edihe, Eimpr)."""
    e9 = [float(x) for x in energies9]
    if len(e9) != 9:
        raise ProtocolError(f"energies needs 9 values, got {len(e9)}")
    header = pack_header(IMDHeaderType.ENERGIES, 1)
    body = struct.pack(f"{endian.value}i9f", int(step), *e9)
    return header + body


def parse_energies(body: bytes, endian: Endian) -> tuple[int, np.ndarray]:
    if len(body) < _ENERGIES_BODY:
        raise ProtocolError(f"energies needs {_ENERGIES_BODY} bytes, got {len(body)}")
    vals = struct.unpack(f"{endian.value}i9f", body[:_ENERGIES_BODY])
    return int(vals[0]), np.asarray(vals[1:], dtype=np.float64)


# ------------------------------------------------------------------------ box
def pack_box(vectors, endian: Endian) -> bytes:
    """Pack a 3x3 (or flat-9) triclinic box, row-major [aX,aY,aZ,bX,...]."""
    arr = np.asarray(vectors, dtype=np.float64).reshape(-1)
    if arr.size != 9:
        raise ProtocolError(f"box needs 9 values, got {arr.size}")
    header = pack_header(IMDHeaderType.BOX, 1)
    return header + arr.astype(_f4(endian)).tobytes()


def parse_box(body: bytes, endian: Endian) -> np.ndarray:
    if len(body) < _BOX_BODY:
        raise ProtocolError(f"box needs {_BOX_BODY} bytes, got {len(body)}")
    arr = np.frombuffer(body[:_BOX_BODY], dtype=_f4(endian)).astype(np.float64)
    return arr.reshape(3, 3)


# ----------------------------------------------- coords / velocities / forces
_ARRAY_TYPES = (IMDHeaderType.COORDS, IMDHeaderType.VELOCITIES, IMDHeaderType.FORCES)


def pack_array(htype: IMDHeaderType, arr_n3, endian: Endian) -> bytes:
    """Pack an (N,3) float array packet (coords / velocities / forces)."""
    if htype not in _ARRAY_TYPES:
        raise ProtocolError(f"{htype!r} is not an (N,3) array packet")
    arr = np.asarray(arr_n3, dtype=np.float64).reshape(-1, 3)
    header = pack_header(htype, arr.shape[0])
    return header + arr.reshape(-1).astype(_f4(endian)).tobytes()


def parse_array(body: bytes, n_atoms: int, endian: Endian) -> np.ndarray:
    expected = n_atoms * 12
    if len(body) < expected:
        raise ProtocolError(f"array needs {expected} bytes for {n_atoms} atoms, got {len(body)}")
    arr = np.frombuffer(body[:expected], dtype=_f4(endian)).astype(np.float64)
    return arr.reshape(n_atoms, 3)


# --------------------------------------------------------------------- mdcomm
def pack_mdcomm(indices, forces_n3, endian: Endian) -> bytes:
    """Pack an MDComm force packet (receiver -> engine).

    ``indices``: atom indices (int32 on the wire). ``forces_n3``: (N,3) forces
    already in the IMDv3 force unit (kJ/mol/angstrom) — unit conversion is the
    caller's job.
    """
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    frc = np.asarray(forces_n3, dtype=np.float64).reshape(-1, 3)
    if idx.shape[0] != frc.shape[0]:
        raise ProtocolError(f"mdcomm index/force mismatch: {idx.shape[0]} != {frc.shape[0]}")
    n = idx.shape[0]
    header = pack_header(IMDHeaderType.MDCOMM, n)
    idx_bytes = idx.astype(_i4(endian)).tobytes()
    frc_bytes = frc.reshape(-1).astype(_f4(endian)).tobytes()
    return header + idx_bytes + frc_bytes


def parse_mdcomm(body: bytes, n: int, endian: Endian) -> tuple[np.ndarray, np.ndarray]:
    expected = n * 4 + n * 12
    if len(body) < expected:
        raise ProtocolError(f"mdcomm needs {expected} bytes for n={n}, got {len(body)}")
    idx = np.frombuffer(body[: n * 4], dtype=_i4(endian)).astype(np.int64)
    frc = np.frombuffer(body[n * 4 : expected], dtype=_f4(endian)).astype(np.float64)
    return idx, frc.reshape(n, 3)


# -------------------------------------------------------------------- control
def pack_control(htype: IMDHeaderType, dataslot: int = 0) -> bytes:
    """Pack a header-only control packet (Go / Disconnect / Pause / Resume /
    Kill / TRate[slot=rate] / Wait[slot=blocking])."""
    return pack_header(htype, dataslot)
