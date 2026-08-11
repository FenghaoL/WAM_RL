"""Length-prefixed TCP message helpers used by deployment scripts.

Messages are serialized with Python pickle for simplicity and compatibility with
NumPy arrays. Only use this protocol on trusted local robot networks: unpickling
messages from untrusted peers is unsafe.
"""

from __future__ import annotations

import io
import pickle
import socket
import struct
from typing import Any


class _NumpyCompatibleUnpickler(pickle.Unpickler):
    """Load ndarray pickles across NumPy's ``core`` to ``_core`` rename."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def loads_compatible_pickle(payload: bytes) -> Any:
    """Deserialize trusted payloads, accepting NumPy 2 arrays on NumPy 1.26."""
    return _NumpyCompatibleUnpickler(io.BytesIO(payload)).load()


def recv_exact(conn: socket.socket, size: int) -> bytes:
    """Receive exactly ``size`` bytes or raise if the peer closes."""
    data = bytearray()
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            raise ConnectionError("remote closed connection")
        data.extend(packet)
    return bytes(data)


def recv_message(conn: socket.socket) -> tuple[Any, int]:
    """Receive one length-prefixed pickle message and return ``(message, bytes)``."""
    header = recv_exact(conn, 4)
    (size,) = struct.unpack("!I", header)
    payload = recv_exact(conn, size)
    return loads_compatible_pickle(payload), size


def send_message(conn: socket.socket, message: Any) -> int:
    """Send one length-prefixed pickle message and return payload byte count."""
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("!I", len(payload)) + payload)
    return len(payload)
