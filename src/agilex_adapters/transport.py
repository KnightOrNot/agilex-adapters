"""Bounded, per-client requests to an owner-private local worker."""

from __future__ import annotations

import math
import stat
from pathlib import Path
from typing import Any

import zmq


class BridgeClient:
    def __init__(self, endpoint: str, timeout_s: float = 1.0) -> None:
        if not endpoint.startswith("ipc:///") or endpoint.startswith("ipc:///@"):
            raise ValueError("an absolute filesystem ipc:// endpoint is required")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.endpoint, self.timeout_s = endpoint, timeout_s
        self._context = zmq.Context()
        self._socket: zmq.Socket | None = None

    def _open(self) -> zmq.Socket:
        import os

        path = Path(self.endpoint.removeprefix("ipc://"))
        if path.exists():
            info = path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                raise PermissionError("IPC socket must belong to the current user")
            if info.st_mode & 0o077:
                raise PermissionError("IPC socket must be owner-private (0600)")
        if self._socket is None:
            self._socket = self._context.socket(zmq.REQ)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.connect(self.endpoint)
        return self._socket

    def call(self, method: str, **args: Any) -> Any:
        socket = self._open()
        try:
            socket.send_pyobj({"method": method, "args": args})
            if not socket.poll(max(1, round(self.timeout_s * 1000))):
                raise TimeoutError(f"worker request timed out: {method}")
            response = socket.recv_pyobj()
            if not isinstance(response, dict) or "ok" not in response:
                raise RuntimeError("invalid worker response")
            if response["ok"] is not True:
                raise RuntimeError(response.get("error", "worker request failed"))
            return response["result"]
        except BaseException:
            # A timed-out REQ must be replaced before sending another request.
            socket.close()
            self._socket = None
            raise

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._context.term()
