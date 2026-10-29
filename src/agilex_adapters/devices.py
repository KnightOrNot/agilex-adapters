"""LeRobot protocols over IPC; hardware lifecycle belongs to the worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator

from .transport import BridgeClient

JOINTS = tuple(f"joint_{i}.pos" for i in range(1, 7)) + ("gripper.pos",)


@RobotConfig.register_subclass("agilex_piperx")
@dataclass(kw_only=True)
class PiperXConfig(RobotConfig):
    endpoint: str = "ipc:///tmp/agilex/devices.sock"
    timeout_s: float = 1.0


@TeleoperatorConfig.register_subclass("agilex_gello")
@dataclass(kw_only=True)
class GelloConfig(TeleoperatorConfig):
    endpoint: str = "ipc:///tmp/agilex/devices.sock"
    timeout_s: float = 1.0


def _positions(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError("expected seven finite joint values")
    if not 0 <= result[-1] <= 1:
        raise ValueError("gripper must be in [0, 1]")
    return result


class _Device:
    _client: BridgeClient | None = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    @property
    def is_calibrated(self) -> bool:
        return self.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("device is already connected")
        client = BridgeClient(self.config.endpoint, self.config.timeout_s)
        try:
            if client.call("status").get("ready") is not True:
                raise RuntimeError("worker is not ready")
        except BaseException:
            client.close()
            raise
        self._client = client

    def _call(self, method: str, **args: Any) -> Any:
        if self._client is None:
            raise RuntimeError("device is not connected")
        return self._client.call(method, **args)

    def disconnect(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            client.close()

    def calibrate(self) -> None:
        raise NotImplementedError("calibration is supplied by the worker's config")

    def configure(self) -> None:
        if not self.is_connected:
            raise RuntimeError("device is not connected")


class PiperX(_Device, Robot):
    config_class = PiperXConfig
    name = "agilex_piperx"

    def __init__(self, config: PiperXConfig) -> None:
        super().__init__(config)
        self.config = config

    @property
    def observation_features(self) -> dict:
        return dict.fromkeys(JOINTS, float)

    @property
    def action_features(self) -> dict:
        return dict.fromkeys(JOINTS, float)

    def get_observation(self) -> dict:
        state = self._call("observation")
        return dict(zip(JOINTS, _positions(state["positions"]), strict=True))

    def send_action(self, action: dict) -> dict:
        if set(action) != set(JOINTS):
            raise ValueError("action must contain exactly the seven joint features")
        positions = _positions([action[name] for name in JOINTS])
        sent = _positions(self._call("command", positions=positions))
        return dict(zip(JOINTS, sent, strict=True))


class Gello(_Device, Teleoperator):
    config_class = GelloConfig
    name = "agilex_gello"

    def __init__(self, config: GelloConfig) -> None:
        super().__init__(config)
        self.config = config
        self.last_timestamp_ns: int | None = None

    @property
    def action_features(self) -> dict:
        return dict.fromkeys(JOINTS, float)

    @property
    def feedback_features(self) -> dict:
        return {}

    def get_action(self) -> dict:
        state = self._call("action")
        self.last_timestamp_ns = state["timestamp_ns"]
        return dict(zip(JOINTS, _positions(state["positions"]), strict=True))

    def send_feedback(self, feedback: dict) -> None:
        if feedback:
            raise NotImplementedError("force feedback is not supported")
