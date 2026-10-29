"""Real local IPC and the official LeRobot protocols, without hardware."""

import os
import threading

import numpy as np
import pytest
import zmq
from lerobot.robots.robot import Robot
from lerobot.teleoperators.teleoperator import Teleoperator

from agilex_adapters import Gello, GelloConfig, PiperX, PiperXConfig
from agilex_adapters.devices import JOINTS
from agilex_adapters.transport import BridgeClient


@pytest.fixture
def worker(tmp_path):
    endpoint = f"ipc://{tmp_path}/devices.sock"
    ready, stop = threading.Event(), threading.Event()
    context = zmq.Context()
    state = {"positions": np.array([0.1] * 6 + [0.5]), "commands": []}

    def serve():
        with context.socket(zmq.REP) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(endpoint)
            os.chmod(tmp_path / "devices.sock", 0o600)
            ready.set()
            while not stop.is_set():
                if not socket.poll(10):
                    continue
                request = socket.recv_pyobj()
                method = request["method"]
                if method == "status":
                    result = {"ready": True}
                elif method in {"action", "observation"}:
                    result = {"positions": state["positions"], "timestamp_ns": 123}
                elif method == "command":
                    result = request["args"]["positions"]
                    state["commands"].append(result)
                else:
                    socket.send_pyobj({"ok": False, "error": "worker failure"})
                    continue
                socket.send_pyobj({"ok": True, "result": result})

    thread = threading.Thread(target=serve)
    thread.start()
    assert ready.wait(2)
    try:
        yield endpoint, state
    finally:
        stop.set()
        thread.join(2)
        assert not thread.is_alive()
        context.term()


def test_protocols_preserve_joint_units_and_echo_applied_action(worker):
    endpoint, state = worker
    robot, leader = (
        PiperX(PiperXConfig(endpoint=endpoint)),
        Gello(GelloConfig(endpoint=endpoint)),
    )
    assert isinstance(robot, Robot) and isinstance(leader, Teleoperator)
    try:
        robot.connect()
        leader.connect()
        action = leader.get_action()
        assert list(action) == list(JOINTS)
        assert action == robot.get_observation()
        assert robot.send_action(action) == action
        assert leader.last_timestamp_ns == 123
        np.testing.assert_array_equal(state["commands"][0], state["positions"])
        with pytest.raises(NotImplementedError):
            leader.calibrate()
    finally:
        leader.disconnect()
        robot.disconnect()
    assert not robot.is_connected and not leader.is_connected


def test_official_lerobot_factories_resolve_registered_configs():
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    robot = make_robot_from_config(PiperXConfig())
    leader = make_teleoperator_from_config(GelloConfig())
    assert isinstance(robot, PiperX) and isinstance(leader, Gello)
    assert not robot.is_connected and not leader.is_connected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 2.0])
def test_invalid_gripper_never_reaches_worker(worker, bad):
    endpoint, state = worker
    robot = PiperX(PiperXConfig(endpoint=endpoint))
    try:
        robot.connect()
        action = dict.fromkeys(JOINTS, 0.0)
        action[JOINTS[-1]] = bad
        with pytest.raises(ValueError):
            robot.send_action(action)
        assert not state["commands"]
    finally:
        robot.disconnect()


def test_timeout_replaces_req_socket_and_worker_errors_propagate(tmp_path, worker):
    client = BridgeClient(f"ipc://{tmp_path}/missing.sock", timeout_s=0.01)
    try:
        for _ in range(2):
            with pytest.raises(TimeoutError):
                client.call("status")
    finally:
        client.close()
    client = BridgeClient(worker[0])
    try:
        with pytest.raises(RuntimeError, match="worker failure"):
            client.call("unsupported")
        assert client.call("status")["ready"]
    finally:
        client.close()


@pytest.mark.parametrize(
    "endpoint", ["tcp://127.0.0.1:5555", "ipc://relative", "ipc:///@abstract"]
)
def test_rejects_nonprivate_transports(endpoint):
    with pytest.raises(ValueError):
        BridgeClient(endpoint)


def test_rejects_world_readable_socket(worker):
    endpoint, _ = worker
    os.chmod(endpoint.removeprefix("ipc://"), 0o666)
    client = BridgeClient(endpoint)
    try:
        with pytest.raises(PermissionError):
            client.call("status")
    finally:
        client.close()
