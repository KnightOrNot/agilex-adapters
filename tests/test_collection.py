"""Cross-interpreter worker and real LeRobot RGB-D persistence regression."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from agilex_data_collection.config import AppConfig  # noqa: E402
from agilex_data_collection.session import Session  # noqa: E402


def wait(session, phase, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = session.status()
        assert status["phase"] != "error", status
        if status["phase"] == phase:
            return status
        time.sleep(0.02)
    pytest.fail(f"timed out waiting for {phase}: {session.status()}")


def test_real_worker_save_discard_and_readback(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    (tmp_path / "src").symlink_to(PROJECT / "src", target_is_directory=True)
    (tmp_path / ".venv").symlink_to(PROJECT / ".venv", target_is_directory=True)
    config_path = tmp_path / "config.toml"
    config_path.write_text("[collection]\ndataset_fps = 10\n")
    config = AppConfig.load(config_path)
    session = Session(tmp_path, config_path, config, simulate=True)
    try:
        session.start(confirmed=False, task="synthetic regression")
        wait(session, "following")
        for command in ("save", "discard"):
            session.command("record")
            wait(session, "recording")
            deadline = time.monotonic() + 10
            while session.status()["frames"] < 4 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert session.status()["frames"] >= 4
            session.command(command)
            wait(session, "following")
        assert session.status()["saved_episodes"] == 1
    finally:
        session.stop(wait=True)
    status = wait(session, "completed")
    dataset = LeRobotDataset(
        f"local/{Path(status['dataset']).name}",
        root=status["dataset"],
        depth_output_unit="m",
    )
    assert dataset.num_episodes == 1 and len(dataset) >= 4
    assert tuple(dataset[0]["action"].shape) == (7,)
    depth = dataset[0]["observation.images.front_depth"]
    assert tuple(depth.shape) == (1, 48, 64)
    assert float(depth.mean()) == pytest.approx(1.0)
    raw = Path(status["raw"])
    assert [
        json.loads(p.read_text())["result"]
        for p in sorted(raw.glob("take_*/result.json"))
    ] == ["save", "discard"]
    native = np.load(next(raw.glob("take_*/depth_*.npy")), allow_pickle=False)
    assert native.dtype == np.uint16 and np.all(native == 1000)


def test_runtime_routes_and_all_preview_modes(tmp_path):
    from agilex_data_collection.runtime import create_app
    from fastapi.testclient import TestClient

    (tmp_path / "src").symlink_to(PROJECT / "src", target_is_directory=True)
    (tmp_path / ".venv").symlink_to(PROJECT / ".venv", target_is_directory=True)
    config_path = tmp_path / "config.toml"
    config_path.write_text("[collection]\ndataset_fps = 10\n")
    with TestClient(create_app(tmp_path, config_path, simulate=True)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/dataset-browser").status_code == 200
        assert client.get("/edit-dataset").status_code == 200
        assert client.get("/collection-assets/pointcloud.js").status_code == 200
        assert client.get("/jobs").status_code == 404
        assert (
            client.post("/api/start", json={"task": "preview test"}).status_code == 200
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = client.get("/api/status").json()
            assert status["phase"] != "error", status
            if status["phase"] == "following":
                break
            time.sleep(0.02)
        assert status["phase"] == "following"
        for mode in ("rgb", "depth", "rgbd", "cloud"):
            with client.websocket_connect(f"/api/preview/ws?mode={mode}") as socket:
                info = socket.receive_json()
                assert info["type"] == "frame" and info["mode"] == mode
                payload = socket.receive_bytes()
                assert payload
                if mode == "cloud":
                    assert len(payload) % 12 == 0
        assert client.get("/api/preview").status_code == 200
        assert client.post("/api/control/stop").status_code == 200
