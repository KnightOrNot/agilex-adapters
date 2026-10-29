# agilex-adapters

LeRobot `Robot` / `Teleoperator` adapters for PiPER-X and GELLO. Hardware access,
calibration, motion limits and lifecycle remain in the application-owned Python
3.11 worker. These Python 3.12 adapters use only an owner-private filesystem IPC
socket; they do not open CAN, serial or cameras.

`PiperXConfig(endpoint=...)` and `GelloConfig(endpoint=...)` select the worker.
Actions expose six `joint_N.pos` radians and `gripper.pos` in `[0, 1]`.
Unknown, nonfinite or incomplete actions are rejected. Calibration is explicit
in the controller TOML; adapters never automatically recalibrate hardware.

```bash
uv sync --frozen --extra app --python 3.12
uv run --frozen --extra app pytest
uv run --frozen ruff check src tests
```

The `app` extra provides the controller's browser runtime with LeRobot 0.6.0,
CPU PyTorch and LeLab pinned to `9b5afaee33baa1d773e499937564f3831d25de22`.
Installing LeLab builds its frontend using the isolated Node build dependency.
