# Leukerbad Object Detection Robotics

Base project for combining computer-vision object detection with a simple robotic arm.

## Project layout

```text
src/leukerbad/
  detection/    Detection interfaces and model adapters
  robotics/     Robotic-arm interfaces and motion commands
  cli.py        Minimal command-line entry point
tests/          Tests
```

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --extra dev
uv run leukerbad-detect
```

Optional integrations can be installed with:

```bash
uv sync --extra ml --extra robotics --extra dev
```

The current implementation provides interfaces and safe placeholders. Camera, model, and
robot-specific wiring can be added without changing the project layout.

## Development

```bash
uv run ruff check .
uv run pytest
```

## License

MIT

## SO-101 follower: point-to-point motion

The first Cartesian motion experiment is in `puff_pipeline/move_to_xy.py`. It uses the
`mon_bras` follower calibration and table-frame mapping from the `kash` branch. It does
not use the leader, camera, or object detector. The follower must be connected to an
environment where LeRobot is installed and the calibration belongs to that physical arm.

Coordinates are millimetres from the existing table origin: `+X` is right and `+Y` is
away from the robot base. The command defaults to 50 mm above the table reference point,
so the first movement does not intentionally lower the gripper to the surface.

First check a target without connecting to the robot:

```bash
python puff_pipeline/move_to_xy.py 0 0 --dry-run
```

Then, with the follower serial port substituted, explicitly enable hardware and confirm
the move at the keyboard:

```bash
python puff_pipeline/move_to_xy.py 0 0 --port COM5 --hardware
```

After moving, the script asks the operator to confirm whether the gripper TCP is visibly above the selected reference point. One successful point only checks that location; repeat at several points to validate the whole work area. The script checks the saved calibration, IK reachability, and servo operating modes,
interpolates the motion at a limited speed, and keeps motor torque enabled at the end.
Start with the arm in the validated rest pose, the work area clear, and a supervised,
small target. The allowed XY/Z region is enforced by the calibration module.
