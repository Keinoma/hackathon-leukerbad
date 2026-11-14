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
