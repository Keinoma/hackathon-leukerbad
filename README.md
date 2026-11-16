# Leukerbad Object Detection Robotics

Base project for combining computer-vision object detection with a simple robotic arm.

## Project layout

```text
src/leukerbad/
  detection/    Detection interfaces and model adapters
  robotics/     Robotic-arm interfaces and motion commands
  cli.py        Image/webcam detection, calibration, and manual JSON export
tests/          Tests
```

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --extra ml
uv run --extra ml leukerbad-detect --camera 0 --weights models/vape_seg.pt --device cpu
```

Optional integrations can be installed with:

```bash
uv sync --extra ml --extra robotics --extra dev
```

## Local vape detector

The image and camera CLI uses Ultralytics YOLO. Robotics remains a placeholder.
The demo checkpoint `models/vape_seg.pt` was extracted from
`stage1_pretrain/weights/best.pt` in the provided training archive. This checkpoint
and its training arguments are included in this branch; other weights, datasets,
local calibration files, and snapshots stay local. See [models/README.md](models/README.md).

From this repository root:

```bash
uv sync --extra ml
uv run --extra ml leukerbad-detect --image path/to/vape.jpg --weights models/vape_seg.pt --conf 0.25 --save annotated.png
uv run --extra ml leukerbad-detect --camera 0 --weights models/vape_seg.pt --conf 0.25
```

Provide your own test image in the image command. The CLI prints the checkpoint's
class names and each detection's class, confidence, and box.
The saved image shows masks and a yellow grasp center/width line. This grasp overlay
is a 2D minimum-rectangle heuristic, not a calibrated robot command.

On macOS, double-click **Start Webcam.command** to launch the camera from Terminal.
Allow camera access when prompted. If necessary, enable Terminal or Codex in
System Settings > Privacy & Security > Camera, then relaunch.
Use `--camera 1` to try a different device. The launcher also accepts an index:
`./"Start Webcam.command" 1`.

In the live window, press **s** to save numbered PNGs under `snapshots/`, and **q**
or Escape to quit. Existing snapshots are preserved. The display includes inference FPS.
Apple Silicon uses MPS when available; `--device cpu` selects CPU explicitly.
Use `--imgsz 416` if a smaller inference size is needed for speed.
The camera keeps its native capture format and reads continuously in a background
thread so inference does not interrupt capture or build up a queue of old frames.
Brief dropped frames are retried; a sustained outage produces a connection error.

### Puff measurements

For each valid segmentation mask, the image overlay and terminal report include
`x`, `y` (oriented rectangle center in pixels), `theta` (long-axis angle in degrees),
and `width` (the rectangle's short side in pixels). Image coordinates start at the
top left: x increases rightward and y downward. Theta increases clockwise from the
horizontal, in `[0, 180)`; 0 degrees is horizontal and 90 degrees is vertical.
The mask does not identify the mouthpiece end, so this is an undirected orientation.
The yellow line shows width; the cyan line shows the longitudinal axis used for theta.
Invalid or missing masks produce no geometric measurements.

These are projected image measurements. A fixed camera makes the image reference
stable; millimetres and robot-plane angles still require camera/plane calibration.
Restart the live detector after updating the code to see these measurements.

### Calibrated workspace and JSON

See [QUICKSTART.md](QUICKSTART.md) for lens correction, five-point paper calibration,
the 1 cm grid, and click-to-measure. The current AUKEY setup uses a rectangle with
385 mm horizontally and 280 mm vertically. Its centre mark is `(0,0)`, **+X is up**,
and **+Y is left**. Angles are positive anticlockwise from vertical, signed around
zero in `(-90,90]` degrees. These calibrated measurements differ from the raw pixel
measurements described above.

Press `j` to export one detected puff to `target.json`. This is a manual handoff:
wait for the robot to finish and clear the view before exporting again. Automatic
robot acknowledgements and robot motion control are not implemented.

On Windows, install dependencies with `uv sync --extra ml` and use the Python CLI
commands above or in QUICKSTART. `.command` launchers are macOS-only. Calibrate
the actual demo camera/PC before using coordinates. Windows hardware is not yet tested.

## Development

```bash
uv run ruff check .
uv run pytest
```

## License

MIT
