# Leukerbad Object Detection Robotics

Two-stage disposable vape & e-waste detection and analytical 2D grasp pose estimation (`YOLOv11-seg` + Rotating Calipers OBB + optional `Gemini 2.5 Flash` lookalike verifier) built for Dual SO-101 robotic arms at the Leukerbad Robotics Hackathon.

## Overview

Discarded lithium-ion disposable vapes and e-cigarettes present severe fire hazards during automated waste handling. This project implements a real-time, high-precision detection and robotic grasp planning system:

1. **Stage 1 (Local Real-Time Segmentation)**: Ultralytics `YOLOv11-seg` detects and segments irregular vape bodies and pods against cluttered conveyor and tabletop backgrounds.
2. **Analytical Grasp Pose Estimation**: Rotating Calipers minimum-area bounding box calculation extracts object orientation, major/minor axes, and an antipodal grasp point (`GraspCandidate`) with optimal gripper approach angle and opening width.
3. **Stage 2 (Multimodal VLM Verifier)**: An optional crop-level `Gemini 2.5 Flash` verifier inspects fine structural cues (mouthpieces, airflow holes, seams, pocket clips) to reliably reject benign tabletop lookalikes (AirPods cases, pens, highlighters, lipsticks, USB drives, lighters).
4. **Physical Trial Logging & Robotics Protocol**: Complete trial logging for empirical grasp success evaluation (Dataset B) and protocol interfaces (`Pose`, `RobotArm`) for Dual SO-101 arm motion execution.

## Project Layout

```text
src/leukerbad/
  detection/
    base.py             Detection & GraspCandidate interfaces, Detector protocol
    geometry.py         Rotating Calipers OBB, axial angle normalization, 2D grasp pose math
    yolo_seg.py         YoloSegDetector wrapping Ultralytics YOLOv11-seg inference
    vlm_verifier.py     VlmVerifiedDetector using Gemini 2.5 Flash for lookalike filtering
    visualize.py        OpenCV overlay renderer (masks, OBB, major axis, antipodal jaw markers)
  dataset/
    capture.py          Dataset A interactive webcam capture CLI with split tagging
    sam_masker.py       SAM 2 batch bbox-to-polygon upgrader and 1-click interactive annotator
    validate.py         Split hygiene, dHash duplicate detection, and held-out tag validator
  evaluation/
    grasp_logger.py     Dataset B physical grasp trial logger (raw/overlay PNGs, JSONL, CSV)
  robotics/
    base.py             Pose dataclass and RobotArm control protocol
  cli.py                Command-line entry point (leukerbad-detect)
notebooks/
  train_vape_seg.ipynb  2-stage Google Colab yolo11s-seg training notebook
docs/
  project_overview.pdf  4-page technical overview & architecture specification
tests/                  Unit and integration test suite
```

## Setup (`uv`)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python >= 3.11), then sync dependencies:

- **Minimal dev / test install**:
  ```bash
  uv sync --extra dev
  ```
- **Full ML + VLM + Robotics + Dev install**:
  ```bash
  uv sync --extra ml --extra vlm --extra robotics --extra dev
  ```

### API Keys
If using the Gemini 2.5 Flash lookalike verifier (`--verifier gemini`), set your API key:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
# or export GOOGLE_API_KEY="your-google-api-key"
```

## Detection & Grasp Visualization CLI (`leukerbad-detect`)

Run detection and grasp visualization across synthetic scenes, static images, or live webcam feeds.

### Examples

```bash
# Offline synthetic conveyor demo with rotated vapes (no camera or GPU required)
uv run leukerbad-detect --synthetic-demo

# Synthetic demo with real-world millimeter scale and saved output
uv run leukerbad-detect --synthetic-demo --mm-per-px 0.5 --save outputs/synthetic_grasp.png

# Detect on a static tabletop image with custom weights and confidence threshold
uv run leukerbad-detect --image path/to/tabletop.jpg --weights yolo11s-seg.pt --conf 0.35 --save annotated.png

# Live webcam feed with Gemini VLM verifier and vape-only filtering
uv run leukerbad-detect --camera 0 --verifier gemini --only-vapes
```

### CLI Flags

| Flag | Type / Default | Description |
|---|---|---|
| `--synthetic-demo` | flag | Run offline on a generated synthetic tabletop scene with rotated vapes and distractors. |
| `--image <path>` | Path | Path to an input image file. |
| `--camera <index>` | int | OpenCV webcam device index (e.g. `0`). Interactive keys: `q` quit, `s` save numbered snapshot, `v` force VLM verify. |
| `--save <path>` | Path | Save the annotated visualization frame to this path. |
| `--weights <path>` | str (`yolo11s-seg.pt`) | Ultralytics segmentation model checkpoint path or name. |
| `--conf <float>` | float (`0.25`) | Minimum confidence threshold for Stage-1 detections. |
| `--labels <csv>` | str | Comma-separated whitelist of allowed class labels (e.g. `vape,disposable_vape`). |
| `--mm-per-px <float>` | float | Tabletop scale (mm per pixel) to compute physical SO-101 jaw opening in mm. |
| `--verifier {none,gemini}` | choice (`none`) | Stage-2 multimodal verifier backend (`none` or `gemini`). |
| `--verify-min <float>` | float (`0.20`) | Lower confidence bound for sending Stage-1 candidates to Gemini VLM. |
| `--verify-max <float>` | float (`0.85`) | Upper confidence bound for VLM verification (candidates above this threshold are accepted directly). |
| `--verify-every <int>` | int (`30`) | In camera mode, run periodic VLM verification every N frames. |
| `--only-vapes` | flag | Filter out non-vape detections and distractors. |

## Dataset A & Training Workflow

### 1. Overhead Webcam Capture
Capture overhead tabletop images with deterministic split-and-tag naming:
```bash
uv run python -m leukerbad.dataset.capture --camera 0 --split train --tag AD_vapes
```
*Controls: `[SPACE]` save frame and metadata, `[1/2/3]` switch split (`train`/`val`/`test`), `[q]` quit.*

### 2. SAM 2 Batch Bounding-Box to Polygon Upgrade
Convert rectangular YOLO bounding boxes into precise segmentation mask polygons using Segment Anything 2:
```bash
uv run python -m leukerbad.dataset.sam_masker --images datasets/dataset_a_overhead/images/train --labels datasets/dataset_a_overhead/labels/train
```

### 3. Interactive 1-Click SAM 2 Annotator
Annotate new frames by clicking objects to prompt SAM 2 and assigning class labels with hotkeys:
```bash
# 1-class setup (vapes only)
uv run python -m leukerbad.dataset.sam_masker --interactive --images datasets/dataset_a_overhead/images/val --labels datasets/dataset_a_overhead/labels/val --num-classes 1

# 2-class setup (class 0: vape, class 1: non_vape lookalikes)
uv run python -m leukerbad.dataset.sam_masker --interactive --images datasets/dataset_a_overhead/images/val --labels datasets/dataset_a_overhead/labels/val --num-classes 2
```
*Controls: Click object -> `0` (vape), `1` (non_vape), `b` (empty background), `u` (undo last label), `n` / `p` (next / previous image), `q` (quit).*

### 4. Dataset Hygiene Validation
Validate split isolation, dHash image deduplication, and held-out test device leakage:
```bash
uv run python -m leukerbad.dataset.validate --dataset datasets/dataset_a_overhead
```

### 5. Two-Stage YOLOv11-seg Training
Train on Google Colab (free T4 GPU) or locally using `notebooks/train_vape_seg.ipynb`:
- **Stage 1**: Train base `yolo11s-seg` on overhead vape dataset.
- **Stage 2**: Hard-negative mining and fine-tuning on lookalike distractors.

### 6. Dataset B Physical Grasp Trial Logging
Log real-world tabletop grasp attempts and calculate empirical success metrics:
```python
from pathlib import Path
from leukerbad.evaluation.grasp_logger import GraspTrialLogger, FailureReason

logger = GraspTrialLogger(output_dir=Path("datasets/dataset_b_physical_trials"))
record = logger.log_trial(
    frame=frame,
    detection=detection,
    object_design_id="vape_A",
    succeeded=True,
    failure_reason=FailureReason.NONE,
    gripper_opening_mm=32.0,
    approx_width_mm=28.5,
)
print(logger.summarize())
```

## Development & Testing

```bash
uv run ruff check .
uv run pytest
```

## Tracked Hardware Integration Milestone

The robotics control interfaces under `src/leukerbad/robotics/` define the protocol contract for physical hardware integration:
- **`Pose` & `RobotArm` protocol**: Standard Cartesian pose representation `Pose(x, y, z, rotation)` and gripper actions (`move_to`, `open_gripper`, `close_gripper`).
- **Roadmap Milestone (Item #10 - Full Dual SO-101 Hardware Loop)**:
  - **Planar Homography Calibration**: 4-point planar homography calibration mapping pixel coordinates (`GraspCandidate.center_px`), rotation angle (`gripper_angle_deg`), and resolution (`mm_per_px`) into arm base coordinates `Pose(x, y, z, rotation)`.
  - **Serial Hardware Driver**: Serial driver over PySerial communicating with Feetech STS3215 bus servos on the SO-101 robotic arms, implementing the `RobotArm` protocol.

## License

MIT
