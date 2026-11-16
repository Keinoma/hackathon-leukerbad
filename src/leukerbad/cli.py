"""Image and webcam inference for a local YOLO checkpoint."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from leukerbad.detection.camera import LatestFrames
from leukerbad.detection.yolo import YOLODetector


def save_image(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Could not save {path}")
    print(f"Saved {path.resolve()}", flush=True)


def next_snapshot(directory: Path) -> Path:
    """Pick a filename without overwriting an earlier session's snapshots."""
    index = 1
    while (directory / f"snapshot_{index:04d}.png").exists():
        index += 1
    return directory / f"snapshot_{index:04d}.png"


def report(detections) -> None:
    if not detections:
        print("No detections above the confidence threshold.", flush=True)
    for detection in detections:
        geometry = detection.geometry
        left, top, right, bottom = detection.bounding_box
        measurements = (
            f" x={geometry.x_px:.1f}px y={geometry.y_px:.1f}px"
            f" theta={geometry.theta_deg:.1f}deg width={geometry.width_px:.1f}px"
            if geometry is not None else (
                f" x={(left + right) / 2:.1f}px y={(top + bottom) / 2:.1f}px"
                " center=box mask=unavailable"
            )
        )
        print(
            f"{detection.label}: confidence={detection.confidence:.3f} "
            f"box={detection.bounding_box}{measurements}", flush=True,
        )


def run_camera(args, detector) -> None:
    from leukerbad.lens import check_workspace_lens, load_lens
    from leukerbad.workspace import (
        PositionProbe,
        draw_workspace,
        export_target,
        measure_targets,
        orient_frame,
    )

    json_out = getattr(args, "json_out", None)
    calibration_path = getattr(args, "calibration", None)
    calibration = (json.loads(calibration_path.read_text(encoding="utf-8"))
                   if calibration_path is not None and calibration_path.is_file() else None)
    lens = load_lens(getattr(args, "lens_file", None))
    check_workspace_lens(calibration, lens)
    rotation = (calibration.get("rotation_deg", 0)
                if calibration is not None else getattr(args, "rotate", 0))
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.camera, backend)
    frames = None
    title = "Vape detector | j: JSON target | s: snapshot | q: quit"
    probe = PositionProbe(calibration) if calibration is not None else None
    if probe is not None:
        title += " | click: position"
    try:
        if not camera.isOpened():
            raise RuntimeError(
                f"Cannot open camera {args.camera}. Allow camera access in macOS "
                "System Settings > Privacy & Security > Camera, or try --camera 1."
            )
        # Keep the camera's negotiated format. Changing dimensions after opening
        # can stop AVFoundation frame delivery on external USB webcams.
        frames = LatestFrames(camera)
        if probe is not None:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(title, probe.on_mouse)
        print(f"Camera {args.camera} open. Press s to save a snapshot; q to quit.", flush=True)
        last_report = 0.0
        failed_since = None
        while True:
            ok, frame = frames.read()
            if not ok:
                now = time.monotonic()
                if failed_since is None:
                    failed_since = now
                    print("Waiting for camera frames...", flush=True)
                if now - failed_since >= 3.0:
                    raise RuntimeError(
                        f"Camera {args.camera} stopped delivering frames for 3 seconds. "
                        "Check the connection, close other camera apps, or try another index."
                    )
                if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                    break
                continue
            failed_since = None
            if lens is not None:
                frame = lens.apply(frame)
            frame = orient_frame(frame, rotation)
            start = time.monotonic()
            targets = []
            if calibration is None:
                annotated, detections = detector.annotate(frame)
            else:
                annotated, detections, polygons = detector.analyse(frame, draw_geometry=False)
                targets = measure_targets(calibration, (frame.shape[1], frame.shape[0]),
                                          detections, polygons)
                annotated = draw_workspace(annotated, calibration, targets)
            fps = 1.0 / max(time.monotonic() - start, 1e-6)
            cv2.putText(
                annotated, f"{fps:.1f} FPS | j: JSON | s: save | q: quit"
                + (" | grid: 10 mm" if calibration is not None else ""),
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            )
            if lens is not None:
                cv2.putText(annotated, "Lens correction ON", (12, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if probe is not None:
                probe.draw(annotated)
            cv2.imshow(title, annotated)
            if time.monotonic() - last_report >= 2.0:
                if calibration is None:
                    report(detections)
                else:
                    for target in targets:
                        print(f"Paper measurement: {target}", flush=True)
                last_report = time.monotonic()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("s"):
                save_image(next_snapshot(args.snapshots), annotated)
                report(detections)
            if key == ord("j"):
                if json_out is None:
                    print("Enable JSON with --json-out target.json --calibration workspace.json")
                else:
                    payload = export_target(json_out, calibration,
                                            (frame.shape[1], frame.shape[0]), detections, polygons)
                    print(f"Wrote {json_out}: {payload}", flush=True)
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if frames is not None:
            frames.close()
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Image to sanity-check")
    source.add_argument("--camera", type=int, help="Webcam index, usually 0")
    parser.add_argument("--weights", type=Path, default=Path("models/vape_seg.pt"))
    parser.add_argument("--segment-weights", type=Path,
                        help="Optional segmentation model to add masks to local detections")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", type=Path, help="Annotated image output (image mode)")
    parser.add_argument("--snapshots", type=Path, default=Path("snapshots"))
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or CUDA index")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0,
                        help="Clockwise image rotation; calibration saves its chosen rotation")
    parser.add_argument("--calibrate", type=float, nargs=2, metavar=("WIDTH_MM", "HEIGHT_MM"),
                        help="Click four corners and the origin mark; t rotates the calibration view")
    parser.add_argument("--calibration", type=Path, default=Path("workspace.json"))
    parser.add_argument("--json-out", type=Path, help="Press j in the camera window to write one target")
    parser.add_argument("--lens-file", type=Path, default=Path("lens.json"),
                        help="Automatically apply measured lens correction when this file exists")
    parser.add_argument("--lens-calibrate", action="store_true",
                        help="Capture a printed 10x7-square checkerboard (9x6 internal corners)")
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        parser.error("--conf must be greater than 0 and at most 1")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if not args.calibrate and not args.lens_calibrate and not args.weights.is_file():
        parser.error(f"Weights not found: {args.weights}")
    if args.segment_weights is not None and not args.segment_weights.is_file():
        parser.error(f"Segmentation weights not found: {args.segment_weights}")
    if args.camera is not None and args.save:
        parser.error("--save is for image mode; use s to save camera snapshots")
    if args.image and not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if args.json_out and (args.image or args.calibrate):
        parser.error("--json-out is for calibrated live camera mode")
    if args.json_out and not args.calibration.is_file():
        parser.error("Calibrate first: --calibrate WIDTH_MM HEIGHT_MM")
    if args.lens_calibrate and (args.image or args.calibrate or args.json_out):
        parser.error("Lens calibration uses --camera and cannot run with other calibration/export modes")
    try:
        from leukerbad.lens import load_lens

        if args.lens_calibrate:
            from leukerbad.lens import calibrate_lens

            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            calibrate_lens(args.camera, args.lens_file, backend)
            return
        lens = load_lens(args.lens_file) if args.calibrate or args.image else None
        if args.calibrate:
            from leukerbad.workspace import calibrate

            if not all(0 < value < float("inf") for value in args.calibrate):
                parser.error("Calibration dimensions must be positive millimetres")
            if args.image:
                frame = cv2.imread(str(args.image))
                if frame is None:
                    raise ValueError("Could not decode calibration image")
            else:
                backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
                camera = cv2.VideoCapture(args.camera, backend)
                try:
                    deadline = time.monotonic() + 15
                    while True:
                        ok, frame = camera.read()
                        if ok:
                            break
                        if time.monotonic() > deadline:
                            raise RuntimeError("No camera frames for calibration")
                        cv2.waitKey(30)
                finally:
                    camera.release()
            if lens is not None:
                frame = lens.apply(frame)
            calibrate(frame, *args.calibrate, args.calibration, rotation=args.rotate,
                      lens_id=lens.id if lens else None)
            return
        detector = YOLODetector(
            args.weights, args.conf, args.device, args.imgsz, args.segment_weights,
        )
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise ValueError(f"Could not decode image: {args.image}")
            if lens is not None:
                frame = lens.apply(frame)
            if args.rotate:
                from leukerbad.workspace import orient_frame

                frame = orient_frame(frame, args.rotate)
            annotated, detections = detector.annotate(frame)
            report(detections)
            save_image(args.save or Path("annotated.png"), annotated)
        else:
            run_camera(args, detector)
    except KeyboardInterrupt:
        print("Stopped.")
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
