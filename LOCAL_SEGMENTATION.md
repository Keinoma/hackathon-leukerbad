# Local detection with supplementary segmentation

The locally trained YOLO26s checkpoint still supplies the class, confidence, and
bounding box. `models/vape_seg.pt`, extracted from the supplied stage1_pretrain
archive, supplies masks. This does not turn the local detection checkpoint into
a segmentation checkpoint.

Run `./"Start iPhone Local Model.command" 0` for the iPhone currently at index 0.
The launcher accepts another camera index as its first argument. Python dependencies
are stored under `/private/tmp/vape-detector-runtime` to avoid macOS offloading
files from Documents. The launcher uses CPU while the training process remains
on MPS. Two inference models can reduce the video frame rate.

Masks are assigned to local detections of the same class with bounding-box IoU
of at least 0.3. Each mask is used once; unmatched segmentation detections are
not added to the local detector's output.

- Green: predicted mask matched to a local detection.
- Yellow center: center of the oriented rectangle enclosing that mask.
- Yellow width line and cyan longitudinal axis: image-space grasp measurements.
- Orange center marked `center=box (no mask)`: midpoint of a local bounding box
  when no corresponding mask was found. Angle and grasp width are unavailable.

The terminal includes center coordinates in pixels, plus theta and width when a
valid mask exists. Coordinates start at the image's top left; theta is clockwise
from horizontal modulo 180 degrees. Physical units require calibration.
