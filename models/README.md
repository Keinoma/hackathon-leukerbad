# Demo checkpoint

`vape_seg.pt` is the segmentation model currently used by the demo. It is the
`stage1_pretrain/weights/best.pt` file from the supplied
`stage1_pretrain-20261115T101010Z-1-001.zip` archive. The model was trained as
YOLO11s segmentation; its class is `vape`. Training arguments are included here.
It is distinct from the separately trained local YOLO26 detection checkpoint.

To update weights, copy a new segmentation checkpoint into `models/` and pass
`--weights models/new_seg.pt`. To keep the existing launchers unchanged, replace
`models/vape_seg.pt`. Only the demo checkpoint is tracked by default; new filenames
need an explicit Git ignore exception if you want to publish them.

Detection-only checkpoints provide boxes and confidence, but cannot supply mask
geometry alone. They can use `--segment-weights models/vape_seg.pt` to associate
supplementary masks. Updating weights does not require changing the JSON schema.
