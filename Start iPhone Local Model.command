#!/bin/zsh
cd -- "${0:A:h}" || exit 1
export YOLO_CONFIG_DIR=/private/tmp/vape-detector-settings
export MPLCONFIGDIR=/private/tmp/vape-detector-matplotlib
uv_bin="$(command -v uv)"
[[ -n "$uv_bin" ]] || uv_bin=/opt/homebrew/bin/uv
# Keep dependencies outside Documents to avoid macOS offloading library files.
export UV_PROJECT_ENVIRONMENT=/private/tmp/vape-detector-runtime
"$uv_bin" run --offline --extra ml leukerbad-detect \
    --camera "${1:-0}" \
    --weights ../runs_local/vape_yolo26_handphotos/weights/best.pt \
    --segment-weights models/vape_seg.pt --conf 0.25 --device cpu
result=$?
if (( result != 0 )); then
    read "?Press Enter to close."
fi
exit "$result"
