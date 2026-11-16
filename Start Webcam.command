#!/bin/zsh
# Double-click in Finder, or pass a camera index from Terminal.
cd -- "${0:A:h}" || exit 1
export YOLO_CONFIG_DIR="$PWD/.ultralytics"
export MPLCONFIGDIR="$PWD/.uv/matplotlib"
export UV_CACHE_DIR="$PWD/.uv-cache"
uv_bin="$(command -v uv)"
if [[ -z "$uv_bin" && -x /opt/homebrew/bin/uv ]]; then
    uv_bin=/opt/homebrew/bin/uv
fi
if [[ -z "$uv_bin" ]]; then
    print "uv was not found. Install uv, then run this launcher again."
    read "?Press Enter to close."
    exit 1
fi
"$uv_bin" run --offline --extra ml leukerbad-detect \
    --camera "${1:-0}" --weights models/vape_seg.pt --conf 0.60
result=$?
if (( result != 0 )); then
    print "See the error above. For camera errors, check the connection, close other"
    print "camera apps, or check System Settings > Privacy & Security > Camera."
    read "?Press Enter to close."
fi
exit "$result"
