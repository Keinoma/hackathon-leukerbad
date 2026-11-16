#!/bin/zsh
cd -- "${0:A:h}" || exit 1
export UV_PROJECT_ENVIRONMENT=/private/tmp/vape-detector-runtime
export YOLO_CONFIG_DIR=/private/tmp/vape-detector-settings
export MPLCONFIGDIR=/private/tmp/vape-detector-matplotlib
export CLANG_MODULE_CACHE_PATH=/private/tmp/vape-camera-module-cache
uv_bin="$(command -v uv)"
[[ -n "$uv_bin" ]] || uv_bin=/opt/homebrew/bin/uv
camera_index=$(/usr/bin/swift -suppress-warnings -e '
import AVFoundation
let devices = (AVCaptureDevice.devices(for: .video) + AVCaptureDevice.devices(for: .muxed)).sorted { $0.uniqueID < $1.uniqueID }
for (index, device) in devices.enumerated() {
    if device.localizedName == "AUKEY PC-W1" { print(index) }
}')
if [[ ! "$camera_index" =~ ^[0-9]+$ ]]; then
    print "AUKEY PC-W1 not found. Reconnect it, then run this again."
    exit 1
fi
print "Selected AUKEY PC-W1 (camera $camera_index)"
case "${1:-run}" in
    lens)
        "$uv_bin" run --extra ml leukerbad-detect --camera "$camera_index" \
            --lens-calibrate --lens-file lens.json ;;
    preview)
        "$uv_bin" run --extra ml leukerbad-detect --camera "$camera_index" \
            --weights models/vape_seg.pt --device cpu --rotate 180 ;;
    calibrate)
        "$uv_bin" run --extra ml leukerbad-detect --camera "$camera_index" \
            --calibrate 385 280 --calibration workspace.json --rotate 180 ;;
    run)
        "$uv_bin" run --extra ml leukerbad-detect --camera "$camera_index" \
            --weights models/vape_seg.pt --device cpu --conf 0.5 \
            --calibration workspace.json --json-out target.json ;;
    *) print "Usage: ./AUKEY.command [lens|preview|calibrate|run]"; exit 1 ;;
esac
