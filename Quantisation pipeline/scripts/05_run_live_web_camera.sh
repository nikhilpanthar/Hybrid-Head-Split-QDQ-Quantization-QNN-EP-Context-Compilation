#!/usr/bin/env bash
# Step 5: Launch Live Web Camera Detection Server on Qualcomm Hexagon NPU (Port 8080)
# Usage: ./05_run_live_web_camera.sh [refined|3class|2class]

MODEL_CHOICE=${1:-"refined"}
PORT=8080

echo "[*] Killing any existing processes on port ${PORT}..."
fuser -k ${PORT}/tcp 2>/dev/null || true

case "${MODEL_CHOICE}" in
    "refined")
        MODEL_PATH="/home/devuser/fire-smoke-q6a/models/refined_3class-yolov8m-htp-ctx.onnx"
        echo "[*] Launching Refined 3-Class Model (YOLOv8m) on Hexagon NPU..."
        ;;
    "3class")
        MODEL_PATH="/home/devuser/fire-smoke-q6a/models/3class-yolov8m-htp-ctx.onnx"
        echo "[*] Launching Baseline 3-Class Model (YOLOv8m) on Hexagon NPU..."
        ;;
    "2class")
        MODEL_PATH="/home/devuser/fire-smoke-q6a/models/fire-smoke-yolov8n-htp-ctx.onnx"
        echo "[*] Launching Rabahdev 2-Class Model (YOLOv8n) on Hexagon NPU..."
        ;;
    *)
        echo "[!] Unknown model choice: ${MODEL_CHOICE}. Use: refined, 3class, or 2class"
        exit 1
        ;;
esac

echo "[+] Model: ${MODEL_PATH}"
echo "[+] Starting Web Camera service on port ${PORT}..."
~/fire-smoke-q6a/venv/bin/python3 ~/fire-smoke-q6a/scripts/run_web_camera_npu.py \
    --model "${MODEL_PATH}" \
    --port "${PORT}"
