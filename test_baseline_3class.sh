#!/usr/bin/env bash
# =============================================================
# KavachX — Baseline 3-Class NPU Model Tester
# Hardware: Radxa Dragon Q6A (Qualcomm QCS6490, Hexagon HTP V68)
# Model:    3class-yolov8m-htp-ctx.onnx (YOLOv8m INT8, 25.5 MB)
# Classes:  0: Person | 1: Fire | 2: Smoke
# =============================================================

set -e

MODEL="3class"
PORT=8080

echo "=========================================="
echo "  KavachX NPU Model Tester"
echo "  Baseline 3-Class: 3class-yolov8m-htp-ctx.onnx"
echo "=========================================="

# --- Check virtual environment ---
VENV=~/fire-smoke-q6a/venv/bin/python3
if [ ! -f "$VENV" ]; then
    echo "[!] ERROR: Python venv not found at $VENV"
    echo "    Run: python3 -m venv ~/fire-smoke-q6a/venv && ~/fire-smoke-q6a/venv/bin/pip install onnxruntime onnxruntime-qnn opencv-python-headless numpy"
    exit 1
fi

# --- Check model file ---
MODEL_PATH="$HOME/fire-smoke-q6a/models/3class-yolov8m-htp-ctx.onnx"
if [ ! -f "$MODEL_PATH" ]; then
    echo "[!] ERROR: Model not found at $MODEL_PATH"
    echo "    Expected files in ~/fire-smoke-q6a/models/:"
    ls -lh ~/fire-smoke-q6a/models/ 2>/dev/null || echo "    (directory not found)"
    exit 1
fi

echo "[*] Model: $MODEL_PATH ($(du -h "$MODEL_PATH" | cut -f1))"
echo "[*] Port:  $PORT"

# --- Kill any existing service on port ---
echo "[*] Clearing port $PORT..."
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 1

# --- Launch web camera NPU service ---
echo "[*] Starting NPU inference server..."
echo ""
~/fire-smoke-q6a/venv/bin/python3 ~/fire-smoke-q6a/scripts/run_web_camera_npu.py \
    --model "$MODEL_PATH" \
    --port "$PORT"