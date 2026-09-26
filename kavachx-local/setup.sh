#!/usr/bin/env bash
# =============================================================
# KavachX Local Setup — MacBook (CPU Inference)
# =============================================================
set -e

echo "=========================================="
echo "  KavachX Local Setup"
echo "=========================================="

# --- Create virtual environment ---
if [ ! -d "venv" ]; then
    echo "[*] Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "[*] Installing dependencies..."
./venv/bin/pip install --upgrade pip
./venv/bin/pip install \
    onnxruntime \
    opencv-python-headless \
    numpy

echo ""
echo "[+] Setup complete!"
echo ""
echo "Run benchmark:  ./run.sh benchmark baseline"
echo "Run webcam:     ./run.sh webcam baseline"