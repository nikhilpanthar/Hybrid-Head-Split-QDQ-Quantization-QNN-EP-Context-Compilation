#!/usr/bin/env bash
# =============================================================
# KavachX Local Runner — MacBook
# Usage:
#   ./run.sh benchmark [baseline|refined|fp32]
#   ./run.sh webcam [baseline|refined|fp32]
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[!] Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

ACTION=${1:-"help"}
MODEL=${2:-"baseline"}

case "$ACTION" in
    "benchmark")
        echo "[*] Running benchmark with model: $MODEL"
        "$VENV_PYTHON" "$SCRIPT_DIR/scripts/benchmark.py" --model "$MODEL" --save "${@:3}"
        ;;
    "webcam")
        echo "[*] Starting webcam server with model: $MODEL"
        "$VENV_PYTHON" "$SCRIPT_DIR/scripts/run_webcam.py" --model "$MODEL" "${@:3}"
        ;;
    "help"|*)
        echo "=========================================="
        echo "  KavachX Local Runner"
        echo "=========================================="
        echo ""
        echo "Usage:"
        echo "  ./run.sh benchmark [model]        Run benchmark on test images"
        echo "  ./run.sh webcam [model]           Start live webcam server"
        echo ""
        echo "Models: baseline (default), refined, fp32"
        echo ""
        echo "Examples:"
        echo "  ./run.sh benchmark baseline       Benchmark baseline INT8 model"
        echo "  ./run.sh benchmark refined        Benchmark refined INT8 model"
        echo "  ./run.sh benchmark fp32           Benchmark FP32 reference"
        echo "  ./run.sh webcam baseline          Webcam with baseline model"
        echo "  ./run.sh webcam refined           Webcam with refined model"
        echo ""
        echo "Extra options (passed to Python):"
        echo "  ./run.sh benchmark baseline --iterations 50 --conf 0.20"
        echo "  ./run.sh webcam baseline --camera 1 --port 9090"
        ;;
esac