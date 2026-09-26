# KavachX Local — Fire/Smoke Detection on MacBook

Run the KavachX 3-class YOLOv8m INT8 model locally on your MacBook's CPU for testing and validation.

**Classes:** Person | Fire | Smoke

---

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url>
cd kavachx-local

# 2. Setup (installs Python dependencies in venv)
./setup.sh

# 3. Run benchmark on test images
./run.sh benchmark baseline

# 4. Start live webcam detection
./run.sh webcam baseline
# Open http://localhost:8080
```

---

## Available Models

Models are included in `models/` directory:

| Name | File | Size | Description |
|------|------|------|-------------|
| `baseline` | `3class_split_qdq.onnx` | ~25 MB | INT8 quantized, production-ready |
| `refined` | `refined_3class_split_qdq.onnx` | ~26 MB | INT8 quantized, fire/smoke confusion issue |
| `fp32` | `new_3class_best_FP32.fixed.onnx` | ~99 MB | FP32 reference (ground truth) |

> **Note:** The NPU models (`*-htp-ctx.onnx`) are hardware-locked to Qualcomm NPU and won't run on MacBook. Use the QDQ models above for CPU inference.

---

## Commands

### Benchmark

```bash
./run.sh benchmark baseline              # Default 30 iterations
./run.sh benchmark baseline --iterations 50
./run.sh benchmark refined --conf 0.20   # Lower confidence threshold
./run.sh benchmark fp32                  # FP32 ground truth
```

Output: latency, FPS, detection results per image. Saves annotated images to `output/`.

### Webcam Server

```bash
./run.sh webcam baseline                 # Default camera 0, port 8080
./run.sh webcam baseline --camera 1      # Use external camera
./run.sh webcam baseline --port 9090     # Custom port
./run.sh webcam refined --conf 0.20      # Lower threshold
```

Open http://localhost:8080 in your browser.

---

## Directory Structure

```
kavachx-local/
├── setup.sh              # Install dependencies
├── run.sh                # Main entry point
├── README.md             # This file
├── models/               # ONNX model files (included)
├── test_images/          # Test images (included)
├── output/               # Benchmark output images
├── scripts/
│   ├── benchmark.py      # Benchmark on test images
│   └── run_webcam.py     # Live webcam server
└── venv/                 # Python virtual environment
```

---

## Performance Expectations (MacBook CPU vs NPU)

| Device | Latency | FPS |
|--------|---------|-----|
| **Qualcomm NPU** (on device) | ~27 ms | ~37 |
| **MacBook CPU** (local) | ~100-170 ms | ~6-9 |

CPU inference is slower but functionally identical for testing accuracy.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Camera not found` | Check System Settings > Privacy > Camera |
| `Port in use` | `lsof -ti:8080 \| xargs kill` or use `--port 9090` |
| `venv not found` | Run `./setup.sh` |