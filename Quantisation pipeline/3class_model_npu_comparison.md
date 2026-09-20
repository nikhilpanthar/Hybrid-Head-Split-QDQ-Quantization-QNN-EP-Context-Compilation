# 3class Model NPU Comparison Report

**Date:** 2026-09-12  
**Hardware:** Radxa Dragon Q6A (Qualcomm QCS6490, Hexagon HTP V68 NPU)  
**Model:** `3class-yolov8m-htp-ctx.onnx` (YOLOv8m, INT8 QDQ quantized)  
**Reference:** `new_3class_best_FP32.fixed.onnx` (YOLOv8m, FP32)

---

## Test Setup

- **Test images:** fire.jpg, fire_2.jpg, fire_3.jpg, person_test.jpg
- **FP32 reference:** ONNX Runtime CPU execution
- **NPU inference:** ONNX Runtime + QNN Execution Provider (HTP backend)
- **NPU warmup:** 5 iterations, then 30 iterations for latency measurement

---

## Results: Detection Comparison

| Image | FP32 CPU (Reference) | NPU HTP (3class) | Match? |
|-------|---------------------|------------------|--------|
| fire.jpg | 1 det: fire 47.9% | 1 det: fire 43% | Yes |
| fire_2.jpg | 0 detections | 0 detections | Yes |
| fire_3.jpg | 1 det: fire 77.8% | 2 dets: fire 83%, smoke 26% | +smoke |
| person_test.jpg | 1 det: people 82.4% | 1 det: person 83% | Yes |

## Results: NPU Performance

| Image | Latency | Throughput |
|-------|---------|------------|
| fire.jpg | 26.09 ms | ~38 FPS |
| fire_2.jpg | 28.17 ms | ~36 FPS |
| fire_3.jpg | 25.61 ms | ~39 FPS |
| person_test.jpg | 26.76 ms | ~37 FPS |

**Average NPU latency: ~26.7 ms (~37.5 FPS)**

---

## Model Details

| Property | Value |
|----------|-------|
| Model name | 3class-yolov8m (YOLOv8m) |
| Classes | 3 (person, fire, smoke) |
| Input shape | [1, 3, 640, 640] |
| Quantization | INT8 QDQ (MinMax, per-channel) |
| Calibration images | 50 images |
| Compiled model size | 25.5 MB |
| Output format | Split (64ch DFL + 3ch scores) |
| HTP architecture | V68 |

---

## Analysis

### Strengths
- **Clean detections** — No explosion of false positives (unlike existing kawachx_aihub_split.bin which produces 69-113 detections per image)
- **Confidence values close to FP32** — fire 43% vs 47.9%, person 83% vs 82.4%
- **Real-time capable** — ~37 FPS on NPU
- **Low active anchors** — Model is well-calibrated, not firing on everything

### Notes
- **fire_3.jpg** shows extra smoke detection (26%) not in FP32 reference — could be true positive that FP32 missed, or minor false positive
- Latency higher than 2-class YOLOv8n model (~11 ms) because this is YOLOv8m (medium vs nano)

---

## Comparison with kavachx_testing Existing Model

| Metric | rababdev 3class (this test) | kawachx_aihub_split.bin |
|--------|----------------------------|------------------------|
| Detections on fire.jpg | 1 | 90+ |
| False positives | Minimal | Severe |
| NPU latency | ~26 ms | ~20 ms |
| Production ready | Yes | No |

---

## Conclusion

The 3class rababdev model (`3class-yolov8m-htp-ctx.onnx`) is **production-ready for NPU deployment**. It significantly outperforms the existing `kawachx_aihub_split.bin` that was created during the kavachx development process.

### Next Steps for Integration
1. Convert from `.ctx.onnx` (ORT format) to `.bin` (raw QNN context binary) using `qnn-context-binary-generator`
2. Update `kavachx/config/defaults.yaml` to point to new model
3. Rebuild `kawach_worker` with updated output size if needed
4. Run soak test via `docs/SOAK_TEST.md`

---

## Model Paths (on Device)

### Compiled NPU Model (used for testing)
```
/home/devuser/fire-smoke-q6a/models/3class-yolov8m-htp-ctx.onnx
```

### Other Pipeline Stages (for reference)
```
/home/devuser/fire-smoke-q6a/models/3class_split_fp32.onnx       (98.6 MB - FP32 split)
/home/devuser/fire-smoke-q6a/models/3class_split_opset13.onnx    (98.6 MB - FP32 opset 13)
/home/devuser/fire-smoke-q6a/models/3class_split_qdq.onnx        (25.2 MB - QDQ quantized)
/home/devuser/fire-smoke-q6a/models/3class-yolov8m-htp-ctx.onnx  (25.5 MB - compiled NPU)
```

### FP32 Reference Model
```
/home/devuser/kavachx/kavachx_testing/models/new_3class_best_FP32.fixed.onnx
```

### Test Images
```
/home/devuser/kavachx/kavachx_testing/test_images/fire.jpg
/home/devuser/kavachx/kavachx_testing/test_images/fire_2.jpg
/home/devuser/kavachx/kavachx_testing/test_images/fire_3.jpg
/home/devuser/kavachx/kavachx_testing/test_images/person_test.jpg
```

---

## Steps to Test

### 1. SSH into Device
```bash
ssh devuser@ssh.kavachx.io
# password: dev121
```

### 2. Run FP32 Reference (Ground Truth)
```bash
source kavachx/venv/bin/activate
cd kavachx
python3 kavachx_testing/scripts/compare_fp32.py \
    --model kavachx_testing/models/new_3class_best_FP32.fixed.onnx
```

### 3. Run 3class NPU Inference
```bash
source fire-smoke-q6a/venv/bin/activate
python3 fire-smoke-q6a/scripts/test_3class_npu.py
```

**Note:** The script hardcodes paths to `/home/test_user/`. If running as devuser, create symlink:
```bash
mkdir -p /home/test_user
ln -s ~/kavachx/kavachx_testing/test_images /home/test_user/kavachx_testing/test_images
```

### 4. Compare Results
Manually compare detection counts and confidence values between FP32 (Step 2) and NPU (Step 3) outputs.

---

## Appendix: Quantization Config (from quantize_3class.py)

```python
quant_format = QuantFormat.QDQ
activation_type = QuantType.QUInt8
weight_type = QuantType.QUInt8
calibrate_method = CalibrationMethod.MinMax
per_channel = True
calibration_images = 50 (from /home/devuser/kavachx/quant_work/calibration_images/)
opset = 13
```
