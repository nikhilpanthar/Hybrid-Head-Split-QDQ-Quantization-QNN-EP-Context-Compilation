# Qualcomm Q6A NPU YOLOv8 Deployment & Progress Report

**Author / Team:** KavachX Edge AI Team  
**Target Hardware:** Radxa Dragon Q6A (Qualcomm QCS6490, Hexagon HTP V68 NPU)  
**Objective:** Resolve the HTP V68 context generation blocker, validate INT8 QDQ quantization, verify NPU inference accuracy and latency, and deploy on the live streaming pipeline.

---

## 1. Executive Summary & Progress Update

### What Was Handed Over (Initial Junior Status):
- **Fine-Tuning:** YOLOv8m fine-tuned on Google Colab NVIDIA L4 (Precision 0.787).
- **FP32 Export:** Model exported as `refined_3class_L4_FP32.onnx` (98.8 MB).
- **Blocker:** Standalone `qnn-context-binary-generator` failed with DSP/FastRPC skeleton error `0x80000406` (Device Creation failure). The pipeline was blocked at context model generation.

### What Has Been Achieved & Completed:
1. **Unblocked HTP V68 Context Model Generation:**  
   Bypassed the failing CLI binary generator by using the proven **ONNX Runtime QNN Execution Provider (`ORT-QNN`)** context compiler with `htp_arch=68` and `soc_model=60`.
2. **Clean DFL / Class Split:**  
   Separated YOLOv8 concatenated outputs into raw regression anchors `[1, 64, 8400]` and classification probabilities `[1, 3, 8400]`, avoiding costly on-NPU non-maximum suppression or post-processing ops.
3. **Static INT8 QDQ Quantization:**  
   Quantized with opset 13, per-channel MinMax calibration over 50 real-world frames. Reduced model footprint from **98.8 MB down to 26.4 MB**.
4. **Hexagon HTP Compilation:**  
   Compiled to `refined_3class-yolov8m-htp-ctx.onnx` (26.7 MB). Initial compilation: 8.8s; subsequent load time on Hexagon NPU: **~176 ms**.
5. **NPU Hardware Benchmark & Parity Verification:**  
   - Latency: **~25.96 ms** (~38.5 FPS) sustained on Hexagon Tensor Processor.
   - Verification: Accurately detected `[PERSON] (84.5%)` and `[SMOKE] (78.2%)` with only 5–10 active anchors (clean, zero exploding false positives).
6. **Web Streaming Deployment:**  
   Integrated with live web camera service on port 8080 with colored bounding boxes and real-time confidence scores.

---

## 2. Directory Structure

Inside this folder (`personal/`):

```text
personal/
├── README.md                           <-- This handover report & instructions
├── 3class_model_npu_comparison.md      <-- Baseline vs Rabahdev benchmark report
└── scripts/
    ├── 01_split_heads.py               <-- Step 1: Split raw YOLOv8 heads into [1,64,8400] & [1,3,8400]
    ├── 02_quantize_int8_qdq.py         <-- Step 2: Static INT8 QDQ quantization with opset 13
    ├── 03_compile_htp_context.py       <-- Step 3: ORT-QNN HTP V68 context binary generator (unblocks pipeline)
    ├── 04_benchmark_npu.py             <-- Step 4: 30-iteration NPU latency & accuracy benchmark
    └── 05_run_live_web_camera.sh       <-- Step 5: Web camera server launcher on port 8080
```

---

## 3. Step-by-Step Execution Guide for Junior Developer

If you need to reproduce or deploy any future refined model checkpoint, follow these exact steps:

### Step 1: Split YOLOv8 Model Heads
```bash
python3 scripts/01_split_heads.py
```
- **Input:** `~/fire-smoke-q6a/models/refined_3class_L4_FP32.onnx`
- **Output:** `~/fire-smoke-q6a/models/refined_3class_split_fp32.onnx`
- **Purpose:** Extracts `/model.22/Concat_output_0` (DFL 64ch) and `/model.22/Sigmoid_output_0` (Scores 3ch).

---

### Step 2: INT8 QDQ Quantization
```bash
python3 scripts/02_quantize_int8_qdq.py
```
- **Input:** `refined_3class_split_fp32.onnx`
- **Output:** `refined_3class_split_qdq.onnx`
- **Details:** Patches opset to 13, runs MinMax per-channel calibration using 50 images from `~/kavachx/quant_work/calibration_images/`.

---

### Step 3: Compile HTP V68 Context Model (The Solution to the Blocker)
```bash
python3 scripts/03_compile_htp_context.py
```
- **Input:** `refined_3class_split_qdq.onnx`
- **Output:** `refined_3class-yolov8m-htp-ctx.onnx`
- **Why this works:** Rather than invoking `qnn-context-binary-generator` (which fails with FastRPC error `0x80000406`), this script uses `onnxruntime_qnn` to configure:
  ```python
  session_options.add_session_config_entry("ep.context_enable", "1")
  session_options.add_session_config_entry("ep.context_file_path", ctx_model_path)
  session_options.add_session_config_entry("ep.context_embed_mode", "1")
  ep_options = {"backend_path": qnn_ep.get_qnn_htp_path(), "htp_arch": "68", "soc_model": "60"}
  ```

---

### Step 4: Benchmark on NPU Test Images
```bash
python3 scripts/04_benchmark_npu.py
```
- Tests on `fire.jpg`, `person_test.jpg`, etc.
- Measures NPU warmup and 30-iteration inference latency.
- Validates active anchors and post-NMS detections.

---

### Step 5: Run Live Web Camera Stream
To start the live stream on port 8080:

```bash
# Refined 3-class model:
./scripts/05_run_live_web_camera.sh refined

# Baseline 3-class model:
./scripts/05_run_live_web_camera.sh 3class

# Rabahdev 2-class nano model:
./scripts/05_run_live_web_camera.sh 2class
```

Then open an SSH tunnel from your laptop:
```bash
ssh -L 8080:localhost:8080 devuser@ssh.kavachx.io
```
And navigate to: **`http://localhost:8080`**

---

## 4. Performance & Validation Metrics

| Model | Architecture | Size | NPU Latency | Throughput | Target Hardware | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rabahdev 2-Class** | YOLOv8n (nano) | 3.3 MB | ~9.5 ms | ~105 FPS | Qualcomm HTP V68 | Deployed |
| **Baseline 3-Class** | YOLOv8m (medium)| 25.5 MB| ~26.7 ms| ~37.5 FPS | Qualcomm HTP V68 | Deployed |
| **Refined 3-Class (L4)**| YOLOv8m (medium)| 26.7 MB| **25.96 ms**| **~38.5 FPS**| Qualcomm HTP V68 | **Validated & Deployed** |
