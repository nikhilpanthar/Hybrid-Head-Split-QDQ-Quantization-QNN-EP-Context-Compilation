# Baseline 3-Class NPU Model — Testing Guide

**Model:** `3class-yolov8m-htp-ctx.onnx`  
**Hardware:** Radxa Dragon Q6A (Qualcomm QCS6490, Hexagon HTP V68 NPU)  
**Performance:** ~27ms latency | ~37 FPS | 25.5 MB  

---

## Quick Start (3 Steps)

### Step 1 — SSH into the device

```bash
ssh devuser@ssh.kavachx.io
# Password: dev121
```

### Step 2 — Run the test script

```bash
bash ~/fire-smoke-q6a/test_baseline_3class.sh
```

You should see:
```
[*] Model: /home/devuser/fire-smoke-q6a/models/3class-yolov8m-htp-ctx.onnx (26M)
[*] Port:  8080
[*] Initializing QNN Execution Provider on Hexagon HTP...
[+] HTP NPU Session ready in ~35 ms
[+] Server is LIVE at http://0.0.0.0:8080/
```

### Step 3 — Open in browser

Open a **second terminal** on your laptop and run:

```bash
ssh -L 8080:localhost:8080 devuser@ssh.kavachx.io
# Password: dev121
```

Then open in your browser: **http://localhost:8080**

You should see the live webcam feed with real-time fire/smoke/person detection bounding boxes.

---

## What This Tests

| Class | What it detects |
|-------|----------------|
| Person | People in frame |
| Fire | Flames, fire |
| Smoke | Smoke, haze |

- Bounding boxes with confidence percentages are drawn on the live video feed
- Inference runs entirely on the Qualcomm Hexagon NPU (not CPU)
- Expect ~35-39 FPS with ~25-29ms per-frame latency

---

## Manual Run (Without Script)

If the script doesn't work, run commands manually:

```bash
# SSH into device
ssh devuser@ssh.kavachx.io

# Clear port
fuser -k 8080/tcp 2>/dev/null

# Start the server
~/fire-smoke-q6a/venv/bin/python3 ~/fire-smoke-q6a/scripts/run_web_camera_npu.py \
    --model ~/fire-smoke-q6a/models/3class-yolov8m-htp-ctx.onnx \
    --port 8080
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `port 8080 already in use` | Run `fuser -k 8080/tcp` then retry |
| `Model not found` | Check: `ls -lh ~/fire-smoke-q6a/models/` |
| `venv not found` | Run: `python3 -m venv ~/fire-smoke-q6a/venv && ~/fire-smoke-q6a/venv/bin/pip install -r ~/fire-smoke-q6a/requirements.txt` |
| `HTP init failed` | Device may need reboot: `sudo reboot` |
| Browser shows nothing | Make sure SSH tunnel is running in a separate terminal |

---

## Available Models on Device

```bash
ls -lh ~/fire-smoke-q6a/models/
```

| File | Size | Description |
|------|------|-------------|
| `3class-yolov8m-htp-ctx.onnx` | 25.5 MB | **Baseline 3-class** (this test) |
| `refined_3class-yolov8m-htp-ctx.onnx` | 26.7 MB | Refined 3-class (fire/smoke confusion issue) |
| `fire-smoke-yolov8n-htp-ctx.onnx` | 3.3 MB | 2-class nano model |

To test a different model, change the `--model` path:

```bash
# Refined model
~/fire-smoke-q6a/venv/bin/python3 ~/fire-smoke-q6a/scripts/run_web_camera_npu.py \
    --model ~/fire-smoke-q6a/models/refined_3class-yolov8m-htp-ctx.onnx --port 8080

# 2-class nano model
~/fire-smoke-q6a/venv/bin/python3 ~/fire-smoke-q6a/scripts/run_web_camera_npu.py \
    --model ~/fire-smoke-q6a/models/fire-smoke-yolov8n-htp-ctx.onnx --port 8080
```

---

## Stop the Server

Press `Ctrl+C` in the terminal running the server, or from another terminal:

```bash
fuser -k 8080/tcp
```