#!/usr/bin/env python3
"""
KavachX Local — Mobile Camera Detection Server
Runs the INT8 QDQ model locally on your MacBook's CPU.
Connect your phone's browser to use your phone camera as the live video source!
"""
import os
import sys
import time
import json
import argparse
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASS_NAMES = {0: "person", 1: "fire", 2: "smoke"}
CLASS_COLORS = {
    0: {"hex": "#00e5ff", "name": "Person"},   # Bright Cyan for Person
    1: {"hex": "#ff1744", "name": "Fire"},     # Vivid Red for Fire
    2: {"hex": "#ffea00", "name": "Smoke"}     # Bright Yellow for Smoke
}

MODEL_MAP = {
    "baseline": "3class_split_qdq.onnx",
    "refined": "refined_3class_split_qdq.onnx",
    "fp32": "new_3class_best_FP32.fixed.onnx",
}

# Precompute anchor grids and strides
def generate_anchors():
    anchor_x = np.zeros(8400, dtype=np.float32)
    anchor_y = np.zeros(8400, dtype=np.float32)
    stride = np.zeros(8400, dtype=np.float32)
    idx = 0
    for g, s in [(80, 8), (40, 16), (20, 32)]:
        for y in range(g):
            for x in range(g):
                anchor_x[idx] = x + 0.5
                anchor_y[idx] = y + 0.5
                stride[idx] = s
                idx += 1
    return anchor_x, anchor_y, stride

ANCHOR_X, ANCHOR_Y, STRIDE = generate_anchors()

def fast_dfl_and_nms(bbox_dist, cls_scores, conf_thresh=0.35, iou_thresh=0.30, ios_thresh=0.38, orig_shape=(480, 640)):
    h_orig, w_orig = orig_shape
    scores_all = cls_scores[0]  # [3, 8400]
    max_scores = np.max(scores_all, axis=0)  # [8400]
    cands = np.where(max_scores >= conf_thresh)[0]

    if len(cands) == 0:
        return []

    c_scores = scores_all[:, cands]
    cls_ids = np.argmax(c_scores, axis=0)
    confidences = np.max(c_scores, axis=0)

    # Candidate-only DFL computation
    c_dist_raw = bbox_dist[0, :, cands].reshape(4, 16, -1)
    c_exp = np.exp(c_dist_raw - np.max(c_dist_raw, axis=1, keepdims=True))
    c_prob = c_exp / np.sum(c_exp, axis=1, keepdims=True)
    weights = np.arange(16, dtype=np.float32).reshape(1, 16, 1)
    c_dist = np.sum(c_prob * weights, axis=1)  # [4, N]

    # Convert to bounding boxes in 640x640 space
    x1 = (ANCHOR_X[cands] - c_dist[0]) * STRIDE[cands]
    y1 = (ANCHOR_Y[cands] - c_dist[1]) * STRIDE[cands]
    x2 = (ANCHOR_X[cands] + c_dist[2]) * STRIDE[cands]
    y2 = (ANCHOR_Y[cands] + c_dist[3]) * STRIDE[cands]

    # Scale directly to original image space
    scale_x = w_orig / 640.0
    scale_y = h_orig / 640.0
    x1 = np.clip(x1 * scale_x, 0, w_orig)
    y1 = np.clip(y1 * scale_y, 0, h_orig)
    x2 = np.clip(x2 * scale_x, 0, w_orig)
    y2 = np.clip(y2 * scale_y, 0, h_orig)

    boxes = np.column_stack([x1, y1, x2, y2])

    keep = []
    # Per-class NMS with IoU + IoS containment suppression
    for target_cls in np.unique(cls_ids):
        cls_mask = np.where(cls_ids == target_cls)[0]
        cls_boxes = boxes[cls_mask]
        cls_confs = confidences[cls_mask]
        areas = (cls_boxes[:, 2] - cls_boxes[:, 0]) * (cls_boxes[:, 3] - cls_boxes[:, 1])
        order = cls_confs.argsort()[::-1]

        while order.size > 0:
            i = order[0]
            keep.append(cls_mask[i])
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[rest, 0])
            yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[rest, 1])
            xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[rest, 2])
            yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[rest, 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[rest] - inter
            iou = np.where(union > 0, inter / union, 0.0)
            min_area = np.minimum(areas[i], areas[rest])
            ios = np.where(min_area > 0, inter / min_area, 0.0)

            # Suppress if IoU overlap OR containment (IoS) is too high
            suppress = (iou > iou_thresh) | (ios > ios_thresh)
            remaining_inds = np.where(~suppress)[0]
            order = rest[remaining_inds]

    detections = []
    for idx in keep:
        cid = int(cls_ids[idx])
        detections.append({
            "class_id": cid,
            "class_name": CLASS_NAMES.get(cid, f"class_{cid}"),
            "confidence": float(round(confidences[idx], 3)),
            "color": CLASS_COLORS.get(cid, {}).get("hex", "#00ff00"),
            "bbox": [float(round(boxes[idx, 0], 1)), float(round(boxes[idx, 1], 1)),
                     float(round(boxes[idx, 2], 1)), float(round(boxes[idx, 3], 1))]
        })
    return detections

SESSION = None
INP_NAME = None
LOCK = threading.Lock()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>KavachX — Mobile Camera Detection</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: #0b0f19;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 12px;
            touch-action: manipulation;
        }
        .header { text-align: center; margin-bottom: 10px; width: 100%; max-width: 640px; }
        .badge {
            display: inline-block;
            background: #2563eb;
            color: white;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
            box-shadow: 0 0 10px rgba(37, 99, 235, 0.4);
        }
        h1 { font-size: 20px; font-weight: 700; color: #f8fafc; }
        .sub { font-size: 11px; color: #94a3b8; }
        .stage {
            position: relative;
            max-width: 640px;
            width: 100%;
            border-radius: 12px;
            overflow: hidden;
            background: #000;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
            border: 1px solid #1e293b;
        }
        #camera-video { display: none; }
        #canvas-view { width: 100%; height: auto; display: block; }
        .hud-top {
            position: absolute;
            top: 8px; left: 8px; right: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            pointer-events: none;
            z-index: 10;
        }
        .telemetry {
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(6px);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 6px;
            padding: 5px 10px;
            font-size: 12px;
            font-weight: 700;
            color: #38bdf8;
            font-family: monospace;
        }
        .alert-badge {
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .status-normal { background: rgba(16, 185, 129, 0.85); color: #fff; }
        .status-fire { background: rgba(239, 68, 68, 0.95); color: #fff; animation: pulse 0.8s infinite; }
        .status-smoke { background: rgba(245, 158, 11, 0.95); color: #fff; }
        .status-person { background: rgba(14, 165, 233, 0.95); color: #fff; }
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
        .controls {
            max-width: 640px;
            width: 100%;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
            background: #131b2e;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 13px;
        }
        .btn-group { display: flex; gap: 8px; }
        .btn {
            background: #2563eb;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s;
        }
        .btn:hover { background: #1d4ed8; }
        .btn.stop { background: #dc2626; }
        .btn.flip { background: #475569; }
        .slider-group { display: flex; align-items: center; gap: 8px; color: #cbd5e1; font-size: 12px; }
        input[type="range"] { accent-color: #38bdf8; width: 100px; }
        .legend { display: flex; gap: 14px; margin-top: 8px; font-size: 12px; }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    </style>
</head>
<body>
    <div class="header">
        <span class="badge">MacBook CPU &bull; INT8 QDQ YOLOv8m</span>
        <h1>KavachX Mobile Camera</h1>
        <div class="sub">Live Edge Detection &bull; Person &bull; Fire &bull; Smoke</div>
    </div>

    <div class="stage">
        <video id="camera-video" playsinline autoplay muted></video>
        <canvas id="canvas-view" width="640" height="480"></canvas>
        <div class="hud-top">
            <div class="telemetry" id="telemetry-bar">CPU: -- ms | FPS: --</div>
            <div class="alert-badge status-normal" id="status-badge">STANDBY</div>
        </div>
    </div>

    <div class="controls">
        <div class="btn-group">
            <button class="btn" id="start-btn" onclick="toggleStream()">Start Camera</button>
            <button class="btn flip" id="flip-btn" onclick="switchCamera()">Flip Camera</button>
        </div>
        <div class="slider-group">
            <label for="conf-slider">Sens:</label>
            <input type="range" id="conf-slider" min="0.15" max="0.80" step="0.05" value="0.35" oninput="updateConf(this.value)">
            <span id="conf-val">0.35</span>
        </div>
    </div>

    <div class="legend">
        <div class="legend-item"><span class="dot" style="background:#00e5ff;"></span> Person</div>
        <div class="legend-item"><span class="dot" style="background:#ff1744;"></span> Fire</div>
        <div class="legend-item"><span class="dot" style="background:#ffea00;"></span> Smoke</div>
    </div>

    <script>
        const video = document.getElementById('camera-video');
        const canvas = document.getElementById('canvas-view');
        const ctx = canvas.getContext('2d', { alpha: false });
        const telemetry = document.getElementById('telemetry-bar');
        const statusBadge = document.getElementById('status-badge');
        const startBtn = document.getElementById('start-btn');

        let isRunning = false;
        let stream = null;
        let confThresh = 0.35;
        let currentFacingMode = 'environment'; // Rear camera default on mobile
        let offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = 640;
        offscreenCanvas.height = 480;
        let offscreenCtx = offscreenCanvas.getContext('2d');
        let inFlight = false;
        let fpsCounter = 0;
        let lastFpsTime = performance.now();
        let currentFps = 0;
        let latestDetections = [];

        function updateConf(v) {
            confThresh = parseFloat(v);
            document.getElementById('conf-val').innerText = confThresh.toFixed(2);
        }

        async function switchCamera() {
            currentFacingMode = (currentFacingMode === 'environment') ? 'user' : 'environment';
            if (isRunning) {
                stopStream();
                await startStream();
            }
        }

        async function toggleStream() {
            if (isRunning) {
                stopStream();
            } else {
                await startStream();
            }
        }

        async function startStream() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: currentFacingMode
                    },
                    audio: false
                });
                video.srcObject = stream;
                await video.play();
                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;
                isRunning = true;
                startBtn.innerText = "Stop Camera";
                startBtn.classList.add("stop");
                requestAnimationFrame(loop);
            } catch (err) {
                alert("Camera error: " + err.message);
            }
        }

        function stopStream() {
            isRunning = false;
            latestDetections = [];
            if (stream) stream.getTracks().forEach(t => t.stop());
            startBtn.innerText = "Start Camera";
            startBtn.classList.remove("stop");
            statusBadge.className = "alert-badge status-normal";
            statusBadge.innerText = "OFFLINE";
        }

        async function loop() {
            if (!isRunning) return;

            // 1. Draw raw live camera frame
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // 2. Draw persistent detection bounding boxes on EVERY frame
            drawDetections(latestDetections);

            // 3. Send frame for Mac CPU inference asynchronously
            if (!inFlight) {
                inFlight = true;
                offscreenCtx.drawImage(video, 0, 0, 640, 480);
                offscreenCanvas.toBlob(async (blob) => {
                    try {
                        const t0 = performance.now();
                        const formData = new FormData();
                        formData.append("frame", blob);
                        formData.append("conf", confThresh);
                        formData.append("w", canvas.width);
                        formData.append("h", canvas.height);

                        const res = await fetch("/infer", { method: "POST", body: formData });
                        const data = await res.json();
                        const rtt = Math.round(performance.now() - t0);

                        fpsCounter++;
                        const now = performance.now();
                        if (now - lastFpsTime >= 1000) {
                            currentFps = Math.round((fpsCounter * 1000) / (now - lastFpsTime));
                            fpsCounter = 0;
                            lastFpsTime = now;
                        }

                        telemetry.innerText = `CPU: ${data.cpu_ms}ms | Net: ${rtt}ms | ${currentFps} FPS`;
                        latestDetections = data.detections || [];
                        updateStatusBadge(latestDetections);
                    } catch (e) {
                    } finally {
                        inFlight = false;
                    }
                }, "image/jpeg", 0.60);
            }
            requestAnimationFrame(loop);
        }

        function drawDetections(dets) {
            if (!dets || dets.length === 0) return;
            dets.forEach(d => {
                const [x1, y1, x2, y2] = d.bbox;
                const w = x2 - x1;
                const h = y2 - y1;

                // Glowing bounding box
                ctx.strokeStyle = d.color;
                ctx.lineWidth = 4;
                ctx.strokeRect(x1, y1, w, h);

                // Label badge with class & percentage
                ctx.fillStyle = d.color;
                const label = ` ${d.class_name.toUpperCase()} ${(d.confidence * 100).toFixed(0)}% `;
                ctx.font = "bold 15px monospace";
                const textWidth = ctx.measureText(label).width;
                const badgeY = Math.max(0, y1 - 24);
                ctx.fillRect(x1, badgeY, textWidth, 24);

                ctx.fillStyle = "#000000";
                ctx.fillText(label, x1, badgeY + 17);
            });
        }

        function updateStatusBadge(dets) {
            let hasFire = false;
            let hasSmoke = false;
            let hasPerson = false;

            dets.forEach(d => {
                if (d.class_name === "fire") hasFire = true;
                if (d.class_name === "smoke") hasSmoke = true;
                if (d.class_name === "person") hasPerson = true;
            });

            if (hasFire) {
                statusBadge.className = "alert-badge status-fire";
                statusBadge.innerText = "🔥 FIRE DETECTED";
            } else if (hasSmoke) {
                statusBadge.className = "alert-badge status-smoke";
                statusBadge.innerText = "💨 SMOKE DETECTED";
            } else if (hasPerson) {
                statusBadge.className = "alert-badge status-person";
                statusBadge.innerText = "👤 PERSON DETECTED";
            } else {
                statusBadge.className = "alert-badge status-normal";
                statusBadge.innerText = "MONITORING";
            }
        }
    </script>
</body>
</html>
"""

class InferHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            content = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/infer":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            content_type = self.headers.get("Content-Type", "")
            boundary = content_type.split("boundary=")[-1].encode()

            parts = body.split(b"--" + boundary)
            jpeg_bytes = None
            conf_val = 0.35
            orig_w = 640
            orig_h = 480

            for p in parts:
                if b'name="frame"' in p:
                    head, data = p.split(b"\r\n\r\n", 1)
                    jpeg_bytes = data.rstrip(b"\r\n")
                elif b'name="conf"' in p:
                    head, data = p.split(b"\r\n\r\n", 1)
                    try:
                        conf_val = float(data.strip())
                    except:
                        pass
                elif b'name="w"' in p:
                    head, data = p.split(b"\r\n\r\n", 1)
                    try:
                        orig_w = int(data.strip())
                    except:
                        pass
                elif b'name="h"' in p:
                    head, data = p.split(b"\r\n\r\n", 1)
                    try:
                        orig_h = int(data.strip())
                    except:
                        pass

            if not jpeg_bytes:
                self.send_response(400)
                self.end_headers()
                return

            # Decode JPEG
            np_arr = np.frombuffer(jpeg_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                self.send_response(400)
                self.end_headers()
                return

            # Resize to 640x640 for YOLOv8
            resized = cv2.resize(img, (640, 640))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            blob = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
            blob = np.expand_dims(blob, axis=0)

            # MacBook CPU Inference
            t0 = time.perf_counter()
            with LOCK:
                outputs = SESSION.run(None, {INP_NAME: blob})
            cpu_ms = round((time.perf_counter() - t0) * 1000, 1)

            # DFL + NMS
            dets = fast_dfl_and_nms(
                outputs[0], outputs[1],
                conf_thresh=conf_val,
                orig_shape=(orig_h, orig_w)
            )

            resp_payload = json.dumps({"cpu_ms": cpu_ms, "detections": dets}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(resp_payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    global SESSION, INP_NAME

    parser = argparse.ArgumentParser(description="KavachX Mobile Camera Local Server")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--model", default="baseline", choices=list(MODEL_MAP.keys()), help="Model choice")
    args = parser.parse_args()

    model_filename = MODEL_MAP[args.model]
    model_path = os.path.join(MODELS_DIR, model_filename)

    if not os.path.exists(model_path):
        print(f"[!] Error: Model not found at {model_path}")
        sys.exit(1)

    print("==================================================================")
    print("      KavachX Local — Mobile Camera Edge Detection Server         ")
    print(f"      Model: {model_filename} (INT8 QDQ) on CPU                 ")
    print("      Classes: Person | Fire | Smoke                              ")
    print("==================================================================")

    print(f"[*] Loading ONNX model into CPUExecutionProvider...")
    t0 = time.time()
    SESSION = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print(f"[+] Loaded model in {(time.time() - t0)*1000:.1f} ms!")
    INP_NAME = SESSION.get_inputs()[0].name

    # Warmup
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    SESSION.run(None, {INP_NAME: dummy})
    print("[+] Model warmup completed!")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), InferHandler)
    print(f"\n[+] Local Server LIVE on port {args.port}!")
    print(f"[*] Connect via browser on Mac or Mobile.")
    print("==================================================================\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
