#!/usr/bin/env python3
"""
KavachX Local Webcam Server — Live fire/smoke/person detection on your MacBook.
Uses INT8 QDQ model on CPU via ONNX Runtime.
Serves on http://localhost:8080
"""
import os
import sys
import time
import argparse
import cv2
import numpy as np
import onnxruntime as ort
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASSES = ["person", "fire", "smoke"]
COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (128, 128, 128)}

MODEL_MAP = {
    "baseline": "3class_split_qdq.onnx",
    "refined": "refined_3class_split_qdq.onnx",
    "fp32": "new_3class_best_FP32.fixed.onnx",
}

# Global state
current_frame = None
current_stats = {"fps": 0, "latency_ms": 0, "detections": []}
lock = threading.Lock()


def get_model_path(model_name):
    filename = MODEL_MAP.get(model_name)
    if not filename:
        print(f"[!] Unknown model: {model_name}. Choose from: {list(MODEL_MAP.keys())}")
        sys.exit(1)
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f"[!] Model not found: {path}")
        print(f"    Copy from device:")
        print(f"    scp devuser@ssh.kavachx.io:~/fire-smoke-q6a/models/{filename} ./models/")
        sys.exit(1)
    return path


def dfl_decode(dfl_tensor, reg_max=16):
    b, c, num_anchors = dfl_tensor.shape
    tensor = dfl_tensor.reshape(b, 4, reg_max, num_anchors)
    exp_t = np.exp(tensor - np.max(tensor, axis=2, keepdims=True))
    prob = exp_t / np.sum(exp_t, axis=2, keepdims=True)
    weights = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1)
    return np.sum(prob * weights, axis=2)


def make_anchors(strides=(8, 16, 32), grid_sizes=((80, 80), (40, 40), (20, 20))):
    anchor_points, stride_tensor = [], []
    for stride, (h, w) in zip(strides, grid_sizes):
        sx = np.arange(w) + 0.5
        sy = np.arange(h) + 0.5
        gx, gy = np.meshgrid(sx, sy)
        anchor_points.append(np.stack([gx.ravel(), gy.ravel()], axis=-1))
        stride_tensor.append(np.full((h * w, 1), stride))
    return np.concatenate(anchor_points, axis=0), np.concatenate(stride_tensor, axis=0)


def preprocess(frame):
    h0, w0 = frame.shape[:2]
    r = min(640 / h0, 640 / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (640 - nw) / 2, (640 - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob, (h0, w0), (r, dw, dh)


def postprocess(outs, h0, w0, r, dw, dh, conf_threshold=0.25, nms_threshold=0.45):
    dfl_out, scores_out = outs[0], outs[1]
    scores = scores_out[0].T
    bbox_delta = dfl_decode(dfl_out)[0]
    anchors, strides = make_anchors()

    x1 = (anchors[:, 0] - bbox_delta[0, :]) * strides[:, 0]
    y1 = (anchors[:, 1] - bbox_delta[1, :]) * strides[:, 0]
    x2 = (anchors[:, 0] + bbox_delta[2, :]) * strides[:, 0]
    y2 = (anchors[:, 1] + bbox_delta[3, :]) * strides[:, 0]
    boxes = np.stack([x1, y1, x2, y2], axis=-1)

    cls_ids = np.argmax(scores, axis=1)
    max_scores = np.max(scores, axis=1)
    mask = max_scores >= conf_threshold
    boxes = boxes[mask]
    scores_filt = max_scores[mask]
    cls_ids = cls_ids[mask]

    if len(boxes) > 0:
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - dw) / r, 0, w0)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - dh) / r, 0, h0)
        xywh = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes]
        indices = cv2.dnn.NMSBoxes(xywh, scores_filt.tolist(), conf_threshold, nms_threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            return boxes[indices], scores_filt[indices], cls_ids[indices]
    return np.array([]), np.array([]), np.array([])


def draw_detections(image, boxes, scores, cls_ids, fps, latency_ms):
    for box, score, cid in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = COLORS.get(cid, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASSES[cid].upper()} {score * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # FPS + latency overlay
    info = f"CPU | {fps:.1f} FPS | {latency_ms:.0f}ms"
    cv2.putText(image, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global current_frame, current_stats
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode())
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while True:
                with lock:
                    if current_frame is None:
                        continue
                    _, jpeg = cv2.imencode(".jpg", current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    data = jpeg.tobytes()
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
                    time.sleep(0.03)
                except BrokenPipeError:
                    break
        elif self.path == "/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            with lock:
                self.wfile.write(json.dumps(current_stats).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>KavachX - Live Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #1a1a2e; color: #eee; font-family: monospace; display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 20px; }
        h1 { margin-bottom: 10px; color: #0f3460; font-size: 24px; }
        h1 span { color: #e94560; }
        #stats { margin-bottom: 15px; font-size: 14px; color: #aaa; }
        #stream { border: 2px solid #333; border-radius: 8px; max-width: 100%; }
        .legend { margin-top: 15px; display: flex; gap: 20px; font-size: 13px; }
        .legend span { display: flex; align-items: center; gap: 5px; }
        .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
        .dot-person { background: #00ff00; }
        .dot-fire { background: #0000ff; }
        .dot-smoke { background: #808080; }
    </style>
</head>
<body>
    <h1>KavachX <span>Live Detection</span></h1>
    <div id="stats">Loading...</div>
    <img id="stream" src="/stream" width="800" />
    <div class="legend">
        <span><span class="dot dot-person"></span> Person</span>
        <span><span class="dot dot-fire"></span> Fire</span>
        <span><span class="dot dot-smoke"></span> Smoke</span>
    </div>
    <script>
        setInterval(() => {
            fetch('/stats').then(r => r.json()).then(d => {
                let detStr = d.detections.map(x => x.class + ' ' + (x.confidence*100).toFixed(0) + '%').join(', ') || 'None';
                document.getElementById('stats').textContent =
                    `FPS: ${d.fps.toFixed(1)} | Latency: ${d.latency_ms.toFixed(0)}ms | Detections: ${detStr}`;
            }).catch(() => {});
        }, 1000);
    </script>
</body>
</html>"""


def inference_loop(session, camera_id, conf_threshold):
    global current_frame, current_stats

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[!] ERROR: Cannot open camera {camera_id}")
        print("    On macOS, grant camera permission in System Settings > Privacy > Camera")
        sys.exit(1)

    print(f"[+] Camera opened: {camera_id}")

    # Warmup
    ret, frame = cap.read()
    if ret:
        blob, _, _ = preprocess(frame)
        for _ in range(3):
            session.run(None, {"images": blob})

    frame_count = 0
    fps_timer = time.time()
    display_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        blob, (h0, w0), (r, dw, dh) = preprocess(frame)

        t0 = time.time()
        outs = session.run(None, {"images": blob})
        latency = (time.time() - t0) * 1000

        boxes, scores, cls_ids = postprocess(outs, h0, w0, r, dw, dh, conf_threshold)

        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            display_fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        result = draw_detections(frame.copy(), boxes, scores, cls_ids, display_fps, latency)

        detections = []
        if len(boxes) > 0:
            for box, score, cid in zip(boxes, scores, cls_ids):
                detections.append({
                    "class": CLASSES[cid],
                    "confidence": float(score),
                    "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                })

        with lock:
            current_frame = result
            current_stats = {
                "fps": display_fps,
                "latency_ms": latency,
                "detections": detections
            }

    cap.release()


def main():
    parser = argparse.ArgumentParser(description="KavachX Local Webcam Server")
    parser.add_argument("--model", default="baseline", choices=list(MODEL_MAP.keys()),
                        help="Model to use (default: baseline)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Server port (default: 8080)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID (default: 0)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    model_path = get_model_path(args.model)
    model_size = os.path.getsize(model_path) / 1e6

    print(f"==========================================")
    print(f"  KavachX Local Webcam Server")
    print(f"  Model: {os.path.basename(model_path)} ({model_size:.1f} MB)")
    print(f"  Device: CPU (ONNX Runtime)")
    print(f"  Camera: {args.camera}")
    print(f"  Port: {args.port}")
    print(f"==========================================")

    print(f"\n[*] Loading model...")
    t0 = time.time()
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    load_time = (time.time() - t0) * 1000
    print(f"[+] Model loaded in {load_time:.0f} ms")

    # Start inference in background thread
    thread = threading.Thread(target=inference_loop, args=(session, args.camera, args.conf), daemon=True)
    thread.start()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", args.port), StreamHandler)
    print(f"\n[+] Server is LIVE at http://localhost:{args.port}")
    print(f"[+] Open in browser: http://localhost:{args.port}")
    print(f"[+] Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()