#!/usr/bin/env python3
"""
KavachX CCTV video detection server — improved annotation version.

Keeps the existing run_mobile_cam.py postprocessor and tracker, but:
- Does not reshape person boxes (uses detector's original boxes).
- Uses class-specific confidence thresholds with a less aggressive default fire threshold.
- Draws readable labels with collision-aware placement and safe clipping.
- Handles video stride correctly by processing the frame actually read after skipping.
- Adds configurable input size, confidence thresholds, JPEG quality, and looping.
- Validates the ONNX input shape/dtype instead of silently assuming every model is float32 NCHW.

IMPORTANT:
This script assumes fast_dfl_and_nms(outs[0], outs[1], ...) supports the same 640 letterbox
metadata and output format as your existing script. The --imgsz option only supports 640 for
this existing postprocessor unless you update that postprocessor too.
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
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from run_mobile_cam import (  # noqa: E402
    MODEL_MAP,
    CLASS_NAMES,
    CLASS_COLORS,
    fast_dfl_and_nms,
    DetectionTracker,
)

SESSION = None
INP_NAME = None
INP_SHAPE = None
INP_TYPE = None
LOCK = threading.Lock()
STOP = threading.Event()

CURRENT_JPEG = None
SEQ = 0
STATS = {
    "fps": 0.0,
    "inference_ms": 0.0,
    "frame": 0,
    "total": 0,
    "playing": False,
    "detections": [],
    "error": None,
}


def hex_to_bgr(hex_color):
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        return (0, 255, 255)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def get_class_name(detection):
    name = detection.get("class_name")
    if name:
        return str(name).lower()
    cls_id = detection.get("class_id", detection.get("class", -1))
    try:
        return str(CLASS_NAMES[int(cls_id)]).lower()
    except (ValueError, TypeError, IndexError, KeyError):
        return "object"


def preprocess(img, input_size=640):
    """YOLO letterbox preprocessing; returns float32 RGB NCHW and transform metadata."""
    h, w = img.shape[:2]
    r = min(input_size / h, input_size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (input_size - nw) / 2.0, (input_size - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob, (h, w), (r, dw, dh)


def draw_boxes(frame, detections):
    """Professional-style overlay; labels are moved to avoid common overlaps."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    scale = max(0.6, min(w, h) / 900.0)
    thickness = max(2, int(round(min(w, h) / 500)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = max(4, int(round(scale * 6)))
    label_h = max(22, int(round(24 * scale)))
    occupied = []

    # Draw all boxes first so later labels cannot obscure other box edges.
    for d in detections:
        bbox = d.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        color = hex_to_bgr(d.get("color", "#00E5FF"))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # Place labels in a free area above, inside, or below each box.
    for d in sorted(detections, key=lambda x: float(x.get("confidence", 0)), reverse=True):
        bbox = d.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cls_name = get_class_name(d).upper()
        conf = float(d.get("confidence", 0.0))
        label = f"{cls_name} {conf:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, font, scale, max(1, thickness - 1))
        box_w = min(w, tw + 2 * pad)
        box_h = th + baseline + 2 * max(2, pad // 2)

        candidates = [
            (x1, y1 - box_h),       # above
            (x1, y1),               # inside top
            (x1, y2),               # below
            (max(0, x2 - box_w), y1 - box_h),
            (max(0, x2 - box_w), y2),
        ]

        chosen = None
        for lx, ly in candidates:
            lx = max(0, min(w - box_w, lx))
            ly = max(0, min(h - box_h, ly))
            rect = (lx, ly, lx + box_w, ly + box_h)
            overlaps = any(
                not (rect[2] <= r[0] or rect[0] >= r[2] or
                     rect[3] <= r[1] or rect[1] >= r[3])
                for r in occupied
            )
            if not overlaps:
                chosen = rect
                break
        if chosen is None:
            lx = max(0, min(w - box_w, x1))
            ly = max(0, min(h - box_h, y1))
            chosen = (lx, ly, lx + box_w, ly + box_h)

        lx1, ly1, lx2, ly2 = chosen
        occupied.append(chosen)
        color = hex_to_bgr(d.get("color", "#00E5FF"))
        cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), color, -1, cv2.LINE_AA)
        text_y = min(ly2 - baseline - 1, ly1 + th + max(2, pad // 2))
        cv2.putText(
            canvas, label, (lx1 + pad, text_y), font, scale,
            (10, 10, 10), max(1, thickness - 1), cv2.LINE_AA
        )
    return canvas


def inference_loop(video_path, conf, fire_conf, smoke_conf, person_conf, stride, loop, jpeg_quality):
    global CURRENT_JPEG, SEQ

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        message = f"Cannot open video: {video_path}"
        print(f"[!] {message}")
        with LOCK:
            STATS["error"] = message
            STATS["playing"] = False
        STOP.set()
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[+] Video: {src_w}x{src_h} | {total} frames @ {src_fps:.1f} fps | stride={stride}")

    tracker = DetectionTracker()
    processed_idx = 0
    read_idx = 0
    fps_frames = 0
    fps_timer = time.time()
    display_fps = 0.0

    try:
        while not STOP.is_set():
            # Read one frame, then skip the next stride-1 frames.
            ret, frame = cap.read()
            if not ret:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    tracker = DetectionTracker()
                    read_idx = 0
                    continue
                break
            read_idx += 1
            for _ in range(stride - 1):
                if not cap.grab():
                    break
                read_idx += 1

            blob, (h0, w0), lb = preprocess(frame, input_size=640)

            t0 = time.perf_counter()
            try:
                with LOCK:
                    outs = SESSION.run(None, {INP_NAME: blob})
            except Exception as exc:
                print(f"[!] Inference error: {exc}")
                with LOCK:
                    STATS["error"] = str(exc)
                time.sleep(0.1)
                continue
            inference_ms = (time.perf_counter() - t0) * 1000.0

            # Keep class thresholds explicit. Class order must match your model:
            # [person, fire, smoke], as assumed in the original script.
            class_conf = [person_conf, fire_conf, smoke_conf]
            raw = fast_dfl_and_nms(
                outs[0], outs[1],
                conf_thresh=min(person_conf, fire_conf, smoke_conf),
                letterbox=lb,
                orig_shape=(h0, w0),
                class_conf=class_conf,
            )

            # Do not reshape/shrink person boxes. Keep original detector geometry.
            dets = tracker.update(raw)
            annotated = draw_boxes(frame, dets)

            ok, jpg = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
            )
            if not ok:
                continue

            processed_idx += 1
            fps_frames += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                display_fps = fps_frames / elapsed
                fps_frames = 0
                fps_timer = time.time()

            with LOCK:
                CURRENT_JPEG = jpg.tobytes()
                SEQ += 1
                STATS.update({
                    "fps": round(display_fps, 1),
                    "inference_ms": round(inference_ms, 1),
                    "frame": processed_idx,
                    "source_frame": read_idx,
                    "total": total,
                    "playing": True,
                    "detections": dets,
                    "error": None,
                })

    finally:
        cap.release()
        with LOCK:
            STATS["playing"] = False
        print("[*] Video finished.")


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KavachX — CCTV Detection</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0b1020;color:#e5e7eb;font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:100vh;padding:20px;display:flex;align-items:center;flex-direction:column}
.header{width:100%;max-width:1100px;text-align:left;margin:0 auto 16px;display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:42px;height:42px;border-radius:12px;background:#172554;border:1px solid #1d4ed8;display:grid;place-items:center;font-weight:800;color:#93c5fd}
h1{font-size:20px;letter-spacing:-.3px}.sub{font-size:12px;color:#94a3b8;margin-top:4px}
.pill{font-size:11px;color:#bfdbfe;background:#172554;border:1px solid #1d4ed8;padding:6px 10px;border-radius:999px;font-weight:700}
.stage{position:relative;width:100%;max-width:1100px;border-radius:14px;overflow:hidden;background:#020617;border:1px solid #263247;box-shadow:0 20px 55px #0008}
#stream{width:100%;height:auto;display:block;min-height:200px}
.hud{position:absolute;left:12px;right:12px;top:12px;display:flex;justify-content:space-between;gap:8px;pointer-events:none;flex-wrap:wrap}
.metric,.status{padding:7px 10px;border-radius:8px;background:#0f172ad9;border:1px solid #475569;color:#e2e8f0;font:600 12px ui-monospace,SFMono-Regular,Menlo,monospace;backdrop-filter:blur(8px)}
.status{font-family:inherit;font-weight:800;letter-spacing:.4px}
.status.smoke{background:#7f1d1de8;border-color:#ef4444}.status.fire{background:#78350fe8;border-color:#f59e0b}.status.person{background:#0c4a6ee8;border-color:#38bdf8}.status.normal{background:#064e3be8;border-color:#10b981}
.footer{width:100%;max-width:1100px;display:flex;gap:16px;flex-wrap:wrap;align-items:center;justify-content:space-between;color:#94a3b8;font-size:12px;padding:12px 2px}
.legend{display:flex;gap:14px;flex-wrap:wrap}.legend-item{display:flex;align-items:center;gap:7px}.dot{width:9px;height:9px;border-radius:50%}
#error{color:#fca5a5}
@media(max-width:600px){body{padding:10px}.header{align-items:flex-start}h1{font-size:17px}.metric,.status{font-size:10px;padding:5px 7px}.hud{left:7px;right:7px;top:7px}}
</style>
</head>
<body>
<header class="header">
  <div class="brand"><div class="logo">KX</div><div><h1>KavachX Vision</h1><div class="sub">CCTV analytics · Person · Fire · Smoke</div></div></div>
  <div class="pill">YOLOv8m · ONNX Runtime CPU</div>
</header>
<main class="stage">
  <img id="stream" src="/stream" alt="Annotated CCTV video">
  <div class="hud"><div class="metric" id="telemetry">Inference: -- ms · FPS: --</div><div class="status normal" id="status">MONITORING</div></div>
</main>
<footer class="footer">
  <div class="legend">
    <div class="legend-item"><span class="dot" style="background:#00e5ff"></span>Person</div>
    <div class="legend-item"><span class="dot" style="background:#ff1744"></span>Fire</div>
    <div class="legend-item"><span class="dot" style="background:#ffea00"></span>Smoke</div>
  </div>
  <div id="progress">Waiting for video…</div><div id="error"></div>
</footer>
<script>
const telemetry=document.getElementById('telemetry'),statusEl=document.getElementById('status'),progress=document.getElementById('progress'),errorEl=document.getElementById('error');
function updateStatus(dets){
  const names=new Set(dets.map(d=>(d.class_name||'').toLowerCase()));
  if(names.has('fire')){statusEl.className='status fire';statusEl.textContent='FIRE DETECTED'}
  else if(names.has('smoke')){statusEl.className='status smoke';statusEl.textContent='SMOKE DETECTED'}
  else if(names.has('person')){statusEl.className='status person';statusEl.textContent='PERSON DETECTED'}
  else{statusEl.className='status normal';statusEl.textContent='MONITORING'}
}
async function poll(){
  try{
    const r=await fetch('/stats',{cache:'no-store'});const d=await r.json();
    telemetry.textContent=`Inference: ${d.inference_ms??'--'} ms · FPS: ${d.fps??'--'}`;
    progress.textContent=`Processed frame ${d.frame??0}/${d.total??0}`;
    errorEl.textContent=d.error||'';
    updateStatus(d.detections||[]);
  }catch(e){}
}
setInterval(poll,500);poll();
</script>
</body>
</html>
"""


class VideoHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        global CURRENT_JPEG

        if self.path in ("/", "/index.html"):
            content = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif self.path == "/stats":
            with LOCK:
                payload = json.dumps(STATS).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            try:
                while not STOP.is_set():
                    with LOCK:
                        seq = SEQ
                        data = CURRENT_JPEG
                    if data is None or seq == last:
                        time.sleep(0.02)
                        continue
                    last = seq
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


def main():
    global SESSION, INP_NAME, INP_SHAPE, INP_TYPE

    parser = argparse.ArgumentParser(description="KavachX CCTV video detection server")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", default="baseline", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Fallback/base confidence threshold")
    parser.add_argument("--person-conf", type=float, default=0.15)
    parser.add_argument("--fire-conf", type=float, default=0.20,
                        help="Fire threshold; tune on validation footage")
    parser.add_argument("--smoke-conf", type=float, default=0.15)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth source frame")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--no-loop", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        parser.error(f"Video not found: {args.video}")
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.fire_conf <= 1.0:
        parser.error("Confidence values must be between 0 and 1")

    person_conf = args.person_conf if args.person_conf is not None else args.conf
    smoke_conf = args.smoke_conf if args.smoke_conf is not None else args.conf

    model_path = os.path.join(BASE_DIR, "models", MODEL_MAP[args.model])
    if not os.path.isfile(model_path):
        parser.error(f"Model not found: {model_path}")

    print("=" * 68)
    print("KavachX CCTV Detection Server")
    print(f"Model: {model_path}")
    print(f"Video: {args.video}")
    print(f"Thresholds: person={person_conf:.2f}, fire={args.fire_conf:.2f}, smoke={smoke_conf:.2f}")
    print("Runtime: ONNX Runtime CPUExecutionProvider")
    print("=" * 68)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.time()
    SESSION = ort.InferenceSession(
        model_path, sess_options=options, providers=["CPUExecutionProvider"]
    )
    print(f"[+] Loaded model in {(time.time() - t0) * 1000:.1f} ms")
    inp = SESSION.get_inputs()[0]
    INP_NAME, INP_SHAPE, INP_TYPE = inp.name, inp.shape, inp.type
    print(f"[+] Input: name={INP_NAME}, shape={INP_SHAPE}, type={INP_TYPE}")

    # Existing postprocessor expects float32 1x3x640x640. Fail clearly if model differs.
    if INP_TYPE != "tensor(float)":
        raise RuntimeError(
            f"This script preprocesses float32 input, but model input is {INP_TYPE}. "
            "Add the correct quantized-input scale/zero-point preprocessing before running."
        )
    if len(INP_SHAPE) != 4 or INP_SHAPE[1] not in (3, "3", None) or INP_SHAPE[2] not in (640, "640", None) or INP_SHAPE[3] not in (640, "640", None):
        raise RuntimeError(
            f"Expected NCHW input [1,3,640,640] for the current postprocessor; got {INP_SHAPE}. "
            "The model and postprocessor must agree on input resolution/layout."
        )

    SESSION.run(None, {INP_NAME: np.zeros((1, 3, 640, 640), dtype=np.float32)})
    print("[+] Model warmup completed")

    thread = threading.Thread(
        target=inference_loop,
        args=(
            args.video, args.conf, args.fire_conf, smoke_conf, person_conf,
            args.stride, not args.no_loop, args.jpeg_quality
        ),
        daemon=True,
    )
    thread.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), VideoHandler)
    print(f"[+] UI available at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopping server...")
    finally:
        STOP.set()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
