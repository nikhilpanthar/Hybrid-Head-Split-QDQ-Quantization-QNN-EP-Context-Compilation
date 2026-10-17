#!/usr/bin/env python3
"""
KavachX Local Benchmark — Run INT8 QDQ model on CPU and measure latency/accuracy.
Supports baseline and refined 3-class models.
"""
import os
import sys
import time
import glob
import argparse
import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")

CLASSES = ["person", "smoke", "fire"]

MODEL_MAP = {
    "baseline": "3class_split_qdq.onnx",
    "refined": "refined_3class_split_qdq.onnx",
    "fp32": "new_3class_best_FP32.fixed.onnx",
}


def get_model_path(model_name):
    filename = MODEL_MAP.get(model_name)
    if not filename:
        print(f"[!] Unknown model: {model_name}. Choose from: {list(MODEL_MAP.keys())}")
        sys.exit(1)
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f"[!] Model not found: {path}")
        print(f"    Available models in {MODELS_DIR}:")
        for f in glob.glob(os.path.join(MODELS_DIR, "*.onnx")):
            print(f"      - {os.path.basename(f)} ({os.path.getsize(f) / 1e6:.1f} MB)")
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


def preprocess(img_path):
    orig = cv2.imread(img_path)
    if orig is None:
        raise ValueError(f"Could not read {img_path}")
    h0, w0 = orig.shape[:2]
    resized = cv2.resize(orig, (640, 640))
    blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob, (h0, w0), (1.0, 0.0, 0.0), orig


def postprocess(outs, h0, w0, r, dw, dh, conf_threshold=0.25, nms_threshold=0.45):
    dfl_out, scores_out = outs[0], outs[1]
    scores = scores_out[0].T  # [8400, 3]
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
        scale_x = w0 / 640.0
        scale_y = h0 / 640.0
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
        xywh = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes]
        indices = cv2.dnn.NMSBoxes(xywh, scores_filt.tolist(), conf_threshold, nms_threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            return boxes[indices], scores_filt[indices], cls_ids[indices]
    return np.array([]), np.array([]), np.array([])


def draw_detections(image, boxes, scores, cls_ids):
    COLORS = {0: (0, 255, 0), 1: (128, 128, 128), 2: (0, 0, 255)}
    for box, score, cid in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = COLORS.get(cid, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASSES[cid].upper()} {score * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return image


def main():
    parser = argparse.ArgumentParser(description="KavachX Local Benchmark")
    parser.add_argument("--model", default="baseline", choices=list(MODEL_MAP.keys()),
                        help="Model to test (default: baseline)")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Inference iterations for latency measurement (default: 30)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations (default: 5)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save", action="store_true",
                        help="Save detection images to output/ directory")
    args = parser.parse_args()

    model_path = get_model_path(args.model)
    model_size = os.path.getsize(model_path) / 1e6

    print(f"==========================================")
    print(f"  KavachX Local Benchmark")
    print(f"  Model: {os.path.basename(model_path)} ({model_size:.1f} MB)")
    print(f"  Device: CPU (ONNX Runtime)")
    print(f"==========================================")

    print(f"\n[*] Loading model...")
    t0 = time.time()
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    load_time = (time.time() - t0) * 1000
    print(f"[+] Model loaded in {load_time:.0f} ms")
    print(f"[+] Provider: {session.get_providers()}")

    images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")) + glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpeg")) + glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png")))
    if not images:
        print(f"[!] No test images found in {TEST_IMAGES_DIR}/")
        print(f"    Add .jpg/.jpeg/.png images to the test_images/ folder.")
        sys.exit(1)

    print(f"\n[*] Found {len(images)} test images")

    if args.save:
        os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)

    total_latencies = []

    for img_path in images:
        name = os.path.basename(img_path)
        blob, (h0, w0), (r, dw, dh), orig = preprocess(img_path)

        # Warmup
        for _ in range(args.warmup):
            session.run(None, {"images": blob})

        # Latency measurement
        latencies = []
        for _ in range(args.iterations):
            t_start = time.time()
            outs = session.run(None, {"images": blob})
            latencies.append((time.time() - t_start) * 1000)

        avg_lat = np.mean(latencies)
        min_lat = np.min(latencies)
        max_lat = np.max(latencies)
        fps = 1000.0 / avg_lat
        total_latencies.extend(latencies)

        # Postprocess
        boxes, scores, cls_ids = postprocess(outs, h0, w0, r, dw, dh, args.conf)

        print(f"\n=======================================================")
        print(f" Image: {name} ({w0}x{h0})")
        print(f" Latency: {avg_lat:.1f} ms (min: {min_lat:.1f}, max: {max_lat:.1f}) | FPS: {fps:.1f}")
        print(f" Active anchors (>25%): {len(np.max(outs[1][0].T, axis=1) >= 0.25)}")
        print(f" Final Detections (after NMS): {len(boxes)}")

        if len(boxes) > 0:
            for i, (box, score, cid) in enumerate(zip(boxes, scores, cls_ids)):
                print(f"   #{i+1} [{CLASSES[cid].upper()}] Conf: {score*100:.1f}% | Box: [{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]")

            if args.save:
                result = draw_detections(orig.copy(), boxes, scores, cls_ids)
                out_path = os.path.join(BASE_DIR, "output", f"detected_{name}")
                cv2.imwrite(out_path, result)
                print(f"   Saved: {out_path}")
        else:
            print("   No detections")
        print(f"=======================================================")

    # Summary
    overall_avg = np.mean(total_latencies)
    overall_fps = 1000.0 / overall_avg
    print(f"\n==========================================")
    print(f"  SUMMARY")
    print(f"==========================================")
    print(f"  Model:     {os.path.basename(model_path)} ({model_size:.1f} MB)")
    print(f"  Device:    CPU (ONNX Runtime)")
    print(f"  Images:    {len(images)}")
    print(f"  Iterations: {args.iterations} per image")
    print(f"  Avg Latency: {overall_avg:.1f} ms")
    print(f"  Avg FPS:     {overall_fps:.1f}")
    print(f"==========================================")


if __name__ == "__main__":
    main()