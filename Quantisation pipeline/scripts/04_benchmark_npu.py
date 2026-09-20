#!/usr/bin/env python3
"""
Step 4: Benchmark inference latency (ms), FPS, and accuracy detections directly
on the Qualcomm Hexagon NPU using sample test images.
"""
import os
import time
import glob
import cv2
import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

CLASSES = ["person", "fire", "smoke"]
MODEL_PATH = os.path.expanduser("~/fire-smoke-q6a/models/refined_3class-yolov8m-htp-ctx.onnx")
TEST_IMAGES_DIR = os.path.expanduser("~/kavachx/kavachx_testing/test_images")

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
    r = min(640 / h0, 640 / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (640 - nw) / 2, (640 - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    resized = cv2.resize(orig, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob, (h0, w0), (r, dw, dh)

def main():
    print(f"[*] Initializing QNN Execution Provider on Hexagon HTP...")
    ep_lib_path = qnn_ep.get_library_path()
    ort.register_execution_provider_library("QNNExecutionProvider", ep_lib_path)
    selected_ep = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    
    session_options = ort.SessionOptions()
    ep_options = {
        "backend_path": qnn_ep.get_qnn_htp_path(),
        "htp_arch": "68",
        "soc_model": "60",
        "htp_performance_mode": "burst"
    }
    session_options.add_provider_for_devices(selected_ep, ep_options)
    
    t0 = time.time()
    session = ort.InferenceSession(MODEL_PATH, sess_options=session_options)
    print(f"[+] Loaded on Hexagon NPU in {(time.time() - t0)*1000:.2f} ms!")

    images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")))
    if not images:
        images = sorted(glob.glob("/home/test_user/kavachx_testing/test_images/*.jpg"))

    for img_path in images:
        name = os.path.basename(img_path)
        blob, (h0, w0), (r, dw, dh) = preprocess(img_path)

        # Warmup
        for _ in range(5):
            session.run(None, {'images': blob})

        # Latency Benchmark (30 iterations)
        latencies = []
        for _ in range(30):
            t_start = time.time()
            outs = session.run(None, {'images': blob})
            latencies.append((time.time() - t_start) * 1000)

        avg_lat = np.mean(latencies)
        min_lat = np.min(latencies)
        fps = 1000.0 / avg_lat

        # Postprocess detections
        dfl_out, scores_out = outs[0], outs[1]
        scores = scores_out[0].T  # [8400, 3]
        bbox_delta = dfl_decode(dfl_out)[0]
        anchors, strides = make_anchors()

        x1 = (anchors[:, 0] - bbox_delta[0, :]) * strides[:, 0]
        y1 = (anchors[:, 1] - bbox_delta[1, :]) * strides[:, 0]
        x2 = (anchors[:, 0] + bbox_delta[2, :]) * strides[:, 0]
        y2 = (anchors[:, 1] + bbox_delta[3, :]) * strides[:, 0]
        boxes = np.stack([x1, y1, x2, y2], axis=-1)

        # Confidence filtering
        cls_ids = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)
        mask = max_scores >= 0.25
        boxes = boxes[mask]
        scores_filt = max_scores[mask]
        cls_ids = cls_ids[mask]

        print(f"\n=======================================================")
        print(f" Image: {name}")
        print(f" NPU Hardware Latency: {avg_lat:.2f} ms (Min: {min_lat:.2f} ms) | Throughput: {fps:.1f} FPS")
        print(f" Active anchors (>25%): {len(boxes)}")

        if len(boxes) > 0:
            # Scale to original
            boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - dw) / r, 0, w0)
            boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - dh) / r, 0, h0)
            xywh = [[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in boxes]
            indices = cv2.dnn.NMSBoxes(xywh, scores_filt.tolist(), 0.25, 0.45)
            
            print(f" Final Detections (after NMS): {len(indices)}")
            if len(indices) > 0:
                for idx in indices.flatten():
                    c_name = CLASSES[cls_ids[idx]]
                    print(f"   #{idx+1} [{c_name.upper()}] Conf: {scores_filt[idx]*100:.1f}% | Box: {boxes[idx].tolist()}")
        else:
            print(" Final Detections: 0")
        print("=======================================================")

if __name__ == "__main__":
    main()
