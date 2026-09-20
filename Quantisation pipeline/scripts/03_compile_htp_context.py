#!/usr/bin/env python3
"""
Step 3: Compile the INT8 QDQ ONNX model into a Qualcomm Hexagon HTP V68 Context Model (.ctx.onnx)
using ONNX Runtime with Qualcomm QNN Execution Provider (ORT-QNN).
This solves the FastRPC/DSP skeleton loading error encountered with standalone qnn-context-binary-generator.
"""
import os
import time
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

ep_lib_path = qnn_ep.get_library_path()
lib_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(lib_registration_name, ep_lib_path)

all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [d for d in all_ep_devices if d.ep_name == lib_registration_name]

qdq_model_path = os.path.expanduser("~/fire-smoke-q6a/models/refined_3class_split_qdq.onnx")
ctx_model_path = os.path.expanduser("~/fire-smoke-q6a/models/refined_3class-yolov8m-htp-ctx.onnx")

session_options = ort.SessionOptions()
session_options.add_session_config_entry("ep.context_enable", "1")
session_options.add_session_config_entry("ep.context_file_path", ctx_model_path)
session_options.add_session_config_entry("ep.context_embed_mode", "1")

ep_options = {
    "backend_path": qnn_ep.get_qnn_htp_path(),
    "htp_arch": "68",                 # V68 architecture for QCS6490
    "soc_model": "60",
    "htp_performance_mode": "burst"
}
session_options.add_provider_for_devices(selected_ep_devices, ep_options)

print(f"[*] Compiling QNN context binary: {qdq_model_path} -> {ctx_model_path}...")
t0 = time.time()
session = ort.InferenceSession(qdq_model_path, sess_options=session_options)
t_compile = (time.time() - t0) * 1000
print(f"[+] Compilation complete in {t_compile:.2f} ms!")
if os.path.exists(ctx_model_path):
    print(f"[+] Generated HTP context model size: {os.path.getsize(ctx_model_path)} bytes")

# Test loading the generated context model
print("\n[*] Testing direct load of compiled context model on Hexagon HTP...")
opts2 = ort.SessionOptions()
opts2.add_provider_for_devices(selected_ep_devices, ep_options)
t0 = time.time()
session2 = ort.InferenceSession(ctx_model_path, sess_options=opts2)
t_load = (time.time() - t0) * 1000
print(f"[+] Context model load time: {t_load:.2f} ms (down from {t_compile:.2f} ms!)")
print(f"[+] Active providers: {session2.get_providers()}")
