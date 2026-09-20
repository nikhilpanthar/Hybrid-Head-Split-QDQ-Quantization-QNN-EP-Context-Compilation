#!/usr/bin/env python3
"""
Step 2: Static INT8 QDQ Quantization with MinMax per-channel calibration.
Sets opset to 13 and runs calibration over sample images.
"""
import os
import glob
import time
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    CalibrationMethod
)

input_fp32 = os.path.expanduser('~/fire-smoke-q6a/models/refined_3class_split_fp32.onnx')
opset13_fp32 = os.path.expanduser('~/fire-smoke-q6a/models/refined_3class_split_opset13.onnx')
output_qdq = os.path.expanduser('~/fire-smoke-q6a/models/refined_3class_split_qdq.onnx')

print(f'[*] Loading FP32 model from {input_fp32}...')
m = onnx.load(input_fp32)

for n in m.graph.node:
    if n.op_type == "Reshape":
        attrs = [a for a in n.attribute if a.name != "allowzero"]
        del n.attribute[:]
        n.attribute.extend(attrs)

print('[*] Setting opset_import to 13...')
del m.opset_import[:]
opset = m.opset_import.add()
opset.domain = ''
opset.version = 13
m.ir_version = 8

onnx.checker.check_model(m)
onnx.save(m, opset13_fp32)
print(f'[+] Saved opset 13 model to {opset13_fp32}')

class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.image_paths)
        try:
            img_path = next(self.enum_data)
            img = cv2.imread(img_path)
            if img is None:
                return self.get_next()
            img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
            blob = np.expand_dims(chw, axis=0)
            return {'images': blob}
        except StopIteration:
            return None

    def rewind(self):
        self.enum_data = None

calib_dir = os.path.expanduser('~/kavachx/quant_work/calibration_images/calibration_images')
calib_images = sorted(glob.glob(os.path.join(calib_dir, '*.jpg')))[:50]
print(f'[*] Found {len(calib_images)} calibration images')

data_reader = ImageCalibrationDataReader(calib_images)

print('[*] Starting Static QDQ INT8 Quantization (MinMax, per-channel)...')
t0 = time.time()
quantize_static(
    model_input=opset13_fp32,
    model_output=output_qdq,
    calibration_data_reader=data_reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
    calibrate_method=CalibrationMethod.MinMax,
    per_channel=True
)
t1 = time.time()
print(f'[+] Quantization complete in {t1 - t0:.2f}s!')
print(f'[+] Saved INT8 QDQ model to {output_qdq} ({os.path.getsize(output_qdq)} bytes)')

# Verify CPU load
sess = ort.InferenceSession(output_qdq, providers=['CPUExecutionProvider'])
print('[+] Verified CPU load! Inputs:', [i.name for i in sess.get_inputs()])
print('[+] Outputs:', [o.name for o in sess.get_outputs()])
