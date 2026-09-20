#!/usr/bin/env python3
"""
Step 1: Split YOLOv8m detection heads into:
  - Bounding box regression / DFL tensor: [1, 64, 8400]
  - Classification scores: [1, 3, 8400] (person, fire, smoke)
"""
import os
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

orig_path = os.path.expanduser('~/fire-smoke-q6a/models/refined_3class_L4_FP32.onnx')
split_path = os.path.expanduser('~/fire-smoke-q6a/models/refined_3class_split_fp32.onnx')

print(f'Loading {orig_path}...')
model = onnx.load(orig_path)
graph = model.graph

target_output_names = {'/model.22/Concat_output_0', '/model.22/Sigmoid_output_0'}
needed_tensors = set(target_output_names)
nodes_to_keep = []

for node in reversed(graph.node):
    if any(out in needed_tensors for out in node.output):
        nodes_to_keep.append(node)
        for inp in node.input:
            if inp:
                needed_tensors.add(inp)

nodes_to_keep.reverse()
initializers_to_keep = [init for init in graph.initializer if init.name in needed_tensors]

boxes_out = helper.make_tensor_value_info(
    '/model.22/Concat_output_0',
    TensorProto.FLOAT,
    [1, 64, 8400]
)
scores_out = helper.make_tensor_value_info(
    '/model.22/Sigmoid_output_0',
    TensorProto.FLOAT,
    [1, 3, 8400]
)

new_graph = helper.make_graph(
    nodes=nodes_to_keep,
    name='refined_3class_split',
    inputs=list(graph.input),
    outputs=[boxes_out, scores_out],
    initializer=initializers_to_keep
)

new_model = helper.make_model(new_graph, producer_name='refined_split_generator', opset_imports=model.opset_import)
onnx.checker.check_model(new_model)
onnx.save(new_model, split_path)
print(f'[+] Saved split model to {split_path}')
print(f'    Nodes kept: {len(nodes_to_keep)} (out of {len(graph.node)})')

# Verify with CPU inference
sess = ort.InferenceSession(split_path, providers=['CPUExecutionProvider'])
dummy_inp = np.random.randn(1, 3, 640, 640).astype(np.float32)
outs = sess.run(None, {'images': dummy_inp})
print('[+] Verified CPU inference test:')
print(f'    boxes shape : {outs[0].shape}')
print(f'    scores shape: {outs[1].shape}')
