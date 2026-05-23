#!/usr/bin/env python3
"""
Convert an ONNX model to the .json + .bin format consumed by ModelRunner.

Usage:
  python3 engine/onnx_to_json.py models/hand/hand_landmark_large/source/hand_landmark_sparse_Nx3x224x224.onnx \
    --output-dir models/hand/hand_landmark_large \
    --name hand_landmark_large

Outputs:
  <output-dir>/<name>.json  -- graph structure, weight metadata
  <output-dir>/<name>.bin   -- flat float32 weight buffer
"""

import sys, os, argparse, json
import numpy as np
import onnx
from onnx import numpy_helper


def convert(onnx_path, output_dir, name):
    model = onnx.load(onnx_path)
    graph = model.graph

    # Collect initializers (weights/constants)
    initializers = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init).astype(np.float32)
        initializers[init.name] = arr

    # Find true inputs (not initializers)
    init_names = set(initializers.keys())
    true_inputs = []
    for inp in graph.input:
        if inp.name not in init_names:
            dims = []
            for d in inp.type.tensor_type.shape.dim:
                if d.dim_value > 0:
                    dims.append(d.dim_value)
                else:
                    dims.append(1)  # replace dynamic dims (N) with 1
            true_inputs.append({"name": inp.name, "shape": dims})

    if len(true_inputs) != 1:
        print(f"Warning: expected 1 input, found {len(true_inputs)}: {true_inputs}")
    model_input = true_inputs[0]

    # Outputs
    outputs = []
    for out in graph.output:
        dims = []
        for d in out.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                dims.append(1)
        outputs.append({"name": out.name, "shape": dims})

    # Pack all weights into a single flat buffer, tracking offsets
    weight_meta = {}
    weight_arrays = []
    offset = 0
    for wname, arr in initializers.items():
        flat = arr.flatten()
        length = len(flat)
        weight_meta[wname] = {
            "shape": list(arr.shape),
            "offset": offset,
            "length": length,
            "byte_offset": offset * 4,
            "byte_size": length * 4
        }
        weight_arrays.append(flat)
        offset += length

    all_weights = np.concatenate(weight_arrays) if weight_arrays else np.array([], dtype=np.float32)

    # Build graph node list
    nodes = []
    for node in graph.node:
        n = {
            "op": node.op_type,
            "name": node.name or node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attrs": {}
        }
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INTS:
                n["attrs"][attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                n["attrs"][attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.FLOATS:
                n["attrs"][attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.FLOAT:
                n["attrs"][attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                n["attrs"][attr.name] = attr.s.decode("utf-8")
            elif attr.type == onnx.AttributeProto.TENSOR:
                t = numpy_helper.to_array(attr.t)
                n["attrs"][attr.name] = t.tolist()
        nodes.append(n)

    # Assemble JSON
    result = {
        "model": name,
        "input": model_input,
        "outputs": outputs,
        "weight_buffer": {
            "total_floats": int(offset),
            "total_bytes": int(offset * 4)
        },
        "weights": weight_meta,
        "graph": nodes
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{name}.json")
    bin_path = os.path.join(output_dir, f"{name}.bin")

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    all_weights.astype(np.float32).tofile(bin_path)

    print(f"Model: {name}")
    print(f"Input: {model_input}")
    print(f"Outputs: {[o['name'] for o in outputs]}")
    print(f"Nodes: {len(nodes)}")
    print(f"Weights: {len(weight_meta)} tensors, {offset:,} floats, {offset*4:,} bytes")
    print(f"Ops: {sorted(set(n['op'] for n in nodes))}")
    print(f"Written: {json_path} ({os.path.getsize(json_path):,} bytes)")
    print(f"Written: {bin_path} ({os.path.getsize(bin_path):,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX model to ModelRunner .json + .bin format")
    parser.add_argument("onnx_path", help="Path to the .onnx file")
    parser.add_argument("--output-dir", required=True, help="Output directory for .json and .bin")
    parser.add_argument("--name", required=True, help="Model name (used for filenames)")
    args = parser.parse_args()
    convert(args.onnx_path, args.output_dir, args.name)
