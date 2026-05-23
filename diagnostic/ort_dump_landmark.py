#!/usr/bin/env python3
"""
Dump all intermediate activations from hand_landmark_sparse via ORT.

Usage:
  python3 diagnostic/ort_dump_landmark.py images/test_images/hand_images/hand_000.png

The image is resized to 224x224 NCHW [0,1] -- simulating a perfect crop.
Outputs go to diagnostic/dumps/landmark_<image_stem>/ as .bin files.
"""

import sys, os, pathlib, json
import numpy as np
from PIL import Image
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand',
                          'hand_landmark_large', 'source',
                          'hand_landmark_sparse_Nx3x224x224.onnx')
SIZE = 224


def preprocess(img_path):
    """Resize to 224x224, normalize [0,1], return NCHW float32."""
    img = Image.open(img_path).convert('RGB').resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0   # [224, 224, 3]
    nchw = arr.transpose(2, 0, 1).reshape(1, 3, SIZE, SIZE)  # NCHW
    return nchw


def make_debug_model(model_path):
    """Clone the model and add every intermediate tensor as an output."""
    model = onnx.load(model_path)
    existing_outputs = {o.name for o in model.graph.output}
    intermediate_names = []

    for node in model.graph.node:
        for out in node.output:
            if out and out not in existing_outputs:
                intermediate_names.append(out)
                model.graph.output.append(
                    helper.make_tensor_value_info(out, TensorProto.FLOAT, None)
                )

    return model, intermediate_names


def run_dump(img_path, out_dir=None):
    img_stem = pathlib.Path(img_path).stem
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), 'dumps', f'landmark_{img_stem}')
    os.makedirs(out_dir, exist_ok=True)

    nchw = preprocess(img_path)
    nchw.tofile(os.path.join(out_dir, '_input_nchw.bin'))
    print(f"Input: {nchw.shape}, range [{nchw.min():.3f}, {nchw.max():.3f}]")

    debug_model, intermediate_names = make_debug_model(MODEL_PATH)
    debug_model_path = os.path.join(out_dir, '_debug_model.onnx')
    onnx.save(debug_model, debug_model_path)
    print(f"Debug model with {len(intermediate_names)} intermediate outputs saved")

    sess = ort.InferenceSession(debug_model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    results = sess.run(output_names, {input_name: nchw})

    name_to_idx = {}
    for i, node in enumerate(debug_model.graph.node):
        for out in node.output:
            if out:
                name_to_idx[out] = i

    manifest = {'image': os.path.basename(img_path), 'model': 'hand_landmark_sparse', 'layers': []}

    saved = 0
    for name, data in zip(output_names, results):
        arr = np.array(data, dtype=np.float32)
        idx = name_to_idx.get(name, -1)
        safe_name = name.replace('/', '_').replace(':', '_').replace(';', '_')[:80]
        bin_file = f"{idx:03d}_{safe_name}.bin"
        arr.tofile(os.path.join(out_dir, bin_file))
        manifest['layers'].append({
            'idx': idx, 'name': name, 'shape': list(arr.shape),
            'min': float(arr.min()), 'max': float(arr.max()),
            'bin': bin_file,
        })
        saved += 1

    # Print final outputs
    original_outputs = onnx.load(MODEL_PATH).graph.output
    orig_names = {o.name for o in original_outputs}
    for name, data in zip(output_names, results):
        if name in orig_names:
            arr = np.array(data)
            print(f"  Output '{name}': shape={arr.shape} range=[{arr.min():.4f}, {arr.max():.4f}] first5={arr.flatten()[:5]}")

    with open(os.path.join(out_dir, '_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {saved} activation tensors + manifest to {out_dir}")
    return out_dir


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path>")
        sys.exit(1)
    for path in sys.argv[1:]:
        print(f"\n{'='*60}")
        print(f"Processing: {path}")
        print(f"{'='*60}")
        run_dump(path)
