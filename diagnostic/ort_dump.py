#!/usr/bin/env python3
"""
Dump all intermediate activations from palm_detection_lite.onnx via ORT.

Usage:
  python3 diagnostic/ort_dump.py images/test_images/hand_images/hand_000.png

Outputs go to diagnostic/dumps/<image_stem>/ as .npy files.
"""

import sys, os, pathlib
import numpy as np
from PIL import Image
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand', 'palm_detection_lite', 'source', 'palm_detection_lite.onnx')
PALM_SIZE = 192

def letterbox(img_path):
    """Replicate the browser's GPU letterbox: resize to fit 192x192, pad, normalize [0,1], NHWC."""
    img = Image.open(img_path).convert('RGB')
    src_w, src_h = img.size
    scale = PALM_SIZE / max(src_w, src_h)
    dst_w, dst_h = round(src_w * scale), round(src_h * scale)
    offset_x = (PALM_SIZE - dst_w) // 2
    offset_y = (PALM_SIZE - dst_h) // 2

    resized = img.resize((dst_w, dst_h), Image.BILINEAR)
    canvas = Image.new('RGB', (PALM_SIZE, PALM_SIZE), (0, 0, 0))
    canvas.paste(resized, (offset_x, offset_y))

    arr = np.array(canvas, dtype=np.float32) / 255.0  # [192, 192, 3] in [0,1]
    tensor = arr.reshape(1, PALM_SIZE, PALM_SIZE, 3)   # NHWC

    letterbox_info = {
        'src_w': src_w, 'src_h': src_h,
        'scale': scale, 'dst_w': dst_w, 'dst_h': dst_h,
        'offset_x': offset_x, 'offset_y': offset_y,
    }
    return tensor, letterbox_info


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
        out_dir = os.path.join(os.path.dirname(__file__), 'dumps', img_stem)
    os.makedirs(out_dir, exist_ok=True)

    # Preprocess
    tensor, lb_info = letterbox(img_path)
    np.save(os.path.join(out_dir, '_input_nhwc.npy'), tensor)

    # Also save as NCHW raw .bin for direct browser loading
    nchw = tensor.transpose(0, 3, 1, 2)  # [1,192,192,3] -> [1,3,192,192]
    nchw.tofile(os.path.join(out_dir, '_input_nchw.bin'))

    print(f"Input: {tensor.shape}, range [{tensor.min():.3f}, {tensor.max():.3f}]")
    print(f"Letterbox: {lb_info}")

    # Build debug model (all intermediates exposed)
    debug_model, intermediate_names = make_debug_model(MODEL_PATH)
    debug_model_path = os.path.join(out_dir, '_debug_model.onnx')
    onnx.save(debug_model, debug_model_path)
    print(f"Debug model with {len(intermediate_names)} intermediate outputs saved")

    # Run ORT (CPU for exact reproducibility)
    sess = ort.InferenceSession(debug_model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    results = sess.run(output_names, {input_name: tensor})

    # Save all activations
    name_to_idx = {}
    for i, node in enumerate(debug_model.graph.node):
        for out in node.output:
            if out:
                name_to_idx[out] = i

    # Build manifest for browser-side comparison
    manifest = {'image': os.path.basename(img_path), 'letterbox': lb_info, 'layers': []}

    saved = 0
    for name, data in zip(output_names, results):
        arr = np.array(data, dtype=np.float32)
        idx = name_to_idx.get(name, -1)
        safe_name = name.replace('/', '_').replace(':', '_').replace(';', '_')[:80]
        npy_file = f"{idx:03d}_{safe_name}.npy"
        bin_file = f"{idx:03d}_{safe_name}.bin"
        np.save(os.path.join(out_dir, npy_file), arr)
        arr.tofile(os.path.join(out_dir, bin_file))
        manifest['layers'].append({
            'idx': idx, 'name': name, 'shape': list(arr.shape),
            'min': float(arr.min()), 'max': float(arr.max()),
            'bin': bin_file,
        })
        saved += 1

    # Also save final outputs with clear names
    for name, data in zip(output_names[:2], results[:2]):
        arr = np.array(data)
        if arr.shape[-1] == 18:
            np.save(os.path.join(out_dir, '_regressors.npy'), arr)
            arr.astype(np.float32).tofile(os.path.join(out_dir, '_regressors.bin'))
            print(f"Regressors: {arr.shape}")
        elif arr.shape[-1] == 1:
            np.save(os.path.join(out_dir, '_scores.npy'), arr)
            arr.astype(np.float32).tofile(os.path.join(out_dir, '_scores.bin'))
            scores_sigmoid = 1 / (1 + np.exp(-arr.flatten()))
            top5 = np.sort(scores_sigmoid)[::-1][:5]
            print(f"Scores: {arr.shape}, top 5 after sigmoid: {top5}")

    import json
    with open(os.path.join(out_dir, '_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {saved} activation tensors + manifest to {out_dir}")
    return out_dir


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path> [image_path2 ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f"\n{'='*60}")
        print(f"Processing: {path}")
        print(f"{'='*60}")
        run_dump(path)
