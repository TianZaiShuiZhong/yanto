#!/usr/bin/env python3
"""
Simple ONNX inference tester for exported YOLOv8 model.

Usage:
  python test_onnx_infer.py -m runs\weights\best.onnx --size 640

This script loads the ONNX model with `onnxruntime`, creates a dummy input
matching the model's first input shape (replacing dynamic dims with `1` or
the provided `--size`), runs a forward pass and prints output shapes.
"""
import argparse
import numpy as np
import onnxruntime as ort
import os
import sys


def make_dummy_input(shape, size=640):
    # Convert shape entries to ints; replace dynamic or symbolic dims
    new_shape = []
    for d in shape:
        if d is None:
            new_shape.append(1)
        else:
            try:
                di = int(d)
                # if channel dim == 3 keep it
                new_shape.append(di)
            except Exception:
                # symbolic like 'batch' or 'num' or negative
                # treat spatial dims as `size` and batch as 1
                new_shape.append(1 if len(new_shape) == 0 else size if len(new_shape) >= 2 else 1)
    # Ensure at least 4 dims [N,C,H,W]
    if len(new_shape) == 3:
        new_shape = [1] + new_shape
    if len(new_shape) == 2:
        new_shape = [1, new_shape[0], size, size]
    return np.random.rand(*new_shape).astype(np.float32), new_shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", required=True, help="Path to ONNX model")
    p.add_argument("--size", type=int, default=640, help="Spatial size for dummy input (default 640)")
    args = p.parse_args()

    if not os.path.isfile(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(2)

    print(f"Loading ONNX model: {args.model}")
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    inp = sess.get_inputs()[0]
    name = inp.name
    shape = inp.shape
    print(f"Model first input name: {name}, shape: {shape}, type: {inp.type}")

    dummy, used_shape = make_dummy_input(shape, size=args.size)
    print(f"Using dummy input shape: {used_shape}, dtype: {dummy.dtype}")

    print("Running inference...")
    out = sess.run(None, {name: dummy})

    print(f"Got {len(out)} output(s)")
    for i, o in enumerate(out):
        try:
            arr = np.array(o)
            print(f" output[{i}] shape: {arr.shape}, dtype: {arr.dtype}")
            # print a small summary
            flat = arr.flatten()
            print(f"  sample values: min={flat.min():.6f}, max={flat.max():.6f}, mean={flat.mean():.6f}")
        except Exception as e:
            print(f"  Could not parse output[{i}]: {e}")


if __name__ == "__main__":
    main()
