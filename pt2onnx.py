#!/usr/bin/env python3
"""
pt2onnx.py

Convert a YOLOv8 `.pt` model to ONNX using `ultralytics.YOLO.export`.

Usage (PowerShell):
  python .\pt2onnx.py -i "runs\weights\best.pt" -o "runs\weights\best.onnx" --opset 12 --dynamic

Requirements:
  pip install ultralytics onnx==1.16.0
"""
import argparse
import os
import sys
from ultralytics import YOLO


def convert(pt_path: str, out_path: str = None, opset: int = 12, dynamic: bool = False, simplify: bool = False):
    if not os.path.isfile(pt_path):
        print(f"Error: file not found: {pt_path}")
        sys.exit(2)

    print(f"Loading model from: {pt_path}")
    model = YOLO(pt_path)

    kwargs = {"format": "onnx", "opset": opset}
    if dynamic:
        # ultralytics export accepts `dynamic` to enable dynamic axes
        kwargs["dynamic"] = True
    if simplify:
        # some ultralytics versions accept `simplify` to simplify the ONNX graph
        kwargs["simplify"] = True

    print(f"Exporting to ONNX with opset={opset}, dynamic={dynamic}, simplify={simplify} ...")
    result = model.export(**kwargs)

    # `model.export` returns path(s) or a list; try to determine saved path
    saved = None
    if isinstance(result, (list, tuple)) and result:
        saved = result[0]
    elif isinstance(result, str):
        saved = result

    if saved:
        # If user provided an output path, move/rename exported file
        if out_path:
            try:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                if os.path.abspath(saved) != os.path.abspath(out_path):
                    os.replace(saved, out_path)
                    saved = out_path
            except Exception as e:
                print(f"Warning: couldn't move exported file to desired output: {e}")

        print(f"ONNX exported to: {saved}")
    else:
        print("Export finished but couldn't determine output path. Check ultralytics export output.")


def parse_args():
    p = argparse.ArgumentParser(description="Convert YOLOv8 .pt to .onnx (ultralytics)")
    p.add_argument("-i", "--input", required=True, help="Path to input .pt file")
    p.add_argument("-o", "--output", required=False, help="Desired output .onnx path (optional)")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset version (default: 12)")
    p.add_argument("--dynamic", action="store_true", help="Enable dynamic axes for variable input size")
    p.add_argument("--simplify", action="store_true", help="Attempt to simplify the ONNX graph after export")
    return p.parse_args()


def main():
    args = parse_args()
    convert(args.input, args.output, opset=args.opset, dynamic=args.dynamic, simplify=args.simplify)


if __name__ == "__main__":
    main()
