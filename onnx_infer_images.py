#!/usr/bin/env python3
"""
Run inference using an exported ONNX YOLOv8 model via `ultralytics.YOLO`.

This script loads the ONNX model (Ultralytics supports ONNX) and runs
predictions on a folder of images, saving results to `runs/infer/`.

Usage (PowerShell):
  .venv\Scripts\Activate.ps1
  python .\onnx_infer_images.py -m .\runs\weights\best.onnx -s .\dataset\images\train --device cpu --save
"""
import argparse
from ultralytics import YOLO


def run_infer(onnx_path: str, source: str, device: str = 'cpu', imgsz: int = 640, conf: float = 0.25, save: bool = True, save_txt: bool = False):
    print(f"Loading model: {onnx_path}")
    model = YOLO(onnx_path)

    print(f"Running predict on source: {source}")
    results = model.predict(source=source, device=device, imgsz=imgsz, conf=conf, save=save, save_txt=save_txt)

    print("Done. Predictions saved to runs/infer/... (or printed).")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", required=True, help="Path to ONNX model")
    p.add_argument("-s", "--source", required=True, help="Image or folder path to run inference on")
    p.add_argument("--device", default="cpu", help="Device to run on, e.g., cpu or 0 for GPU id")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--save-txt", action="store_true", dest="save_txt", help="Save results to txt")
    p.add_argument("--no-save", dest="save", action="store_false", help="Do not save annotated images")
    return p.parse_args()


def main():
    args = parse_args()
    run_infer(args.model, args.source, device=args.device, imgsz=args.imgsz, conf=args.conf, save=args.save, save_txt=args.save_txt)


if __name__ == '__main__':
    main()
