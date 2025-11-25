#!/usr/bin/env python3
"""
Simple training wrapper for Ultralytics YOLOv8.

Usage examples:
  # attach to GPU tmux session first (if needed):
  # tmux attach -t gpu
  python train_yolo.py --epochs 100 --batch 16 --model yolov8s.pt

"""
import argparse
from ultralytics import YOLO
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data.yaml', help='path to data.yaml')
    p.add_argument('--model', default='yolov8s.pt', help='pretrained model or yaml')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0', help="CUDA device (e.g. '0' or 'cuda:0'), or 'cpu'")
    p.add_argument('--project', default='runs/train', help='save to project/name')
    p.add_argument('--name', default='yolov8s-cigarette', help='run name')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--exist_ok', action='store_true', help='overwrite existing run')
    return p.parse_args()


def main():
    args = parse_args()

    print('Training with config:')
    print(' data:', args.data)
    print(' model:', args.model)
    print(' epochs:', args.epochs)
    print(' batch:', args.batch)
    print(' imgsz:', args.imgsz)
    print(' device:', args.device)

    # Create project folder if needed
    os.makedirs(args.project, exist_ok=True)

    # Initialize model
    model = YOLO(args.model)

    # Start training
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == '__main__':
    main()
