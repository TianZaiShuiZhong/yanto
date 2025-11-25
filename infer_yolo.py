#!/usr/bin/env python3
#!/usr/bin/env python3
"""
YOLOv8 香烟检测模型推理脚本。

使用示例：
  # 检测单张图片
  python infer_yolo.py --source dataset/images/test/001.jpg
  
  # 检测测试集所有图片
  python infer_yolo.py --source dataset/images/test/
  
  # 检测视频文件
  python infer_yolo.py --source video.mp4
  
  # 使用摄像头实时检测 (0 = 默认摄像头)
  python infer_yolo.py --source 0
  
  # 使用自定义训练模型
  python infer_yolo.py --source dataset/images/test/ --model runs/train/yolov8s-cigarette/weights/best.pt
"""
import argparse
from ultralytics import YOLO
import os


def parse_args():
    p = argparse.ArgumentParser(description='YOLOv8 Inference')
    p.add_argument('--source', type=str, required=True, help='image/video path, folder, or camera id (0)')
    p.add_argument('--model', type=str, default='runs/train/yolov8s-cigarette/weights/best.pt',
                   help='path to trained model weights')
    p.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    p.add_argument('--iou', type=float, default=0.45, help='NMS IOU threshold')
    p.add_argument('--device', type=str, default='0', help='CUDA device (0) or cpu')
    p.add_argument('--save', action='store_true', help='save detection results')
    p.add_argument('--save_txt', action='store_true', help='save results as txt')
    p.add_argument('--save_conf', action='store_true', help='save confidence scores')
    p.add_argument('--project', type=str, default='runs/infer', help='save folder')
    p.add_argument('--name', type=str, default='exp', help='run name')
    p.add_argument('--line_width', type=int, default=2, help='bounding box line width')
    p.add_argument('--visualize', action='store_true', help='visualize features')
    return p.parse_args()


def main():
    args = parse_args()
    
    print('Inference config:')
    print(' source:', args.source)
    print(' model:', args.model)
    print(' conf:', args.conf)
    print(' device:', args.device)
    print()
    
    # Load model
    model = YOLO(args.model)
    
    # Run inference
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
        line_width=args.line_width,
        visualize=args.visualize,
    )
    
    # Print summary
    print(f'\nDetection complete. {len(results)} image(s) processed.')
    if args.save:
        print(f'Results saved to: {args.project}/{args.name}/')


if __name__ == '__main__':
    main()
