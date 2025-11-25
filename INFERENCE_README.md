# YOLOv8s 烟头检测模型 — 推理仓库

## 快速开始

### 1. 克隆仓库
```bash
git clone <your-repo-url>
cd yanto
```

### 2. 创建虚拟环境并安装依赖
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 推理
```bash
# 推理图片文件夹
python infer_yolo.py --source dataset/images/test/ --save

# 推理单张图片
python infer_yolo.py --source your_image.jpg --save

# 推理视频
python infer_yolo.py --source your_video.mp4 --save

# 保存检测结果为 TXT 和置信度
python infer_yolo.py --source dataset/images/test/ --save --save_txt --save_conf
```

**结果保存在** `runs/infer/exp/` 中。

---

## 文件说明

- `infer_yolo.py` — 推理脚本（主要使用）
- `data.yaml` — 数据集配置（包含类别映射）
- `requirements.txt` — Python 依赖
- `runs/train/yolov8s-cigarette/weights/best.pt` — 训练好的模型权重
- `TECHNICAL_GUIDE.md` — 完整技术文档（包含训练、高级推理、导出等）

---

## 类别说明

模型检测 **2 个类别**：
- `0: yanto` — 烟头
- `1: w` — 其他

---

## 常见推理命令

| 需求 | 命令 |
|------|------|
| 推理文件夹（保存图片） | `python infer_yolo.py --source ./images/ --save` |
| 推理并保存标签 | `python infer_yolo.py --source ./images/ --save --save_txt` |
| 降低置信度（检测更多） | `python infer_yolo.py --source ./images/ --conf 0.3 --save` |
| 提高置信度（减少误检） | `python infer_yolo.py --source ./images/ --conf 0.5 --save` |
| 推理视频 | `python infer_yolo.py --source video.mp4 --save` |
| 指定 GPU 设备 | `python infer_yolo.py --source ./images/ --device 1 --save` |
| 使用 CPU | `python infer_yolo.py --source ./images/ --device cpu --save` |

---

## 推理参数说明

```bash
python infer_yolo.py \
  --source <path>              # 图片/视频/文件夹路径（必需）
  --model best.pt              # 模型权重（默认：最佳模型）
  --conf 0.25                  # 置信度阈值（0-1，默认 0.25）
  --iou 0.45                   # NMS IOU 阈值（默认 0.45）
  --device 0                   # GPU ID (0/1/...) 或 cpu
  --save                       # 保存结果图片
  --save_txt                   # 保存检测结果为 TXT
  --save_conf                  # 保存置信度
  --project runs/infer         # 输出文件夹
  --name exp                   # 实验名称
```

---

## 系统要求

- Python >= 3.8
- PyTorch >= 2.0（GPU 版本需要 CUDA 11.8+）
- 可选：GPU（NVIDIA）用于加速

---

## 故障排除

**问题：`ModuleNotFoundError: No module named 'ultralytics'`**
```bash
source .venv/bin/activate
pip install ultralytics torch torchvision
```

**问题：找不到模型文件**
```bash
# 检查权重文件
ls -la runs/train/yolov8s-cigarette/weights/best.pt
```

**问题：GPU 不可用**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# 若返回 False，使用 CPU：--device cpu
```

---

## 更多信息

- 查看完整文档：`TECHNICAL_GUIDE.md`
- 官方文档：https://docs.ultralytics.com/
- 问题反馈：提交 Issue

---

**模型信息**
- 架构：YOLOv8s
- 训练数据：烟头检测数据集
- 精度：见 `TECHNICAL_GUIDE.md` 第 5 节或 `runs/train/yolov8s-cigarette/results.png`
