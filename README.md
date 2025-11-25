# YOLOv8s 烟头检测模型 — 推理仓库

## 快速开始

**项目结构**

> ```
> yanto/
> ├── README.md                      # 推理快速开始指南 
> ├── infer_yolo.py                  # 推理脚本
> ├── dataset                        # 训练集
> ├── train_yolo.py                  # 训练脚本（可选）
> ├── data.yaml                      # 数据配置
> ├── requirements.txt               # 依赖
> ├── setup.py                       # 包安装配置 
> ├── .gitignore                     # Git 忽略规则 
> ├── dataset/images/test/           # 示例测试图片（可选）
> └── runs/train/yolov8s-cigarette/
>     └── weights/
>         └── best.pt                # 模型权重 
> ```

## 系统要求

- Python >= 3.8
- PyTorch >= 2.0（GPU 版本需要 CUDA 11.8+）
- 可选：GPU（NVIDIA）用于加速

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd yanto
```

### 2. 创建虚拟环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

**总步骤**

```
python -m venv .venv

.venv\Scripts\activate; pip install --upgrade pip

.venv\Scripts\python.exe -m pip install --upgrade pip

.venv\Scripts\python.exe -m pip install -r requirements.txt

.venv\Scripts\python.exe -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"          #确认CUDA不可用：


.venv\Scripts\python.exe infer_yolo.py --source dataset/images/train/ --model runs/weights/best.pt --save --device cpu
```

### 3. 推理

**不用gpu的话**：

```
infer_yolo.py --source dataset/images/train/ --model runs/weights/best.pt --save --device cpu
```



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

**==结果保存==在** `runs/infer/exp/` 中。

---

## 文件说明

- `infer_yolo.py` — 推理脚本（主要使用）
- `data.yaml` — 数据集配置（包含类别映射）
- `requirements.txt` — Python 依赖
- `runs/train/yolov8s-cigarette/weights/best.pt` — 训练好的模型权重
- `TECHNICAL_GUIDE.md` — 完整技术文档（包含训练、高级推理、导出等）

---

## 类别说明

模型检测 **1个类别**：

- `0: yanto` — 烟头

---

## 常见推理命令

| 需求                   | 命令                                                         |
| ---------------------- | ------------------------------------------------------------ |
| 推理文件夹（保存图片） | `python infer_yolo.py --source ./images/ --save`             |
| 推理并保存标签         | `python infer_yolo.py --source ./images/ --save --save_txt`  |
| 降低置信度（检测更多） | `python infer_yolo.py --source ./images/ --conf 0.3 --save`  |
| 提高置信度（减少误检） | `python infer_yolo.py --source ./images/ --conf 0.5 --save`  |
| 推理视频               | `python infer_yolo.py --source video.mp4 --save`             |
| 指定 GPU 设备          | `python infer_yolo.py --source ./images/ --device 1 --save`  |
| 使用 CPU               | `python infer_yolo.py --source ./images/ --device cpu --save` |

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
