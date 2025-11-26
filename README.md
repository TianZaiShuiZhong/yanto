# YOLOv8s 烟头检测模型 — 推理仓库

## 快速开始

**项目结构**

> ```
> yanto/
> ├── README.md                      # 推理快速开始指南 
> ├── infer_yolo.py                  # .bt模型推理脚本
> ├── dataset                        # 训练集
> ├── train_yolo.py                  # 训练脚本（可选）
> ├── data.yaml                      # 数据配置
> ├── requirements.txt               # 依赖
> ├── setup.py                       # 包安装配置 
> ├── .gitignore                     # Git 忽略规则 
> ├── dataset/images/test/           # 示例测试图片（可选）
> ├── pt2onnx.py                     # 用于导出 .pt 到 .onnx 格式的辅助脚本
> ├── test_onnx_infer.py             # 简单的 ONNX 运行时前向传播测试程序
> ├── onnx_infer_images.py           # 使用 ONNX 进行批量推理
> └── runs/
>     ├── weights/
>     |   ├── best.pt                # 训练得到的 PyTorch 权重
>     |   └── best.onnx               # 导出的 ONNX 权重(由 `pt2onnx.py` / export 生成
>     └── train/yolov8s-cigarette/
>         └── weights/                # 训练过程中保存的权重目录
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


bt推理：
.venv\Scripts\python.exe infer_yolo.py --source dataset/images/train/ --model runs/weights/best.pt --save --device cpu

onnx推理：
python .\onnx_infer_images.py -m .\runs\weights\best.onnx -s .\dataset\images\train --device cpu --imgsz 640 --conf 0.25
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

## 导出为 ONNX（补充）

下面给出将 `runs\weights\best.pt` 导出为 ONNX 的方法与在 Windows PowerShell 下的详细命令。仓库中已包含辅助脚本 `pt2onnx.py`，可以直接使用。

1) 在仓库根目录创建并激活虚拟环境：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2) 安装依赖（若 `onnx==1.16.0` 在你的 Python 版本上无法直接安装，ultralytics 会尝试安装兼容版本）：

```powershell
pip install -r requirements.txt
pip install ultralytics onnx==1.16.0 onnxruntime onnxslim
```

3) 使用仓库脚本导出（推荐）或直接调用 `ultralytics` API：

```powershell
# 推荐（仓库自带脚本，支持指定输出路径与参数）
python .\pt2onnx.py -i ".\runs\weights\best.pt" -o ".\runs\weights\best.onnx" --opset 12 --dynamic

# 或直接使用 ultralytics API（Python 交互式或脚本）：
python - <<'PY'
from ultralytics import YOLO
model = YOLO(r"runs\weights\best.pt")
model.export(format='onnx', opset=12, dynamic=True)
PY
```

导出成功后示例文件：`runs\weights\best.onnx`。

提示：如果 pip 在安装 `onnx` 时尝试从源码编译失败，请尝试使用 Python 3.10/3.11 或直接使用 `ultralytics` 的 AutoUpdate（脚本运行时会自动安装兼容的 onnx/onnxruntime）。

## 使用 ONNX 运行推理（补充）

仓库中已包含两个用于测试和批量推理的脚本：
- `test_onnx_infer.py`：用 `onnxruntime` 执行一次 dummy forward，验证模型可加载并检查输出形状。
- `onnx_infer_images.py`：通过 `ultralytics.YOLO` 加载 ONNX 模型并对图片文件夹做批量推理，结果保存到 `runs/detect/predict`（或 `runs/infer`，取决于 ultralytics 版本）。

示例 PowerShell 命令（在激活的虚拟环境中运行）：

```powershell
.venv\Scripts\Activate.ps1

# 使用 ultralytics 脚本对图片文件夹推理并保存结果
python .\onnx_infer_images.py -m .\runs\weights\best.onnx -s .\dataset\images\train --device cpu --imgsz 640 --conf 0.25

# 快速验证 ONNX 模型（dummy 输入）
python .\test_onnx_infer.py -m .\runs\weights\best.onnx --size 640
```

可选：若机器有 NVIDIA GPU 并且你已安装 `onnxruntime-gpu`，可在 `--device` 中指定 GPU 加速（或使用 `ultralytics` CLI 指定 `device=0`）。

可视化：要查看 ONNX 模型结构，请使用 Netron（https://netron.app）并打开 `runs\weights\best.onnx`。
