# 🚗 YOLOv8-RoadSegmentation — 基于 YOLOv8 的道路实例分割系统

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/CUDA-GPU加速-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/🏆-软件杯竞赛作品-gold" alt="Competition">
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License">
</p>

> 🏆 **软件杯竞赛作品** — 基于 **YOLOv8-seg** 模型的道路场景实例分割系统，实现道路标线（实线）的像素级精准检测与分割。支持视频流实时推理、FPS 显示、检测结果视频输出。

---

## ✨ 功能特性

- 🎯 **实例分割** — 基于 YOLOv8n-seg 预训练模型微调，像素级分割精度
- 🎬 **视频推理** — 支持视频文件和摄像头的实时目标检测与分割
- ⚡ **GPU 加速** — 支持 CUDA 加速推理，实时显示 FPS
- 📊 **掩码处理** — 自定义掩码后处理脚本 `maskmask.py`
- 💾 **结果输出** — 自动保存检测结果视频

---

## 🏗️ 项目结构

```
YOLOv8-RoadSegmentation/
├── train.py                 # 模型训练脚本
├── detectvedio.py           # 视频检测推理脚本（支持摄像头/视频文件）
├── maskmask.py              # 掩码后处理工具
├── myseg.yaml               # 自定义数据集配置文件
├── yolov8-seg.yaml          # YOLOv8-seg 模型架构配置
├── requirements.txt         # Python 依赖
├── data/                    # 数据集目录
│   └── dataset_A/           # 训练数据（images + labels）
├── Label/                   # 标注文件
├── weights/                 # 预训练权重
│   └── yolov8n-seg.pt       # YOLOv8n 分割预训练模型
├── runs/                    # 训练输出（权重、日志、可视化）
├── ultralytics/             # Ultralytics 框架源码
└── rear/                    # 后处理相关资源
```

---

## 🚀 快速开始

### 环境要求
- Python 3.8+
- CUDA 11.x + cuDNN（推荐 GPU 加速）
- 显存 ≥ 4GB

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 2️⃣ 模型训练

```bash
python train.py
# 使用 yolov8n-seg.pt 预训练权重
# 训练 100 epochs，输入尺寸 1280
# 训练结果保存在 runs/segment/train/
```

### 3️⃣ 视频推理

```bash
python detectvedio.py
# 默认读取 data/example.mp4
# 使用 CUDA GPU 加速
# 置信度阈值：0.6
# 检测结果保存为 output.mp4
```

---

## ⚙️ 配置说明

### 数据集配置 (`myseg.yaml`)
```yaml
path: data/dataset_A       # 数据根目录
train: images/train         # 训练集
val: images/val             # 验证集
test: images/test           # 测试集

names:
  0: solid_line             # 实线检测类别
```

### 推理参数调整
在 `detectvedio.py` 中修改：
- `conf=0.6` — 置信度阈值
- `device="cuda:0"` — 推理设备
- `cap = cv2.VideoCapture(...)` — 输入源（视频路径或摄像头 ID）

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
