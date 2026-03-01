# YOLOv8-RoadSegmentation — 基于 YOLOv8 的道路实例分割系统

[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)](https://ultralytics.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-GPU加速-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

> 软件杯竞赛参赛作品

## 项目背景

自动驾驶和辅助驾驶场景中，道路标线（车道线、实线等）的检测是基础且关键的任务。传统基于图像处理的方法（如霍夫变换）在复杂路况下鲁棒性不足，难以应对光照变化、标线磨损、遮挡等实际问题。

本项目使用 YOLOv8-seg 实例分割模型，对道路实线进行 **像素级** 的精准检测与分割，能够在视频流中实时运行并输出检测结果。通过 CUDA GPU 加速，在保证精度的同时达到了实时推理的要求。

## 主要功能

- 基于 YOLOv8n-seg 预训练模型微调，实现道路实线的像素级分割
- 支持视频文件和摄像头的实时检测推理
- CUDA GPU 加速，实时显示推理帧率
- 自定义掩码后处理（`maskmask.py`）
- 自动保存检测结果视频

## 项目结构

```
YOLOv8-RoadSegmentation/
├── train.py                 # 模型训练
├── detectvedio.py           # 视频推理（摄像头/视频文件）
├── maskmask.py              # 掩码后处理
├── myseg.yaml               # 数据集配置
├── yolov8-seg.yaml          # 模型架构配置
├── requirements.txt         # 依赖
├── data/
│   └── dataset_A/           # 训练数据（images + labels）
├── weights/
│   └── yolov8n-seg.pt       # 预训练权重
├── runs/                    # 训练输出
└── ultralytics/             # Ultralytics 框架
```

## 快速开始

**环境要求：** Python 3.8+、CUDA 11.x + cuDNN、显存 ≥ 4GB

```bash
# 安装依赖
pip install -r requirements.txt

# 训练模型（使用 yolov8n-seg.pt 预训练权重，100 epochs，输入 1280）
python train.py

# 视频推理（默认读取 data/example.mp4，CUDA 加速，置信度 0.6）
python detectvedio.py
```

## 配置说明

数据集配置（`myseg.yaml`）：

```yaml
path: data/dataset_A
train: images/train
val: images/val
test: images/test
names:
  0: solid_line
```

推理参数在 `detectvedio.py` 中调整：`conf`（置信度阈值）、`device`（推理设备）、`VideoCapture`（输入源）。

## 开源协议

MIT License
