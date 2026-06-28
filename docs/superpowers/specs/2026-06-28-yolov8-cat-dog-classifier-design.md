# YOLOv8 猫狗分类器设计文档

## 目标

搭建一个可以在本机 Mac 上训练的 YOLOv8 分类项目，用来判断一张动物图片是猫还是狗。

本项目使用 Ultralytics YOLOv8 的分类权重 `yolov8n-cls.pt`。当前训练和预测都应使用 Conda 环境 `yolo8`，不要直接使用全局 Python 环境。

## 当前 Conda 环境

已确认当前机器存在 Conda 环境：

- 环境名：`yolo8`
- Python：3.8.20
- Ultralytics：8.4.80
- Torch：2.4.1
- MPS：用户终端中已确认可用

因此脚本默认使用 `device=auto`。运行时会优先使用 Apple GPU，也就是 `mps`；如果当前运行上下文中 MPS 不可用，则自动退回 `cpu`。

## 已有数据集

原始数据集已经在项目中：

- `dataset/PetImages/Cat`：12,500 张 JPG 图片
- `dataset/PetImages/Dog`：12,500 张 JPG 图片

数据集中包含 `Thumbs.db` 等非图片文件，也可能包含少量损坏图片。数据准备脚本必须验证图片能否被打开，坏图片只跳过，不中断整个流程。

原始数据集保持不变。

## 推荐方案

使用 YOLOv8 分类模型，而不是 YOLOv8 检测模型。

原因是当前目标是“判断整张图片是猫还是狗”。分类模型回答“这张图属于哪个类别”，检测模型回答“物体在哪里”。检测模型需要额外标注框，对这个任务来说成本更高，也没有必要。

## 项目结构

核心文件：

- `README.md`：中文新手教程，包含 Conda 环境、准备数据、训练、预测和验证说明。
- `requirements.txt`：备用依赖清单，不作为当前训练入口。
- `scripts/prepare_dataset.py`：验证图片并生成训练/验证数据集。
- `scripts/train.py`：调用 Ultralytics YOLOv8 分类训练。
- `scripts/predict.py`：加载训练好的权重并预测单张图片。
- `scripts/train_yolo8_conda.sh`：使用 `yolo8` Conda 环境训练。
- `scripts/predict_yolo8_conda.sh`：使用 `yolo8` Conda 环境预测。
- `.gitignore`：忽略数据集副本、训练输出、模型权重、缓存和虚拟环境。

准备后的训练数据目录：

- `data/cat_dog_cls/train/Cat`
- `data/cat_dog_cls/train/Dog`
- `data/cat_dog_cls/val/Cat`
- `data/cat_dog_cls/val/Dog`

## 数据准备

`scripts/prepare_dataset.py` 负责：

1. 从 `dataset/PetImages/Cat` 和 `dataset/PetImages/Dog` 读取原始图片。
2. 忽略 `Thumbs.db` 等非 JPG 文件。
3. 使用 Pillow 验证图片是否可读。
4. 使用固定随机种子拆分训练集和验证集。
5. 把可用图片复制到 YOLOv8 分类目录结构。
6. 输出每个类别的可用、跳过、训练、验证数量。

默认拆分比例：

- 训练集：80%
- 验证集：20%

## 训练

训练入口优先使用包装脚本：

```bash
./scripts/train_yolo8_conda.sh
```

它内部执行：

```bash
conda run -n yolo8 python scripts/train.py
```

训练默认参数：

- 模型：`yolov8n-cls.pt`
- 数据集：`data/cat_dog_cls`
- 设备：`auto`，优先 MPS，失败时退回 CPU
- 图片尺寸：`224`
- 轮数：`10`
- 批大小：`32`
- 运行名：`cat_dog_yolov8n`
- 默认输出：`runs/classify/cat_dog_yolov8n`

## 预测

预测入口优先使用包装脚本：

```bash
./scripts/predict_yolo8_conda.sh path/to/image.jpg
```

默认模型路径：

```text
runs/classify/cat_dog_yolov8n/weights/best.pt
```

脚本会输出预测类别和置信度。

## 错误处理

数据准备时，单张坏图不会导致整个任务失败，只会被统计为 skipped。

训练前会检查准备后的数据集目录是否存在，并检查 `train` 和 `val` 目录是否存在。

预测前会检查模型文件和图片文件是否存在，并用中文错误说明缺少哪个文件。

## 验证

当前项目已经完成：

1. 使用真实数据集准备 `data/cat_dog_cls`。
2. 确认猫狗训练/验证目录都存在。
3. 使用临时小数据集完成 YOLOv8 CPU 冒烟训练。
4. 使用冒烟训练模型完成一次单图预测。
5. 确认 `yolo8` Conda 环境中有 Ultralytics 和 Torch。

冒烟训练只证明流程能跑通，不代表模型已经训练好。正式模型需要使用完整数据集运行训练命令。
