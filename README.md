# YOLOv8 猫狗分类训练项目

这个项目会训练一个模型，用来判断一张动物图片是猫还是狗。

我们使用的是 YOLOv8 的“分类模型”，不是“检测模型”。分类模型回答“这张图是什么类别”，检测模型回答“图中物体在哪里”。你的需求是判断猫还是狗，所以分类模型更合适。

## 数据集

原始数据放在：

- `dataset/PetImages/Cat`
- `dataset/PetImages/Dog`

原始数据不会被修改。脚本会把可用图片复制到 `data/cat_dog_cls`，并自动拆分训练集和验证集。

## 安装依赖

建议使用 Python 3.11 或 3.12 创建虚拟环境。当前机器默认 Python 是 3.14，很多机器学习库可能还没有完全适配，所以不建议直接用系统默认 Python 训练。

先检查你是否已经安装了合适的 Python：

```bash
python3.12 --version
python3.11 --version
```

如果其中一个命令能正常显示版本号，就用对应版本创建虚拟环境。下面以 `python3.12` 为例：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

如果你的电脑暂时只有 `python3`，可以先继续阅读项目结构，但真正训练前建议先安装 Python 3.11 或 3.12。

## 准备数据

```bash
python scripts/prepare_dataset.py
```

这一步会检查图片能不能被正常打开，然后把可用图片复制到 YOLOv8 分类训练需要的目录结构里。

## 训练模型

```bash
python scripts/train.py
```

训练完成后，最好的模型通常会保存在：

```text
runs/classify/cat_dog_yolov8n/weights/best.pt
```

## 预测单张图片

```bash
python scripts/predict.py dataset/PetImages/Cat/0.jpg
```

脚本会输出模型认为这张图片是 `Cat` 还是 `Dog`，以及对应置信度。
