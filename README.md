# YOLOv8 猫狗分类训练项目

这个项目会训练一个模型，用来判断一张动物图片是猫还是狗。

我们使用的是 YOLOv8 的“分类模型”，不是“检测模型”。分类模型回答“这张图是什么类别”，检测模型回答“图中物体在哪里”。你的需求是判断猫还是狗，所以分类模型更合适。

## 数据集

原始数据放在：

- `dataset/PetImages/Cat`
- `dataset/PetImages/Dog`

原始数据不会被修改。脚本会把可用图片复制到 `data/cat_dog_cls`，并自动拆分训练集和验证集。

## 安装依赖

建议使用 Python 3.11 或 3.12 创建虚拟环境。当前机器默认 Python 是 3.14；本项目已经在这台机器的 Python 3.14 虚拟环境里完成了依赖安装和 YOLOv8 冒烟训练，但如果你以后遇到兼容性问题，优先换成 Python 3.11 或 3.12。

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

如果你的电脑暂时只有 `python3`，也可以这样创建当前项目的虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 准备数据

```bash
python scripts/prepare_dataset.py
```

这一步会检查图片能不能被正常打开，然后把可用图片复制到 YOLOv8 分类训练需要的目录结构里。

在当前数据集上的验证结果是：

- `Cat`: 12,499 张可用图片，1 张损坏图片被跳过，9,999 张用于训练，2,500 张用于验证。
- `Dog`: 12,499 张可用图片，1 张损坏图片被跳过，9,999 张用于训练，2,500 张用于验证。

## 训练模型

```bash
python scripts/train.py
```

当前脚本默认使用 `device=cpu`。原因是这台机器上已经验证过 Torch 可以训练，但 `torch.backends.mps.is_available()` 返回 `False`，也就是当前 Python/Torch 环境暂时不能使用 Apple Silicon 的 MPS 加速。

如果你之后换了支持 MPS 的 PyTorch 环境，可以这样尝试加速：

```bash
python scripts/train.py --device mps
```

如果 MPS 报错，就继续使用默认 CPU。CPU 会慢一些，但更稳定。

训练完成后，最好的模型通常会保存在：

```text
runs/classify/cat_dog_yolov8n/weights/best.pt
```

## 预测单张图片

```bash
python scripts/predict.py dataset/PetImages/Cat/0.jpg
```

脚本会输出模型认为这张图片是 `Cat` 还是 `Dog`，以及对应置信度。

预测脚本默认读取训练输出中的 `best.pt`。如果你的模型文件在其他位置，可以这样指定：

```bash
python scripts/predict.py path/to/image.jpg --model path/to/best.pt
```

如果你用 MPS 训练出了模型，也可以预测时指定 MPS：

```bash
python scripts/predict.py path/to/image.jpg --model path/to/best.pt --device mps
```

## 已完成的本地验证

当前项目已经完成这些验证：

- 单元测试通过：数据准备、训练前路径检查、预测前路径检查。
- 真实数据准备通过：已生成 `data/cat_dog_cls/train` 和 `data/cat_dog_cls/val`。
- YOLOv8 冒烟训练通过：使用临时小数据集完成了 1 个 epoch 的 CPU 训练。
- YOLOv8 权重下载通过：`yolov8n-cls.pt` 已能被 Ultralytics 正常下载和加载。
- 预测脚本通过：使用冒烟训练得到的 `best.pt` 成功完成单张图片预测。

冒烟训练只证明训练流程能跑通，不代表模型已经学好了。真正训练要使用完整的 `data/cat_dog_cls` 数据集。
