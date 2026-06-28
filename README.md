# YOLOv8 猫狗分类训练项目

这个项目会训练一个模型，用来判断一张动物图片是猫还是狗。

我们使用的是 YOLOv8 的“分类模型”，不是“检测模型”。分类模型回答“这张图是什么类别”，检测模型回答“图中物体在哪里”。你的需求是判断猫还是狗，所以分类模型更合适。

如果你是模型训练新手，请先看这份手把手教程：

- [从零训练第一个猫狗分类模型](docs/beginner-guide.md)

## 数据集

原始数据放在：

- `dataset/PetImages/Cat`
- `dataset/PetImages/Dog`

原始数据不会被修改。脚本会把可用图片复制到 `data/cat_dog_cls`，并自动拆分训练集和验证集。

## Conda 环境

本项目训练和预测只使用 Conda 环境 `yolo8`。

不要直接使用全局 Python 环境，也不需要在 README 的训练流程里创建新的 `venv`。这样做是为了避免机器学习依赖混在一起，导致 YOLOv8、Torch 或 OpenCV 版本互相冲突。

当前电脑的 Conda 里已经有一个 YOLOv8 环境，名字是 `yolo8`。

我已经验证过这个环境：

- Python: 3.8.20
- Ultralytics: 8.4.80
- Torch: 2.4.1
- MPS: 不可用，所以默认继续用 CPU 训练

你可以这样确认环境是否存在：

```bash
conda env list
conda run -n yolo8 python --version
conda run -n yolo8 python -c "import ultralytics; print(ultralytics.__version__)"
```

项目里仍然保留 `requirements.txt`，它只是依赖清单和备用参考，不是当前推荐入口。正式训练和预测请使用 `yolo8` Conda 环境。

最常用的三个命令是：

```bash
conda run -n yolo8 python scripts/prepare_dataset.py
./scripts/train_yolo8_conda.sh
./scripts/predict_yolo8_conda.sh dataset/PetImages/Cat/0.jpg
```

## 准备数据

```bash
conda run -n yolo8 python scripts/prepare_dataset.py
```

这一步会检查图片能不能被正常打开，然后把可用图片复制到 YOLOv8 分类训练需要的目录结构里。

在当前数据集上的验证结果是：

- `Cat`: 12,499 张可用图片，1 张损坏图片被跳过，9,999 张用于训练，2,500 张用于验证。
- `Dog`: 12,499 张可用图片，1 张损坏图片被跳过，9,999 张用于训练，2,500 张用于验证。

## 训练模型

```bash
./scripts/train_yolo8_conda.sh
```

上面这个包装脚本内部实际执行的是：

```bash
conda run --no-capture-output -n yolo8 python scripts/train.py
```

`--no-capture-output` 的作用是让 YOLOv8 的训练日志实时显示在终端里。否则 Conda 可能会暂时缓存输出，看起来就像训练脚本卡住了。

训练开始后，你会先看到类似下面的日志：

```text
[1/2] 使用 Conda 环境 yolo8 启动训练...
[2/2] 下面开始显示 YOLOv8 训练日志，请保持终端窗口打开。
开始训练 YOLOv8 猫狗分类模型
数据集: data/cat_dog_cls
模型: yolov8n-cls.pt
训练轮数: 10
图片尺寸: 224
批大小: 32
设备: cpu
```

后面 Ultralytics 会继续输出每个 epoch 的训练进度、loss、准确率和模型保存路径。CPU 训练时某些阶段会比较慢，尤其是扫描图片、建立缓存、验证集评估时，短时间没有新输出不一定代表程序死掉。

当前脚本默认使用 `device=cpu`。原因是 `yolo8` Conda 环境中已经验证过 Torch 可以训练，但 `torch.backends.mps.is_available()` 返回 `False`，也就是当前 Conda 环境暂时不能使用 Apple Silicon 的 MPS 加速。

如果你之后换了支持 MPS 的 PyTorch 环境，可以这样尝试加速：

```bash
./scripts/train_yolo8_conda.sh --device mps
```

如果 MPS 报错，就继续使用默认 CPU。CPU 会慢一些，但更稳定。

训练完成后，最好的模型通常会保存在：

```text
runs/classify/cat_dog_yolov8n/weights/best.pt
```

## 预测单张图片

```bash
./scripts/predict_yolo8_conda.sh dataset/PetImages/Cat/0.jpg
```

脚本会输出模型认为这张图片是 `Cat` 还是 `Dog`，以及对应置信度。

预测脚本默认读取训练输出中的 `best.pt`。如果你的模型文件在其他位置，可以这样指定：

```bash
./scripts/predict_yolo8_conda.sh path/to/image.jpg --model path/to/best.pt
```

如果你用 MPS 训练出了模型，也可以预测时指定 MPS：

```bash
./scripts/predict_yolo8_conda.sh path/to/image.jpg --model path/to/best.pt --device mps
```

## 已完成的本地验证

当前项目已经完成这些验证：

- 单元测试通过：数据准备、训练前路径检查、预测前路径检查。
- 真实数据准备通过：已生成 `data/cat_dog_cls/train` 和 `data/cat_dog_cls/val`。
- YOLOv8 冒烟训练通过：使用临时小数据集完成了 1 个 epoch 的 CPU 训练。
- YOLOv8 权重下载通过：`yolov8n-cls.pt` 已能被 Ultralytics 正常下载和加载。
- 预测脚本通过：使用冒烟训练得到的 `best.pt` 成功完成单张图片预测。

冒烟训练只证明训练流程能跑通，不代表模型已经学好了。真正训练要使用完整的 `data/cat_dog_cls` 数据集。
