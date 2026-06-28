# 从零训练第一个猫狗分类模型

这份文档是给完全没有模型训练经验的人看的。你不需要先懂 YOLO、PyTorch、训练集、验证集这些词。我们会从“这个项目里每个目录是什么”开始，一步一步训练出第一个能判断猫或狗的模型。

## 先理解我们要做什么

我们要训练的是一个“分类模型”。

分类模型要回答的问题是：

> 这张图片更像 `Cat`，还是更像 `Dog`？

它不是在图片里画框，也不是找动物的位置。它只输出一个类别和一个置信度。

举例：

```text
预测结果: Cat
置信度: 0.9821
```

意思是：模型认为这张图片是猫，而且它比较有把握。

## 项目目录总览

你当前项目大概长这样：

```text
cat_or_dog/
├── README.md
├── requirements.txt
├── dataset/
├── data/
├── runs/
├── scripts/
├── tests/
├── docs/
├── .cache/
├── .venv/
├── .gitignore
└── yolov8n-cls.pt
```

下面逐个解释。

## 根目录文件

### `README.md`

这是项目首页说明。

你以后忘记怎么运行时，先看这个文件。它会告诉你：

- 用哪个 Conda 环境
- 怎么准备数据
- 怎么训练
- 怎么预测一张图片
- 更详细的新手教程在哪里

### `requirements.txt`

这是 Python 依赖清单。

当前项目正式训练使用 Conda 环境 `yolo8`，所以你一般不需要直接用这个文件安装依赖。它的作用是记录这个项目大概需要哪些包：

- `ultralytics`：YOLOv8 的主要库
- `pillow`：用来检查图片是否能打开
- `pytest`：用来跑测试

### `.gitignore`

这个文件告诉 Git：哪些东西不要提交。

为什么需要它？因为训练项目会产生很多大文件和临时文件，比如：

- 原始数据集
- 准备后的数据集
- 训练输出
- 模型权重
- 缓存
- 虚拟环境

这些文件很大，或者可以重新生成，不适合放进 Git。

### `yolov8n-cls.pt`

这是 YOLOv8 的预训练分类模型权重。

你可以把它理解成“别人已经提前训练过的基础模型”。我们不是从零开始教一个完全空白的模型认识图片，而是在这个基础模型上继续训练，让它专门学会区分猫和狗。

这叫“迁移学习”。它的好处是：

- 训练更快
- 需要的数据更少
- 效果通常更好

这个文件已经被 `.gitignore` 忽略，不会提交到 Git。

## 数据目录

### `dataset/`

这是原始数据集目录。

当前原始图片在：

```text
dataset/PetImages/Cat/
dataset/PetImages/Dog/
```

它的意思很直观：

- `Cat/` 里面都是猫图片
- `Dog/` 里面都是狗图片

这个目录是“原材料”。脚本不会修改它，只会读取它。

### `data/`

这是准备后的训练数据目录。

运行数据准备脚本后，会生成：

```text
data/cat_dog_cls/
├── train/
│   ├── Cat/
│   └── Dog/
└── val/
    ├── Cat/
    └── Dog/
```

这里有两个关键词：

### `train`

训练集。

模型会用这些图片学习猫和狗的区别。你可以把它想象成“教材”。

### `val`

验证集。

模型训练过程中，会用这些图片检查自己学得怎么样。你可以把它想象成“模拟考试”。

为什么不能所有图片都拿来训练？因为如果只看教材，不考试，我们不知道模型是真的学会了，还是只记住了训练图片。

### 当前数据准备结果

你的数据集已经验证过：

```text
Cat: 12,499 张可用，1 张损坏，9,999 张训练，2,500 张验证
Dog: 12,499 张可用，1 张损坏，9,999 张训练，2,500 张验证
```

这说明原始数据里猫和狗各有 1 张坏图，脚本已经自动跳过了。

## 训练输出目录

### `runs/`

这是训练结果目录。

正式训练完成后，模型和训练日志会保存在：

```text
runs/classify/cat_dog_yolov8n/
```

里面最重要的是：

```text
runs/classify/cat_dog_yolov8n/weights/best.pt
```

### `best.pt`

这是训练过程中表现最好的模型。

以后预测图片时，默认就会读取这个文件。

你可以把 `best.pt` 理解成最终交付物：我们训练模型，最终就是为了得到它。

## 脚本目录

### `scripts/`

这里放的是你真正会运行的脚本。

### `scripts/prepare_dataset.py`

数据准备脚本。

它做这些事：

1. 读取 `dataset/PetImages/Cat` 和 `dataset/PetImages/Dog`
2. 检查图片能不能正常打开
3. 跳过坏图片
4. 按比例拆成训练集和验证集
5. 复制到 `data/cat_dog_cls`

运行方式：

```bash
conda run -n yolo8 python scripts/prepare_dataset.py
```

你可以把这一步理解成：把乱放的原材料整理成 YOLOv8 能直接吃的格式。

### `scripts/train.py`

真正调用 YOLOv8 开始训练的 Python 脚本。

它会：

1. 检查 `data/cat_dog_cls` 是否存在
2. 加载 `yolov8n-cls.pt`
3. 用你的猫狗数据继续训练
4. 把结果保存到 `runs/classify/cat_dog_yolov8n`

一般不建议你直接运行它，而是使用下面的 Conda 包装脚本。

### `scripts/train_yolo8_conda.sh`

训练入口脚本。

推荐用这个训练：

```bash
./scripts/train_yolo8_conda.sh
```

它内部实际做的是：

```bash
conda run -n yolo8 python scripts/train.py
```

为什么要多包一层？因为你是新手时，很容易不小心用错 Python 环境。这个脚本把环境固定成 `yolo8`，减少出错。

### `scripts/predict.py`

预测脚本。

它会：

1. 读取训练好的模型 `best.pt`
2. 读取你给它的一张图片
3. 输出 `Cat` 或 `Dog`
4. 输出置信度

一般也不建议你直接运行它，而是使用下面的 Conda 包装脚本。

### `scripts/predict_yolo8_conda.sh`

预测入口脚本。

推荐用这个预测：

```bash
./scripts/predict_yolo8_conda.sh dataset/PetImages/Cat/0.jpg
```

它内部实际做的是：

```bash
conda run -n yolo8 python scripts/predict.py dataset/PetImages/Cat/0.jpg
```

### `scripts/runtime.py`

运行环境辅助脚本。

它主要是设置缓存目录，让 Ultralytics 和 Matplotlib 这类库把缓存放到项目的 `.cache/` 里。

你不需要手动运行它。

## 测试目录

### `tests/`

这里放的是自动测试。

自动测试不是训练模型用的，它是检查我们的脚本有没有写错。

比如：

- 数据准备脚本能不能跳过坏图
- 训练脚本能不能发现缺少数据目录
- 预测脚本能不能发现模型文件不存在
- Conda 包装脚本是不是确实使用了 `yolo8`

运行测试：

```bash
.venv/bin/python -m pytest -v
```

注意：测试目前使用项目里的 `.venv`，训练和预测使用 Conda 的 `yolo8`。这是因为 `yolo8` 里没有安装 `pytest`，而我们不想随便改你的 Conda 环境。

## 文档目录

### `docs/`

这里放项目文档。

### `docs/beginner-guide.md`

就是你正在看的这份新手教程。

### `docs/superpowers/specs/`

这里放设计文档。

设计文档回答：

> 我们为什么要这样做？

### `docs/superpowers/plans/`

这里放实现计划。

实现计划回答：

> 当时是按什么步骤一步步做出来的？

你作为新手，不需要每次都看这两个目录。以后如果你想理解项目演进过程，再看它们。

## 缓存和环境目录

### `.cache/`

缓存目录。

一些库会把临时配置、字体缓存等放在这里。它不是代码，也不是数据集，可以删，删了以后下次运行会自动生成。

### `.venv/`

项目测试用的 Python 虚拟环境。

当前正式训练不用它，正式训练用 Conda 的 `yolo8` 环境。

### `__pycache__/` 和 `.pytest_cache/`

Python 和 pytest 自动生成的缓存目录。

你不需要管它们。

## 第一次训练：手把手流程

下面是完整流程。建议你按顺序执行。

### 第 1 步：进入项目目录

```bash
cd /Users/chenshibo/Documents/Workspace/AI/cat_or_dog
```

为什么要这样做？

因为后面的脚本都假设你在项目根目录运行。项目根目录就是有 `README.md`、`scripts/`、`dataset/` 的这个目录。

### 第 2 步：确认 Conda 环境存在

```bash
conda env list
```

你应该能看到：

```text
yolo8    /Users/chenshibo/miniconda3/envs/yolo8
```

为什么要这样做？

因为训练不能乱用全局 Python。机器学习依赖比较复杂，环境错了就容易出现各种奇怪错误。

### 第 3 步：确认 YOLOv8 能导入

```bash
conda run -n yolo8 python -c "import ultralytics; print(ultralytics.__version__)"
```

如果输出类似：

```text
8.4.80
```

说明 YOLOv8 库已经在 `yolo8` 环境里。

### 第 4 步：准备数据

```bash
conda run -n yolo8 python scripts/prepare_dataset.py
```

你会看到类似：

```text
Cat: valid=12499, skipped=1, train=9999, val=2500
Dog: valid=12499, skipped=1, train=9999, val=2500
准备完成: data/cat_dog_cls
```

这一步在做什么？

它把原始数据从：

```text
dataset/PetImages/Cat
dataset/PetImages/Dog
```

整理成：

```text
data/cat_dog_cls/train/Cat
data/cat_dog_cls/train/Dog
data/cat_dog_cls/val/Cat
data/cat_dog_cls/val/Dog
```

为什么要这样做？

YOLOv8 分类训练需要这种目录格式。它通过文件夹名字知道类别：放在 `Cat` 文件夹里的就是猫，放在 `Dog` 文件夹里的就是狗。

### 第 5 步：开始训练

```bash
./scripts/train_yolo8_conda.sh
```

这一步在做什么？

它会用 `yolov8n-cls.pt` 作为起点，然后用你的猫狗数据继续训练。

默认训练参数是：

```text
模型: yolov8n-cls.pt
图片尺寸: 224
训练轮数: 10
批大小: 32
设备: cpu
```

### 什么是训练轮数 `epochs`

`epochs=10` 的意思是：模型会把训练集大致看 10 遍。

轮数太少，可能学不够。

轮数太多，可能训练时间长，也可能记住训练图片而不是学会规律。

第一次训练先用默认 10 轮就好。

### 什么是批大小 `batch`

`batch=32` 的意思是：模型一次看 32 张图片，再更新一次自己的参数。

批大小越大，可能更快，但更吃内存。

你的电脑是 16 GB 内存，默认 32 是一个比较温和的起点。如果训练时内存压力大，可以改小：

```bash
./scripts/train_yolo8_conda.sh --batch 16
```

### 什么是图片尺寸 `imgsz`

`imgsz=224` 的意思是：训练时图片会被缩放到 224 x 224 左右。

尺寸越大，模型能看到更多细节，但训练更慢。

分类猫狗时，224 是常见起点。

### 为什么现在用 CPU

当前 `yolo8` 环境里：

```text
torch.backends.mps.is_available() = False
```

这说明 PyTorch 暂时不能使用 Apple Silicon 的 MPS 加速。

所以默认使用 CPU。CPU 慢一点，但更稳定。

### 第 6 步：找到训练好的模型

训练完成后，重点看这个文件：

```text
runs/classify/cat_dog_yolov8n/weights/best.pt
```

这就是你训练出来的第一个模型。

### 第 7 步：预测一张图片

```bash
./scripts/predict_yolo8_conda.sh dataset/PetImages/Cat/0.jpg
```

如果训练已经完成，脚本会默认读取：

```text
runs/classify/cat_dog_yolov8n/weights/best.pt
```

输出类似：

```text
预测结果: Cat
置信度: 0.9821
```

如果你要预测狗图片：

```bash
./scripts/predict_yolo8_conda.sh dataset/PetImages/Dog/0.jpg
```

## 如何判断模型有没有训练好

训练时你会看到一些指标。对新手来说，先关注这几个：

### `loss`

损失值。

你可以把它理解成“模型错得有多离谱”。一般来说，训练过程中 loss 越来越低是好事。

### `top1_acc`

Top-1 准确率。

它表示模型最有把握的那个答案是不是对的。

猫狗分类只有两个类别，所以你可以把它近似理解成准确率。

比如：

```text
top1_acc = 0.95
```

大致表示验证集上 95% 图片分类正确。

### `best.pt`

训练过程中表现最好的模型。

不要随便拿最后一轮模型当最终模型，优先用 `best.pt`。

## 常见问题

### 为什么要先准备数据，不能直接训练？

因为 YOLOv8 分类训练需要固定目录结构：

```text
train/类别名/
val/类别名/
```

原始数据不是完整训练格式，所以要先整理。

### 为什么坏图片要跳过？

因为只要训练过程中读到一张坏图片，训练就可能中断。提前检查并跳过坏图，可以让训练更稳定。

### 为什么不用全局 Python？

机器学习项目依赖很多，比如 PyTorch、TorchVision、Ultralytics、OpenCV。不同版本之间可能互相影响。

Conda 环境就像一个独立小房间。我们把 YOLOv8 相关依赖放在 `yolo8` 这个房间里，训练时只进这个房间，减少环境混乱。

### 训练很慢怎么办？

当前默认 CPU 训练会比较慢。可以先用少量轮数试跑：

```bash
./scripts/train_yolo8_conda.sh --epochs 1
```

确认流程没问题后，再跑完整训练：

```bash
./scripts/train_yolo8_conda.sh --epochs 10
```

### 我可以中断训练吗？

可以。终端里按 `Ctrl+C` 可以中断。

中断后，已经写入 `runs/` 的部分结果可能还在，但不一定是完整模型。第一次建议先用 `--epochs 1` 试跑，确认流程通了再长时间训练。

## 推荐的新手练习顺序

1. 先读完这份文档。
2. 运行 `conda env list`，确认 `yolo8` 存在。
3. 运行数据准备命令。
4. 运行 `./scripts/train_yolo8_conda.sh --epochs 1`，先训练 1 轮感受流程。
5. 用预测脚本预测一张猫图和一张狗图。
6. 再运行 `./scripts/train_yolo8_conda.sh --epochs 10`，训练一个更像样的模型。
7. 再次预测猫图和狗图，对比结果有没有变好。

第一轮目标不是追求最高准确率，而是完整理解流程：数据准备、训练、生成模型、预测图片。
