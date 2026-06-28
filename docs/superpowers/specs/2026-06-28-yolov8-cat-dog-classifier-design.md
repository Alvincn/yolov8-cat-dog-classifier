# YOLOv8 Cat/Dog Classifier Design

## Goal

Build a small, reproducible YOLOv8 classification project that trains a model to decide whether an image contains a cat or a dog.

The project will train locally on this MacBook Pro with Apple M4 and 16 GB memory. It will use Ultralytics YOLOv8 classification weights, starting with `yolov8n-cls.pt`, and target Apple Silicon MPS acceleration through `device=mps`.

## Existing Dataset

The raw dataset is already present in the repository:

- `dataset/PetImages/Cat`: 12,500 JPG files
- `dataset/PetImages/Dog`: 12,500 JPG files

The dataset also contains non-image `Thumbs.db` files. This dataset is known to include occasional corrupt images, so preparation must verify that files can be opened before using them for training.

The raw dataset will remain unchanged.

## Recommended Approach

Use YOLOv8 classification rather than YOLOv8 object detection.

The user goal is binary image classification: decide whether the animal is a cat or a dog. Object detection would require bounding-box labels and would add unnecessary annotation and training complexity. A YOLOv8 classification model accepts class-folder data directly and fits this problem well.

## Project Structure

The implementation will add:

- `requirements.txt`: Python dependencies for Ultralytics and image validation.
- `scripts/prepare_dataset.py`: Validate raw images and create a train/validation classification dataset.
- `scripts/train.py`: Train `yolov8n-cls.pt` on the prepared dataset with Mac-friendly defaults.
- `scripts/predict.py`: Run inference on a single image and print the predicted label and confidence.
- `README.md`: Setup, preparation, training, and prediction commands.
- `.gitignore`: Ignore virtual environments, generated datasets, training outputs, and model artifacts.

Generated training data will live under:

- `data/cat_dog_cls/train/Cat`
- `data/cat_dog_cls/train/Dog`
- `data/cat_dog_cls/val/Cat`
- `data/cat_dog_cls/val/Dog`

## Data Preparation

`scripts/prepare_dataset.py` will:

1. Read source images from `dataset/PetImages/Cat` and `dataset/PetImages/Dog`.
2. Ignore non-JPG files such as `Thumbs.db`.
3. Open and verify images using Pillow.
4. Split valid images into train and validation sets with a fixed random seed.
5. Copy valid images into the YOLOv8 classification folder layout.
6. Print a summary of accepted and skipped files per class.

Default split:

- Train: 80%
- Validation: 20%

The script will be idempotent enough for normal use: it can recreate or update the prepared dataset from the raw dataset.

## Training

`scripts/train.py` will use the Ultralytics Python API:

- Model: `yolov8n-cls.pt`
- Dataset: `data/cat_dog_cls`
- Device default: `mps`
- Image size default: `224`
- Epochs default: `10`
- Batch default: `32`
- Project output: `runs/classify`
- Run name default: `cat_dog_yolov8n`

These defaults prioritize a reliable first local run on the Mac. Users can override them from command-line arguments for longer or larger training runs.

## Prediction

`scripts/predict.py` will:

1. Load a trained model path, defaulting to `runs/classify/cat_dog_yolov8n/weights/best.pt`.
2. Run prediction on a user-supplied image.
3. Print the top class name and confidence.

The script will fail with a clear message if the model file or image file is missing.

## Error Handling

Preparation should report skipped files rather than stopping the whole run for one bad image.

Training should validate that the prepared dataset exists and contains `train` and `val` folders before invoking Ultralytics.

Prediction should validate input paths before loading the model.

## Verification

Before considering the implementation complete:

1. Run the dataset preparation script on the local dataset.
2. Confirm the prepared dataset has both classes in train and validation folders.
3. Run a lightweight smoke test for training with a very small epoch count if dependencies are available.
4. Run prediction with a sample image if a trained model exists.

If dependencies cannot be installed or model weights cannot be downloaded because of network restrictions, the code and documentation will still make that requirement explicit.
