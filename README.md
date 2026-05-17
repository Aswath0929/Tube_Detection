# Tube Lid Detector

Detect tube lids and estimate their tab angles with a YOLOv8 keypoint model.

## Overview

The system uses a two-stage approach:

1. Train a YOLOv8 detector to localize lids with bounding boxes.
2. Train a YOLOv8 pose model to predict two keypoints per lid (center and tab tip).

The tab angle is computed from the predicted keypoints:

$$
	heta = \mathrm{atan2}(-(y_{tab}-y_{center}),\ x_{tab}-x_{center})
$$

## Data and Annotations

- Input images are 640x480 RGB.
- Labels in [data/annotations.csv](data/annotations.csv) include bounding boxes, lid centers, and tab angles in degrees.

## Project Structure

- [data/](data/) raw images and CSV annotations
- [yolo_dataset/](yolo_dataset/) detection dataset in YOLO format
- [yolo_kp_dataset/](yolo_kp_dataset/) keypoint dataset in YOLO pose format
- [src/](src/) data prep, training, inference, evaluation, and debug scripts
- [runs/](runs/) training outputs and checkpoints
- [outputs/](outputs/) predictions and evaluation artifacts
- [labels/](labels/) generated label files for train/val/test splits

## Scripts and Tasks

1. Detection dataset preparation: [src/prepare_dataset.py](src/prepare_dataset.py)
   - Converts CSV annotations into YOLO detection labels (normalized bbox format).
   - Splits data 80/20 by image with seed 42.
   - Writes [yolo_dataset/dataset.yaml](yolo_dataset/dataset.yaml).

2. Detection model training: [src/train.py](src/train.py)
   - Trains a YOLOv8 detector to localize lids (bbox only).
   - Outputs saved under [runs/](runs/).

3. Keypoint dataset preparation: [src/prepare_kp_dataset.py](src/prepare_kp_dataset.py)
   - Computes a tab-tip keypoint from center and angle labels.
   - Creates YOLO keypoint labels with two keypoints: center and tab tip.
   - Splits data 80/20 by image with seed 42.
   - Writes [yolo_kp_dataset/dataset.yaml](yolo_kp_dataset/dataset.yaml).

4. Keypoint model training: [src/train_kp.py](src/train_kp.py)
   - Trains a YOLOv8 pose model to predict the two keypoints per lid.

5. Inference: [src/detector.py](src/detector.py)
   - Runs the keypoint model to get center and tab tip per detection.
   - Computes the tab angle from the keypoints.
   - Writes [outputs/results/predictions.csv](outputs/results/predictions.csv).

6. Evaluation: [src/evaluate.py](src/evaluate.py)
   - Matches predictions to ground truth with the Hungarian algorithm.
   - Applies a distance threshold to determine TP/FP/FN.
   - Reports precision, recall, F1, and mean angular error.

7. Debug utilities: [src/debug_*.py](src/)
   - Visual checks for angles, crops, thresholds, and classical CV behavior.

## Quick Start

1. Prepare detection dataset

```bash
python src/prepare_dataset.py
```

2. Train YOLOv8 detector (requires [yolov8n.pt](yolov8n.pt))

```bash
python src/train.py
```

3. Prepare keypoint dataset

```bash
python src/prepare_kp_dataset.py
```

4. Train YOLOv8 keypoint model (requires [yolov8n-pose.pt](yolov8n-pose.pt))

```bash
python src/train_kp.py
```

5. Run inference on images

```bash
python src/detector.py --images-dir data/images --output-csv outputs/results/predictions.csv
```

6. Evaluate predictions

```bash
python src/evaluate.py --gt data/annotations.csv --pred outputs/results/predictions.csv
```

## Dependencies

Install dependencies from [requirements.txt](requirements.txt).
