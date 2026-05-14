# Tube Lid Detector

This project detects tube lids and estimates their tab angles using a YOLOv8 keypoint model.

Flow-by-flow breakdown (techniques + data flow):

1. Data + annotations
	- Input images are 640x480 RGB.
	- Labels in [data/annotations.csv](data/annotations.csv) include bounding boxes, lid centers, and tab angles in degrees.

2. Detection dataset preparation ([src/prepare_dataset.py](src/prepare_dataset.py))
	- Convert CSV annotations into YOLO detection labels (normalized bbox format).
	- Split 80/20 by image with seed 42 to avoid label leakage.
	- Write [yolo_dataset/dataset.yaml](yolo_dataset/dataset.yaml) for training.

3. Detection model training ([src/train.py](src/train.py))
	- Train a YOLOv8 detector to localize lids (box-only).
	- Outputs saved under the runs directory.

4. Keypoint dataset preparation ([src/prepare_kp_dataset.py](src/prepare_kp_dataset.py))
	- Compute a tab-tip keypoint from the center and angle labels.
	- Create YOLO keypoint labels with two keypoints: center and tab tip.
	- Split 80/20 by image with seed 42.
	- Write [yolo_kp_dataset/dataset.yaml](yolo_kp_dataset/dataset.yaml) for keypoint training.

5. Keypoint model training ([src/train_kp.py](src/train_kp.py))
	- Train a YOLOv8 pose model to predict the two keypoints per lid.
	- The model learns geometry directly instead of relying on classical CV.

6. Inference (prediction) ([src/detector.py](src/detector.py))
	- Run the keypoint model to get center and tab tip per detection.
	- Compute angle with $\theta = \mathrm{atan2}(-(y_{tab}-y_{center}), x_{tab}-x_{center})$.
	- Save outputs to [outputs/results/predictions.csv](outputs/results/predictions.csv).

7. Evaluation ([src/evaluate.py](src/evaluate.py))
	- Match predictions to ground truth per image using the Hungarian algorithm.
	- Apply a distance threshold to determine TP/FP/FN.
	- Report precision/recall/F1 and mean angular error, optionally saving matches/metrics.

Project structure overview:
- [data/](data/) raw images and CSV annotations
- [yolo_dataset/](yolo_dataset/) detection dataset (YOLO format)
- [yolo_kp_dataset/](yolo_kp_dataset/) keypoint dataset (YOLO pose format)
- [src/](src/) scripts for data prep, training, inference, evaluation, and debug
- [runs/](runs/) training outputs and weights
- [outputs/](outputs/) predictions and evaluation artifacts

Quick steps:

1. Prepare detection dataset

```bash
python src/prepare_dataset.py
```

2. Train YOLOv8 detector (requires `yolov8n.pt` checkpoint)

```bash
python src/train.py
```

3. Prepare keypoint dataset

```bash
python src/prepare_kp_dataset.py
```

4. Train YOLOv8 keypoint model (requires `yolov8n-pose.pt` checkpoint)

```bash
python src/train_kp.py
```

5. Run detector on images

```bash
python src/detector.py --images-dir data/images --output-csv outputs/results/predictions.csv
```

6. Evaluate

```bash
python src/evaluate.py --gt data/annotations.csv --pred outputs/results/predictions.csv
```

Dependencies: see `requirements.txt`.
