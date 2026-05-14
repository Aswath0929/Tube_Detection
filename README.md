# Tube Lid Detector

This project detects tube lids and estimates their tab angles using a YOLOv8 keypoint model.

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
