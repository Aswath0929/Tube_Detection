from ultralytics import YOLO


def main():
    model = YOLO('yolov8n.pt')
    model.train(
        data='yolo_dataset/dataset.yaml',
        epochs=30,
        imgsz=448,
        batch=32,
        project='runs/detect',
        name='tube_lid',
        exist_ok=True,
        patience=8,
        weight_decay=0.002,
        label_smoothing=0.15,
        freeze=18,
        flipud=0.5,
        fliplr=0.5,
        degrees=180,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        translate=0.2,
        scale=0.4,
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.2,
        close_mosaic=5,
        dropout=0.1,
    )


if __name__ == '__main__':
    main()
