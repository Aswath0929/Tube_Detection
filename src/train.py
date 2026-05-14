from ultralytics import YOLO


def main():
    model = YOLO('yolov8n.pt')
    model.train(
        data='yolo_dataset/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        project='runs/detect',
        name='tube_lid',
        exist_ok=True,
        flipud=0.5,
        fliplr=0.5,
        degrees=180,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        translate=0.1,
        scale=0.3,
    )


if __name__ == '__main__':
    main()
