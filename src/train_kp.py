from ultralytics import YOLO


def main():
    model = YOLO('yolov8n-pose.pt')
    model.train(
        data='yolo_kp_dataset/dataset.yaml',
        epochs=200,
        imgsz=640,
        batch=8,
        project='runs/pose/detect',
        name='tube_lid_kp',
        exist_ok=True,
        flipud=0.0,
        fliplr=0.0,
        degrees=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        translate=0.1,
        scale=0.3,
    )


if __name__ == '__main__':
    main()
