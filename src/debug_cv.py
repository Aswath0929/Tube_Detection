import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='data/images/fc1715f8-color.png')
    parser.add_argument('--weights', default='runs/detect/runs/tube_lid/weights/best.pt')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--pad', type=int, default=10)
    parser.add_argument('--output-dir', default='outputs/debug')
    parser.add_argument('--scale', type=float, default=2.0)
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')

    gt = pd.read_csv('data/annotations.csv')
    gt = gt[gt['image'] == image_path.name]

    model = YOLO(args.weights)
    results = model(image, conf=args.conf, iou=args.iou)

    out_root = Path(args.output_dir) / image_path.stem
    out_root.mkdir(parents=True, exist_ok=True)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        print('No detections found.')
        return

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cx1 = max(0, x1 - args.pad)
        cy1 = max(0, y1 - args.pad)
        cx2 = min(image.shape[1], x2 + args.pad)
        cy2 = min(image.shape[0], y2 + args.pad)

        crop = image[cy1:cy2, cx1:cx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # show intermediate steps
        scale = max(0.1, float(args.scale))
        crop_show = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        thresh_show = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        closed_show = cv2.resize(closed, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('crop', crop_show)
        cv2.imshow('thresh', thresh_show)
        cv2.imshow('closed', closed_show)

        debug = crop.copy()
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                ccx = int(M['m10'] / M['m00'])
                ccy = int(M['m01'] / M['m00'])
                pts = cnt[:, 0, :]
                dists = np.sqrt((pts[:, 0] - ccx) ** 2 + (pts[:, 1] - ccy) ** 2)
                tab = pts[np.argmax(dists)]

                cv2.drawContours(debug, [cnt], -1, (0, 255, 0), 1)
                cv2.circle(debug, (ccx, ccy), 3, (255, 0, 0), -1)
                cv2.circle(debug, tuple(tab), 3, (0, 0, 255), -1)
                dx = tab[0] - ccx
                dy = tab[1] - ccy
                angle = np.degrees(np.arctan2(-dy, dx)) % 360
                cv2.putText(
                    debug,
                    f'{angle:.1f}',
                    (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

        result_show = cv2.resize(debug, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('result', result_show)

        det_dir = out_root / f'det_{idx:02d}'
        det_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(det_dir / 'crop.png'), crop)
        cv2.imwrite(str(det_dir / 'thresh.png'), thresh)
        cv2.imwrite(str(det_dir / 'closed.png'), closed)
        cv2.imwrite(str(det_dir / 'result.png'), debug)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
