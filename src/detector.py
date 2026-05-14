import argparse
from pathlib import Path
import csv
import math
from ultralytics import YOLO


def run(images_dir, out_csv, conf=0.25, iou=0.45):
    model = YOLO('runs/pose/runs/detect/tube_lid_kp/weights/best.pt')
    images_dir = Path(images_dir)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(images_dir.glob('*')):
        if p.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        res = model(str(p), conf=conf, iou=iou)
        if len(res) == 0:
            continue
        keypoints = getattr(res[0], 'keypoints', None)
        if keypoints is None or keypoints.xy is None:
            continue
        kps = keypoints.xy.cpu().numpy()  # shape: (n, 2, 2)
        if kps.size == 0:
            continue
        for kp in kps:
            if kp.shape[0] < 2:
                continue
            center_x, center_y = kp[0]
            tab_x, tab_y = kp[1]
            if any(map(lambda v: v is None or math.isnan(v), [center_x, center_y, tab_x, tab_y])):
                continue
            angle = (math.degrees(math.atan2(-(tab_y - center_y), tab_x - center_x))) % 360
            rows.append((p.name, float(center_x), float(center_y), float(angle)))

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'center_x', 'center_y', 'angle_deg'])
        for r in rows:
            writer.writerow(r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    args = parser.parse_args()
    run(args.images_dir, args.output_csv, conf=args.conf, iou=args.iou)
