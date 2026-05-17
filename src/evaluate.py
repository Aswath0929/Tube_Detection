import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import json


def angular_error(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _load_image_set(images_dir):
    if images_dir is None:
        return None
    image_paths = Path(images_dir).glob('*')
    return {p.name for p in image_paths if p.suffix.lower() in ['.png', '.jpg', '.jpeg']}


def evaluate(gt_path, pred_path, dist_thresh=20.0, matches_out=None, metrics_out=None, images_dir=None):
    gt = pd.read_csv(gt_path)
    preds = pd.read_csv(pred_path)

    image_set = _load_image_set(images_dir)
    if image_set is not None:
        gt = gt[gt['image'].isin(image_set)]
        preds = preds[preds['image'].isin(image_set)]

    images = sorted(set(gt['image'].unique()).union(set(preds['image'].unique())))

    TP = 0
    FP = 0
    FN = 0
    ang_errors = []

    matches_rows = []

    for im in images:
        gt_rows = gt[gt['image'] == im]
        pred_rows = preds[preds['image'] == im]

        gt_pts = gt_rows[['center_x', 'center_y', 'angle_deg']].values if len(gt_rows) > 0 else np.zeros((0, 3))
        pred_pts = pred_rows[['center_x', 'center_y', 'angle_deg']].values if len(pred_rows) > 0 else np.zeros((0, 3))

        if gt_pts.shape[0] == 0 and pred_pts.shape[0] == 0:
            continue

        if gt_pts.shape[0] == 0:
            # all preds are false positives
            for p in pred_pts:
                matches_rows.append([im, '', '', '', float(p[0]), float(p[1]), float(p[2]), '', '', 'FP'])
            FP += len(pred_pts)
            continue

        if pred_pts.shape[0] == 0:
            for g in gt_pts:
                matches_rows.append([im, float(g[0]), float(g[1]), float(g[2]), '', '', '', '', '', 'FN'])
            FN += len(gt_pts)
            continue

        # compute distance matrix
        D = np.zeros((gt_pts.shape[0], pred_pts.shape[0]), dtype=float)
        for i, g in enumerate(gt_pts):
            for j, p in enumerate(pred_pts):
                D[i, j] = math.hypot(g[0] - p[0], g[1] - p[1])

        row_ind, col_ind = linear_sum_assignment(D)

        matched_gt = set()
        matched_pred = set()
        for i, j in zip(row_ind, col_ind):
            dist = D[i, j]
            if dist <= dist_thresh:
                g = gt_pts[i]
                p = pred_pts[j]
                ae = angular_error(float(p[2]), float(g[2]))
                matches_rows.append([im, float(g[0]), float(g[1]), float(g[2]), float(p[0]), float(p[1]), float(p[2]), float(dist), float(ae), 'TP'])
                TP += 1
                ang_errors.append(ae)
                matched_gt.add(i)
                matched_pred.add(j)

        # unmatched GT -> FN
        for i in range(gt_pts.shape[0]):
            if i not in matched_gt:
                g = gt_pts[i]
                matches_rows.append([im, float(g[0]), float(g[1]), float(g[2]), '', '', '', '', '', 'FN'])
                FN += 1

        # unmatched pred -> FP
        for j in range(pred_pts.shape[0]):
            if j not in matched_pred:
                p = pred_pts[j]
                matches_rows.append([im, '', '', '', float(p[0]), float(p[1]), float(p[2]), '', '', 'FP'])
                FP += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_ae = float(np.mean(ang_errors)) if ang_errors else None

    print(f"TP={TP}, FP={FP}, FN={FN}")
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    print(f"Mean angular error={mean_ae}")

    if matches_out:
        cols = ['image', 'gt_cx', 'gt_cy', 'gt_angle', 'pred_cx', 'pred_cy', 'pred_angle', 'distance', 'angular_error', 'match_type']
        pd.DataFrame(matches_rows, columns=cols).to_csv(matches_out, index=False)

    if metrics_out:
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_angular_error': mean_ae,
            'TP': int(TP),
            'FP': int(FP),
            'FN': int(FN),
        }
        with open(metrics_out, 'w') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, help='ground truth CSV (data/annotations.csv)')
    parser.add_argument('--pred', required=True, help='predictions CSV (outputs/results/predictions.csv)')
    parser.add_argument('--distance-threshold', type=float, default=20.0)
    parser.add_argument('--images-dir', help='limit evaluation to images in this directory (e.g., yolo_dataset/images/val)')
    parser.add_argument('--matches-out', default='outputs/results/matches.csv')
    parser.add_argument('--metrics-out', default='outputs/results/metrics.json')
    args = parser.parse_args()

    Path(args.matches_out).parent.mkdir(parents=True, exist_ok=True)
    evaluate(
        args.gt,
        args.pred,
        dist_thresh=args.distance_threshold,
        matches_out=args.matches_out,
        metrics_out=args.metrics_out,
        images_dir=args.images_dir,
    )
