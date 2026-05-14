import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


def select_contour(contours, crop_shape):
    if not contours:
        return None, None, None
    h, w = crop_shape[:2]
    center_x = w / 2.0
    center_y = h / 2.0
    candidates = []
    for c in contours:
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        dist = math.hypot(cx - center_x, cy - center_y)
        candidates.append((dist, c, cx, cy))
    if not candidates:
        return None, None, None
    _, c, cx, cy = min(candidates, key=lambda x: x[0])
    return c, cx, cy


def longest_edge(approx):
    verts = approx[:, 0, :].astype(float)
    max_len = -1.0
    v1 = None
    v2 = None
    for i in range(len(verts)):
        p1 = verts[i]
        p2 = verts[(i + 1) % len(verts)]
        seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if seg_len > max_len:
            max_len = seg_len
            v1, v2 = p1, p2
    return v1, v2, max_len


def compute_angle_from_edge(v1, v2, cx, cy):
    mid_x = (v1[0] + v2[0]) / 2.0
    mid_y = (v1[1] + v2[1]) / 2.0
    dx = v2[0] - v1[0]
    dy = v2[1] - v1[1]
    norm = math.hypot(dx, dy)
    if norm == 0:
        return None, None, None
    dx /= norm
    dy /= norm
    perp1 = (-dy, dx)
    perp2 = (dy, -dx)
    to_center = (cx - mid_x, cy - mid_y)
    dot1 = perp1[0] * to_center[0] + perp1[1] * to_center[1]
    dot2 = perp2[0] * to_center[0] + perp2[1] * to_center[1]
    perp_dx, perp_dy = perp1 if dot1 >= dot2 else perp2
    angle = (math.degrees(math.atan2(-perp_dy, perp_dx))) % 360
    return (mid_x, mid_y), (perp_dx, perp_dy), angle


def main():
    image_path = 'data/images/fc1715f8-color.png'
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')

    gt = pd.read_csv('data/annotations.csv')
    gt = gt[gt['image'] == 'fc1715f8-color.png']

    model = YOLO('runs/detect/runs/tube_lid/weights/best.pt')
    results = model(image, conf=0.25, iou=0.45)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        print('No detections found.')
        return

    h_img, w_img = image.shape[:2]
    pad = 4

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w_img, x2 + pad)
        cy2 = min(h_img, y2 + pad)

        crop = image[cy1:cy2, cx1:cx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c, ccx, ccy = select_contour(contours, crop.shape)

        if c is None:
            full_cx = (x1 + x2) / 2.0
            full_cy = (y1 + y2) / 2.0
            angle = 0.0
            midpoint = (crop.shape[1] / 2.0, crop.shape[0] / 2.0)
            perp = (1.0, 0.0)
            approx = None
        else:
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if approx is None or len(approx) < 2:
                full_cx = cx1 + ccx
                full_cy = cy1 + ccy
                angle = 0.0
                midpoint = (ccx, ccy)
                perp = (1.0, 0.0)
            else:
                v1, v2, max_len = longest_edge(approx)
                if v1 is None or v2 is None or max_len <= 0:
                    full_cx = cx1 + ccx
                    full_cy = cy1 + ccy
                    angle = 0.0
                    midpoint = (ccx, ccy)
                    perp = (1.0, 0.0)
                else:
                    midpoint, perp, angle = compute_angle_from_edge(v1, v2, ccx, ccy)
                    full_cx = cx1 + ccx
                    full_cy = cy1 + ccy

        gt_angle = None
        if not gt.empty:
            gt_pts = gt[['center_x', 'center_y', 'angle_deg']].values
            dists = np.sqrt((gt_pts[:, 0] - full_cx) ** 2 + (gt_pts[:, 1] - full_cy) ** 2)
            gt_angle = float(gt_pts[int(np.argmin(dists))][2])

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(closed, cmap='gray')
        axes[0].set_title('Binary Mask')
        axes[0].axis('off')

        contour_vis = crop.copy()
        if c is not None:
            cv2.drawContours(contour_vis, [c], -1, (0, 255, 0), 1)
            if approx is not None:
                for p in approx[:, 0, :]:
                    cv2.circle(contour_vis, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)
                v1, v2, _ = longest_edge(approx)
                if v1 is not None and v2 is not None:
                    v1i = (int(v1[0]), int(v1[1]))
                    v2i = (int(v2[0]), int(v2[1]))
                    cv2.line(contour_vis, v1i, v2i, (0, 0, 255), 2)
                    mid_i = (int((v1[0] + v2[0]) / 2.0), int((v1[1] + v2[1]) / 2.0))
                    cv2.circle(contour_vis, mid_i, 3, (0, 0, 255), -1)

        axes[1].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Contour + Flat Edge')
        axes[1].axis('off')

        angle_vis = crop.copy()
        if c is not None:
            cv2.circle(angle_vis, (int(ccx), int(ccy)), 3, (255, 0, 0), -1)
            mid_x, mid_y = midpoint
            perp_dx, perp_dy = perp
            arrow_end = (int(mid_x + perp_dx * 20), int(mid_y + perp_dy * 20))
            cv2.arrowedLine(angle_vis, (int(mid_x), int(mid_y)), arrow_end, (0, 255, 255), 2)
            angle_rad = math.radians(angle)
            end_x = int(ccx + 20 * math.cos(angle_rad))
            end_y = int(ccy - 20 * math.sin(angle_rad))
            cv2.arrowedLine(angle_vis, (int(ccx), int(ccy)), (end_x, end_y), (0, 0, 255), 2)

        title = f'Pred: {angle:.1f} deg'
        if gt_angle is not None:
            title = f'{title} | GT: {gt_angle:.1f} deg'
        axes[2].imshow(cv2.cvtColor(angle_vis, cv2.COLOR_BGR2RGB))
        axes[2].set_title(title)
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
        input('Press Enter for next lid...')


if __name__ == '__main__':
    main()
