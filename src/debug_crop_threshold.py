import cv2
import matplotlib.pyplot as plt
import pandas as pd


def main():
    image_path = 'data/images/fc1715f8-color.png'
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')

    gt = pd.read_csv('data/annotations.csv')
    gt = gt[gt['image'] == 'fc1715f8-color.png']
    if gt.empty:
        print('No annotations for image.')
        return

    h, w = image.shape[:2]
    pad = 10

    for idx, row in gt.iterrows():
        x1 = int(max(0, row['bbox_x'] - pad))
        y1 = int(max(0, row['bbox_y'] - pad))
        x2 = int(min(w, row['bbox_x'] + row['bbox_w'] + pad))
        y2 = int(min(h, row['bbox_y'] + row['bbox_h'] + pad))

        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        otsu_thresh, otsu_mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        print(f'Crop {idx}: Otsu threshold = {otsu_thresh:.2f}')

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Crop')
        axes[0].axis('off')

        axes[1].hist(gray.ravel(), bins=256, range=(0, 256))
        axes[1].set_title('Grayscale Histogram')
        axes[1].set_xlabel('Intensity')
        axes[1].set_ylabel('Count')

        axes[2].imshow(otsu_mask, cmap='gray')
        axes[2].set_title('Otsu Mask')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
