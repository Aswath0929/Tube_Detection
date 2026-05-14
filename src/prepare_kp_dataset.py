from pathlib import Path
import shutil
import random
import math
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data'
    images_dir = data_dir / 'images'
    ann_path = data_dir / 'annotations.csv'

    out_root = root / 'yolo_kp_dataset'
    img_out = out_root / 'images'
    lbl_out = out_root / 'labels'

    df = pd.read_csv(ann_path)
    groups = df.groupby('image')
    images = sorted(groups.groups.keys())

    random.seed(42)
    random.shuffle(images)
    n = len(images)
    n_train = int(n * 0.8)
    train_images = set(images[:n_train])
    val_images = set(images[n_train:])

    for p in [img_out / 'train', img_out / 'val', lbl_out / 'train', lbl_out / 'val']:
        p.mkdir(parents=True, exist_ok=True)

    def write_image_and_labels(img_name, split):
        src_img = images_dir / img_name
        dst_img = img_out / split / img_name
        shutil.copyfile(src_img, dst_img)

        rows = groups.get_group(img_name)
        label_lines = []
        for _, r in rows.iterrows():
            bbox_cx = (r['bbox_x'] + r['bbox_w'] / 2.0) / 640.0
            bbox_cy = (r['bbox_y'] + r['bbox_h'] / 2.0) / 480.0
            bbox_w = r['bbox_w'] / 640.0
            bbox_h = r['bbox_h'] / 480.0

            center_x = r['center_x'] / 640.0
            center_y = r['center_y'] / 480.0

            ang = math.radians(r['angle_deg'])
            tab_x = (r['center_x'] + 20.0 * math.cos(ang)) / 640.0
            tab_y = (r['center_y'] - 20.0 * math.sin(ang)) / 480.0

            label_lines.append(
                f"0 {bbox_cx:.6f} {bbox_cy:.6f} {bbox_w:.6f} {bbox_h:.6f} "
                f"{center_x:.6f} {center_y:.6f} 2 {tab_x:.6f} {tab_y:.6f} 2"
            )

        lbl_path = lbl_out / split / (Path(img_name).stem + '.txt')
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(label_lines))

    for img in images:
        if img in train_images:
            write_image_and_labels(img, 'train')
        else:
            write_image_and_labels(img, 'val')

    yaml_text = (
        "path: yolo_kp_dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['tube_lid']\n"
        "kpt_shape: [2, 3]\n"
    )
    (out_root / 'dataset.yaml').write_text(yaml_text)

    print(f"Total images: {n}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    print(f"Total annotations: {len(df)}")


if __name__ == '__main__':
    main()
