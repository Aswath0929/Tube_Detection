from pathlib import Path
import shutil
import random
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data'
    images_dir = data_dir / 'images'
    ann_path = data_dir / 'annotations.csv'

    out_root = root / 'yolo_dataset'
    img_out = out_root / 'images'
    lbl_out = out_root / 'labels'

    # read annotations
    df = pd.read_csv(ann_path)
    # expected columns: image, center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h, bbox_rotation, angle_deg

    # group by image
    groups = df.groupby('image')
    images = sorted(groups.groups.keys())

    random.seed(42)
    random.shuffle(images)
    n = len(images)
    n_train = int(n * 0.8)
    train_images = set(images[:n_train])
    val_images = set(images[n_train:])

    # create dirs
    for p in [img_out / 'train', img_out / 'val', lbl_out / 'train', lbl_out / 'val']:
        p.mkdir(parents=True, exist_ok=True)

    total_ann = 0

    def write_image_and_labels(img_name, split):
        nonlocal total_ann
        src_img = images_dir / img_name
        dst_img = img_out / split / img_name
        shutil.copyfile(src_img, dst_img)

        rows = groups.get_group(img_name)
        label_lines = []
        for _, r in rows.iterrows():
            cx = (r['bbox_x'] + r['bbox_w'] / 2.0) / 640.0
            cy = (r['bbox_y'] + r['bbox_h'] / 2.0) / 480.0
            w = r['bbox_w'] / 640.0
            h = r['bbox_h'] / 480.0
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_ann += 1

        lbl_path = lbl_out / split / (Path(img_name).stem + '.txt')
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(label_lines))

    for img in images:
        if img in train_images:
            write_image_and_labels(img, 'train')
        else:
            write_image_and_labels(img, 'val')

    # write dataset.yaml
    yaml_text = (
        "path: yolo_dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['tube_lid']\n"
    )
    (out_root / 'dataset.yaml').write_text(yaml_text)

    print(f"Total images: {n}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    print(f"Total annotations: {total_ann}")


if __name__ == '__main__':
    main()
