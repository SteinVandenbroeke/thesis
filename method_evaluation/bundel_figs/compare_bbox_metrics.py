import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from PIL import Image


def get_dataset_paths(dataset):
    """Matches the dataset paths defined in the original WSIS_metric_analyse.py script."""
    if dataset == "voc_val" or dataset == "voc_val_gt":
        label_file = "./data/VOC2012/annotations/voc_2012_val.json"
        root = './data/VOC2012/JPEGImages'
    elif dataset == "voc_train" or dataset == "voc_train_gt":
        label_file = "./data/VOC2012/annotations/voc_2012_trainaug.json"
        root = './data/VOC2012/JPEGImages'
    elif dataset == "coco_val" or dataset == "coco_val_gt":
        label_file = "./datasets/coco_dataset/coco2017/annotations/instances_val2017.json"
        root = './datasets/coco_dataset/coco2017/val2017'
    elif dataset == "cub_val":
        label_file = "./datasets/CUB_200_2011/CUB_as_COCO/annotations/instances_val2017.json"
        root = "./datasets/CUB_200_2011/CUB_200_2011/images_combined"
    else:
        raise ValueError(f"Dataset {dataset} path not configured in this script.")
    return label_file, root


def visualize_bbox_comparison(coco, root_dir, img_id, save_dir):
    """Generates a visual comparison between IoU and IoTA for bounding boxes."""

    # 1. Load Image
    img_info = coco.loadImgs([img_id])[0]
    img_path = os.path.join(root_dir, img_info['file_name'])

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    with Image.open(img_path).convert("RGB") as img_pil:
        img_arr = np.array(img_pil)

    h, w = img_info['height'], img_info['width']

    # 2. Load Ground Truths
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)

    # Filter out crowd annotations
    anns = [ann for ann in anns if ann.get('iscrowd', 0) == 0]

    if len(anns) < 2:
        print(f"Image {img_id} has fewer than 2 instances. Cannot visualize overlap.")
        return

    # Select the largest object as 'Target' and second largest as 'Intruder' for a 1-on-1 comparison
    anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    target_ann = anns[1]
    intruder_ann = anns[0]

    # Extract Bounding Boxes [x, y, width, height] -> [x1, y1, x2, y2]
    tx, ty, tw, th = target_ann['bbox']
    target_bbox = [int(tx), int(ty), int(tx + tw), int(ty + th)]

    ix, iy, iw, ih = intruder_ann['bbox']
    intruder_bbox = [int(ix), int(iy), int(ix + iw), int(iy + ih)]

    # --- SETUP BOOLEAN MASKS FOR EASY AREA CALCULATION ---
    t_box_mask = np.zeros((h, w), dtype=bool)
    t_box_mask[target_bbox[1]:target_bbox[3], target_bbox[0]:target_bbox[2]] = True

    i_box_mask = np.zeros((h, w), dtype=bool)
    i_box_mask[intruder_bbox[1]:intruder_bbox[3], intruder_bbox[0]:intruder_bbox[2]] = True

    # Calculate Intersection and Union
    intersection_mask = np.logical_and(t_box_mask, i_box_mask)
    union_mask = np.logical_or(t_box_mask, i_box_mask)

    intersection_area = intersection_mask.sum()
    target_area = t_box_mask.sum()
    union_area = union_mask.sum()

    # Compute Metrics
    iou_ratio = intersection_area / union_area if union_area > 0 else 0.0
    IoTA_ratio = intersection_area / target_area if target_area > 0 else 0.0

    # --- SET UP 1x3 GRID PLOT ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(f"Bounding Box Overlap: IoU vs. IoTA - Image ID: {img_id}", fontsize=20, fontweight='bold')

    # [Col 1]: Original Image + BBoxes
    bbox_viz = img_arr.copy()
    cv2.rectangle(bbox_viz, (target_bbox[0], target_bbox[1]), (target_bbox[2], target_bbox[3]), (0, 255, 0), 4)
    cv2.rectangle(bbox_viz, (intruder_bbox[0], intruder_bbox[1]), (intruder_bbox[2], intruder_bbox[3]), (255, 0, 0), 4)

    axes[0].imshow(bbox_viz)
    axes[0].set_title("1. Bounding Boxes\n(Target=Green, Intruder=Red)", fontsize=16)
    axes[0].axis('off')

    # [Col 2]: IoU (Intersection over Union)
    iou_viz = np.zeros_like(img_arr)
    # Color the union blue
    iou_viz[union_mask] = [50, 150, 255]
    # Color the intersection yellow
    iou_viz[intersection_mask] = [255, 255, 0]

    # Overlay onto grayscale background for context
    gray_bg = cv2.cvtColor(cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB) * 0.3
    iou_viz_combined = np.where(iou_viz > 0, iou_viz, gray_bg)

    axes[1].imshow(np.uint8(iou_viz_combined))
    axes[1].set_title(f"2. IoU Metric: {iou_ratio:.2%}\nYellow (Intersect) / Blue (Total Union)", fontsize=16,
                      fontweight='bold')
    axes[1].axis('off')

    # [Col 3]: IoTA (Intersection over Area)
    IoTA_viz = np.zeros_like(img_arr)
    # Color the target area green
    IoTA_viz[t_box_mask] = [0, 200, 50]
    # Color the intersection yellow
    IoTA_viz[intersection_mask] = [255, 255, 0]

    # Draw the red outline of the intruder to show where it's overlapping
    cv2.rectangle(IoTA_viz, (intruder_bbox[0], intruder_bbox[1]), (intruder_bbox[2], intruder_bbox[3]), (255, 0, 0), 3)

    IoTA_viz_combined = np.where(IoTA_viz > 0, IoTA_viz, gray_bg)

    axes[2].imshow(np.uint8(IoTA_viz_combined))
    axes[2].set_title(f"3. IoTA Metric: {IoTA_ratio:.2%}\nYellow (Intersect) / Green (Target Area)", fontsize=16,
                      fontweight='bold')
    axes[2].axis('off')

    # --- SAVE OUTPUT ---
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"bbox_iou_vs_IoTA_{img_id}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization for image {img_id} -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Bounding Box IoU vs IoTA.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., coco_val, voc_train)")
    parser.add_argument("--img_ids", type=str, required=True, help="Comma-separated list of image IDs")
    parser.add_argument("--save_dir", type=str, default="./bbox_comparisons", help="Where to save the outputs")

    args = parser.parse_args()

    # Fetch paths based on dataset mapping
    label_file, root_dir = get_dataset_paths(args.dataset)
    print(f"Loading annotations from: {label_file}")

    # Initialize COCO API
    coco = COCO(label_file)

    # Parse image IDs
    target_ids = [int(i.strip()) for i in args.img_ids.split(',')]

    for img_id in target_ids:
        visualize_bbox_comparison(coco, root_dir, img_id, args.save_dir)