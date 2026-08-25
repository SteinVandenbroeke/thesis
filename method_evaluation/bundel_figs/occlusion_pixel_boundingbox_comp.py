import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
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


def extract_mask(ann, h, w):
    """Extracts a binary mask from a COCO annotation."""
    if type(ann['segmentation']) == list:
        rle = maskUtils.frPyObjects(ann['segmentation'], h, w)
        mask = maskUtils.decode(rle)
        if len(mask.shape) > 2:
            mask = np.max(mask, axis=2)
    elif type(ann['segmentation']['counts']) == list:
        rle = maskUtils.frPyObjects([ann['segmentation']], h, w)
        mask = maskUtils.decode(rle)[:, :, 0]
    else:
        rle = [ann['segmentation']]
        mask = maskUtils.decode(rle)[:, :, 0]
    return mask.astype(bool)


def visualize_occlusion_steps(coco, root_dir, img_id, save_dir, border_thickness=5):
    """Generates a two-row step-by-step visualization for a given image ID."""

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
        print(f"Image {img_id} has fewer than 2 instances. Cannot visualize overlapping occlusion.")
        return

    # Select the largest object as the 'Target' and combine all others as 'Intruders'
    anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    target_ann = anns[0]
    intruder_anns = anns[1:]

    # Extract Masks
    target_mask = extract_mask(target_ann, h, w)
    intruder_mask_combined = np.zeros((h, w), dtype=bool)
    for ann in intruder_anns:
        intruder_mask_combined = np.logical_or(intruder_mask_combined, extract_mask(ann, h, w))

    # Extract Bounding Boxes
    tx, ty, tw, th = target_ann['bbox']
    target_bbox = [int(tx), int(ty), int(tx + tw), int(ty + th)]

    intruder_bboxes = []
    for ann in intruder_anns:
        ix, iy, iw, ih = ann['bbox']
        intruder_bboxes.append([int(ix), int(iy), int(ix + iw), int(iy + ih)])

    # --- MASK CALCULATION (Halo Method) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_thickness * 2 + 1, border_thickness * 2 + 1))
    dilated_mask = cv2.dilate(target_mask.astype(np.uint8), kernel, iterations=1)
    halo_mask = (dilated_mask > 0) & (target_mask == 0)
    halo_overlap = np.logical_and(halo_mask, intruder_mask_combined)

    halo_area = halo_mask.sum()
    mask_occ_ratio = halo_overlap.sum() / halo_area if halo_area > 0 else 0.0

    # --- SET UP 2x3 GRID PLOT ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Occlusion Analysis: Pixels vs. Bounding Boxes - Image ID: {img_id}", fontsize=20, fontweight='bold')

    # ==========================================
    # ROW 1: PIXEL (MASK) METHOD
    # ==========================================

    # [Row 1, Col 1]: GT Masks Overlay
    gt_viz = img_arr.copy()
    gt_viz[target_mask] = gt_viz[target_mask] * 0.5 + np.array([0, 255, 0]) * 0.5  # Green Target
    gt_viz[intruder_mask_combined] = gt_viz[intruder_mask_combined] * 0.5 + np.array([255, 0, 0]) * 0.5  # Red Intruders
    axes[0, 0].imshow(np.uint8(gt_viz))
    axes[0, 0].set_title("1. Ground Truth Masks\n(Target=Green, Intruders=Red)", fontsize=14)
    axes[0, 0].axis('off')

    # [Row 1, Col 2]: Halo Calculation
    halo_viz = np.zeros_like(img_arr)
    halo_viz[target_mask] = [50, 50, 50]  # Dim target
    halo_viz[halo_mask] = [0, 255, 255]  # Cyan Halo
    axes[0, 1].imshow(halo_viz)
    axes[0, 1].set_title(f"2. Target Halo Expansion\n(Border Area: {halo_area}px)", fontsize=14)
    axes[0, 1].axis('off')

    # [Row 1, Col 3]: Mask Occlusion
    occ_viz = halo_viz.copy()
    occ_viz[intruder_mask_combined] = [255, 0, 0]  # Red intruders
    occ_viz[halo_overlap] = [255, 255, 0]  # Yellow highlight where they touch the halo
    axes[0, 2].imshow(occ_viz)
    axes[0, 2].set_title(f"3. Mask Occlusion Ratio: {mask_occ_ratio:.2%}\n(Yellow shows intrusion)", fontsize=14,
                         fontweight='bold')
    axes[0, 2].axis('off')

    # ==========================================
    # ROW 2: BOUNDING BOX (IoA) METHOD
    # ==========================================

    # [Row 2, Col 1]: Original Image for baseline
    axes[1, 0].imshow(img_arr)
    axes[1, 0].set_title("4. Original Image Baseline", fontsize=14)
    axes[1, 0].axis('off')

    # [Row 2, Col 2]: Target & Intruder BBoxes
    bbox_viz = img_arr.copy()
    cv2.rectangle(bbox_viz, (target_bbox[0], target_bbox[1]), (target_bbox[2], target_bbox[3]), (0, 255, 0), 3)
    for ibox in intruder_bboxes:
        cv2.rectangle(bbox_viz, (ibox[0], ibox[1]), (ibox[2], ibox[3]), (255, 0, 0), 3)
    axes[1, 1].imshow(bbox_viz)
    axes[1, 1].set_title("5. Target & Intruder BBoxes\n(Target=Green, Intruders=Red)", fontsize=14)
    axes[1, 1].axis('off')

    # [Row 2, Col 3]: Bounding Box Occlusion (IoA)
    bbox_occ_viz = bbox_viz.copy()
    target_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])

    # Create empty masks just for the boxes to visually highlight intersection
    t_box_mask = np.zeros((h, w), dtype=bool)
    t_box_mask[target_bbox[1]:target_bbox[3], target_bbox[0]:target_bbox[2]] = True

    i_box_mask = np.zeros((h, w), dtype=bool)
    for ibox in intruder_bboxes:
        i_box_mask[ibox[1]:ibox[3], ibox[0]:ibox[2]] = True

    box_overlap = np.logical_and(t_box_mask, i_box_mask)
    bbox_ioa = box_overlap.sum() / target_area if target_area > 0 else 0.0

    # Flood the overlapping box area with yellow
    bbox_occ_viz[box_overlap] = bbox_occ_viz[box_overlap] * 0.3 + np.array([255, 255, 0]) * 0.7

    axes[1, 2].imshow(np.uint8(bbox_occ_viz))
    axes[1, 2].set_title(f"6. BBox Occlusion (IoA): {bbox_ioa:.2%}\n(Yellow shows intersection)", fontsize=14,
                         fontweight='bold')
    axes[1, 2].axis('off')

    # --- SAVE OUTPUT ---
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"occlusion_vis_pixels_vs_bboxes_{img_id}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization for image {img_id} -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize occlusion calculations (Pixels vs. Bounding Boxes).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., coco_val, voc_train)")
    parser.add_argument("--img_ids", type=str, required=True, help="Comma-separated list of image IDs")
    parser.add_argument("--save_dir", type=str, default="./occlusion_visualizations", help="Where to save the outputs")

    args = parser.parse_args()

    # Fetch paths based on dataset mapping
    label_file, root_dir = get_dataset_paths(args.dataset)
    print(f"Loading annotations from: {label_file}")

    # Initialize COCO API
    coco = COCO(label_file)

    # Parse image IDs
    target_ids = [int(i.strip()) for i in args.img_ids.split(',')]

    for img_id in target_ids:
        visualize_occlusion_steps(coco, root_dir, img_id, args.save_dir)