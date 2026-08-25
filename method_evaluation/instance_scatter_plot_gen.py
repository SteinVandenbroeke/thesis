# -*- coding: utf-8 -*-
import os
import json
import argparse
import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm


# ==============================================================================
# HELPER FUNCTIONS (Preserved from original evaluation script)
# ==============================================================================

def calculate_instance_occlusion(gt_masks, gt_labels, neighbor_dist=10):
    all_occ_scores_any = []
    all_occ_scores_same = []

    kernel_size = neighbor_dist * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for masks, labels in zip(gt_masks, gt_labels):
        n = masks.shape[0]
        scores_any = []
        scores_same = []

        dilated_borders = []
        border_areas = []

        for i in range(n):
            mask_i = masks[i].astype(np.uint8)
            if mask_i.sum() == 0:
                dilated_borders.append(np.zeros_like(mask_i, dtype=bool))
                border_areas.append(0)
                continue

            dilated_mask = cv2.dilate(mask_i, kernel, iterations=1)
            border = (dilated_mask > 0) & (mask_i == 0)

            dilated_borders.append(border)
            border_areas.append(border.sum())

        for i in range(n):
            if border_areas[i] == 0:
                scores_any.append(0.0)
                scores_same.append(0.0)
                continue

            border_i = dilated_borders[i]
            max_occ_any = 0.0
            max_occ_same = 0.0

            for j in range(n):
                if i == j: continue
                mask_j = masks[j].astype(bool)

                overlap_pixels = np.logical_and(border_i, mask_j).sum()
                occ_ratio = overlap_pixels / border_areas[i]

                if occ_ratio > max_occ_any:
                    max_occ_any = occ_ratio

                if labels[i] == labels[j]:
                    if occ_ratio > max_occ_same:
                        max_occ_same = occ_ratio

            scores_any.append(max_occ_any)
            scores_same.append(max_occ_same)

        all_occ_scores_any.append(np.array(scores_any))
        all_occ_scores_same.append(np.array(scores_same))

    return all_occ_scores_any, all_occ_scores_same


def get_chainercv_format(coco_api, imgIds, is_gt=True, cat2idx=None, max_dets=100, score_thr=0.3):
    all_masks, all_labels, all_scores = [], [], []

    if cat2idx is None:
        catIds = sorted(coco_api.getCatIds())
        cat2idx = {cat: i for i, cat in enumerate(catIds)}

    for img_id in tqdm(imgIds, desc=f"Converting {'GT' if is_gt else 'Pred'} formats"):
        ann_ids = coco_api.getAnnIds(imgIds=[img_id])
        anns = coco_api.loadAnns(ann_ids)

        if not is_gt:
            anns = sorted(anns, key=lambda x: x.get('score', 0), reverse=True)
            anns = [ann for ann in anns if ann.get('score', 0) >= score_thr]
            anns = anns[:max_dets]

        img_info = coco_api.loadImgs(img_id)[0]
        h, w = img_info['height'], img_info['width']

        masks, labels, scores = [], [], []

        for ann in anns:
            if is_gt and ann.get('iscrowd', 0) == 1:
                continue

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

            masks.append(mask.astype(bool))
            labels.append(cat2idx[ann['category_id']])
            if not is_gt:
                scores.append(ann['score'])

        if len(masks) > 0:
            all_masks.append(np.stack(masks, axis=0))
            all_labels.append(np.array(labels, dtype=np.int32))
            if not is_gt:
                all_scores.append(np.array(scores, dtype=np.float32))
        else:
            all_masks.append(np.empty((0, h, w), dtype=bool))
            all_labels.append(np.empty((0,), dtype=np.int32))
            if not is_gt:
                all_scores.append(np.empty((0,), dtype=np.float32))

    if is_gt:
        return all_masks, all_labels, cat2idx
    else:
        return all_masks, all_labels, all_scores


# ==============================================================================
# MAIN SCRIPT LOGIC
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scatter Plot for Instance Counts")
    parser.add_argument("--dataset", type=str, required=True, help="E.g., voc_val, coco_val")
    parser.add_argument("--result_files", type=str, required=True, help="Path to predicted JSON results")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the plots")
    parser.add_argument("--num", type=int, default=None, help="Number of images to process")
    args = parser.parse_args()

    # 1. Dataset Selection
    if args.dataset == "voc_val":
        label_file = "./data/VOC2012/annotations/voc_2012_val.json"
    elif args.dataset == "coco_val":
        label_file = "./datasets/coco_dataset/coco2017/annotations/instances_val2017.json"
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not fully configured in this snippet. Please add its path.")

    os.makedirs(args.save_dir, exist_ok=True)

    # 2. Load GT and Pred Annotations
    print(f"Loading GT annotations from {label_file}...")
    cocoGt = COCO(label_file)
    imgIds = sorted(cocoGt.getImgIds())
    if args.num is not None:
        imgIds = imgIds[:args.num]

    print(f"Loading Predictions from {args.result_files}...")
    try:
        res = json.load(open(args.result_files))
        if "annotations" in res.keys():
            temp_filename = os.path.join(args.save_dir, 'temp_eval.json')
            with open(temp_filename, 'w') as file_obj:
                json.dump(res['annotations'], file_obj)
            result_file = temp_filename
        else:
            result_file = args.result_files
    except:
        result_file = args.result_files

    cocoDt = cocoGt.loadRes(result_file)

    # 3. Extract masks and calculate occ metrics
    gt_masks, gt_labels, cat2idx = get_chainercv_format(cocoGt, imgIds, is_gt=True)
    p_masks, p_labels, p_scores = get_chainercv_format(cocoDt, imgIds, is_gt=False, cat2idx=cat2idx)

    print("\nCalculating Occlusion Scores...")
    occ_scores_any, occ_scores_same = calculate_instance_occlusion(gt_masks, gt_labels, neighbor_dist=10)

    import matplotlib.colors as mcolors
    import matplotlib.colors as mcolors

    # 4. Count instances based on conditions
    gt_all, pred_all = [], []
    gt_occ_none, pred_occ_none = [], []
    gt_occ_any, pred_occ_any = [], []
    gt_occ_same, pred_occ_same = [], []

    for i in range(len(imgIds)):
        n_gt = len(gt_masks[i])
        n_pred = len(p_masks[i])

        # Max occlusion score for the image
        max_any = np.max(occ_scores_any[i]) if len(occ_scores_any[i]) > 0 else 0
        max_same = np.max(occ_scores_same[i]) if len(occ_scores_same[i]) > 0 else 0

        # Condition 1: All Images
        gt_all.append(n_gt)
        pred_all.append(n_pred)

        # Condition 2: No Occlusion
        if max_any == 0.0:
            gt_occ_none.append(n_gt)
            pred_occ_none.append(n_pred)

        # Condition 3: Any Occlusion
        if max_any > 0.0:
            gt_occ_any.append(n_gt)
            pred_occ_any.append(n_pred)

        # Condition 4: Same-Class Occlusion
        if max_same > 0.0:
            gt_occ_same.append(n_gt)
            pred_occ_same.append(n_pred)

    # 5. Plotting in a 2x2 grid
    print("\nGenerating 2D Heatmaps...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    titles = [
        "Plot 1: All Images",
        "Plot 2: Images with No Occlusion",
        "Plot 3: Images with Any Class Occlusion",
        "Plot 4: Images with Same Class Occlusion"
    ]

    data_pairs = [
        (gt_all, pred_all),
        (gt_occ_none, pred_occ_none),
        (gt_occ_any, pred_occ_any),
        (gt_occ_same, pred_occ_same)
    ]

    # Use .flatten() to loop through the 2x2 axes array as a flat list
    for idx, ax in enumerate(axes.flatten()):
        gt_data, pred_data = data_pairs[idx]

        if not gt_data:
            ax.set_title(titles[idx])
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            continue

        # Hardcoded maximum value
        max_val = 20

        # Create discrete bins centered on the integers (-0.5 to 0.5, 0.5 to 1.5, ... up to 10.5)
        # Using max_val + 1.5 ensures the final bin edge is 10.5
        bins = np.arange(-0.5, max_val + 1.5, 1)

        # Plot 2D Histogram (Heatmap)
        h, xedges, yedges, image = ax.hist2d(
            gt_data, pred_data,
            bins=[bins, bins],
            cmap='Blues',
            cmin=1,  # Don't color empty bins
            norm=mcolors.LogNorm()
        )

        # Add a colorbar for this specific subplot
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Number of Images', fontsize=10)

        # Plot ideal diagonal
        ax.plot([-0.5, max_val + 0.5], [-0.5, max_val + 0.5], 'r--', label='Ideal (GT = Pred)')

        ax.set_title(titles[idx], fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of GT Instances', fontsize=12)
        ax.set_ylabel('Number of Predicted Instances', fontsize=12)
        ax.set_xlim(-0.5, max_val + 0.5)
        ax.set_ylim(-0.5, max_val + 0.5)

        # Force integer ticks on axes up to max_val
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.get_major_locator().set_params(integer=True)

        ax.legend(loc='upper left')

    plt.tight_layout()
    save_path = os.path.join(args.save_dir, "instance_counts_heatmap_2x2.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Heatmaps successfully saved to {save_path}")