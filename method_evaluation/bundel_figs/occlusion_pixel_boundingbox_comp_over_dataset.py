import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
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


def generate_visualization(img_arr, target_mask, intruder_mask_combined, target_bbox, intruder_bboxes,
                           halo_mask, halo_overlap, box_overlap, halo_area, mask_occ_ratio,
                           target_area, bbox_IoTA, img_id, save_path):
    """Generates the 2x3 visualization grid for the agreed-upon target."""
    h, w = img_arr.shape[:2]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    diff = bbox_IoTA - mask_occ_ratio
    fig.suptitle(f"Occlusion Analysis - Image ID: {img_id} | Abs Difference: {abs(diff):.2%}", fontsize=20,
                 fontweight='bold')

    # --- ROW 1: PIXEL (MASK) METHOD ---
    gt_viz = img_arr.copy()
    gt_viz[target_mask] = gt_viz[target_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    gt_viz[intruder_mask_combined] = gt_viz[intruder_mask_combined] * 0.5 + np.array([255, 0, 0]) * 0.5
    axes[0, 0].imshow(np.uint8(gt_viz))
    axes[0, 0].set_title("1. Ground Truth Masks\n(Target=Green, Intruders=Red)", fontsize=14)
    axes[0, 0].axis('off')

    halo_viz = np.zeros_like(img_arr)
    halo_viz[target_mask] = [50, 50, 50]
    halo_viz[halo_mask] = [0, 255, 255]
    axes[0, 1].imshow(halo_viz)
    axes[0, 1].set_title(f"2. Target Halo Expansion\n(Border Area: {halo_area}px)", fontsize=14)
    axes[0, 1].axis('off')

    occ_viz = halo_viz.copy()
    occ_viz[intruder_mask_combined] = [255, 0, 0]
    occ_viz[halo_overlap] = [255, 255, 0]
    axes[0, 2].imshow(occ_viz)
    axes[0, 2].set_title(f"3. Mask Occlusion Ratio: {mask_occ_ratio:.2%}\n(Yellow shows intrusion)", fontsize=14,
                         fontweight='bold')
    axes[0, 2].axis('off')

    # --- ROW 2: BOUNDING BOX (IoTA) METHOD ---
    axes[1, 0].imshow(img_arr)
    axes[1, 0].set_title("4. Original Image Baseline", fontsize=14)
    axes[1, 0].axis('off')

    bbox_viz = img_arr.copy()
    cv2.rectangle(bbox_viz, (target_bbox[0], target_bbox[1]), (target_bbox[2], target_bbox[3]), (0, 255, 0), 3)
    for ibox in intruder_bboxes:
        cv2.rectangle(bbox_viz, (ibox[0], ibox[1]), (ibox[2], ibox[3]), (255, 0, 0), 3)
    axes[1, 1].imshow(bbox_viz)
    axes[1, 1].set_title("5. Target & Intruder BBoxes\n(Target=Green, Intruders=Red)", fontsize=14)
    axes[1, 1].axis('off')

    bbox_occ_viz = bbox_viz.copy()
    bbox_occ_viz[box_overlap] = bbox_occ_viz[box_overlap] * 0.3 + np.array([255, 255, 0]) * 0.7
    axes[1, 2].imshow(np.uint8(bbox_occ_viz))
    axes[1, 2].set_title(f"6. BBox Occlusion (IoTA): {bbox_IoTA:.2%}\n(Yellow shows intersection)", fontsize=14,
                         fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


def analyze_dataset_occlusion_subdivided(dataset_name, limit=None, plot_limit=20, border_thickness=5,
                                         save_dir="./dataset_metrics"):
    os.makedirs(save_dir, exist_ok=True)
    examples_dir = os.path.join(save_dir, "generated_examples")
    os.makedirs(examples_dir, exist_ok=True)

    label_file, root_dir = get_dataset_paths(dataset_name)
    print(f"Loading annotations from: {label_file}")
    coco = COCO(label_file)

    img_ids = coco.getImgIds()
    if limit:
        img_ids = img_ids[:limit]
        print(f"Limiting analysis to first {limit} images...")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_thickness * 2 + 1, border_thickness * 2 + 1))

    results = {
        "Image_ID": [],
        "Target_Instance_ID": [],
        "Max_Mask_Halo_Occlusion": [],
        "Max_BBox_IoTA": []
    }

    images_processed = 0
    images_kept = 0
    images_filtered = 0

    # Initialize trackers for the expanded subdirectories
    categories = [
        "Diff_00_to_10_percent",
        "Diff_10_to_30_percent",
        "Diff_30_to_40_percent",
        "Diff_40_to_50_percent",
        "Diff_50_to_60_percent",
        "Diff_60_to_70_percent",
        "Diff_70_to_80_percent",
        "Diff_gt_80_percent"
    ]
    plots_generated = {cat: 0 for cat in categories}

    for img_id in tqdm(img_ids, desc="Processing Dataset"):
        img_info = coco.loadImgs([img_id])[0]
        h, w = img_info['height'], img_info['width']

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        anns = [ann for ann in anns if ann.get('iscrowd', 0) == 0]

        if len(anns) < 2:
            continue

        images_processed += 1
        masks = [extract_mask(ann, h, w) for ann in anns]

        img_mask_occs = []
        img_bbox_IoTAs = []
        instance_ids = []

        for i, target_ann in enumerate(anns):
            t_mask = masks[i]
            tx, ty, tw, th = target_ann['bbox']
            t_bbox = [int(tx), int(ty), int(tx + tw), int(ty + th)]
            t_area = max(0, tw * th)

            if t_area == 0:
                continue

            dilated_mask = cv2.dilate(t_mask.astype(np.uint8), kernel, iterations=1)
            halo_mask = (dilated_mask > 0) & (t_mask == 0)
            halo_area = halo_mask.sum()

            intruder_masks = [masks[j] for j in range(len(anns)) if j != i]
            combined_intruders = np.any(intruder_masks, axis=0) if intruder_masks else np.zeros((h, w), dtype=bool)

            halo_overlap = np.logical_and(halo_mask, combined_intruders)
            halo_overlap_sum = halo_overlap.sum()
            mask_occ_ratio = halo_overlap_sum / halo_area if halo_area > 0 else 0.0

            t_box_mask = np.zeros((h, w), dtype=bool)
            x1, y1, x2, y2 = max(0, t_bbox[0]), max(0, t_bbox[1]), min(w, t_bbox[2]), min(h, t_bbox[3])
            t_box_mask[y1:y2, x1:x2] = True

            i_box_mask = np.zeros((h, w), dtype=bool)
            for j, i_ann in enumerate(anns):
                if i == j: continue
                ix, iy, iw, ih = i_ann['bbox']
                ix1, iy1, ix2, iy2 = max(0, int(ix)), max(0, int(iy)), min(w, int(ix + iw)), min(h, int(iy + ih))
                i_box_mask[iy1:iy2, ix1:ix2] = True

            box_overlap = np.logical_and(t_box_mask, i_box_mask)
            box_overlap_sum = box_overlap.sum()
            actual_t_area = t_box_mask.sum()
            bbox_IoTA = box_overlap_sum / actual_t_area if actual_t_area > 0 else 0.0

            img_mask_occs.append(mask_occ_ratio)
            img_bbox_IoTAs.append(bbox_IoTA)
            instance_ids.append(target_ann['id'])

        if img_mask_occs and img_bbox_IoTAs:
            max_mask_idx = np.argmax(img_mask_occs)
            max_bbox_idx = np.argmax(img_bbox_IoTAs)

            # If both metrics identify the exact same instance as the most occluded
            if instance_ids[max_mask_idx] == instance_ids[max_bbox_idx]:
                mask_val = img_mask_occs[max_mask_idx]
                bbox_val = img_bbox_IoTAs[max_bbox_idx]

                results["Image_ID"].append(img_id)
                results["Target_Instance_ID"].append(instance_ids[max_mask_idx])
                results["Max_Mask_Halo_Occlusion"].append(mask_val)
                results["Max_BBox_IoTA"].append(bbox_val)
                images_kept += 1

                # Determine category based on absolute difference with expanded bins
                abs_diff = abs(bbox_val - mask_val)
                if abs_diff <= 0.10:
                    cat = categories[0]
                elif abs_diff <= 0.30:
                    cat = categories[1]
                elif abs_diff <= 0.40:
                    cat = categories[2]
                elif abs_diff <= 0.50:
                    cat = categories[3]
                elif abs_diff <= 0.60:
                    cat = categories[4]
                elif abs_diff <= 0.70:
                    cat = categories[5]
                elif abs_diff <= 0.80:
                    cat = categories[6]
                else:
                    cat = categories[7]

                # Plot the image if we haven't hit the plotting limit for THIS category
                if plot_limit is None or plots_generated[cat] < plot_limit:
                    img_path = os.path.join(root_dir, img_info['file_name'])
                    if os.path.exists(img_path):
                        with Image.open(img_path).convert("RGB") as img_pil:
                            img_arr = np.array(img_pil)

                        target_ann = anns[max_mask_idx]
                        t_mask = masks[max_mask_idx]

                        tx, ty, tw, th = target_ann['bbox']
                        t_bbox = [int(tx), int(ty), int(tx + tw), int(ty + th)]

                        intruder_bboxes = []
                        intruder_masks = []
                        for j, i_ann in enumerate(anns):
                            if max_mask_idx == j: continue
                            ix, iy, iw, ih = i_ann['bbox']
                            intruder_bboxes.append([int(ix), int(iy), int(ix + iw), int(iy + ih)])
                            intruder_masks.append(masks[j])

                        combined_intruders = np.any(intruder_masks, axis=0)

                        dilated_mask = cv2.dilate(t_mask.astype(np.uint8), kernel, iterations=1)
                        halo_mask = (dilated_mask > 0) & (t_mask == 0)
                        halo_overlap = np.logical_and(halo_mask, combined_intruders)

                        t_box_mask = np.zeros((h, w), dtype=bool)
                        x1, y1, x2, y2 = max(0, t_bbox[0]), max(0, t_bbox[1]), min(w, t_bbox[2]), min(h, t_bbox[3])
                        t_box_mask[y1:y2, x1:x2] = True

                        i_box_mask = np.zeros((h, w), dtype=bool)
                        for ibox in intruder_bboxes:
                            ix1, iy1, ix2, iy2 = max(0, ibox[0]), max(0, ibox[1]), min(w, ibox[2]), min(h, ibox[3])
                            i_box_mask[iy1:iy2, ix1:ix2] = True

                        box_overlap = np.logical_and(t_box_mask, i_box_mask)

                        # Ensure subfolder exists
                        cat_dir = os.path.join(examples_dir, cat)
                        os.makedirs(cat_dir, exist_ok=True)

                        save_path = os.path.join(cat_dir, f"diff_{abs_diff:.2f}_img_{img_id}.png")

                        generate_visualization(
                            img_arr, t_mask, combined_intruders, t_bbox, intruder_bboxes,
                            halo_mask, halo_overlap, box_overlap, halo_mask.sum(), mask_val,
                            t_box_mask.sum(), bbox_val, img_id, save_path
                        )
                        plots_generated[cat] += 1
            else:
                images_filtered += 1

    # --- DATA ANALYSIS & VISUALIZATION ---
    df = pd.DataFrame(results)

    print("\n" + "=" * 50)
    print("SUBDIVIDED IMAGE-LEVEL ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Valid multi-object images processed: {images_processed}")
    print(f"Images FILTERED OUT (Metrics disagreed): {images_filtered}")
    print(f"Images KEPT (Metrics agreed): {images_kept}")
    print("\nPlots generated per category:")
    for cat, count in plots_generated.items():
        print(f"  - {cat}: {count} plots")
    print("-" * 50)

    if len(df) == 0:
        print("No images remained after filtering. Cannot generate plots.")
        return

    csv_path = os.path.join(save_dir, f"{dataset_name}_subdivided_metrics.csv")
    df.to_csv(csv_path, index=False)

    df['Difference'] = df['Max_BBox_IoTA'] - df['Max_Mask_Halo_Occlusion']
    mean_abs_diff = df['Difference'].abs().mean()
    correlation, _ = pearsonr(df['Max_Mask_Halo_Occlusion'], df['Max_BBox_IoTA'])

    print(f"Mean Absolute Difference: {mean_abs_diff:.4f}")
    print(f"Pearson Correlation: {correlation:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Filtered Image-Level Max Occlusion (Same Target): {dataset_name}', fontsize=16, fontweight='bold')

    axes[0].scatter(df['Max_Mask_Halo_Occlusion'], df['Max_BBox_IoTA'], alpha=0.5, color='#d62728', s=20)
    axes[0].plot([0, 1], [0, 1], color='black', linestyle='--', label='Perfect Agreement (y=x)')
    axes[0].set_title('Pixel boundry vs. Max BBox IoTA (Per Image)', fontsize=14)
    axes[0].set_xlabel('Pixel boundry Occlusion', fontsize=12)
    axes[0].set_ylabel('Max BBox IoTA', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    axes[1].hist(df['Difference'], bins=50, color='#1f77b4', edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='dashed', linewidth=2, label='Zero Difference')
    axes[1].set_title('Distribution of Max Differences', fontsize=14)
    axes[1].set_xlabel('Difference (BBox - Mask)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{dataset_name}_subdivided_comparison_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze Occlusion Metrics and subdivide generated examples by value difference.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., coco_val, voc_train)")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N dataset images for faster testing")
    parser.add_argument("--plot_limit", type=int, default=20, help="Max plots to generate PER CATEGORY (Default: 20)")
    parser.add_argument("--save_dir", type=str, default="./dataset_metrics", help="Where to save the outputs")

    args = parser.parse_args()

    analyze_dataset_occlusion_subdivided(args.dataset, limit=args.limit, plot_limit=args.plot_limit,
                                         save_dir=args.save_dir)