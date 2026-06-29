# -*- coding: utf-8 -*-
from pycocotools import mask as COCOMask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from six.moves import cPickle as pickle
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pycocotools import mask as maskUtils
import os
import argparse
from tqdm import tqdm
from mmcv_box.visualization.image import imshow_det_bboxes
import multiprocessing
import os.path as osp
import sys
import pandas as pd
import gc
import cv2
from chainercv.evaluations import eval_instance_segmentation_voc


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)
from datasets.json_inference import coco_inst_seg_eval


def colormap(rgb=False):
    color_list = np.array(
        [
            0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556,
            0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078, 0.184,
            0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
            1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000,
            0.000, 0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000,
            0.333, 0.667, 0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000,
            0.667, 0.667, 0.000, 0.667, 1.000, 0.000, 1.000, 0.333, 0.000,
            1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500,
            0.000, 0.667, 0.500, 0.000, 1.000, 0.500, 0.333, 0.000, 0.500,
            0.333, 0.333, 0.500, 0.333, 0.667, 0.500, 0.333, 1.000, 0.500,
            0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667, 0.667, 0.500,
            0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333, 0.500,
            1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
            0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000,
            0.333, 0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000,
            0.667, 0.000, 1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000,
            0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000,
            1.000, 0.667, 1.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000,
            0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000,
            1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
            0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
            0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333,
            0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
            0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143,
            0.286, 0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571,
            0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 1.000, 1.000, 1.000,
            0.000, 0.447, 0.741,
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3))
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def id_2_clsname(annotation_file_path):
    with open(annotation_file_path, "r") as f:
        content = f.readlines()
    json_dict = json.loads(content[0])
    cls_name_map = [cat["name"] for cat in json_dict["categories"]]

    name_2_index = dict()
    for cat in json_dict["categories"]:
        name_2_index[cat["name"]] = cat["id"]

    return cls_name_map, name_2_index


# ==============================================================================
# OCCLUSION & CONFUSION METRICS
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


def calculate_overlap_confusion(p_masks, p_labels, gt_masks, gt_labels, iou_thresh=0.5):
    counts = {
        "Overlapping same class": 0,
        "Overlapping different class": 0,
        "No overlapping": 0
    }

    for p_m_list, p_l_list, gt_m_list, gt_l_list in zip(p_masks, p_labels, gt_masks, gt_labels):
        if len(p_m_list) == 0:
            continue
        if len(gt_m_list) == 0:
            counts["No overlapping"] += len(p_m_list)
            continue

        for pm, pl in zip(p_m_list, p_l_list):
            max_iou = 0.0
            matched_label = -1

            pm_bool = pm.astype(bool)
            for gm, gl in zip(gt_m_list, gt_l_list):
                gm_bool = gm.astype(bool)
                inter = np.logical_and(pm_bool, gm_bool).sum()
                if inter == 0:
                    continue
                union = np.logical_or(pm_bool, gm_bool).sum()
                iou = inter / (union + 1e-6)

                if iou > max_iou:
                    max_iou = iou
                    matched_label = gl

            if max_iou >= iou_thresh:
                if pl == matched_label:
                    counts["Overlapping same class"] += 1
                else:
                    counts["Overlapping different class"] += 1
            else:
                counts["No overlapping"] += 1

    return counts


def run_occlusion_analysis_with_classes(p_masks, p_labels, p_scores, gt_masks, gt_labels, occ_scores, iou_thresh, bins,
                                        bin_labels):
    bin_results = []

    for (low, high), label in zip(bins, bin_labels):
        filtered_gt_masks = []
        filtered_gt_labels = []
        filtered_p_masks = []
        filtered_p_labels = []
        filtered_p_scores = []

        for p_m, p_l, p_s, gt_m, gt_l, gt_o in zip(p_masks, p_labels, p_scores, gt_masks, gt_labels, occ_scores):
            # Determine which GT instances fall into the current occlusion bin
            keep_gt_idx = np.where((gt_o > low) & (gt_o <= high))[0]
            ignore_gt_idx = np.where((gt_o <= low) | (gt_o > high))[0]

            if len(keep_gt_idx) > 0:
                filtered_gt_masks.append(gt_m[keep_gt_idx])
                filtered_gt_labels.append(gt_l[keep_gt_idx])
                # Track which classes actually have instances in this occlusion bin for this image
                valid_classes = set(gt_l[keep_gt_idx])
            else:
                filtered_gt_masks.append(
                    np.empty((0, gt_m.shape[1] if gt_m.shape[0] > 0 else 1, gt_m.shape[2] if gt_m.shape[0] > 0 else 1),
                             dtype=bool))
                filtered_gt_labels.append(np.array([], dtype=int))
                valid_classes = set()

            if len(p_m) == 0:
                filtered_p_masks.append(p_m)
                filtered_p_labels.append(p_l)
                filtered_p_scores.append(p_s)
                continue

            keep_p = []
            for pi in range(len(p_m)):
                pred_mask = p_m[pi]
                pred_label = p_l[pi]

                # STRICT FILTERING: Only evaluate this prediction if its class has
                # at least one valid GT instance in the current occlusion bin.

                #TODO 29/06 05:33
                # if pred_label not in valid_classes:
                #     continue

                max_iou_ignore = 0.0
                if len(ignore_gt_idx) > 0:
                    for ig_i in ignore_gt_idx:
                        if gt_l[ig_i] == pred_label:
                            inter = np.logical_and(pred_mask, gt_m[ig_i]).sum()
                            union = np.logical_or(pred_mask, gt_m[ig_i]).sum()
                            iou = inter / (union + 1e-6)
                            if iou > max_iou_ignore: max_iou_ignore = iou

                max_iou_keep = 0.0
                if len(keep_gt_idx) > 0:
                    for kg_i in keep_gt_idx:
                        if gt_l[kg_i] == pred_label:
                            inter = np.logical_and(pred_mask, gt_m[kg_i]).sum()
                            union = np.logical_or(pred_mask, gt_m[kg_i]).sum()
                            iou = inter / (union + 1e-6)
                            if iou > max_iou_keep: max_iou_keep = iou

                if max_iou_ignore > 0.1 and max_iou_ignore > max_iou_keep:
                    continue
                else:
                    keep_p.append(pi)

            if len(keep_p) > 0:
                filtered_p_masks.append(p_m[keep_p])
                filtered_p_labels.append(p_l[keep_p])
                filtered_p_scores.append(p_s[keep_p])
            else:
                filtered_p_masks.append(np.empty((0, p_m.shape[1], p_m.shape[2]), dtype=bool))
                filtered_p_labels.append(np.array([], dtype=int))
                filtered_p_scores.append(np.array([], dtype=np.float32))

        print(f"  -> Evaluating instances with {label} occlusion @ IoU {iou_thresh}...")
        res = eval_instance_segmentation_voc(
            filtered_p_masks, filtered_p_labels, filtered_p_scores,
            filtered_gt_masks, filtered_gt_labels, iou_thresh=iou_thresh
        )

        bin_results.append({
            "Occlusion_Category": label,
            "mAP": res['map'] if res['map'] is not None else 0.0,
            "class_aps": res['ap']
        })
        del filtered_gt_masks, filtered_gt_labels, filtered_p_masks, filtered_p_labels, filtered_p_scores
        gc.collect()

    return bin_results

def get_chainercv_format_v3(coco_api, imgIds, is_gt=True, cat2idx=None, max_dets=100, score_thr=0.01):
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

def get_chainercv_format(coco_api, imgIds, is_gt=True, cat2idx=None, max_dets=50, score_thr=0.05):
    return get_chainercv_format_v3(coco_api, imgIds, is_gt, cat2idx, max_dets, score_thr)
    all_masks, all_labels, all_scores = [], [], []

    if cat2idx is None:
        catIds = sorted(coco_api.getCatIds())
        cat2idx = {cat: i for i, cat in enumerate(catIds)}

    # --- THE FIX: Map Continuous IDs (1-80) back to Official COCO IDs ---
    # cat2idx keys are official IDs, values are 0-79. We reverse it so 1-80 maps to official IDs.
    continuous_to_coco = {v + 1: k for k, v in cat2idx.items()}
    # --------------------------------------------------------------------

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

            # --- THE FIX: Translate the category ID ---
            raw_cat_id = ann.get('category_id')
            if not is_gt and raw_cat_id in continuous_to_coco:
                cat_id = continuous_to_coco[raw_cat_id]
            else:
                cat_id = raw_cat_id

            # Safety check: skip if it's somehow still an invalid ID
            if cat_id not in cat2idx:
                continue
            # ------------------------------------------

            # Safety check: skip if there is no segmentation data
            if 'segmentation' not in ann:
                continue

            segm = ann['segmentation']
            if isinstance(segm, list):
                rle = maskUtils.frPyObjects(segm, h, w)
                mask = maskUtils.decode(rle)
                if len(mask.shape) > 2:
                    mask = np.max(mask, axis=2)
            elif isinstance(segm, dict) and isinstance(segm.get('counts'), list):
                rle = maskUtils.frPyObjects([segm], h, w)
                mask = maskUtils.decode(rle)[:, :, 0]
            else:
                rle = [segm]
                mask = maskUtils.decode(rle)[:, :, 0]

            if not is_gt and mask.sum() < 350:
                continue

            masks.append(mask.astype(bool))
            labels.append(cat2idx[cat_id])
            if not is_gt:
                scores.append(ann.get('score', 0.0))

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
# EXISTING VISUALIZATION LOGIC
# ==============================================================================

def gt_dataset_mask(cocoGT, save_dir, root_dir, name_mapping):
    imgIds = sorted(cocoGT.getImgIds())
    Cls_Ids = cocoGT.getCatIds()

    for index in tqdm(range(len(imgIds)), total=len(imgIds)):
        img_id = [imgIds[index]]

        path = cocoGT.loadImgs(img_id)[0]['file_name']
        if os.path.exists(os.path.join(save_dir, path)):
            continue

        gt_ann_ids = cocoGT.getAnnIds(imgIds=img_id)
        anns = cocoGT.loadAnns(gt_ann_ids)
        anns = sorted(anns, key=lambda x: x['area'], reverse=True)

        try:
            with Image.open(os.path.join(root_dir, path)).convert("RGB") as img_pil:
                img = np.array(img_pil)

            polygons = []
            label_list = []
            box_list = []
            score_list = []
            color = []
            color_array = colormap(True)
            w_ratio = .4
            color_array = color_array * (1 - w_ratio) + w_ratio
            for idx, ann in enumerate(anns):
                if idx >= color_array.shape[0]:
                    c = (np.random.random((1, 3)) * 0.6 + 0.2)
                else:
                    c = color_array[idx][None, :]

                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        rle = maskUtils.frPyObjects(ann['segmentation'], img.shape[0], img.shape[1])
                        mask = maskUtils.decode(rle).transpose(2, 0, 1)
                        polygons.append(mask)
                        for _ in range(mask.shape[0]):
                            label_list.append(Cls_Ids.index(ann['category_id']))
                            box_list.append(np.array(ann['bbox']))
                            color.append(c)
                    else:
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], img.shape[0], img.shape[1])
                        else:
                            rle = [ann['segmentation']]

                        mask = maskUtils.decode(rle).transpose(2, 0, 1)
                        if ann['iscrowd'] == 1:
                            continue
                        elif ann['iscrowd'] == 0:
                            polygons.append(mask)
                            label_list.append(Cls_Ids.index(ann['category_id']))
                            box_list.append(np.array(ann['bbox']))
                            color.append(c)

                if 'score' in ann:
                    score_list.append(ann['score'])

            if len(polygons) > 0:
                polygons = np.concatenate(polygons, axis=0)
                box_list = np.concatenate(box_list, axis=0).reshape(-1, 4)
                box_list[:, 2:] = box_list[:, :2] + box_list[:, 2:]
                color = np.concatenate(color, axis=0)

                label_list = np.array(label_list)
                score_list = np.array(score_list).reshape(-1, 1)
                if len(score_list) != 0:
                    box_list = np.concatenate((box_list, score_list), axis=1)

                img = imshow_det_bboxes(img, box_list, labels=label_list, segms=polygons,
                                        class_names=name_mapping, show=False,
                                        bbox_color=color, mask_color=255 * color,
                                        font_size=18, thickness=1, alpha=0.8)

            plt.imshow(np.uint8(img))
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, path), dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close('all')

        except Exception as e:
            print(f"Failed drawing GT for {path}: {e}")
            plt.close('all')

    print(f'\rImage Index: {(index + 1):.0f}/{len(imgIds):.0f}  ', end='')


def analyze_single_instance_splits(p_masks, p_labels, gt_masks, gt_labels, class_names, model_name, save_dir, cocoGt,
                                   imgIds, root_dir, min_overlap_pixels=5):
    """
    Analyzes images with exactly ONE ground truth instance.
    Plots side-by-side comparisons for images where the model predicts 0 or >1 instances.
    """
    split_counts_overall = {}
    split_counts_by_class = {}

    # Create the specific folder for the wrong split visualizations
    error_save_dir = os.path.join(save_dir, "wrong single instance")
    os.makedirs(error_save_dir, exist_ok=True)

    # Get colormap for drawing (using the function already in your script)
    color_array = colormap(True)
    w_ratio = .4
    color_array = color_array * (1 - w_ratio) + w_ratio

    for i in tqdm(range(len(gt_masks)), desc=f"Analyzing Splits for {model_name}"):
        # Only look at images with exactly 1 GT instance
        if len(gt_masks[i]) == 1:
            gt_m = gt_masks[i][0].astype(bool)
            gt_l = gt_labels[i][0]

            if gt_l not in split_counts_by_class:
                split_counts_by_class[gt_l] = {}

            preds_m = p_masks[i]
            preds_l = p_labels[i]

            splits = 0
            split_indices = []

            if len(preds_m) > 0:
                for p_idx, (pm, pl) in enumerate(zip(preds_m, preds_l)):
                    # Check if prediction is the same class and overlaps the GT mask
                    if pl == gt_l:
                        overlap = np.logical_and(pm.astype(bool), gt_m).sum()
                        if overlap >= min_overlap_pixels:
                            splits += 1
                            split_indices.append(p_idx)

            # Tally the counts
            split_counts_overall[splits] = split_counts_overall.get(splits, 0) + 1
            split_counts_by_class[gt_l][splits] = split_counts_by_class[gt_l].get(splits, 0) + 1

            # --- VISUALIZATION FOR ERRORS (Splits != 1) ---
            if splits != 1:
                img_id = imgIds[i]
                img_info = cocoGt.loadImgs([img_id])[0]
                path = img_info['file_name']

                try:
                    with Image.open(os.path.join(root_dir, path)).convert("RGB") as img_pil:
                        img = np.array(img_pil)

                    # 1. Prepare GT Data
                    gt_img_drawn = img.copy()
                    y_indices, x_indices = np.where(gt_m)

                    if len(x_indices) > 0:
                        x1, y1 = np.min(x_indices), np.min(y_indices)
                        x2, y2 = np.max(x_indices), np.max(y_indices)
                        gt_box = np.array([[x1, y1, x2, y2]])

                        c_gt = color_array[gt_l % len(color_array)][None, :]
                        gt_polygons = gt_m[None, :, :]
                        gt_label_arr = np.array([gt_l])

                        gt_img_drawn = imshow_det_bboxes(img.copy(), gt_box, labels=gt_label_arr, segms=gt_polygons,
                                                         class_names=class_names, show=False,
                                                         bbox_color=c_gt, mask_color=255 * c_gt,
                                                         font_size=12, thickness=2, alpha=0.8)

                    # 2. Prepare Pred Data
                    dt_img_drawn = img.copy()
                    if len(split_indices) > 0:
                        dt_polygons, dt_box_list, dt_label_list, dt_color = [], [], [], []

                        for p_idx in split_indices:
                            pm = preds_m[p_idx]
                            py, px = np.where(pm)
                            if len(px) > 0:
                                px1, py1 = np.min(px), np.min(py)
                                px2, py2 = np.max(px), np.max(py)
                                dt_box_list.append([px1, py1, px2, py2])
                                dt_polygons.append(pm)
                                dt_label_list.append(preds_l[p_idx])
                                dt_color.append(color_array[preds_l[p_idx] % len(color_array)][None, :])

                        if len(dt_polygons) > 0:
                            dt_polygons = np.stack(dt_polygons, axis=0)
                            dt_box_list = np.array(dt_box_list)
                            dt_label_list = np.array(dt_label_list)
                            dt_color = np.concatenate(dt_color, axis=0)

                            dt_img_drawn = imshow_det_bboxes(img.copy(), dt_box_list, labels=dt_label_list,
                                                             segms=dt_polygons,
                                                             class_names=class_names, show=False,
                                                             bbox_color=dt_color, mask_color=255 * dt_color,
                                                             font_size=12, thickness=2, alpha=0.8)

                    # 3. Plot Side-by-Side
                    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
                    axes[0].imshow(np.uint8(gt_img_drawn))
                    axes[0].set_title("Ground Truth (1 object)", fontsize=24, fontweight='bold')
                    axes[0].axis("off")

                    axes[1].imshow(np.uint8(dt_img_drawn))
                    axes[1].set_title(f"Prediction ({splits} overlapping objects)", fontsize=24, fontweight='bold')
                    axes[1].axis("off")

                    plt.tight_layout()
                    # Replace slashes to avoid path issues if images are in subfolders
                    safe_path = path.replace('/', '_').replace('\\', '_')
                    save_name = f"{os.path.splitext(safe_path)[0]}_splits_{splits}.png"
                    plt.savefig(os.path.join(error_save_dir, save_name), dpi=150, bbox_inches='tight', pad_inches=0)

                    fig.clf()
                    plt.close(fig)
                    plt.close('all')

                except Exception as e:
                    print(f"Failed to draw split error for {path}: {e}")
                    plt.close('all')

    if not split_counts_overall:
        print(f"No single-instance images found for {model_name}.")
        return

    # --- Plot 1: Overall Counts ---
    max_splits = max(split_counts_overall.keys())
    x_vals = list(range(max_splits + 1))
    y_overall = [split_counts_overall.get(x, 0) for x in x_vals]

    plt.figure(figsize=(10, 6))
    plt.bar(x_vals, y_overall, color='#1f77b4', edgecolor='black', zorder=3)
    plt.title(f'Single Instance Splitting - Overall ({model_name})', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Predicted Instances (Splits)', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(x_vals)
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"splits_overall_{model_name}.png"), dpi=300)
    plt.close('all')

    # --- Plot 2: Counts by Class (Stacked Bar) ---
    plt.figure(figsize=(14, 8))
    bottom = np.zeros(len(x_vals))
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(split_counts_by_class)))

    for idx, (cls_idx, counts) in enumerate(sorted(split_counts_by_class.items())):
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
        y_cls = [counts.get(x, 0) for x in x_vals]
        plt.bar(x_vals, y_cls, bottom=bottom, label=cls_name, color=colors[idx], edgecolor='white', zorder=3)
        bottom += np.array(y_cls)

    plt.title(f'Single Instance Splitting - By Class ({model_name})', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Predicted Instances (Splits)', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(x_vals)
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2 if len(split_counts_by_class) > 15 else 1,
               fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"splits_by_class_{model_name}.png"), dpi=300)
    plt.close('all')

    print(f"Saved split distribution plots and visual errors for {model_name} in {save_dir}")


def analyze_gt_splits(cocoGt, cocoDt, imgIds, model_name, save_dir, iou_thresh=0.5):
    """
    Analyzes ground truth instances to see how many separate polygons (splits)
    they are made of, and checks if the model correctly predicted them.
    """
    splits_counts = {}  # num_splits -> count of instances
    correct_counts = {}  # num_splits -> count of correct predictions
    wrong_counts = {}  # num_splits -> count of wrong (missed) predictions

    for img_id in tqdm(imgIds, desc=f"Analyzing GT splits for {model_name}"):
        gt_ann_ids = cocoGt.getAnnIds(imgIds=[img_id])
        gt_anns = cocoGt.loadAnns(gt_ann_ids)

        dt_ann_ids = cocoDt.getAnnIds(imgIds=[img_id])
        dt_anns = cocoDt.loadAnns(dt_ann_ids)

        # Sort predictions by score so we match the highest confidence ones first
        dt_anns = sorted(dt_anns, key=lambda x: x.get('score', 0), reverse=True)

        img_info = cocoGt.loadImgs(img_id)[0]
        h, w = img_info['height'], img_info['width']

        # Decode dt masks
        dt_masks = []
        dt_cats = []
        for d_ann in dt_anns:
            if type(d_ann['segmentation']) == list:
                rle = maskUtils.frPyObjects(d_ann['segmentation'], h, w)
                mask = maskUtils.decode(rle)
                if len(mask.shape) > 2:
                    mask = np.max(mask, axis=2)
            elif type(d_ann['segmentation']['counts']) == list:
                rle = maskUtils.frPyObjects([d_ann['segmentation']], h, w)
                mask = maskUtils.decode(rle)[:, :, 0]
            else:
                rle = [d_ann['segmentation']]
                mask = maskUtils.decode(rle)[:, :, 0]
            dt_masks.append(mask.astype(bool))
            dt_cats.append(d_ann['category_id'])

        # Check each Ground Truth instance
        for g_ann in gt_anns:
            if g_ann.get('iscrowd', 0) == 1:
                continue

            # 1. Determine number of splits (polygons) in GT
            if type(g_ann['segmentation']) == list:
                num_splits = len(g_ann['segmentation'])
            else:
                num_splits = 1  # RLE format is counted as a single block here

            # Get GT mask
            if type(g_ann['segmentation']) == list:
                rle = maskUtils.frPyObjects(g_ann['segmentation'], h, w)
                mask = maskUtils.decode(rle)
                if len(mask.shape) > 2:
                    mask = np.max(mask, axis=2)
            elif type(g_ann['segmentation']['counts']) == list:
                rle = maskUtils.frPyObjects([g_ann['segmentation']], h, w)
                mask = maskUtils.decode(rle)[:, :, 0]
            else:
                rle = [g_ann['segmentation']]
                mask = maskUtils.decode(rle)[:, :, 0]

            g_mask = mask.astype(bool)
            g_cat = g_ann['category_id']

            # 2. Check if the model correctly predicted it (IoU >= threshold)
            matched = False
            for d_mask, d_cat in zip(dt_masks, dt_cats):
                if d_cat == g_cat:
                    # # === ADD THIS DEBUG SNIPPET ===
                    # if g_mask.shape != d_mask.shape:
                    #     print(f"\n⚠️ Mismatch found! GT: {g_mask.shape}, Pred: {d_mask.shape}")
                    #     import matplotlib.pyplot as plt
                    #     import os
                    #     import sys
                    #
                    #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    #     axes[0].imshow(g_mask, cmap='gray')
                    #     axes[0].set_title(f"Ground Truth {g_mask.shape}", fontsize=14)
                    #     axes[0].axis('off')
                    #
                    #     axes[1].imshow(d_mask, cmap='gray')
                    #     axes[1].set_title(f"Prediction {d_mask.shape}", fontsize=14)
                    #     axes[1].axis('off')
                    #
                    #     debug_path = os.path.join(save_dir, f"mismatch_debug_img_{img_id}.png")
                    #     plt.tight_layout()
                    #     plt.savefig(debug_path)
                    #     plt.close('all')
                    #
                    #     print(f"📸 Saved debug plot to: {debug_path}")
                    #     print("Stopping execution so you can inspect the image...")
                    #     sys.exit(1)  # Halt the script immediately
                    # # ==============================

                    #TODO CIM CHECK
                    # if g_mask.shape != d_mask.shape:
                    #     d_mask_eval = cv2.resize(
                    #         d_mask.astype(np.uint8),
                    #         (g_mask.shape[1], g_mask.shape[0]),
                    #         interpolation=cv2.INTER_NEAREST
                    #     ).astype(bool)
                    # else:
                    #     d_mask_eval = d_mask


                    inter = np.logical_and(g_mask, d_mask).sum()
                    union = np.logical_or(g_mask, d_mask).sum()
                    iou = inter / (union + 1e-6)
                    if iou >= iou_thresh:
                        matched = True
                        break  # Found a correct prediction for this GT

            # Tally results
            splits_counts[num_splits] = splits_counts.get(num_splits, 0) + 1
            if num_splits not in correct_counts:
                correct_counts[num_splits] = 0
                wrong_counts[num_splits] = 0

            if matched:
                correct_counts[num_splits] += 1
            else:
                wrong_counts[num_splits] += 1

    if not splits_counts:
        print(f"No GT annotations found for {model_name}.")
        return

    # --- Plot 1: Splits Distribution ---
    x_vals = sorted(list(splits_counts.keys()))
    y_vals = [splits_counts[x] for x in x_vals]

    plt.figure(figsize=(10, 6))
    plt.bar([str(x) for x in x_vals], y_vals, color='#8c564b', edgecolor='black', zorder=3)
    plt.title(f'Ground Truth Instance Splits Distribution ({model_name})', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Splits (Polygons in GT Instance)', fontsize=12)
    plt.ylabel('Number of Ground Truth Instances', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"gt_splits_overall_distribution_{model_name}.png"), dpi=300)
    plt.close('all')

    # --- Plot 2: Correct vs Wrong Predictions by Splits ---
    y_correct = [correct_counts[x] for x in x_vals]
    y_wrong = [wrong_counts[x] for x in x_vals]

    plt.figure(figsize=(12, 7))
    x_indices = np.arange(len(x_vals))
    width = 0.4

    plt.bar(x_indices - width / 2, y_correct, width=width, label='Correct Predictions (TP)', color='#2ca02c',
            edgecolor='black', zorder=3)
    plt.bar(x_indices + width / 2, y_wrong, width=width, label='Wrong Predictions (FN)', color='#d62728',
            edgecolor='black', zorder=3)

    plt.title(f'Prediction Accuracy by GT Splits (IoU $\geq$ {iou_thresh}) - {model_name}', fontsize=16,
              fontweight='bold')
    plt.xlabel('Number of Splits (Polygons in GT Instance)', fontsize=12)
    plt.ylabel('Number of Ground Truth Instances', fontsize=12)
    plt.xticks(x_indices, [str(x) for x in x_vals])
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"gt_splits_accuracy_{model_name}.png"), dpi=300)
    plt.close('all')

    print(f"Saved GT splits analysis plots for {model_name} in {save_dir}")


def dataset_mask(cocoGT, cocoDt, save_dir, root_dir, imgIds=None, num=None, anno_thr=-1, name_mapping=None,
                 img_category_map=None):
    if imgIds == None:
        imgIds = sorted(cocoGT.getImgIds())

    Cls_Ids = cocoGT.getCatIds()
    color_array = colormap(True)
    w_ratio = .4
    color_array = color_array * (1 - w_ratio) + w_ratio

    for index in tqdm(range(len(imgIds)), total=len(imgIds)):
        img_id = imgIds[index]

        path = cocoGT.loadImgs(img_id)[0]['file_name']
        cat_folder = img_category_map.get(img_id, "") if img_category_map else ""
        current_save_dir = os.path.join(save_dir, cat_folder)
        os.makedirs(current_save_dir, exist_ok=True)
        save_path = os.path.join(current_save_dir, os.path.basename(path))

        gt_ann_ids = cocoGT.getAnnIds(imgIds=[img_id])
        gt_anns = cocoGT.loadAnns(gt_ann_ids)

        ann_ids = cocoDt.getAnnIds(imgIds=[img_id])
        anns = cocoDt.loadAnns(ann_ids)
        anns = sorted(anns, key=lambda x: x.get('score', 0))
        score = np.asarray([x.get("score", 0) for x in anns])

        if anno_thr == -1:
            gt_nums = len(gt_anns)
        else:
            gt_nums = len((score >= anno_thr).nonzero()[0])

        if gt_nums != 0:
            anns = anns[-gt_nums:]
            anns = sorted(anns, key=lambda x: x['area'], reverse=True)
        else:
            anns = []

        try:
            with Image.open(os.path.join(root_dir, path)).convert("RGB") as img_pil:
                img = np.array(img_pil)

            gt_img_drawn = img.copy()
            gt_masks_dict = {}
            if len(gt_anns) > 0:
                gt_polygons, gt_label_list, gt_box_list, gt_color = [], [], [], []
                for idx, ann in enumerate(gt_anns):
                    c = color_array[idx % len(color_array)][None, :]
                    if 'segmentation' in ann:
                        if type(ann['segmentation']) == list:
                            rle = maskUtils.frPyObjects(ann['segmentation'], img.shape[0], img.shape[1])
                            mask = maskUtils.decode(rle)
                            m_for_iou = np.max(mask, axis=2) if len(mask.shape) > 2 else mask
                            mask_t = mask.transpose(2, 0, 1)
                        else:
                            if type(ann['segmentation']['counts']) == list:
                                rle = maskUtils.frPyObjects([ann['segmentation']], img.shape[0], img.shape[1])
                            else:
                                rle = [ann['segmentation']]
                            mask = maskUtils.decode(rle)
                            m_for_iou = mask[:, :, 0]
                            mask_t = mask.transpose(2, 0, 1)

                        if ann.get('iscrowd', 0) == 1: continue

                        cat_id = ann['category_id']
                        if cat_id not in gt_masks_dict:
                            gt_masks_dict[cat_id] = []
                        gt_masks_dict[cat_id].append(m_for_iou.astype(bool))

                        gt_polygons.append(mask_t)
                        for _ in range(mask_t.shape[0]):
                            gt_label_list.append(Cls_Ids.index(cat_id))
                            gt_box_list.append(np.array(ann['bbox']))
                            gt_color.append(c)

                if len(gt_polygons) > 0:
                    gt_polygons = np.concatenate(gt_polygons, axis=0)
                    gt_label_list = np.array(gt_label_list)
                    gt_box_list = np.concatenate(gt_box_list, axis=0).reshape(-1, 4)
                    gt_box_list[:, 2:] = gt_box_list[:, :2] + gt_box_list[:, 2:]
                    gt_color = np.concatenate(gt_color, axis=0)

                    gt_img_drawn = imshow_det_bboxes(img.copy(), gt_box_list, labels=gt_label_list, segms=gt_polygons,
                                                     class_names=name_mapping, show=False,
                                                     bbox_color=gt_color, mask_color=255 * gt_color,
                                                     font_size=12, thickness=1, alpha=0.8)

            dt_img_drawn = img.copy()
            if len(anns) > 0:
                dt_polygons, dt_label_list, dt_box_list, dt_score_list, dt_color = [], [], [], [], []
                dt_custom_names = []

                for idx, ann in enumerate(anns):
                    c = color_array[idx % len(color_array)][None, :]
                    if 'segmentation' in ann:
                        if type(ann['segmentation']) == list:
                            rle = maskUtils.frPyObjects(ann['segmentation'], img.shape[0], img.shape[1])
                            mask = maskUtils.decode(rle)
                            m_for_iou = np.max(mask, axis=2) if len(mask.shape) > 2 else mask
                            mask_t = mask.transpose(2, 0, 1)
                        else:
                            if type(ann['segmentation']['counts']) == list:
                                rle = maskUtils.frPyObjects([ann['segmentation']], img.shape[0], img.shape[1])
                            else:
                                rle = [ann['segmentation']]
                            mask = maskUtils.decode(rle)
                            m_for_iou = mask[:, :, 0]
                            mask_t = mask.transpose(2, 0, 1)

                        if ann.get('iscrowd', 0) == 1: continue

                        cat_id = ann['category_id']
                        max_iou = 0.0
                        if cat_id in gt_masks_dict:
                            m_bool = m_for_iou.astype(bool)
                            for g_mask in gt_masks_dict[cat_id]:
                                intersection = np.logical_and(m_bool, g_mask).sum()
                                union = np.logical_or(m_bool, g_mask).sum()
                                iou = intersection / (union + 1e-6)
                                if iou > max_iou: max_iou = iou

                        orig_cls_name = name_mapping[Cls_Ids.index(cat_id)]
                        custom_name = f"{orig_cls_name} | IoU:{max_iou:.2f}"

                        dt_polygons.append(mask_t)
                        for _ in range(mask_t.shape[0]):
                            dt_label_list.append(len(dt_custom_names))
                            dt_custom_names.append(custom_name)
                            dt_box_list.append(np.array(ann['bbox']))
                            dt_color.append(c)
                            if 'score' in ann:
                                dt_score_list.append(ann['score'])

                if len(dt_polygons) > 0:
                    dt_polygons = np.concatenate(dt_polygons, axis=0)
                    dt_label_list = np.array(dt_label_list)
                    dt_box_list = np.concatenate(dt_box_list, axis=0).reshape(-1, 4)
                    dt_box_list[:, 2:] = dt_box_list[:, :2] + dt_box_list[:, 2:]
                    dt_color = np.concatenate(dt_color, axis=0)
                    dt_score_list = np.array(dt_score_list).reshape(-1, 1)

                    if len(dt_score_list) > 0 and len(dt_score_list) == len(dt_box_list):
                        dt_box_list = np.concatenate((dt_box_list, dt_score_list), axis=1)

                    dt_img_drawn = imshow_det_bboxes(img.copy(), dt_box_list, labels=dt_label_list, segms=dt_polygons,
                                                     class_names=dt_custom_names, show=False,
                                                     bbox_color=dt_color, mask_color=255 * dt_color,
                                                     font_size=10, thickness=1, alpha=0.8)

            fig, axes = plt.subplots(1, 2, figsize=(24, 12))
            axes[0].imshow(np.uint8(gt_img_drawn))
            axes[0].set_title(f"Ground Truth ({len(gt_anns)} objects)", fontsize=20, fontweight='bold')
            axes[0].axis("off")

            axes[1].imshow(np.uint8(dt_img_drawn))
            axes[1].set_title(f"Prediction ({len(anns)} objects)", fontsize=20, fontweight='bold')
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)

            fig.clf()
            plt.close(fig)
            plt.close('all')

        except Exception as e:
            print(f"Failed to process and draw image {path}: {e}")
            plt.close('all')

        if index % 10 == 0:
            gc.collect()

        if num is not None:
            if index == num - 1:
                break


def update_and_generate_latex_table(dataset_arg, model_names, overall_maps, any_overlap_maps, same_overlap_maps,
                                    db_path="latex_table_db.json", txt_path="overview_table.txt"):
    import json
    import os
    import pandas as pd
    import numpy as np

    # Map the argparse dataset to the table columns
    dataset_arg_lower = dataset_arg.lower()
    if "cub" in dataset_arg_lower:
        ds_key = "CUB"
    elif "voc" in dataset_arg_lower:
        ds_key = "VOC"
    elif "coco" in dataset_arg_lower:
        ds_key = "COCO"
    else:
        ds_key = "UNKNOWN"

    # Load existing database if it exists to append/update data
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            db = json.load(f)
    else:
        db = {}

    # Update database with current run's metrics
    for model in model_names:
        if model not in db:
            db[model] = {}
        if ds_key not in db[model]:
            db[model][ds_key] = {}

        db[model][ds_key]["All"] = {
            "30": overall_maps[model].get(0.3, np.nan),
            "50": overall_maps[model].get(0.5, np.nan),
            "75": overall_maps[model].get(0.75, np.nan)
        }
        db[model][ds_key]["Between"] = {
            "30": any_overlap_maps[model].get(0.3, np.nan),
            "50": any_overlap_maps[model].get(0.5, np.nan),
            "75": any_overlap_maps[model].get(0.75, np.nan)
        }
        db[model][ds_key]["Same"] = {
            "30": same_overlap_maps[model].get(0.3, np.nan),
            "50": same_overlap_maps[model].get(0.5, np.nan),
            "75": same_overlap_maps[model].get(0.75, np.nan)
        }

    # Save updated metrics back to JSON
    with open(db_path, 'w') as f:
        json.dump(db, f, indent=4)

    # Build Pandas MultiIndex columns
    columns = pd.MultiIndex.from_tuples([
        ('CUB dataset', 'All samples', 'map@30'),
        ('CUB dataset', 'All samples', 'map@50'),
        ('CUB dataset', 'All samples', 'map@75'),
        ('VOC dataset', 'All samples', 'map@30'),
        ('VOC dataset', 'All samples', 'map@50'),
        ('VOC dataset', 'All samples', 'map@75'),
        ('VOC dataset', 'Overlapping between classes', 'map@30'),
        ('VOC dataset', 'Overlapping between classes', 'map@50'),
        ('VOC dataset', 'Overlapping between classes', 'map@75'),
        ('VOC dataset', 'Overlapping same classes', 'map@30'),
        ('VOC dataset', 'Overlapping same classes', 'map@50'),
        ('VOC dataset', 'Overlapping same classes', 'map@75'),
        # ('COCO dataset', 'All samples', 'map@30'),
        # ('COCO dataset', 'All samples', 'map@50'),
        # ('COCO dataset', 'All samples', 'map@75'),
        # ('COCO dataset', 'Overlapping between classes', 'map@30'),
        # ('COCO dataset', 'Overlapping between classes', 'map@50'),
        # ('COCO dataset', 'Overlapping between classes', 'map@75'),
        # ('COCO dataset', 'Overlapping same classes', 'map@30'),
        # ('COCO dataset', 'Overlapping same classes', 'map@50'),
        # ('COCO dataset', 'Overlapping same classes', 'map@75'),
    ], names=['Model', 'Filtertype', 'Score'])

    # Helper function to format 0-1 scale mAP to 0-100 string format
    def fmt(val):
        if pd.isna(val) or val == "" or val is None:
            return ""
        return f"{float(val) * 100:.1f}"

    data = []
    models = list(db.keys())

    for model in models:
        row = []

        # CUB Data
        cub = db[model].get("CUB", {})
        cub_all = cub.get("All", {})
        row.extend([fmt(cub_all.get('30')), fmt(cub_all.get('50')), fmt(cub_all.get('75'))])

        # VOC Data
        voc = db[model].get("VOC", {})
        for section_key in ["All", "Between", "Same"]:
            section = voc.get(section_key, {})
            row.extend([fmt(section.get('30')), fmt(section.get('50')), fmt(section.get('75'))])

        # # COCO Data
        # coco = db[model].get("COCO", {})
        # for section_key in ["All", "Between", "Same"]:
        #     section = coco.get(section_key, {})
        #     row.extend([fmt(section.get('30')), fmt(section.get('50')), fmt(section.get('75'))])

        data.append(row)

    df = pd.DataFrame(data, index=models, columns=columns)

    # Generate LaTeX using Pandas
    latex_str = df.to_latex(
        column_format="|l|lll|lllllllll|lllllllll|",
        multicolumn=True,
        multicolumn_format="c|",
        multirow=False,
        na_rep=""
    )

    # Write output to the txt file
    with open(txt_path, 'w') as f:
        f.write(latex_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="eval model")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--result_files", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--thr", type=float, default=-1)
    parser = parser.parse_args()

    if "gt" not in parser.dataset:
        if parser.dataset == "voc_val":
            label_file = "./data/VOC2012/annotations/voc_2012_val.json"
            root = './data/VOC2012/JPEGImages'
        elif parser.dataset == "voc_train":
            label_file = "./data/VOC2012/annotations/voc_2012_trainaug.json"
            root = './data/VOC2012/JPEGImages'
        elif parser.dataset == "coco_val":
            label_file = "./datasets/coco_dataset/coco2017/annotations/instances_val2017.json"
            root = './datasets/coco_dataset/coco2017/val2017'
        elif parser.dataset == "coco_test":
            label_file = "./datasets/coco_dataset/coco2017/annotations/image_info_test-dev2017.json"
            root = './datasets/coco_dataset/coco2017/test2017'
        elif parser.dataset == "coco_train":
            label_file = "./datasets/coco_dataset/coco2017/annotations/instances_train2017.json"
            root = './datasets/coco_dataset/coco2017/train2017'
        elif parser.dataset == "cub_val":
            label_file = "./datasets/CUB_200_2011/CUB_as_COCO/annotations/instances_val2017.json"
            root = "./datasets/CUB_200_2011/CUB_200_2011/images_combined"
        elif parser.dataset == "cub_train":
            label_file = "./datasets/CUB_200_2011/CUB_as_COCO/annotations/instances_train2017.json"
            root = "./datasets/CUB_200_2011/CUB_200_2011/images_combined"

        result_files = parser.result_files.split(',')
        base_save_dir = parser.save_dir
        os.makedirs(base_save_dir, exist_ok=True)
        print("Folder created: {}".format(base_save_dir))
        thr = parser.thr

        cocoGt = COCO(label_file)
        imgIds = sorted(cocoGt.getImgIds())

        if parser.num is not None:
            imgIds = imgIds[:parser.num]

        cls_name_map, name_2_index = id_2_clsname(label_file)

        all_models_results_any = {}
        all_models_results_same = {}
        all_models_results_binary = {}
        all_models_results_binary_same = {}
        overall_mAP_results = {}
        overlap_confusion_results = {}

        iou_thresholds = [0.75, 0.5, 0.3]
        overall_ious = [0.25, 0.30, 0.50, 0.70, 0.75]

        # BINS DEFINITIONS:
        bins_any = [(-0.01, 0.0), (0.0, 0.25), (0.25, 0.50), (0.50, 1.0)]
        bin_labels_any = ["No Overlap", "Low (0-25%)", "Medium (25-50%)", "High (>50%)"]

        bins_same = [(-0.01, 0.0), (0.0, 0.25), (0.25, 0.50), (0.50, 1.0)]
        bin_labels_same = ["No Overlap", "Low (0-25%)", "Medium (25-50%)", "High (>50%)"]

        bins_binary = [(-0.01, 0.0), (0.0, 1.0)]
        bin_labels_binary = ["No Overlap", "Overlap"]

        print("\n--- Extracting GT and Calculating True Mask Occlusion ---")
        gt_masks, gt_labels, cat2idx = get_chainercv_format(cocoGt, imgIds, is_gt=True)
        occ_scores_any, occ_scores_same = calculate_instance_occlusion(gt_masks, gt_labels, neighbor_dist=10)

        # ----------------------------------------------------------------------
        # Data Distribution Plot: Any Class (Stacked Bar)
        # ----------------------------------------------------------------------
        print("\n--- Plotting Sample Distribution per Class & Occlusion Category (Any Class) ---")

        cats = cocoGt.loadCats(cocoGt.getCatIds())
        cat_id_to_name = {cat['id']: cat['name'] for cat in cats}
        idx_to_cat_id = {v: k for k, v in cat2idx.items()}
        idx_to_name = {idx: cat_id_to_name[cat_id] for idx, cat_id in idx_to_cat_id.items()}

        class_occ_counts = {idx: {label: 0 for label in bin_labels_any} for idx in idx_to_name.keys()}

        for img_labels, img_occs in zip(gt_labels, occ_scores_any):
            for label, occ in zip(img_labels, img_occs):
                for (low, high), bin_label in zip(bins_any, bin_labels_any):
                    if low < occ <= high:
                        class_occ_counts[label][bin_label] += 1
                        break

        plot_data = {"Class": [], "No Overlap": [], "Low (0-25%)": [], "Medium (25-50%)": [], "High (>50%)": []}

        for idx in sorted(class_occ_counts.keys(), key=lambda x: idx_to_name[x]):
            class_name = idx_to_name[idx]
            plot_data["Class"].append(class_name)
            for label in bin_labels_any:
                plot_data[label].append(class_occ_counts[idx][label])

        df_counts = pd.DataFrame(plot_data)

        fig, ax = plt.subplots(figsize=(18, 8))
        bottom = np.zeros(len(df_counts))
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

        for label, color in zip(bin_labels_any, colors):
            ax.bar(df_counts["Class"], df_counts[label], label=label, bottom=bottom, color=color, alpha=0.85)
            bottom += df_counts[label].values

        ax.set_title('Distribution of Samples per Occlusion Category (Any Class)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Segmentation Class', fontsize=12)
        ax.set_ylabel('Total Number of Instances in GT Dataset', fontsize=12)
        plt.xticks(rotation=90, ha='center')
        ax.legend(title="Occlusion Category")
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        dist_plot_path = os.path.join(base_save_dir, "gt_samples_per_occlusion_category_AnyClass.png")
        plt.savefig(dist_plot_path)
        plt.close('all')
        print(f"Sample distribution plot saved to {dist_plot_path}\n")

        # ----------------------------------------------------------------------
        # Data Distribution Plot: Global Comparison (Any Class vs Same Class)
        # ----------------------------------------------------------------------
        print("\n--- Plotting Global Sample Distribution: Any Class vs Same Class ---")
        global_bins = [(-0.01, 0.0), (0.0, 0.25), (0.25, 0.50), (0.50, 1.0)]
        global_bin_labels = ["No Overlap", "Low (0-25%)", "Medium (25-50%)", "High (>50%)"]

        any_counts = {label: 0 for label in global_bin_labels}
        same_counts = {label: 0 for label in global_bin_labels}

        for img_occs_any, img_occs_same in zip(occ_scores_any, occ_scores_same):
            for occ_any, occ_same in zip(img_occs_any, img_occs_same):
                for (low, high), label in zip(global_bins, global_bin_labels):
                    if low < occ_any <= high:
                        any_counts[label] += 1
                    if low < occ_same <= high:
                        same_counts[label] += 1

        x = np.arange(len(global_bin_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width / 2, [any_counts[l] for l in global_bin_labels], width, label='All Classes',
                        color='#1f77b4')
        rects2 = ax.bar(x + width / 2, [same_counts[l] for l in global_bin_labels], width, label='Same Class',
                        color='#ff7f0e')

        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Global Sample Distribution by Occlusion Amount', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(global_bin_labels)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6)


        # Add labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)


        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        global_dist_plot_path = os.path.join(base_save_dir, "global_samples_per_occlusion_category_comparison.png")
        plt.savefig(global_dist_plot_path)
        plt.close('all')
        print(f"Global distribution comparison plot saved to {global_dist_plot_path}\n")

        # ----------------------------------------------------------------------
        # Model Evaluation Loop
        # ----------------------------------------------------------------------
        for result_file in result_files:
            model_name = os.path.splitext(os.path.basename(result_file))[0]
            print(f"\n{'=' * 50}\nEvaluating Model: {model_name}\n{'=' * 50}")

            model_save_dir = os.path.join(base_save_dir, model_name)
            os.makedirs(model_save_dir, exist_ok=True)

            res = json.load(open(result_file))

            try:
                if "annotations" in res.keys():
                    temp_filename = os.path.join("./data/trash", f'temp_{model_name}.json')
                    os.makedirs(os.path.dirname(temp_filename), exist_ok=True)
                    with open(temp_filename, 'w') as file_obj:
                        json.dump(res['annotations'], file_obj)
                    result_file = temp_filename
            except:
                pass

            cocoDt1 = cocoGt.loadRes(result_file)

            # Raise score threshold to 0.1 and limit to the top 30 detections per image
            p_masks, p_labels, p_scores = get_chainercv_format(cocoDt1, imgIds, is_gt=False, cat2idx=cat2idx)

            print(f"\n--- Calculating Overall mAP for {model_name} at IoUs: {overall_ious} ---")
            model_overall_mAPs = {}
            for iou in overall_ious:
                res = eval_instance_segmentation_voc(
                    p_masks, p_labels, p_scores,
                    gt_masks, gt_labels, iou_thresh=iou
                )
                model_overall_mAPs[iou] = res['map'] if res['map'] is not None else 0.0
            overall_mAP_results[model_name] = model_overall_mAPs

            print(f"\n--- Calculating Overlap Confusion Categories for {model_name} ---")
            overlap_counts = calculate_overlap_confusion(p_masks, p_labels, gt_masks, gt_labels, iou_thresh=0.5)
            overlap_confusion_results[model_name] = overlap_counts
            print(f"Counts: {overlap_counts}")

            run_configurations = [
                ("Any_Class", occ_scores_any, all_models_results_any, bins_any, bin_labels_any),
                ("Same_Class", occ_scores_same, all_models_results_same, bins_same, bin_labels_same),
                #("Binary_Overlap", occ_scores_any, all_models_results_binary, bins_binary, bin_labels_binary),
                #("Binary_Same_Overlap", occ_scores_same, all_models_results_binary_same, bins_binary, bin_labels_binary)
            ]

            for occ_type, occ_scores, target_results_dict, specific_bins, specific_labels in run_configurations:
                print(f"\n--- Running Occlusion Metrics [{occ_type}] for {model_name} ---")
                all_dfs = []
                for iou in iou_thresholds:
                    results = run_occlusion_analysis_with_classes(p_masks, p_labels, p_scores, gt_masks, gt_labels,
                                                                  occ_scores, iou, specific_bins, specific_labels)
                    df = pd.DataFrame(results)
                    df['IoU_Threshold'] = iou
                    all_dfs.append(df)

                    csv_name = f"occlusion_mAP_{occ_type}_results_{str(iou).replace('.', '_')}.csv"
                    df.to_csv(os.path.join(model_save_dir, csv_name), index=False)

                target_results_dict[model_name] = all_dfs
                master_df = pd.concat(all_dfs)

                #TODO duurt fucking lang niet nodig voor CUB
                # print(f"Generating per-class plots in {model_save_dir}/class_plots_{occ_type}...")
                # class_plots_dir = os.path.join(model_save_dir, f"class_plots_{occ_type}")
                # os.makedirs(class_plots_dir, exist_ok=True)
                #
                # for idx, class_name in enumerate(cls_name_map):
                #     plt.figure(figsize=(15, 6))
                #     colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e'][:len(iou_thresholds)]
                #     for i, iou in enumerate(iou_thresholds):
                #         df_iou = master_df[master_df['IoU_Threshold'] == iou]
                #         class_ap = [res[idx] if idx < len(res) and not np.isnan(res[idx]) else 0.0 for res in
                #                     df_iou['class_aps']]
                #         plt.plot(df_iou['Occlusion_Category'], class_ap, label=f'AP @ IoU {iou}', color=colors[i],
                #                  marker='o', markersize=3)
                #
                #     plt.title(f'Performance Analysis: {class_name.upper()} ({model_name} | {occ_type})', fontsize=14)
                #     plt.xlabel('Occlusion Percentage (%)', fontsize=12)
                #     plt.ylabel('Average Precision (AP)', fontsize=12)
                #     plt.ylim(0, 1.1)
                #     plt.legend()
                #     plt.grid(True, linestyle='--', alpha=0.5)
                #     plt.savefig(os.path.join(class_plots_dir, f"{class_name}_occlusion_impact.png"))
                #     plt.close('all')

                categories = all_dfs[0]['Occlusion_Category'].tolist()
                x = np.arange(len(categories))
                width = 0.25

                plt.figure(figsize=(18, 7))
                if len(categories) == 4:
                    plt.bar(x - width, all_dfs[0]['mAP'], width, label='mAP @ IoU 0.7', color='b')
                    plt.bar(x, all_dfs[1]['mAP'], width, label='mAP @ IoU 0.5', color='g')
                    plt.bar(x + width, all_dfs[2]['mAP'], width, label='mAP @ IoU 0.3', color='r')
                else:
                    plt.bar(x - width, all_dfs[0]['mAP'], width, label='mAP @ IoU 0.7', color='b')
                    plt.bar(x, all_dfs[1]['mAP'], width, label='mAP @ IoU 0.5', color='g')
                    plt.bar(x + width, all_dfs[2]['mAP'], width, label='mAP @ IoU 0.3', color='r')

                title_str = 'mAP vs. True Mask Occlusion' if occ_type == "Any_Class" else 'mAP vs. Same-Class Mask Occlusion (Overlapping Items Only)'
                if occ_type == "Binary_Overlap":
                    title_str = 'mAP for No Overlap vs. Overlap'

                plt.title(f'{title_str} ({model_name})', fontsize=14)
                plt.xlabel('Occlusion Category', fontsize=12)
                plt.ylabel('mAP', fontsize=12)
                plt.xticks(x, categories)
                plt.legend()
                plt.grid(axis='y', linestyle='--', alpha=0.6)
                plt.tight_layout()

                combined_plot_path = os.path.join(model_save_dir, f"combined_occlusion_impact_bars_{occ_type}.png")
                plt.savefig(combined_plot_path)
                plt.close('all')

            # ----------------------------------------------------------------------
            # Multiprocessing visualization
            # ----------------------------------------------------------------------
            masks_dir = os.path.join(model_save_dir, "mask_visualizations")
            os.makedirs(masks_dir, exist_ok=True)

            img_category_map = {}
            for i, img_id in enumerate(imgIds):
                img_occs = occ_scores_any[i]
                max_occ = np.max(img_occs) if len(img_occs) > 0 else 0
                if max_occ == 0:
                    occ_cat = "0_No_Overlap"
                elif max_occ <= 0.25:
                    occ_cat = "1_Low_Overlap"
                elif max_occ <= 0.50:
                    occ_cat = "2_Medium_Overlap"
                else:
                    occ_cat = "3_High_Overlap"

                p_m = p_masks[i]
                gt_m = gt_masks[i]

                if len(gt_m) == 0 and len(p_m) == 0:
                    img_iou = 1.0
                elif len(gt_m) == 0 or len(p_m) == 0:
                    img_iou = 0.0
                else:
                    ious = []
                    for g in gt_m:
                        intersection = np.logical_and(p_m, g).sum(axis=(1, 2))
                        union = np.logical_or(p_m, g).sum(axis=(1, 2))
                        iou = np.max(intersection / (union + 1e-6))
                        ious.append(iou)
                    img_iou = np.mean(ious)

                if img_iou < 0.5:
                    iou_cat = "Average_IoU_lt_0.5"
                elif img_iou < 0.75:
                    iou_cat = "Average_IoU_0.5_to_0.75"
                else:
                    iou_cat = "Average_IoU_ge_0.75"

                img_category_map[img_id] = os.path.join(occ_cat, iou_cat)

            worker = 8
            per_len = int(len(imgIds) / worker)

            jobs = []
            print("run dataset_mask")
            for worker_id in range(worker):
                start_idx = worker_id * per_len
                end_idx = (worker_id + 1) * per_len if worker_id + 1 != worker else len(imgIds)

                p = multiprocessing.Process(target=dataset_mask,
                                            args=(cocoGt, cocoDt1, masks_dir, root,
                                                  imgIds[start_idx:end_idx], parser.num,
                                                  thr, cls_name_map, img_category_map))
                jobs.append(p)
                p.start()
            for p in jobs:
                p.join()

            print(f"Categorized side-by-side mask visualization images saved to {masks_dir}")

            mAP, cls_ap, cls_names = coco_inst_seg_eval(label_file, result_file)

            stack_item = []
            for key, value in cls_ap.items():
                stack_item.append(value)

            stack_item = np.concatenate(stack_item, axis=0).reshape((len(cls_ap.keys()), -1)).transpose()

            print(f'\nClass Performance(COCOAPI) for {model_name}: ')
            for idx, _ in enumerate(cls_names):
                print("%-15s -->  %.1f, %.1f, %.1f, %.1f" % (cls_names[idx], 100 * stack_item[idx][0],
                                                             100 * stack_item[idx][1], 100 * stack_item[idx][2],
                                                             100 * stack_item[idx][3]))

            print(f'\nPerformance(COCOAPI) for {model_name}: ')
            for k, v in mAP.items():
                print('mAP@%s: %.1f' % (k, 100 * v))

            print(f"\n--- Analyzing Single Instance Splits for {model_name} ---")
            analyze_single_instance_splits(
                p_masks=p_masks,
                p_labels=p_labels,
                gt_masks=gt_masks,
                gt_labels=gt_labels,
                class_names=cls_name_map,
                model_name=model_name,
                save_dir=model_save_dir,
                cocoGt=cocoGt,  # New
                imgIds=imgIds,  # New
                root_dir=root  # New
            )
            # >>> NEW CODE FOR SINGLE-INSTANCE mAP <<<
            print(f"\n--- Calculating mAP for Single-Instance Images ({model_name}) ---")

            print(f"\n--- GT split ({model_name}) ---")
            analyze_gt_splits(
                cocoGt=cocoGt,
                cocoDt=cocoDt1,
                imgIds=imgIds,
                model_name=model_name,
                save_dir=model_save_dir,
                iou_thresh=0.5
            )

            single_gt_masks = []
            single_gt_labels = []
            single_p_masks = []
            single_p_labels = []
            single_p_scores = []

            # Filter for images that have exactly 1 GT mask
            for i in range(len(gt_masks)):
                if len(gt_masks[i]) == 1:
                    single_gt_masks.append(gt_masks[i])
                    single_gt_labels.append(gt_labels[i])
                    single_p_masks.append(p_masks[i])
                    single_p_labels.append(p_labels[i])
                    single_p_scores.append(p_scores[i])

            if len(single_gt_masks) > 0:
                for iou in overall_ious:  # Uses the overall_ious list from your script (e.g., 0.25, 0.5, 0.7, 0.75)
                    res_single = eval_instance_segmentation_voc(
                        single_p_masks, single_p_labels, single_p_scores,
                        single_gt_masks, single_gt_labels, iou_thresh=iou
                    )
                    single_map = res_single['map'] if res_single['map'] is not None else 0.0
                    print(f"  mAP @ IoU {iou}: {single_map:.4f}")

                # Optional: Save these specific results to a CSV
                single_map_df = pd.DataFrame({"Class": cls_name_map, "AP_at_IoU_0.5": res_single['ap']})
                single_map_df.to_csv(os.path.join(model_save_dir, f"single_instance_mAP_{model_name}.csv"), index=False)
            else:
                print("  No single-instance images found to evaluate.")
            # >>> END OF NEW CODE <<<

            print(f"\n--- Calculating Overall mAP for {model_name} at IoUs: {overall_ious} ---")

        # ======================================================================
        # GENERATE CROSS-MODEL COMPARISON PLOT
        # ======================================================================
        print("\n" + "=" * 50)
        print("Generating Cross-Model Comparison Bar Plots...")
        print("=" * 50)

        cross_model_configs = [
            ("Any_Class", all_models_results_any, 'Model Comparison: mAP vs. True Mask Occlusion (Any Class)'),
            ("Same_Class", all_models_results_same,
             'Model Comparison: mAP vs. Same-Class Occlusion (Overlapping Items Only)'),
            #("Binary_Overlap", all_models_results_binary, 'Model Comparison: mAP for No Overlap vs. Overlap')
        ]

        for occ_type, results_dict, title_str in cross_model_configs:
            if not results_dict: continue

            fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
            fig.suptitle(title_str, fontsize=16, fontweight='bold')

            categories = list(results_dict.values())[0][0]['Occlusion_Category'].tolist()
            x = np.arange(len(categories))

            num_models = len(results_dict)
            total_width = 0.8
            bar_width = total_width / num_models

            for i, iou in enumerate(iou_thresholds):
                ax = axes[i]
                for j, (model_name, model_dfs) in enumerate(results_dict.items()):
                    df_iou = model_dfs[i]
                    offset = x - (total_width / 2) + (j * bar_width) + (bar_width / 2)
                    ax.bar(offset, df_iou['mAP'], width=bar_width, label=model_name, alpha=0.9)

                ax.set_title(f'mAP @ IoU {iou}', fontsize=14)
                ax.set_xlabel('Occlusion Category', fontsize=12)
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
                if i == 0:
                    ax.set_ylabel('mAP', fontsize=12)

                ax.grid(axis='y', linestyle='--', alpha=0.6)
                if i == 0:
                    ax.legend(title="Models")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            comparison_plot_path = os.path.join(base_save_dir, f"compare_all_models_occlusion_bars_{occ_type}.png")
            plt.savefig(comparison_plot_path)
            plt.close('all')
            print(f"Cross-model {occ_type} occlusion comparison saved to {comparison_plot_path}")

        # ======================================================================
        # GENERATE OVERALL mAP COMPARISON PLOT
        # ======================================================================
        print("\nGenerating Overall mAP Comparison Bar Plot...")

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Model Comparison: Overall mAP at Different IoU Thresholds', fontsize=16, fontweight='bold')

        overall_ious_str = [f'IoU {iou}' for iou in overall_ious]
        x_overall = np.arange(len(overall_ious))

        for j, (model_name, maps_dict) in enumerate(overall_mAP_results.items()):
            offset = x_overall - (total_width / 2) + (j * bar_width) + (bar_width / 2)
            y_vals = [maps_dict[iou] for iou in overall_ious]
            ax.bar(offset, y_vals, width=bar_width, label=model_name, alpha=0.9)

        ax.set_xlabel('IoU Threshold', fontsize=12)
        ax.set_ylabel('mAP', fontsize=12)
        ax.set_xticks(x_overall)
        ax.set_xticklabels(overall_ious_str)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.legend(title="Models")

        plt.tight_layout()
        overall_mAP_plot_path = os.path.join(base_save_dir, "compare_all_models_overall_mAP_bars.png")
        plt.savefig(overall_mAP_plot_path)
        plt.close('all')
        print(f"Overall mAP comparison bar plot successfully saved to {overall_mAP_plot_path}\n")

        # ======================================================================
        # GENERATE OVERLAP CONFUSION COMPARISON PLOT
        # ======================================================================
        print("\n" + "=" * 50)
        print("Generating Overlap Confusion Bar Plot and CSV...")
        print("=" * 50)

        confusion_df = pd.DataFrame(overlap_confusion_results).T
        confusion_csv_path = os.path.join(base_save_dir, "overlap_confusion_matrix.csv")
        confusion_df.to_csv(confusion_csv_path, index_label="Model")
        print(f"Overlap confusion matrix CSV saved to {confusion_csv_path}")

        fig, ax = plt.subplots(figsize=(12, 7))
        models = list(overlap_confusion_results.keys())

        same_class = [overlap_confusion_results[m]["Overlapping same class"] for m in models]
        diff_class = [overlap_confusion_results[m]["Overlapping different class"] for m in models]
        no_overlap = [overlap_confusion_results[m]["No overlapping"] for m in models]

        x_pos = np.arange(len(models))
        width = 0.5

        ax.bar(x_pos, same_class, width, label='TP (Same class)', color='#2ca02c')
        ax.bar(x_pos, diff_class, width, bottom=same_class, label='Misclassified (Diff class)', color='#ff7f0e')
        ax.bar(x_pos, no_overlap, width, bottom=np.array(same_class) + np.array(diff_class), label='FP (No overlap)',
               color='#d62728')

        ax.set_ylabel('Number of Predicted Masks', fontsize=12)
        ax.set_title('Prediction Confusion Breakdown by Model (IoU $\geq$ 0.5)', fontsize=16, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        confusion_plot_path = os.path.join(base_save_dir, "compare_all_models_overlap_confusion.png")
        plt.savefig(confusion_plot_path)
        plt.close('all')
        print(f"Overlap confusion bar chart saved to {confusion_plot_path}\n")

        # ======================================================================
        # GENERATE CROSS-MODEL MASK VISUALIZATION COMPARISON WITH LABELS
        # ======================================================================
        from PIL import ImageDraw, ImageFont

        print("\n" + "=" * 50)
        print("Generating Cross-Model Mask Visualization Comparison Grid with Labels...")
        print("=" * 50)

        comparison_masks_dir = os.path.join(base_save_dir, "mask_visualizations_comparison")
        os.makedirs(comparison_masks_dir, exist_ok=True)

        image_to_model_paths = {}
        for result_file in result_files:
            m_name = os.path.splitext(os.path.basename(result_file))[0]
            m_viz_dir = os.path.join(base_save_dir, m_name, "mask_visualizations")

            for root_subdir, _, files in os.walk(m_viz_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if f not in image_to_model_paths:
                            image_to_model_paths[f] = []
                        image_to_model_paths[f].append((m_name, os.path.join(root_subdir, f)))

        for img_name, model_tuples in tqdm(image_to_model_paths.items(), desc="Tiling model comparisons"):
            model_tuples.sort(key=lambda x: x[0])

            processed_tiles = []
            for m_name, m_path in model_tuples:
                try:
                    tile_img = Image.open(m_path).convert("RGB")
                    draw = ImageDraw.Draw(tile_img)
                    img_w, img_h = tile_img.size

                    dynamic_font_size = max(24, int(img_w * 0.05))

                    try:
                        font = ImageFont.truetype("arial.ttf", dynamic_font_size)
                    except:
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                                      dynamic_font_size)
                        except:
                            font = ImageFont.load_default(size=dynamic_font_size)

                    text_label = f" {m_name} "
                    padding = 10

                    bbox = draw.textbbox((10, 10), text_label, font=font)
                    padded_bbox = [bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5]

                    draw.rectangle(padded_bbox, fill="black")
                    draw.text((10, 10), text_label, fill="white", font=font)

                    processed_tiles.append(tile_img)
                except Exception as e:
                    print(f"Error processing {m_path}: {e}")

            if not processed_tiles:
                continue

            num_to_tile = len(processed_tiles)
            cols = 2 if num_to_tile > 1 else 1
            rows = (num_to_tile + cols - 1) // cols

            ref_w, ref_h = processed_tiles[0].size
            grid_img = Image.new('RGB', (ref_w * cols, ref_h * rows), (255, 255, 255))

            for idx, tile_obj in enumerate(processed_tiles):
                r = idx // cols
                c = idx % cols
                grid_img.paste(tile_obj, (c * ref_w, r * ref_h))

            grid_img.save(os.path.join(comparison_masks_dir, img_name), quality=95)

            for t in processed_tiles:
                t.close()

        print(f"Labeled comparison grids saved to {comparison_masks_dir}")

        # ======================================================================
        # UPDATE AND EXPORT LATEX OVERVIEW TABLE
        # ======================================================================
        print("\n" + "=" * 50)
        print("Extracting Table Metrics and Generating overview_table.txt...")
        print("=" * 50)

        any_overlap_maps = {}
        same_overlap_maps = {}

        for model in all_models_results_binary.keys():
            # Extract maps for 'Any Overlap' (Overlapping between classes)
            df_any = pd.concat(all_models_results_binary[model])
            any_overlap_maps[model] = {}
            for iou in [0.3, 0.5, 0.75]:
                val = df_any[(df_any['IoU_Threshold'] == iou) & (df_any['Occlusion_Category'] == 'Overlap')][
                    'mAP'].values
                any_overlap_maps[model][iou] = val[0] if len(val) > 0 and not pd.isna(val[0]) else 0.0

            # Extract maps for 'Same Class Overlap'
            df_same = pd.concat(all_models_results_binary_same[model])
            same_overlap_maps[model] = {}
            for iou in [0.3, 0.5, 0.75]:
                val = df_same[(df_same['IoU_Threshold'] == iou) & (df_same['Occlusion_Category'] == 'Overlap')][
                    'mAP'].values
                same_overlap_maps[model][iou] = val[0] if len(val) > 0 and not pd.isna(val[0]) else 0.0

        update_and_generate_latex_table(
            dataset_arg=parser.dataset,
            model_names=list(all_models_results_binary.keys()),
            overall_maps=overall_mAP_results,
            any_overlap_maps=any_overlap_maps,
            same_overlap_maps=same_overlap_maps,
            db_path=os.path.join(base_save_dir, "latex_table_db.json"),
            txt_path=os.path.join(base_save_dir, "overview_table.txt")
        )
        print(f"LaTeX table successfully exported to {os.path.join(base_save_dir, 'overview_table.txt')}")

    elif parser.dataset == "voc_val_gt":
        label_file = "./data/VOC2012/annotations/voc_2012_val.json"
        root = './data/VOC2012/JPEGImages'
        save_dir = parser.save_dir
        os.makedirs(save_dir, exist_ok=True)
        cls_name_map, name_2_index = id_2_clsname(label_file)
        cocoGt = COCO(label_file)
        gt_dataset_mask(cocoGt, save_dir, root, cls_name_map)

    elif parser.dataset == "voc_train_gt":
        label_file = "./data/VOC2012/annotations/voc_2012_trainaug.json"
        root = './data/VOC2012/JPEGImages'
        save_dir = parser.save_dir
        os.makedirs(save_dir, exist_ok=True)
        cls_name_map, name_2_index = id_2_clsname(label_file)
        cocoGt = COCO(label_file)
        gt_dataset_mask(cocoGt, save_dir, root, cls_name_map)

    elif parser.dataset == "coco_val_gt":
        label_file = "./datasets/coco2017/annotations/instances_val2017.json"
        root = './datasets/coco2017/val2017'
        save_dir = parser.save_dir
        os.makedirs(save_dir, exist_ok=True)
        cls_name_map, name_2_index = id_2_clsname(label_file)
        cocoGt = COCO(label_file)
        gt_dataset_mask(cocoGt, save_dir, root, cls_name_map)

    elif parser.dataset == "coco_train_gt":
        label_file = "./datasets/coco2017/annotations/instances_train2017.json"
        root = './datasets/coco2017/train2017'
        save_dir = parser.save_dir
        os.makedirs(save_dir, exist_ok=True)
        cls_name_map, name_2_index = id_2_clsname(label_file)
        cocoGt = COCO(label_file)
        gt_dataset_mask(cocoGt, save_dir, root, cls_name_map)

    else:
        raise NotImplementedError