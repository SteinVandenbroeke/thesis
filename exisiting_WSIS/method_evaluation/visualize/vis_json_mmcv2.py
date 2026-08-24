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
# OCCLUSION METRICS AND PLOTTING FUNCTIONS
# ==============================================================================

def calculate_instance_occlusion(gt_masks):
    all_occ_scores = []
    for masks in gt_masks:
        n = masks.shape[0]
        scores = []
        for i in range(n):
            mask_i = masks[i].astype(bool)
            pos = np.where(mask_i)
            if pos[0].size == 0:
                scores.append(0)
                continue

            ymin, xmin = np.min(pos, axis=1)
            ymax, xmax = np.max(pos, axis=1)

            max_occ = 0
            for j in range(n):
                if i == j: continue
                mask_j = masks[j].astype(bool)
                overlap = mask_j[ymin:ymax, xmin:xmax].sum()
                area_i = np.sum(mask_i)

                occ_ratio = overlap / (overlap + area_i)
                max_occ = max(max_occ, occ_ratio)
            scores.append(max_occ)
        all_occ_scores.append(np.array(scores))
    return all_occ_scores

def run_occlusion_analysis_with_classes(p_masks, p_labels, p_scores, gt_masks, gt_labels, occ_scores, iou_thresh):
    bins = [(-0.01, 0.0), (0.0, 0.25), (0.25, 0.50), (0.50, 1.0)]
    bin_labels = ["No Overlap", "Low (0-25%)", "Medium (25-50%)", "High (>50%)"]
    bin_results = []

    for (low, high), label in zip(bins, bin_labels):
        filtered_gt_masks = []
        filtered_gt_labels = []
        filtered_p_masks = []
        filtered_p_labels = []
        filtered_p_scores = []

        for p_m, p_l, p_s, gt_m, gt_l, gt_o in zip(p_masks, p_labels, p_scores, gt_masks, gt_labels, occ_scores):
            # 1. Filter Ground Truths
            keep_gt_idx = np.where((gt_o > low) & (gt_o <= high))[0]
            ignore_gt_idx = np.where((gt_o <= low) | (gt_o > high))[0]

            filtered_gt_masks.append(gt_m[keep_gt_idx])
            filtered_gt_labels.append(gt_l[keep_gt_idx])

            # 2. Filter Predictions (Remove predictions matching hidden Ground Truths)
            if len(p_m) == 0:
                filtered_p_masks.append(p_m)
                filtered_p_labels.append(p_l)
                filtered_p_scores.append(p_s)
                continue

            keep_p = []
            for pi in range(len(p_m)):
                pred_mask = p_m[pi]
                pred_label = p_l[pi]

                # Check IoU against ignored/hidden GT objects
                max_iou_ignore = 0.0
                if len(ignore_gt_idx) > 0:
                    for ig_i in ignore_gt_idx:
                        if gt_l[ig_i] == pred_label:
                            inter = np.logical_and(pred_mask, gt_m[ig_i]).sum()
                            union = np.logical_or(pred_mask, gt_m[ig_i]).sum()
                            iou = inter / (union + 1e-6)
                            if iou > max_iou_ignore: max_iou_ignore = iou

                # Check IoU against kept GT objects in current occlusion bin
                max_iou_keep = 0.0
                if len(keep_gt_idx) > 0:
                    for kg_i in keep_gt_idx:
                        if gt_l[kg_i] == pred_label:
                            inter = np.logical_and(pred_mask, gt_m[kg_i]).sum()
                            union = np.logical_or(pred_mask, gt_m[kg_i]).sum()
                            iou = inter / (union + 1e-6)
                            if iou > max_iou_keep: max_iou_keep = iou

                # If this prediction primarily belongs to a hidden GT object, discard it so it isn't counted as a False Positive
                if max_iou_ignore > 0.1 and max_iou_ignore > max_iou_keep:
                    continue
                else:
                    keep_p.append(pi)

            # Numpy array subsetting naturally handles empty lists resulting in (0, H, W)
            filtered_p_masks.append(p_m[keep_p])
            filtered_p_labels.append(p_l[keep_p])
            filtered_p_scores.append(p_s[keep_p])

        print(f"Evaluating instances with {label} occlusion @ IoU {iou_thresh}...")
        res = eval_instance_segmentation_voc(
            filtered_p_masks, filtered_p_labels, filtered_p_scores,
            filtered_gt_masks, filtered_gt_labels, iou_thresh=iou_thresh
        )

        bin_results.append({
            "Occlusion_Category": label,
            "mAP": res['map'] if res['map'] is not None else 0.0,
            "class_aps": res['ap']
        })
    return bin_results
    
def get_chainercv_format(coco_api, imgIds, is_gt=True, cat2idx=None, max_dets=100, score_thr=0.01):
    all_masks, all_labels, all_scores = [], [], []

    if cat2idx is None:
        catIds = sorted(coco_api.getCatIds())
        cat2idx = {cat: i for i, cat in enumerate(catIds)}

    for img_id in tqdm(imgIds, desc=f"Converting {'GT' if is_gt else 'Pred'} formats"):
        ann_ids = coco_api.getAnnIds(imgIds=[img_id])
        anns = coco_api.loadAnns(ann_ids)

        # Memory fix applied:
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


def dataset_mask(cocoGT, cocoDt, save_dir, root_dir, imgIds=None, num=None, anno_thr=-1, name_mapping=None, img_category_map=None):
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

            # ----------------------------------------------------------------------
            # Render Ground Truth Image Copy
            # ----------------------------------------------------------------------
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

            # ----------------------------------------------------------------------
            # Render Prediction Image Copy & compute IoU per object
            # ----------------------------------------------------------------------
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

            # ----------------------------------------------------------------------
            # Stitch GT and Predictions side-by-side
            # ----------------------------------------------------------------------
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
            label_file = "./data/coco2017/annotations/instances_val2017.json"
            root = './data/coco2017/val2017'
        elif parser.dataset == "coco_test":
            label_file = "./data/coco2017/annotations/image_info_test-dev2017.json"
            root = './data/coco2017/test2017'
        elif parser.dataset == "coco_train":
            label_file = "./data/coco2017/annotations/instances_train2017.json"
            root = './data/coco2017/train2017'

        result_files = parser.result_files.split(',')
        base_save_dir = parser.save_dir
        os.makedirs(base_save_dir, exist_ok=True)
        thr = parser.thr

        cocoGt = COCO(label_file)
        imgIds = sorted(cocoGt.getImgIds())

        if parser.num is not None:
            imgIds = imgIds[:parser.num]
            
        cls_name_map, name_2_index = id_2_clsname(label_file)

        # ----------------------------------------------------------------------
        # Data Structures to hold all models' performance for the final plots
        # ----------------------------------------------------------------------
        all_models_results = {}
        overall_mAP_results = {}
        
        iou_thresholds = [0.7, 0.5, 0.3]
        overall_ious = [0.25, 0.50, 0.70, 0.75]

        print("\n--- Extracting GT and Calculating Occlusion ---")
        gt_masks, gt_labels, cat2idx = get_chainercv_format(cocoGt, imgIds, is_gt=True)
        occ_scores = calculate_instance_occlusion(gt_masks)

        # ----------------------------------------------------------------------
        # Data Distribution Plot: GT Samples per Class & Occlusion Category
        # ----------------------------------------------------------------------
        print("\n--- Plotting Sample Distribution per Class & Occlusion Category ---")
        bins = [(-0.01, 0.0), (0.0, 0.25), (0.25, 0.50), (0.50, 1.0)]
        bin_labels = ["No Overlap", "Low (0-25%)", "Medium (25-50%)", "High (>50%)"]

        cats = cocoGt.loadCats(cocoGt.getCatIds())
        cat_id_to_name = {cat['id']: cat['name'] for cat in cats}
        idx_to_cat_id = {v: k for k, v in cat2idx.items()}
        idx_to_name = {idx: cat_id_to_name[cat_id] for idx, cat_id in idx_to_cat_id.items()}

        class_occ_counts = {idx: {label: 0 for label in bin_labels} for idx in idx_to_name.keys()}

        for img_labels, img_occs in zip(gt_labels, occ_scores):
            for label, occ in zip(img_labels, img_occs):
                for (low, high), bin_label in zip(bins, bin_labels):
                    if low < occ <= high:
                        class_occ_counts[label][bin_label] += 1
                        break

        plot_data = {"Class": [], "No Overlap": [], "Low (0-25%)": [], "Medium (25-50%)": [], "High (>50%)": []}
        
        for idx in sorted(class_occ_counts.keys(), key=lambda x: idx_to_name[x]):
            class_name = idx_to_name[idx]
            plot_data["Class"].append(class_name)
            for label in bin_labels:
                plot_data[label].append(class_occ_counts[idx][label])

        df_counts = pd.DataFrame(plot_data)

        fig, ax = plt.subplots(figsize=(18, 8))
        bottom = np.zeros(len(df_counts))
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'] 

        for label, color in zip(bin_labels, colors):
            ax.bar(df_counts["Class"], df_counts[label], label=label, bottom=bottom, color=color, alpha=0.85)
            bottom += df_counts[label].values

        ax.set_title('Distribution of Samples per Occlusion Category per Segmentation Class', fontsize=16, fontweight='bold')
        ax.set_xlabel('Segmentation Class', fontsize=12)
        ax.set_ylabel('Total Number of Instances in GT Dataset', fontsize=12)
        plt.xticks(rotation=90, ha='center')
        ax.legend(title="Occlusion Category")
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        dist_plot_path = os.path.join(base_save_dir, "gt_samples_per_occlusion_category.png")
        plt.savefig(dist_plot_path)
        plt.close('all')
        print(f"Sample distribution plot saved to {dist_plot_path}\n")

        # ----------------------------------------------------------------------
        # Model Evaluation Loop
        # ----------------------------------------------------------------------
        for result_file in result_files:
            model_name = os.path.splitext(os.path.basename(result_file))[0]
            print(f"\n{'='*50}\nEvaluating Model: {model_name}\n{'='*50}")
            
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


            print(f"\n--- Running Occlusion Metrics for {model_name} ---")
            all_dfs = []
            for iou in iou_thresholds:
                results = run_occlusion_analysis_with_classes(p_masks, p_labels, p_scores, gt_masks, gt_labels, occ_scores, iou)
                df = pd.DataFrame(results)
                df['IoU_Threshold'] = iou
                all_dfs.append(df)

                csv_name = f"occlusion_mAP_results_{str(iou).replace('.', '_')}.csv"
                df.to_csv(os.path.join(model_save_dir, csv_name), index=False)

            all_models_results[model_name] = all_dfs

            master_df = pd.concat(all_dfs)
            
            print(f"Generating per-class plots in {model_save_dir}/class_plots...")
            class_plots_dir = os.path.join(model_save_dir, "class_plots")
            os.makedirs(class_plots_dir, exist_ok=True)
            
            for idx, class_name in enumerate(cls_name_map):
                plt.figure(figsize=(15, 6))
                colors = ['#1f77b4', '#2ca02c', '#d62728']
                for i, iou in enumerate(iou_thresholds):
                    df_iou = master_df[master_df['IoU_Threshold'] == iou]
                    class_ap = [res[idx] if idx < len(res) and not np.isnan(res[idx]) else 0.0 for res in df_iou['class_aps']]
                    plt.plot(df_iou['Occlusion_Category'], class_ap, label=f'AP @ IoU {iou}', color=colors[i], marker='o', markersize=3)

                plt.title(f'Performance Analysis: {class_name.upper()} ({model_name})', fontsize=14)
                plt.xlabel('Occlusion Percentage (%)', fontsize=12)
                plt.ylabel('Average Precision (AP)', fontsize=12)
                plt.ylim(0, 1.1)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.savefig(os.path.join(class_plots_dir, f"{class_name}_occlusion_impact.png"))
                plt.close('all')

            categories = all_dfs[0]['Occlusion_Category'].tolist()
            x = np.arange(len(categories))
            width = 0.25

            plt.figure(figsize=(18, 7))
            plt.bar(x - width, all_dfs[0]['mAP'], width, label='mAP @ IoU 0.7', color='b')
            plt.bar(x, all_dfs[1]['mAP'], width, label='mAP @ IoU 0.5', color='g')
            plt.bar(x + width, all_dfs[2]['mAP'], width, label='mAP @ IoU 0.3', color='r')

            plt.title(f'mAP vs. Instance Occlusion at Multiple IoU Thresholds ({model_name})', fontsize=14)
            plt.xlabel('Occlusion Category', fontsize=12)
            plt.ylabel('mAP', fontsize=12)
            plt.xticks(x, categories)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            combined_plot_path = os.path.join(model_save_dir, "combined_occlusion_impact_bars.png")
            plt.savefig(combined_plot_path)
            plt.close('all')

            # ----------------------------------------------------------------------
            # Multiprocessing visualization (Images Category Routing by IoU)
            # ----------------------------------------------------------------------
            masks_dir = os.path.join(model_save_dir, "mask_visualizations")
            os.makedirs(masks_dir, exist_ok=True)
            
            img_category_map = {}
            for i, img_id in enumerate(imgIds):
                img_occs = occ_scores[i]
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

                if img_iou < 0.10:
                    iou_cat = "Average_IoU_lt_0.1"
                elif img_iou < 0.25:
                    iou_cat = "Average_IoU_0.1_to_0.25"
                elif img_iou < 0.5:
                    iou_cat = "Average_IoU_0.25_to_0.5"
                elif img_iou < 0.75:
                    iou_cat = "Average_IoU_0.5_to_0.75"
                else:
                    iou_cat = "Average_IoU_ge_0.75"

                img_category_map[img_id] = os.path.join(occ_cat, iou_cat)
            
            worker = 12
            per_len = int(len(imgIds) / worker)

            jobs = []
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

        # ======================================================================
        # GENERATE CROSS-MODEL COMPARISON PLOT (Occlusion Grouped Bar Charts)
        # ======================================================================
        print("\n" + "="*50)
        print("Generating Cross-Model Comparison Bar Plot...")
        print("="*50)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        fig.suptitle('Model Comparison: mAP vs. Instance Occlusion', fontsize=16, fontweight='bold')

        categories = list(all_models_results.values())[0][0]['Occlusion_Category'].tolist()
        x = np.arange(len(categories))
        
        num_models = len(all_models_results)
        total_width = 0.8
        bar_width = total_width / num_models
        
        for i, iou in enumerate(iou_thresholds):
            ax = axes[i]
            
            for j, (model_name, model_dfs) in enumerate(all_models_results.items()):
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
        
        comparison_plot_path = os.path.join(base_save_dir, "compare_all_models_occlusion_bars.png")
        plt.savefig(comparison_plot_path)
        plt.close('all')
        print(f"Cross-model occlusion comparison bar plot successfully saved to {comparison_plot_path}")


        # ======================================================================
        # GENERATE OVERALL mAP COMPARISON PLOT (Grouped Bar Charts)
        # ======================================================================
        print("Generating Overall mAP Comparison Bar Plot...")
        
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
        label_file = "./data/coco2017/annotations/instances_val2017.json"
        root = './data/coco2017/val2017'
        save_dir = parser.save_dir
        os.makedirs(save_dir, exist_ok=True)
        cls_name_map, name_2_index = id_2_clsname(label_file)
        cocoGt = COCO(label_file)
        gt_dataset_mask(cocoGt, save_dir, root, cls_name_map)

    elif parser.dataset == "coco_train_gt":
        label_file = "./data/coco2017/annotations/instances_train2017.json"
        root = './data/coco2017/train2017'
        save_dir = parser.save_dir
        os.makedirs(save_dir, exist_ok=True)
        cls_name_map, name_2_index = id_2_clsname(label_file)
        cocoGt = COCO(label_file)
        gt_dataset_mask(cocoGt, save_dir, root, cls_name_map)

    else:
        raise NotImplementedError