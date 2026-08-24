import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import hashlib
import scipy.ndimage as ndimage
import colorsys

# --- Dataset Paths including all train/val/test splits ---
DATASET_CONFIGS = {
    "voc": {
        "val": {"label_file": "./data/VOC2012/annotations/voc_2012_val.json", "root": "./data/VOC2012/JPEGImages"},
    },
    "coco": {
        "val": {"label_file": "./datasets/coco_dataset/coco2017/annotations/instances_val2017.json",
                "root": "./datasets/coco_dataset/coco2017/val2017"},
        "test": {"label_file": "./datasets/coco_dataset/coco2017/annotations/image_info_test-dev2017.json",
                 "root": "./datasets/coco_dataset/coco2017/test2017"},
        "train": {"label_file": "./datasets/coco_dataset/coco2017/annotations/instances_train2017.json",
                  "root": "./datasets/coco_dataset/coco2017/train2017"}
    },
    "cub": {
        "val": {"label_file": "./datasets/CUB_200_2011/CUB_as_COCO/annotations/instances_val2017.json",
                "root": "./datasets/CUB_200_2011/CUB_200_2011/images_combined"},
        "train": {"label_file": "./datasets/CUB_200_2011/CUB_as_COCO/annotations/instances_train2017.json",
                  "root": "./datasets/CUB_200_2011/CUB_200_2011/images_combined"}
    }
}


def fix_rle_bytes(coco_obj):
    for ann in coco_obj.dataset.get('annotations', []):
        segm = ann.get('segmentation', None)
        if isinstance(segm, dict) and 'counts' in segm and isinstance(segm['counts'], str):
            segm['counts'] = segm['counts'].encode('utf-8')


def infer_dataset_type(file_path):
    path_lower = file_path.lower()
    if "result_cub" in path_lower: return "cub"
    if "result_voc" in path_lower: return "voc"
    if "result_coco" in path_lower: return "coco"
    raise ValueError(f"Could not infer dataset type (cub, voc, coco) from path: {file_path}")


def load_merged_coco(ds_type):
    configs = DATASET_CONFIGS[ds_type]
    split_cocos = []

    for split, conf in configs.items():
        if os.path.exists(conf["label_file"]):
            print(f"Loading {ds_type.upper()} '{split}' GT from {conf['label_file']}...")
            c = COCO(conf["label_file"])
            for img in c.dataset.get('images', []):
                img['root_path'] = conf["root"]
            fix_rle_bytes(c)
            split_cocos.append(c)
        else:
            print(f"⚠️ Notice: {ds_type.upper()} '{split}' GT not found at {conf['label_file']}. Skipping split.")

    if not split_cocos:
        raise FileNotFoundError(f"Could not find ANY Ground Truth JSON files for dataset type: {ds_type}")

    merged = COCO()
    merged.dataset = {
        'categories': split_cocos[0].dataset['categories'],
        'images': [],
        'annotations': []
    }

    for c in split_cocos:
        merged.dataset['images'].extend(c.dataset.get('images', []))
        merged.dataset['annotations'].extend(c.dataset.get('annotations', []))

    merged.createIndex()
    return merged


def generate_distinct_colors(num_colors, seed_val):
    """Generates maximally distinct RGB colors by stepping around the HSV color wheel."""
    if num_colors <= 0:
        return []

    np.random.seed(seed_val)
    start_hue = np.random.random()
    colors = []

    for i in range(num_colors):
        hue = (start_hue + (i / num_colors) + np.random.uniform(-0.05, 0.05)) % 1.0
        saturation = np.random.uniform(0.7, 1.0)
        value = np.random.uniform(0.8, 1.0)
        rgb = mcolors.hsv_to_rgb([hue, saturation, value])
        colors.append(rgb.tolist())

    np.random.shuffle(colors)
    return colors


def compute_max_iou(pred_ann, gt_anns, img_info):
    """Computes the maximum IoU between a prediction and ground truth annotations of the same class."""
    gt_matches = [gt for gt in gt_anns if gt['category_id'] == pred_ann['category_id']]
    if not gt_matches:
        return 0.0

    def get_rle(ann):
        if 'segmentation' in ann and ann['segmentation']:
            segm = ann['segmentation']
            if isinstance(segm, list):
                # Convert Polygon to RLE
                rles = maskUtils.frPyObjects(segm, img_info['height'], img_info['width'])
                return maskUtils.merge(rles)
            elif isinstance(segm, dict):
                # Check if it's an uncompressed RLE (where counts is a list)
                if isinstance(segm.get('counts'), list):
                    # Convert uncompressed RLE to compressed RLE
                    return maskUtils.frPyObjects([segm], img_info['height'], img_info['width'])[0]

                # Failsafe: Ensure string counts are encoded to bytes
                if isinstance(segm.get('counts'), str):
                    segm['counts'] = segm['counts'].encode('utf-8')

                # Already a compressed RLE
                return segm
        elif 'bbox' in ann:
            # Fallback to Bounding Box RLE
            return maskUtils.frPyObjects([ann['bbox']], img_info['height'], img_info['width'])[0]
        return None

    dt_rle = get_rle(pred_ann)
    if not dt_rle:
        return 0.0

    gt_rles = [get_rle(gt) for gt in gt_matches]
    gt_rles = [r for r in gt_rles if r is not None]

    if not gt_rles:
        return 0.0

    # Calculate IoU using pycocotools
    ious = maskUtils.iou([dt_rle], gt_rles, [0] * len(gt_rles))
    if ious.size > 0:
        return float(np.max(ious))

    return 0.0


def main():
    parser = argparse.ArgumentParser(
        prog='Vis_example',
        description='Generate visualisation image using pregenerated json segmentation files',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("-o", "--Output", required=True, help="Image output path")
    parser.add_argument("-r", "--Result_files", required=True,
                        help="Result files, example input: result_cub/BESTIE_CUB.json,result_voc/CIM-VOC.json")
    parser.add_argument("-v", "--VisualisationString", required=True,
                        help="Format: [image_id]:[file_index][style][draw_mode][source]\n"
                             " -Style: C (Combined), S (Seperated), N (No styling)\n"
                             " -Draw Mode: M (Masks only), B (Boxes only), A (All - Masks & Boxes)\n"
                             " -Source: G (Ground truth), P (Predictions)\n"
                             " -image_id settings:\n"
                             "    R or empty : Random Row ID (uses the same random image across the row)\n"
                             "    C          : Random Col ID (uses the same random image down the column)\n"
                             "    U          : Unique ID (generates a new unique image every time)\n"
                             "    <id>       : Specific image ID (e.g. 123)\n"
                             " Example: \"R:0CMA R:0CBG \\n C:1SMA C:1SBM\"")
    parser.add_argument("-t", "--Title", default="", help="Title of the plot")
    parser.add_argument("-c", "--Classes", default="",
                        help="Optional comma-separated list of class names to display (e.g. 'person,dog').")

    args = parser.parse_args()

    if not os.path.exists(args.Output) and '.png' not in args.Output:
        os.makedirs(args.Output)

    target_classes = [c.strip().lower() for c in args.Classes.split(',')] if args.Classes.strip() else []

    # --- 1. Parse Result Files and Load Merged Ground Truths ---
    result_files = [f.strip() for f in args.Result_files.split(',') if f.strip()]

    gt_objects = {}
    gt_sources = []
    pred_sources = []
    dataset_types = []

    for rf in result_files:
        ds_type = infer_dataset_type(rf)
        dataset_types.append(ds_type)

        if ds_type not in gt_objects:
            gt_objects[ds_type] = load_merged_coco(ds_type)

        gt_src = gt_objects[ds_type]
        gt_sources.append(gt_src)

        print(f"Loading {ds_type.upper()} Predictions from {rf}...")
        pred_coco = gt_src.loadRes(rf)
        fix_rle_bytes(pred_coco)
        pred_sources.append(pred_coco)

    # --- 2. Parse Visualisation String ---
    vis_string = args.VisualisationString.replace('\\n', '\n')
    rows_str = vis_string.split('\n')
    grid = []

    used_ids = set()  # Track all unique IDs assigned to avoid duplicates
    col_ids_map = {}  # Track column IDs (maps col_idx -> img_id)

    def get_unique_random_id(dataset_idx):
        if dataset_idx >= len(gt_sources):
            raise ValueError(f"Dataset index {dataset_idx} out of range.")

        gt_src = gt_sources[dataset_idx]
        all_ids = list(gt_src.imgs.keys())

        # Filter out any IDs already used in the visualization
        available_ids = [vid for vid in all_ids if str(vid) not in used_ids]

        if not available_ids:
            raise ValueError(f"No unused images left in dataset index {dataset_idx}")

        # Shuffle to randomize
        np.random.shuffle(available_ids)

        # Check files until we find one that exists on the file system
        for vid in available_ids:
            img_info = gt_src.imgs[vid]
            img_path = os.path.join(img_info['root_path'], img_info['file_name'])
            if os.path.exists(img_path):
                new_id = str(vid)
                used_ids.add(new_id)
                return new_id

        raise FileNotFoundError(
            f"Could not find any existing physical image files for dataset {dataset_idx} among unused IDs. Check your dataset directories.")

    for r, row in enumerate(rows_str):
        if not row.strip(): continue
        cells = [c.strip() for c in row.strip().split(' ') if c.strip()]

        row_config = []
        row_id = None  # Track the ID for this specific row

        for c, cell in enumerate(cells):
            m = re.match(r'^(.*):(\d+)([cnsCNS])([mbaMBA])([gpGP])$', cell)
            if not m:
                raise ValueError(f"Invalid cell: '{cell}'. Ensure format is [img_id]:[idx][Style][Draw][Source]")

            img_id_str = m.group(1).strip().upper()
            idx = int(m.group(2))

            # --- ID RESOLUTION LOGIC ---
            if img_id_str == 'R' or img_id_str == '':
                if row_id is None:
                    row_id = get_unique_random_id(idx)
                actual_id_str = row_id

            elif img_id_str == 'C':
                if c not in col_ids_map:
                    col_ids_map[c] = get_unique_random_id(idx)
                actual_id_str = col_ids_map[c]

            elif img_id_str == 'U':
                actual_id_str = get_unique_random_id(idx)

            else:
                # Specific explicit ID provided (e.g. 123)
                actual_id_str = m.group(1).strip()
                used_ids.add(str(actual_id_str))

            try:
                img_id = int(actual_id_str)
            except (ValueError, TypeError):
                img_id = actual_id_str

            if idx >= len(result_files):
                raise IndexError(f"Grid requests index '{idx}' but only {len(result_files)} results provided.")

            row_config.append({
                'img_id': img_id,
                'idx': idx,
                'style': m.group(3).upper(),
                'draw_mode': m.group(4).upper(),
                'source': m.group(5).upper()
            })
        if row_config:
            grid.append(row_config)

    if not grid:
        raise ValueError("VisualisationString produced empty grid.")

    num_rows = len(grid)
    num_cols = max(len(row) for row in grid)

    print(f"\nGrid Layout constructed ({num_rows}x{num_cols}). Plotting...")

    # --- 3. Generate Visualization ---
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 7 * num_rows))

    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes[None, :]
    elif num_cols == 1:
        axes = axes[:, None]

    image_cache = {}

    for r in range(num_rows):
        for c in range(num_cols):
            ax = axes[r, c]

            if c >= len(grid[r]):
                ax.axis('off')
                continue

            cfg = grid[r][c]
            img_id = cfg['img_id']
            idx = cfg['idx']
            style = cfg['style']
            draw_mode = cfg['draw_mode']
            source = cfg['source']

            gt_src = gt_sources[idx]
            dtype_name = dataset_types[idx].upper()

            if img_id not in gt_src.imgs:
                if isinstance(img_id, int) and str(img_id) in gt_src.imgs:
                    img_id = str(img_id)
                elif isinstance(img_id, str) and img_id.isdigit() and int(img_id) in gt_src.imgs:
                    img_id = int(img_id)

            if img_id not in image_cache:
                try:
                    img_info = gt_src.imgs[img_id]
                    img_path = os.path.join(img_info['root_path'], img_info['file_name'])
                    if os.path.exists(img_path):
                        image_cache[img_id] = Image.open(img_path).convert("RGB")
                    else:
                        print(f"⚠️ Warning: Image file not found at '{img_path}'.")
                        image_cache[img_id] = None
                except KeyError:
                    print(f"⚠️ Warning: Could not find image ID '{img_id}' in {dtype_name} Ground Truth.")
                    image_cache[img_id] = None

            img = image_cache[img_id]

            if img is None:
                ax.axis('off')
                ax.set_title(f"ID {img_id} NOT FOUND\nDataset: {dtype_name}", color="red")
                continue

            coco_source = gt_src if source == 'G' else pred_sources[idx]
            title_prefix = f"GT [{dtype_name}]" if source == 'G' else f"Pred [{dtype_name}]"

            ann_ids = coco_source.getAnnIds(imgIds=img_id)
            raw_anns = coco_source.loadAnns(ann_ids)

            anns = []
            for ann in raw_anns:
                # Filter out predictions with a score lower than 0.3
                if source == 'P':
                    score = ann.get('score', None)
                    if score is not None and score < 0.3:
                        continue

                if target_classes:
                    cat_name = gt_src.loadCats(ann['category_id'])[0]['name']
                    if cat_name.lower() not in target_classes:
                        continue

                anns.append(ann)

            seed_str = f"{img_id}_{source}_{idx}"
            seed_val = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2 ** 32)
            ann_colors = generate_distinct_colors(len(anns), seed_val)

            has_masks = any('segmentation' in ann and ann['segmentation'] for ann in anns)

            ax.imshow(img)
            ax.set_title(f"ID: {img_id} | {title_prefix}", fontsize=11)
            ax.axis('off')

            def draw_standard_labels(annotations):
                # Load GTs for the current image to calculate IoU
                gt_ann_ids = gt_src.getAnnIds(imgIds=img_id)
                gt_anns_for_iou = gt_src.loadAnns(gt_ann_ids)

                for i_ann, ann in enumerate(annotations):
                    cat_name = gt_src.loadCats(ann['category_id'])[0]['name']
                    score = ann.get('score', None)

                    if source == 'P' and score is not None and score < 0.3:
                        continue

                    if source == 'P':
                        iou_val = compute_max_iou(ann, gt_anns_for_iou, img_info)
                        if score is not None:
                            label_text = f"{cat_name}\nConfidence: {score:.2f}\nIoU: {iou_val:.2f}"
                        else:
                            label_text = f"{cat_name}\nIoU: {iou_val:.2f}"
                    else:
                        label_text = cat_name

                    if 'bbox' in ann:
                        x, y, w, h = ann['bbox']
                    else:
                        try:
                            x, y, w, h = maskUtils.toBbox(ann['segmentation'])
                        except Exception:
                            x, y = 10, 10

                    box_color = 'red' if source == 'P' else 'black'

                    ax.text(x, max(0, y - 4), label_text, color='white', fontsize=9, fontweight='bold',
                            bbox=dict(facecolor=box_color, alpha=0.5, pad=2, edgecolor='none'))

            # --- RENDER BLOCK ---
            draw_boxes = (draw_mode in ['B', 'A'])
            draw_masks = (draw_mode in ['M', 'A'])

            # 1. Bounding Boxes
            if draw_boxes:
                for i, ann in enumerate(anns):
                    if 'bbox' in ann:
                        x, y, w, h = ann['bbox']
                    else:
                        try:
                            x, y, w, h = maskUtils.toBbox(ann['segmentation'])
                        except Exception:
                            continue

                    color = ann_colors[i]
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)

            # 2. Masks & Labels
            if style == 'S':
                if draw_masks:
                    if not has_masks:
                        print(f"ℹ️  Notice: No masks in {title_prefix} for ID {img_id}.")
                    else:
                        try:
                            h_img, w_img = img.size[1], img.size[0]
                            overlay = np.zeros((h_img, w_img, 4))
                            for i, ann in enumerate(anns):
                                if 'segmentation' not in ann or not ann['segmentation']: continue
                                m_mask = coco_source.annToMask(ann)
                                color = ann_colors[i]

                                overlay[m_mask == 1, :3] = color
                                overlay[m_mask == 1, 3] = 0.55
                                ax.contour(m_mask, levels=[0.5], colors=[color], linewidths=1.5, alpha=1.0)
                            ax.imshow(overlay)
                        except Exception as e:
                            print(f"⚠️ Warning: Could not draw 'S' style mask for {img_id}: {e}")

                draw_standard_labels(anns)

            elif style == 'N':
                # No masks drawn, and explicitly skipping labels as requested.
                pass

            elif style == 'C' and anns:
                if draw_masks:
                    if not has_masks:
                        print(f"ℹ️  Notice: No masks in {title_prefix} for ID {img_id}.")
                        draw_standard_labels(anns)
                    else:
                        categories = list(set([ann['category_id'] for ann in anns]))
                        try:
                            h_img, w_img = img.size[1], img.size[0]
                            overlay = np.zeros((h_img, w_img, 4))

                            for cat_id in categories:
                                cat_anns_with_idx = [(i, ann) for i, ann in enumerate(anns) if
                                                     ann['category_id'] == cat_id and 'segmentation' in ann and ann[
                                                         'segmentation']]
                                if not cat_anns_with_idx: continue

                                masks = []
                                color_map = []
                                score_map = []
                                mask_to_orig = []

                                for orig_idx, ann in cat_anns_with_idx:
                                    m_mask = coco_source.annToMask(ann) == 1
                                    labeled_mask, num_features = ndimage.label(m_mask, structure=np.ones((3, 3)))

                                    orig_rgb = ann_colors[orig_idx]
                                    orig_hsv = colorsys.rgb_to_hsv(*orig_rgb)

                                    for i in range(1, num_features + 1):
                                        blob_mask = (labeled_mask == i)
                                        masks.append(blob_mask)
                                        score_map.append(ann.get('score', None))
                                        mask_to_orig.append(orig_idx)

                                        if i == 1:
                                            color_map.append(orig_rgb)
                                        else:
                                            new_h = (orig_hsv[0] + (i - 1) / num_features) % 1.0
                                            new_rgb = colorsys.hsv_to_rgb(new_h, orig_hsv[1], orig_hsv[2])
                                            color_map.append(list(new_rgb))

                                groups = [[i] for i in range(len(masks))]
                                changed = True
                                while changed:
                                    changed = False
                                    for i in range(len(groups)):
                                        for j in range(i + 1, len(groups)):
                                            touch = False
                                            for idx_i in groups[i]:
                                                for idx_j in groups[j]:
                                                    if mask_to_orig[idx_i] == mask_to_orig[idx_j]:
                                                        continue

                                                    m1 = masks[idx_i]
                                                    m2 = masks[idx_j]

                                                    m1_d = m1.copy()
                                                    m1_d[:-1, :] |= m1[1:, :]
                                                    m1_d[1:, :] |= m1[:-1, :]
                                                    m1_d[:, :-1] |= m1[:, 1:]
                                                    m1_d[:, 1:] |= m1[:, :-1]
                                                    m1_d[:-1, :-1] |= m1[1:, 1:]
                                                    m1_d[1:, 1:] |= m1[:-1, :-1]
                                                    m1_d[:-1, 1:] |= m1[1:, :-1]
                                                    m1_d[1:, :-1] |= m1[:-1, 1:]

                                                    if np.any(m1_d & m2):
                                                        touch = True
                                                        break
                                                if touch: break

                                            if touch:
                                                groups[i].extend(groups[j])
                                                groups.pop(j)
                                                changed = True
                                                break
                                        if changed: break

                                cat_name = gt_src.loadCats(cat_id)[0]['name']

                                for grp in groups:
                                    color = color_map[grp[0]]
                                    combined_mask = masks[grp[0]]
                                    for idx in grp[1:]:
                                        combined_mask = combined_mask | masks[idx]

                                    overlay[combined_mask, :3] = color
                                    overlay[combined_mask, 3] = 0.55
                                    if np.any(combined_mask):
                                        ax.contour(combined_mask.astype(float), levels=[0.5], colors=[color],
                                                   linewidths=1.5, alpha=1.0)

                                    y_indices, x_indices = np.where(combined_mask)
                                    if len(y_indices) > 0:
                                        x, y = np.min(x_indices), np.min(y_indices)
                                    else:
                                        x, y = 10, 10

                                    scores = [score_map[idx] for idx in grp]
                                    valid_scores = [s for s in scores if s is not None]

                                    if source == 'P':
                                        # Calculate the max IoU for any segment in this combined group
                                        max_iou = 0.0
                                        gt_ann_ids = gt_src.getAnnIds(imgIds=img_id)
                                        gt_anns_for_iou = gt_src.loadAnns(gt_ann_ids)

                                        for idx in grp:
                                            iou = compute_max_iou(anns[idx], gt_anns_for_iou, img_info)
                                            if iou > max_iou:
                                                max_iou = iou

                                        if valid_scores:
                                            label_text = f"{cat_name}\nConfidence: {max(valid_scores):.2f}\nIoU: {max_iou:.2f}"
                                        else:
                                            label_text = f"{cat_name}\nIoU: {max_iou:.2f}"
                                    else:
                                        label_text = cat_name

                                    box_color = 'red' if source == 'P' else 'black'
                                    ax.text(x, max(0, y - 4), label_text, color='white', fontsize=9, fontweight='bold',
                                            bbox=dict(facecolor=box_color, alpha=0.5, pad=2, edgecolor='none'))

                            ax.imshow(overlay)
                            bbox_only_anns = [a for a in anns if 'segmentation' not in a or not a['segmentation']]
                            if bbox_only_anns:
                                draw_standard_labels(bbox_only_anns)

                        except Exception as e:
                            print(f"⚠️ Warning: Could not draw 'C' style mask for {img_id}: {e}")
                else:
                    # If C style but they didn't want masks drawn, standard labels apply
                    draw_standard_labels(anns)

    if args.Title:
        fig.suptitle(args.Title.strip(), fontsize=18, fontweight='bold')

    plt.tight_layout()
    if args.Title:
        fig.subplots_adjust(top=0.92)

    unique_ids_plotted = list(set([str(c['img_id']) for r in grid for c in r]))
    ids_str = "_".join(unique_ids_plotted)[:50]
    if '.png' not in args.Output:
        out_path = os.path.join(args.Output, f"comparison_plot_{ids_str}.png")
    else:
        out_path = args.Output

    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    print(f"\n✅ Visualization saved to: {out_path}")


if __name__ == "__main__":
    main()