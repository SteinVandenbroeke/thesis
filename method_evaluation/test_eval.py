import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pycocotools import mask as maskUtils
import os
import argparse
from tqdm import tqdm
import multiprocessing
import os.path as osp
import sys
import gc
import cv2
from chainercv.evaluations import eval_instance_segmentation_voc
from pycocotools import mask as COCOMask
from pycocotools.coco import COCO
import json
from pycocotools.cocoeval import COCOeval
# Load your predictions
with open("results_coco/BAS_COCO.json", "r") as f:
    preds = json.load(f)

# Extract only the predictions for image 785
bad_preds = [p for p in preds if p["image_id"] == 785]

print(f"Total predictions for 785: {len(bad_preds)}")
if len(bad_preds) > 0:
    print(f"RLE Size array: {bad_preds[0]['segmentation']['size']}")

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


import json
from tqdm import tqdm

# --- Configuration ---
gt_file = "./datasets/coco_dataset/coco2017/annotations/instances_val2017.json"
pred_file = "results_coco/BAS_COCO.json"

pred_file = "results_coco/BAS_COCO.json"


def test_c_engine():
    print(f"Loading {pred_file}...")
    with open(pred_file, "r") as f:
        preds = json.load(f)

    print("Testing C-backend decoding on all 115,720 predictions...")

    for i, p in enumerate(tqdm(preds)):
        img_id = p['image_id']

        rle = p['segmentation']

        # Format the RLE exactly how COCOeval does right before the C-call
        test_rle = {
            'size': rle['size'],
            'counts': rle['counts'].encode('utf-8') if isinstance(rle['counts'], str) else rle['counts']
        }

        # The dangerous C-call: if the RLE is corrupted, it freezes here.
        try:
            maskUtils.decode(test_rle)
        except Exception as e:
            print(f"\nCaught Python exception at index {i}: {e}: img_id:{img_id}")
            return

    print("\n✅ Success! The C-backend can decode all masks without freezing.")

def check_mask_dimensions():
    print("Loading Ground Truth...")
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    # Create a fast lookup dictionary for GT dimensions: {image_id: [height, width]}
    gt_sizes = {img['id']: [img['height'], img['width']] for img in gt_data['images']}

    print("Loading Predictions...")
    with open(pred_file, 'r') as f:
        preds = json.load(f)

    print("Checking prediction mask sizes against ground truth...")
    mismatched_images = set()
    mismatch_details = []

    for pred in tqdm(preds):
        img_id = pred['image_id']

        # Ensure the prediction has a size array
        if 'segmentation' in pred and isinstance(pred['segmentation'], dict) and 'size' in pred['segmentation']:
            pred_size = pred['segmentation']['size']  # RLE size is [height, width]

            if img_id in gt_sizes:
                gt_size = gt_sizes[img_id]

                # Check if dimensions match
                if pred_size[0] != gt_size[0] or pred_size[1] != gt_size[1]:
                    if img_id not in mismatched_images:
                        mismatched_images.add(img_id)
                        mismatch_details.append({
                            'image_id': img_id,
                            'gt_size_hw': gt_size,
                            'pred_size_hw': pred_size
                        })

    # --- Print Results ---
    print("\n" + "=" * 50)
    if not mismatched_images:
        print("✅ Success! All prediction mask sizes match the ground truth perfectly.")
    else:
        print(f"❌ Found dimension mismatches in {len(mismatched_images)} unique images!")
        print("Showing the first 20 mismatches:")
        for detail in mismatch_details[:20]:
            print(
                f"Image ID: {detail['image_id']:<8} | GT (H,W): {detail['gt_size_hw']} | Pred (H,W): {detail['pred_size_hw']}")
    print("=" * 50)


gt_file = "./datasets/coco_dataset/coco2017/annotations/instances_val2017.json"
pred_file = "results_coco/BAS_COCO.json"


def test_iou_worker(dts, gts, iscrowd, return_dict):
    """
    Worker function that runs the dangerous C-call.
    If it freezes, it will be forcefully terminated by the main process.
    """
    try:
        # The exact C-level math function that COCOeval uses internally
        maskUtils.iou(dts, gts, iscrowd)
        return_dict['status'] = 'success'
    except Exception as e:
        return_dict['status'] = f'error: {e}'


def find_poison_pills_lock():
    print("Loading Ground Truth...")
    dataset = COCO("./datasets/coco_dataset/coco2017/annotations/instances_val2017.json")

    print("Loading Predictions...")
    # FIX 1: Wrap your predictions in dataset.loadRes()
    preds = dataset.loadRes("results_coco/BAS_COCO.json")

    # FIX 2: Convert Ground Truth RLE strings to bytes
    for ann in dataset.dataset.get('annotations', []):
        segm = ann.get('segmentation', None)
        if isinstance(segm, dict) and 'counts' in segm and isinstance(segm['counts'], str):
            segm['counts'] = segm['counts'].encode('utf-8')

    # FIX 2: Convert Prediction RLE strings to bytes
    for ann in preds.dataset.get('annotations', []):
        segm = ann.get('segmentation', None)
        if isinstance(segm, dict) and 'counts' in segm and isinstance(segm['counts'], str):
            segm['counts'] = segm['counts'].encode('utf-8')

    imgIdsSorted = sorted(dataset.getImgIds())

    for img_id in imgIdsSorted:
        # We flush the print statement so it immediately hits the console BEFORE evaluating
        print(f"Testing Image ID {img_id}...", end="", flush=True)

        coco_eval = COCOeval(dataset, preds, 'segm')
        coco_eval.params.imgIds = [img_id]

        # 🚨 IF IT HITS A POISON PILL, IT WILL FREEZE RIGHT HERE 🚨
        coco_eval.evaluate()

        print(" -> Success!")


dataset = None
preds = None


def load_data():
    global dataset, preds
    print("Loading Ground Truth...")
    dataset = COCO("./datasets/coco_dataset/coco2017/annotations/instances_val2017.json")

    print("Loading Predictions...")
    preds = dataset.loadRes("results_coco/BAS_COCO.json")

    # --- Convert RLE strings back to bytes for the Ground Truth ---
    for ann in dataset.dataset.get('annotations', []):
        segm = ann.get('segmentation', None)
        if isinstance(segm, dict) and 'counts' in segm and isinstance(segm['counts'], str):
            segm['counts'] = segm['counts'].encode('utf-8')

    # --- Convert RLE strings back to bytes for the Predictions ---
    for ann in preds.dataset.get('annotations', []):
        segm = ann.get('segmentation', None)
        if isinstance(segm, dict) and 'counts' in segm and isinstance(segm['counts'], str):
            segm['counts'] = segm['counts'].encode('utf-8')
    print("Data loaded and formatted!\n")


def test_evaluate_worker(img_id, return_dict):
    """
    Worker function that runs the exact evaluation your final script will use.
    """
    try:
        # Suppress stdout so COCOeval doesn't print 5,000 times and ruin the progress bar
        sys.stdout = open(os.devnull, 'w')

        coco_eval = COCOeval(dataset, preds, 'segm')
        coco_eval.params.imgIds = [img_id]

        # This is the line that will freeze if the image is corrupted
        coco_eval.evaluate()

        return_dict['status'] = 'success'
    except Exception as e:
        return_dict['status'] = f'error: {e}'

def find_poison_pills():
    load_data()

    img_ids = sorted(dataset.getImgIds())
    poison_pills = []

    print(f"Testing {len(img_ids)} images against COCOeval...")
    manager = multiprocessing.Manager()

    for i, img_id in tqdm(enumerate(img_ids[0:100]), desc="Scanning for freezes"):
        return_dict = manager.dict()
        return_dict['status'] = 'running'

        # Spawn a disposable worker process
        p = multiprocessing.Process(target=test_evaluate_worker, args=(img_id, return_dict))
        p.start()

        # Give COCOeval a maximum of 10 seconds to finish evaluating this single image.
        p.join(timeout=1.0)

        if p.is_alive():
            # 🚨 THE PROCESS FROZE! Kill it immediately.
            p.terminate()
            p.join()
            poison_pills.append(img_id)
            tqdm.write(f"\n⚠️ {i} KILLED STUCK PROCESS: Found Poison Pill at Image ID {img_id}")
        elif return_dict.get('status') != 'success':
            tqdm.write(f"\n❌ {i}, Error on Image ID {img_id}: {return_dict.get('status')}")

    # --- Final Output ---
    print("\n" + "=" * 60)
    print(f"Scan complete. Found {len(poison_pills)} corrupted images out of {len(img_ids)}.")
    if poison_pills:
        print(f"Copy and paste this list into your evaluator's quarantine block:")
        print(f"poison_pills = {poison_pills}")
    else:
        print("Everything looks clean!")
    print("=" * 60)


def check_for_ghosts():
    print(f"Loading predictions from {pred_file}...")
    with open(pred_file, "r") as f:
        preds = json.load(f)

    print(f"Loaded {len(preds)} total predictions.")
    print("Scanning for ghost masks (Area = 0)...")

    ghost_count = 0
    ghost_image_ids = set()

    for p in tqdm(preds, desc="Calculating Pixels"):
        # Format the RLE dict
        rle = {
            'size': p['segmentation']['size'],
            'counts': p['segmentation']['counts']
        }

        # Convert string counts to bytes for the pycocotools C-engine
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('utf-8')

        # maskUtils.area calculates the exact number of active pixels in the mask
        try:
            pixel_area = maskUtils.area(rle)
            if pixel_area == 0:
                ghost_count += 1
                ghost_image_ids.add(p['image_id'])
        except Exception as e:
            print(f"\n⚠️ Encountered a completely malformed mask on Image ID {p['image_id']}: {e}")

    # --- Print Results ---
    print("\n" + "=" * 50)
    print("SCAN COMPLETE")
    print("=" * 50)
    print(f"Total Predictions Checked: {len(preds)}")
    print(f"Total Ghost Masks Found:   {ghost_count}")
    print(f"Unique Images Affected:    {len(ghost_image_ids)}")

    if ghost_count > 0:
        print("\nHere are the first 20 Image IDs infected with ghost masks:")
        print(list(ghost_image_ids)[:20])
    else:
        print("\n✅ No ghost masks found! The freezing issue must be caused by something else.")


import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from PIL import Image


def visualize_predictions_from_json(test_folder="test_predictions",
                                    gt_file="./datasets/coco_dataset/coco2017/annotations/instances_val2017.json",
                                    pred_file="results_coco/BAS_COCO.json",
                                    output_folder="visualizations"):
    """
    Plots the original image, ground truth masks, and predicted masks side-by-side
    ONLY for the images that exist in the prediction JSON, complete with class names and scores.
    """
    print("Loading Ground Truth...")
    coco_gt = COCO(gt_file)

    print("Loading Predictions...")
    coco_pred = coco_gt.loadRes(pred_file)

    # Convert RLE strings to bytes for GT to prevent C-engine freeze
    for ann in coco_gt.dataset.get('annotations', []):
        segm = ann.get('segmentation', None)
        if isinstance(segm, dict) and 'counts' in segm and isinstance(segm['counts'], str):
            segm['counts'] = segm['counts'].encode('utf-8')

    # Convert RLE strings to bytes for Predictions to prevent C-engine freeze
    for ann in coco_pred.dataset.get('annotations', []):
        segm = ann.get('segmentation', None)
        if isinstance(segm, dict) and 'counts' in segm and isinstance(segm['counts'], str):
            segm['counts'] = segm['counts'].encode('utf-8')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract unique image IDs directly from the prediction annotations
    pred_annotations = coco_pred.dataset.get('annotations', [])
    pred_img_ids = list(set([ann['image_id'] for ann in pred_annotations]))

    print(f"\nFound {len(pred_img_ids)} unique images in the prediction JSON. Starting visualization...")

    for img_id in tqdm(pred_img_ids, desc="Plotting Images"):
        # Handle standard COCO 12-digit padded names or custom unpadded names
        filename_padded = f"{str(img_id).zfill(12)}.jpg"
        filename_unpadded = f"{img_id}.jpg"

        img_path = os.path.join(test_folder, filename_padded)
        if not os.path.exists(img_path):
            img_path = os.path.join(test_folder, filename_unpadded)

        if not os.path.exists(img_path):
            tqdm.write(f"⚠️ Warning: Image for ID {img_id} not found in '{test_folder}'. Skipping.")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            tqdm.write(f"❌ Error: Could not load image {img_path}: {e}")
            continue

        # Fetch annotations for this specific image
        gt_ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(gt_ann_ids)

        pred_ann_ids = coco_pred.getAnnIds(imgIds=img_id)
        pred_anns = coco_pred.loadAnns(pred_ann_ids)

        # Set up a 1x3 matplotlib figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        # --- 1. Original Image ---
        axes[0].imshow(img)
        axes[0].set_title(f"Original Image (ID: {img_id})")
        axes[0].axis('off')

        # --- 2. Ground Truth Mask ---
        axes[1].imshow(img)
        axes[1].set_title(f"Ground Truth ({len(gt_anns)} instances)")
        axes[1].axis('off')
        plt.sca(axes[1])
        coco_gt.showAnns(gt_anns, draw_bbox=False)

        # Add labels to Ground Truth
        for ann in gt_anns:
            cat_name = coco_gt.loadCats(ann['category_id'])[0]['name']

            # Find coordinates to place text (use bbox if available, fallback to mask)
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
            else:
                x, y = 10, 10  # Safe fallback

            axes[1].text(x, max(0, y - 4), cat_name, color='white', fontsize=9, fontweight='bold',
                         bbox=dict(facecolor='black', alpha=0.5, pad=2, edgecolor='none'))

        # --- 3. Prediction Mask ---
        axes[2].imshow(img)
        axes[2].set_title(f"Predictions ({len(pred_anns)} instances)")
        axes[2].axis('off')
        plt.sca(axes[2])
        coco_pred.showAnns(pred_anns, draw_bbox=False)

        # Add labels and scores to Predictions
        for ann in pred_anns:
            cat_name = coco_gt.loadCats(ann['category_id'])[0]['name']
            score = ann.get('score', 0.0)
            label_text = f"{cat_name} {score:.2f}"

            # Prediction JSONs sometimes omit the bbox.
            # If missing, calculate it directly from the RLE mask to find text anchor point.
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
            else:
                try:
                    x, y, w, h = maskUtils.toBbox(ann['segmentation'])
                except Exception:
                    x, y = 10, 10

            axes[2].text(x, max(0, y - 4), label_text, color='white', fontsize=9, fontweight='bold',
                         bbox=dict(facecolor='red', alpha=0.6, pad=2, edgecolor='none'))

        # Save and cleanup
        out_path = os.path.join(output_folder, f"{img_id}_comparison.png")
        plt.tight_layout()
        plt.savefig(out_path)

        # Prevent RAM leak
        plt.close(fig)

    print(f"\n✅ All visualizations saved to the '{output_folder}' directory.")


def filter_gt_by_predictions(gt_path, pred_path, output_path):
    """
    Filters the ground truth JSON to only include images and annotations
    that exist in the prediction JSON.
    """
    print(f"Loading predictions from {pred_path}...")
    with open(pred_path, 'r') as f:
        preds = json.load(f)

    # Handle both standard COCO lists and custom dict wrappers
    pred_list = preds.get('annotations', preds) if isinstance(preds, dict) else preds

    # Extract unique image IDs
    valid_image_ids = set([p['image_id'] for p in pred_list])
    print(f"Found {len(valid_image_ids)} unique images in predictions.")

    print(f"Loading ground truth from {gt_path}...")
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    # Filter images and annotations
    original_img_count = len(gt_data['images'])
    gt_data['images'] = [img for img in gt_data['images'] if img['id'] in valid_image_ids]
    gt_data['annotations'] = [ann for ann in gt_data['annotations'] if ann['image_id'] in valid_image_ids]

    print(f"Filtered Ground Truth: Images reduced from {original_img_count} to {len(gt_data['images'])}.")

    # Save the new filtered GT
    print(f"Saving filtered ground truth to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(gt_data, f)

    return output_path

if __name__ == "__main__":
    #check_mask_dimensions()
    #test_c_engine()
    #find_poison_pills()
    #check_for_ghosts()
    #
    from lib.datasets.json_inference import coco_inst_seg_eval

    result_file = "results_coco/coco_annotations_BAS_seperated_instances.json"
    label_file = "./datasets/coco_dataset/coco2017/annotations/instances_train2017.json"
    root = './datasets/coco_dataset/coco2017/train2017'

    filtered_pred = filter_gt_by_predictions(label_file, result_file, "output_gt_predictions.json")

    visualize_predictions_from_json(
        test_folder=root,
        gt_file=label_file,
        pred_file=result_file,
    )

    mAP, cls_ap, cls_names = coco_inst_seg_eval(filtered_pred, result_file)
    print(mAP, cls_ap, cls_names)
    for cls_name in cls_names:
        print(cls_name, ":", cls_ap['0.25'][cls_names.index(cls_name)])