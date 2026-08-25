import argparse
import numpy as np
from pycocotools.coco import COCO
import scipy.ndimage as ndimage


def main():
    parser = argparse.ArgumentParser(description="Find images with split/occluded instance masks.")
    parser.add_argument("--gt_file", default="./datasets/coco_dataset/coco2017/annotations/instances_val2017.json",
                        help="Path to the COCO ground truth JSON.")
    parser.add_argument("--category", default="person", help="Category to search for (e.g., 'person', 'dog').")
    parser.add_argument("--min_blob_size", type=int, default=200,
                        help="Minimum pixel area for a disconnected blob to count (ignores 1-pixel artifacts).")
    parser.add_argument("--max_results", type=int, default=10, help="Stop after finding this many examples.")

    args = parser.parse_args()

    print(f"Loading Ground Truth from {args.gt_file}...")
    coco = COCO(args.gt_file)

    # 1. Get Category ID
    catIds = coco.getCatIds(catNms=[args.category])
    if not catIds:
        print(f"❌ Error: Category '{args.category}' not found in dataset.")
        return
    cat_id = catIds[0]

    # 2. Get all annotations for this category
    annIds = coco.getAnnIds(catIds=[cat_id])
    anns = coco.loadAnns(annIds)

    print(f"Searching through {len(anns)} '{args.category}' annotations for split masks...\n")

    results_found = 0
    candidate_images = []

    for ann in anns:
        # Ignore very small objects (crowds or tiny background people)
        if ann['area'] < 5000:
            continue

        # Ignore 'iscrowd' annotations (these are unsegmented bounding boxes in COCO)
        if ann.get('iscrowd', 0) == 1:
            continue

        # Convert the annotation to a 2D binary mask
        try:
            mask = coco.annToMask(ann)
        except Exception:
            continue

        # 3. Connected-Component Analysis
        # This groups touching pixels into labeled blobs
        labeled_mask, num_features = ndimage.label(mask)

        # If the mask is split into multiple pieces
        if num_features > 1:
            # Calculate the area (number of pixels) of each separated blob
            blob_sizes = ndimage.sum(mask, labeled_mask, range(1, num_features + 1))

            # Filter out tiny pixel artifacts (often caused by rough polygon drawings)
            large_blobs = [size for size in blob_sizes if size > args.min_blob_size]

            if len(large_blobs) > 1:
                img_id = ann['image_id']
                if img_id not in candidate_images:
                    candidate_images.append(img_id)
                    results_found += 1

                    print(f"✅ Found Match!")
                    print(f"   Image ID: {img_id}")
                    print(f"   Annotation ID: {ann['id']}")
                    print(f"   Blobs found: {len(large_blobs)} (Sizes: {[int(s) for s in large_blobs]})")
                    print("-" * 40)

                    if results_found >= args.max_results:
                        break

    print(f"\n🎉 Search complete. Found {len(candidate_images)} unique images with a split '{args.category}'.")
    if candidate_images:
        print(f"Try running your visualizer with one of these IDs:")
        print(" ".join([str(i) for i in candidate_images]))


if __name__ == "__main__":
    main()