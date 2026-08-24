import json
import numpy as np
import pycocotools.mask as maskUtils
from collections import defaultdict


def convert_results_list_to_wsss(input_json, output_json):
    print("Loading JSON list...")
    with open(input_json, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a list of annotations. If this is a dict, use the previous script.")

    # Group annotations by image_id
    img_to_anns = defaultdict(list)
    for ann in data:
        img_to_anns[ann['image_id']].append(ann)

    new_annotations = []
    print(f"Processing {len(img_to_anns)} unique images...")

    for img_id, anns in img_to_anns.items():
        # 1. Determine image dimensions from the RLE size field
        h, w = None, None
        for ann in anns:
            seg = ann.get('segmentation')
            if isinstance(seg, dict) and 'size' in seg:
                h, w = seg['size']
                break

        if h is None or w is None:
            raise ValueError(f"Could not determine image size for image_id {img_id}. "
                             "Annotations must be in RLE format with a 'size' field.")

        # 2. Initialize a blank semantic mask
        semantic_mask = np.zeros((h, w), dtype=np.uint8)

        # Sort annotations by score (if available) so higher confidence instances
        # are drawn last and stay "on top" if they overlap.
        anns_sorted = sorted(anns, key=lambda x: x.get('score', 0.0))

        # 3. Merge all instances onto the semantic mask
        for ann in anns_sorted:
            cat_id = ann['category_id']
            seg = ann['segmentation']

            if isinstance(seg, dict):
                # Handle RLE format
                instance_mask = maskUtils.decode(seg)
            elif isinstance(seg, list):
                # Handle Polygon format
                rles = maskUtils.frPyObjects(seg, h, w)
                rle = maskUtils.merge(rles)
                instance_mask = maskUtils.decode(rle)
            else:
                continue

            semantic_mask[instance_mask > 0] = cat_id

        # 4. Extract merged masks and create new annotations (one per class)
        unique_classes = np.unique(semantic_mask)

        for cat_id in unique_classes:
            if cat_id == 0:  # Skip background
                continue

            class_mask = (semantic_mask == cat_id).astype(np.uint8)

            # Encode back to RLE format
            rle = maskUtils.encode(np.asfortranarray(class_mask))
            rle['counts'] = rle['counts'].decode('utf-8')  # Decode for JSON serialization

            # Create the unified annotation
            new_ann = {
                "image_id": img_id,
                "category_id": int(cat_id),
                "segmentation": rle,
            }

            # WSSS evaluation might require a score. We take the max score among the merged instances.
            scores = [a.get('score', 1.0) for a in anns if a['category_id'] == cat_id]
            if scores:
                new_ann['score'] = float(np.max(scores))

            new_annotations.append(new_ann)

    # 5. Save the new output list
    print(f"Saving semantic JSON to: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(new_annotations, f)

    print("Done!")

if __name__ == "__main__":
    # --- CONFIGURE YOUR PATHS HERE ---
    INPUT_JSON = "result_voc/BAS.json"
    OUTPUT_JSON = "result_voc/BAS_WSSS.json"

    convert_results_list_to_wsss(INPUT_JSON, OUTPUT_JSON)