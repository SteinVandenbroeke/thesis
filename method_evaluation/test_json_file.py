import json
from collections import Counter

# Load your 34MB file
with open("results_coco/BAS_COCO.json", "r") as f:
    results = json.load(f)

print(f"Total predictions: {len(results)}")

# Check 1: Are all predictions piling up on one image?
image_ids = [res['image_id'] for res in results]
counts = Counter(image_ids)
print(f"Number of unique images predicted on: {len(counts)}")
print(f"Most predictions on a single image: {counts.most_common(1)[0]}")

# Check 2: What format is the mask?
print(f"Sample mask format: {type(results[0]['segmentation'])}")
if isinstance(results[0]['segmentation'], dict):
    print("Format is RLE (Good)")
elif isinstance(results[0]['segmentation'], list):
    print("Format is Polygon (Can be slow)")