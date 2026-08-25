import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


def main():
    parser = argparse.ArgumentParser(description="Compare class frequencies between COCO and VOC datasets.")
    parser.add_argument("--coco_gt", default="./datasets/coco_dataset/coco2017/annotations/instances_val2017.json",
                        help="Path to COCO ground truth JSON.")
    parser.add_argument("--voc_gt", default="./data/VOC2012/annotations/voc_2012_val.json",
                        help="Path to VOC ground truth JSON.")
    parser.add_argument("-o", "--output", default="dataset_comparison.png",
                        help="Path to save the output chart.")
    args = parser.parse_args()

    print("Loading COCO dataset...")
    coco = COCO(args.coco_gt)

    print("\nLoading VOC dataset...")
    voc = COCO(args.voc_gt)

    # 1. Map category names to IDs (lowercased to ensure matching)
    coco_name_to_id = {cat['name'].lower(): cat['id'] for cat in coco.loadCats(coco.getCatIds())}
    voc_name_to_id = {cat['name'].lower(): cat['id'] for cat in voc.loadCats(voc.getCatIds())}

    # Find the intersection of classes
    common_classes = sorted(list(set(coco_name_to_id.keys()).intersection(set(voc_name_to_id.keys()))))
    print(f"\nFound {len(common_classes)} common classes between the datasets.")

    # 2. Gather Metrics & Calculate Filtered Totals
    coco_class_stats = {}
    voc_class_stats = {}

    coco_all_valid_imgs = set()
    voc_all_valid_imgs = set()

    coco_total_inst = 0
    voc_total_inst = 0

    for cls_name in common_classes:
        # COCO stats extraction
        coco_cat_id = coco_name_to_id[cls_name]
        coco_ann_ids = coco.getAnnIds(catIds=[coco_cat_id])
        coco_anns = coco.loadAnns(coco_ann_ids)

        c_inst_count = len(coco_anns)
        c_imgs = set(ann['image_id'] for ann in coco_anns)

        coco_class_stats[cls_name] = {'inst': c_inst_count, 'img': len(c_imgs)}
        coco_total_inst += c_inst_count
        coco_all_valid_imgs.update(c_imgs)

        # VOC stats extraction
        voc_cat_id = voc_name_to_id[cls_name]
        voc_ann_ids = voc.getAnnIds(catIds=[voc_cat_id])
        voc_anns = voc.loadAnns(voc_ann_ids)

        v_inst_count = len(voc_anns)
        v_imgs = set(ann['image_id'] for ann in voc_anns)

        voc_class_stats[cls_name] = {'inst': v_inst_count, 'img': len(v_imgs)}
        voc_total_inst += v_inst_count
        voc_all_valid_imgs.update(v_imgs)

    # Derive total unique images that contain AT LEAST one common class
    coco_total_img = len(coco_all_valid_imgs)
    voc_total_img = len(voc_all_valid_imgs)

    print(f"COCO Filtered Instances: {coco_total_inst} | Filtered Images: {coco_total_img}")
    print(f"VOC Filtered Instances: {voc_total_inst} | Filtered Images: {voc_total_img}\n")

    # 3. Calculate Frequencies
    coco_inst_freqs = []
    voc_inst_freqs = []
    coco_img_freqs = []
    voc_img_freqs = []

    coco_total_inst = 0
    coco_total_img = 0
    voc_total_inst = 0
    voc_total_img = 0
    for cls_name in common_classes:
        c_inst = coco_class_stats[cls_name]['inst']
        c_img = coco_class_stats[cls_name]['img']
        coco_total_inst += c_inst
        coco_total_img += c_img

        v_inst = voc_class_stats[cls_name]['inst']
        v_img = voc_class_stats[cls_name]['img']
        voc_total_inst += v_inst
        voc_total_img += v_img

    for cls_name in common_classes:
        c_inst = coco_class_stats[cls_name]['inst']
        c_img = coco_class_stats[cls_name]['img']
        coco_inst_freqs.append((c_inst / coco_total_inst) if coco_total_inst else 0)
        coco_img_freqs.append((c_img / coco_total_img) if coco_total_img else 0)

        v_inst = voc_class_stats[cls_name]['inst']
        v_img = voc_class_stats[cls_name]['img']
        voc_inst_freqs.append((v_inst / voc_total_inst) if voc_total_inst else 0)
        voc_img_freqs.append((v_img / voc_total_img) if voc_total_img else 0)

    # 4. Plotting
    x = np.arange(len(common_classes))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(18, 8))

    # Draw 4 bars per class
    rects1 = ax.bar(x - width * 1.5, coco_inst_freqs, width, label='COCO: Class Instances / Total Instances',
                    color='#1f77b4')
    rects2 = ax.bar(x - width * 0.5, voc_inst_freqs, width, label='VOC: Class Instances / Total Instances',
                    color='#ff7f0e')
    rects3 = ax.bar(x + width * 0.5, coco_img_freqs, width, label='COCO: Images with Class / Total Images',
                    color='#2ca02c')
    rects4 = ax.bar(x + width * 1.5, voc_img_freqs, width, label='VOC: Images with Class / Total Images',
                    color='#d62728')

    # Add text for labels, title and custom x-axis tick labels
    ax.set_ylabel('Frequency (%)', fontweight='bold', fontsize=12)
    ax.set_title('Class Frequency Comparison: COCO vs VOC (Filtered to Shared Classes)', fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in common_classes], rotation=45, ha="right", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Automatically adjust layout and save
    fig.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✅ Success! Chart saved to: {args.output}")


if __name__ == "__main__":
    main()