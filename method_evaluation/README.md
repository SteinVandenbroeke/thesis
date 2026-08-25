# A Study of Weakly Supervised Instance Segmentation Methods

This repository contains the code and resources for the master's thesis "A Study of Weakly Supervised Instance Segmentation Methods" by Stein Vandenbroeke. The research investigates the current state of Weakly Supervised Instance Segmentation (WSIS) by comparatively evaluating different methods and introducing a targeted occlusion-based evaluation metric.

## Repository Structure

* **method_evaluation/**: The evaluation code is in the `method_evaluation` folder. This includes the custom scripts used to test model performance based on instance overlap and standard Mean Average Precision (mAP).
* **exisiting_WSIS/**: The modified method codes can be found in the GitHub repositories linked in `exisiting_WSIS`. Custom dataloaders and code modifications were required for several of these methods to handle updated datasets or resolve deprecated packages.

## Evaluated Methods

This repository evaluates three primary methods:
* **Background Activation Suppression (BAS)**: Originally created as a Weakly Supervised Object Localisation (WSOL) method, this was extended and adapted to serve as a baseline WSIS method by splitting disconnected class masks.
* **Beyond Semantic to Instance Segmentation (BESTIE)**: A framework designed for WSIS that relies strictly on image-level labels and uses Semantic Knowledge Transfer and self-supervised refinement to generate instance labels.
* **Complete Instances Mining (CIM)**: A WSIS framework that utilizes MaskIoU heads and an anti-noise strategy to tackle redundant segmentation problems.

## Datasets Used

The methods were trained and evaluated on the following datasets:
* **Pascal VOC 2012**: A multi-class, multi-instance dataset used to evaluate multi-instance segmentation performance.
* **MS COCO**: A larger dataset with 91 object categories and a high average instance count per image, providing a significantly more difficult segmentation challenge.
* **CUB-200-2011**: A single-instance dataset containing 200 bird species. This was used to test the impact of WSIS methods when applied to a straightforward WSOL task.

## Documentation

For a comprehensive explanation of the methodologies, architectural modifications, dataset challenges, and detailed experiment results, please refer to the thesis_bundle.pdf document.
