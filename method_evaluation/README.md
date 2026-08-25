# Method Evaluation

This section contains the evaluation code for the models, as well as the scripts used to generate most of the plots featured in the thesis book.

## Setup and Prerequisites

Before running the evaluation or visualization scripts, please complete the following setup steps:

*   **Create Python Environment:** Run `sh create_env.sh` to set up the necessary Python environment.
*   **Download Datasets:** Place your downloaded datasets directly into this folder.
*   **Format Datasets:** All datasets must be converted to the default COCO format. You can do this using the conversion scripts provided in the `dataset_converters` folder within this repository.
*   **Configure Paths:** Open the provided `.sh` files and update the paths to link to your downloaded datasets.
*   **Add Result Files:** Download the necessary pre-computed result files from [LINK] and add them to this folder.

## Running Evaluations

You can run the model evaluations by executing the following shell scripts:

*   `sh visual_results_coco.sh`
*   `sh visual_results_cub.sh`
*   `sh visual_results_voc.sh`
*   `sh visual_results_cub_WSOL_metric.sh`

## Creating Plots

To generate the visualizations and plots used in the thesis, run the following script:

*   `sh visualizations/create_visualisations.sh`
