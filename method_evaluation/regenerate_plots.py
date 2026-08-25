import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.gridspec import GridSpec

# ==============================================================================
# GLOBAL PLOT SETTINGS
# ==============================================================================
TITLE_FONT_SIZE = 23
TITLE_FONT_WEIGHT = 'bold'
AXIS_LABEL_FONT_SIZE = 19
TICK_LABEL_FONT_SIZE = 19
LEGEND_FONT_SIZE = 19
ANNOTATION_FONT_SIZE = 19

GRID_LINESTYLE = '--'
GRID_ALPHA = 0.6
BAR_ZORDER = 3


# ==============================================================================

def get_original_figsize(filename):
    """Returns the original (width, height) used in the main script."""
    if "splits_overall" in filename: return (10, 6)
    if "splits_by_class" in filename: return (14, 8)
    if "gt_splits_overall_distribution" in filename: return (10, 6)
    if "gt_splits_accuracy" in filename: return (12, 7)
    if "gt_samples_per_occlusion_category_AnyClass" in filename: return (18, 8)
    if "global_samples_per_occlusion_category_comparison" in filename: return (10, 6)
    if "combined_occlusion_impact_bars" in filename: return (18, 11)
    if "compare_all_models_occlusion_bars" in filename: return (20, 6)
    if "compare_all_models_overall_mAP_bars" in filename: return (10, 6)
    if "compare_all_models_overlap_confusion" in filename: return (12, 7)
    return (10, 6)  # Fallback


def plot_splits_overall(csv_path, ax, model_name):
    df = pd.read_csv(csv_path)
    x_vals = df['Number of Splits'].values
    y_overall = df['Number of Images'].values

    ax.bar(x_vals, y_overall, color='#1f77b4', edgecolor='black', zorder=BAR_ZORDER)
    ax.set_title(f'Single Instance Splitting - Overall\n({model_name})',
                 fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
    ax.set_xlabel('Number of Predicted Instances (Splits)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('Number of Images', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticks(x_vals)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(axis='y', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, zorder=0)


def plot_splits_by_class(csv_path, ax, model_name):
    df = pd.read_csv(csv_path)
    x_vals = df['Number of Splits'].values
    classes = [c for c in df.columns if c != 'Number of Splits']

    bottom = np.zeros(len(x_vals))
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(classes)))

    for idx, cls_name in enumerate(classes):
        y_cls = df[cls_name].values
        ax.bar(x_vals, y_cls, bottom=bottom, label=cls_name, color=colors[idx], edgecolor='white', zorder=BAR_ZORDER)
        bottom += y_cls

    ax.set_title(f'Single Instance Splitting - By Class\n({model_name})',
                 fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
    ax.set_xlabel('Number of Predicted Instances (Splits)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('Number of Images', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticks(x_vals)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(axis='y', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, zorder=0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2 if len(classes) > 15 else 1, fontsize=LEGEND_FONT_SIZE)


def plot_gt_splits_distribution(csv_path, ax, model_name):
    df = pd.read_csv(csv_path)
    x_vals = df['Number of Splits'].values
    y_vals = df['Number of Ground Truth Instances'].values

    ax.bar([str(x) for x in x_vals], y_vals, color='#8c564b', edgecolor='black', zorder=BAR_ZORDER)
    ax.set_title(f'GT Instance Splits Distribution\n({model_name})',
                 fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
    ax.set_xlabel('Number of Splits (Polygons)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('Number of GT Instances', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(axis='y', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, zorder=0)


def plot_gt_splits_accuracy(csv_path, ax, model_name):
    df = pd.read_csv(csv_path)
    x_vals = df['Number of Splits'].values
    y_correct = df['Correct Predictions (TP)'].values
    y_wrong = df['Wrong Predictions (FN)'].values

    x_indices = np.arange(len(x_vals))
    width = 0.4

    ax.bar(x_indices - width / 2, y_correct, width=width, label='Correct (TP)', color='#2ca02c', edgecolor='black',
           zorder=BAR_ZORDER)
    ax.bar(x_indices + width / 2, y_wrong, width=width, label='Wrong (FN)', color='#d62728', edgecolor='black',
           zorder=BAR_ZORDER)

    ax.set_title(f'Prediction Accuracy by GT Splits\n({model_name})',
                 fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
    ax.set_xlabel('Number of Splits (Polygons)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('Number of GT Instances', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(x) for x in x_vals], fontsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.grid(axis='y', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, zorder=0)


def plot_gt_samples_any_class(csv_path, ax):
    df = pd.read_csv(csv_path)
    classes = df['Class'].values
    labels = [col for col in df.columns if col != 'Class']

    bottom = np.zeros(len(df))
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

    for label, color in zip(labels, colors):
        ax.bar(classes, df[label], label=label, bottom=bottom, color=color, alpha=0.85, zorder=BAR_ZORDER)
        bottom += df[label].values

    ax.set_title('Distribution of Samples per Occlusion\n(Any Class)',
                 fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
    ax.set_xlabel('Segmentation Class', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('Total Instances', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticklabels(classes, rotation=90, ha='center', fontsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
    ax.legend(title="Occlusion Category", title_fontsize=LEGEND_FONT_SIZE, fontsize=LEGEND_FONT_SIZE)
    ax.grid(axis='y', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, zorder=0)


def plot_global_samples_comparison(csv_path, ax):
    df = pd.read_csv(csv_path)
    global_bin_labels = df['Occlusion Category'].values
    any_counts = df['All Classes'].values
    same_counts = df['Same Class'].values

    x = np.arange(len(global_bin_labels))
    width = 0.35

    rects1 = ax.bar(x - width / 2, any_counts, width, label='All Classes', color='#1f77b4', zorder=BAR_ZORDER)
    rects2 = ax.bar(x + width / 2, same_counts, width, label='Same Class', color='#ff7f0e', zorder=BAR_ZORDER)

    ax.set_ylabel('Number of Samples', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title('Global Sample Distribution\nby Occlusion Amount',
                 fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
    ax.set_xticks(x)
    ax.set_xticklabels(global_bin_labels, fontsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.grid(axis='y', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, zorder=0)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                        fontsize=ANNOTATION_FONT_SIZE)

    autolabel(rects1)
    autolabel(rects2)


def plot_combined_occlusion_impact(csv_path, ax, model_name, occ_type):
    df = pd.read_csv(csv_path)
    categories = df['Occlusion Category'].values

    x = np.arange(len(categories))
    width = 0.25

    ax.bar(x - width, df['mAP @ IoU 0.75'], width, label='mAP @ 0.75', zorder=BAR_ZORDER)
    ax.bar(x, df['mAP @ IoU 0.5'], width, label='mAP @ 0.5', zorder=BAR_ZORDER)
    ax.bar(x + width, df['mAP @ IoU 0.3'], width, label='mAP @ 0.3', zorder=BAR_ZORDER)

    ax.set_ylim(0, 0.7)
    title_str = 'mAP vs. True Mask Occlusion' if "Any_Class" in occ_type else 'mAP vs. Same-Class Occlusion'
    ax.set_title(f'{title_str}\n({model_name})',
                 fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
    ax.set_xlabel('Occlusion Category', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('mAP', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)

    for tick in ax.get_xticklabels():
        if tick.get_text() == "Overall":
            tick.set_fontweight("bold")

    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.grid(axis='y', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, zorder=0)


def plot_cross_model_occlusion(csv_path, ax, occ_type):
    df = pd.read_csv(csv_path)
    plot_categories = df['Occlusion Category'].values
    models = [col.split(' @ ')[0] for col in df.columns if '@ IoU 0.5' in col]

    title_str = 'Model Comparison @ IoU 0.5\n(Any Class)' if "Any_Class" in occ_type else 'Model Comparison @ IoU 0.5\n(Same Class)'

    x = np.arange(len(plot_categories))
    num_models = len(models)
    total_width = 0.8
    bar_width = total_width / num_models

    for j, model_name in enumerate(models):
        y_vals = df[f'{model_name} @ IoU 0.5'].values
        offset = x - (total_width / 2) + (j * bar_width) + (bar_width / 2)
        ax.bar(offset, y_vals, width=bar_width, label=model_name, alpha=0.9, zorder=BAR_ZORDER)

    ax.set_title(title_str, fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
    ax.set_xlabel('Occlusion Category', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('mAP', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_categories, fontsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)

    for tick in ax.get_xticklabels():
        if tick.get_text() == "Overall":
            tick.set_fontweight("bold")

    ax.grid(axis='y', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, zorder=0)
    ax.legend(title="Models", title_fontsize=LEGEND_FONT_SIZE, fontsize=LEGEND_FONT_SIZE)


def plot_overall_map(csv_path, ax):
    df = pd.read_csv(csv_path)
    models = df['Model'].values
    ious = [col for col in df.columns if col != 'Model']

    x_overall = np.arange(len(ious))
    num_models = len(models)
    total_width = 0.8
    bar_width = total_width / num_models

    for j, model_name in enumerate(models):
        y_vals = df.loc[df['Model'] == model_name, ious].values.flatten()
        offset = x_overall - (total_width / 2) + (j * bar_width) + (bar_width / 2)
        ax.bar(offset, y_vals, width=bar_width, label=model_name, alpha=0.9, zorder=BAR_ZORDER)

    ax.set_title('Model Comparison: Overall mAP',
                 fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
    ax.set_xlabel('IoU Threshold', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('mAP', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticks(x_overall)
    ax.set_xticklabels(ious, fontsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(axis='y', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, zorder=0)
    ax.legend(title="Models", title_fontsize=LEGEND_FONT_SIZE, fontsize=LEGEND_FONT_SIZE)


def plot_overlap_confusion(csv_path, ax):
    df = pd.read_csv(csv_path)
    models = df['Model'].values
    same_class = df['Overlapping same class'].values
    diff_class = df['Overlapping different class'].values
    no_overlap = df['No overlapping'].values

    x_pos = np.arange(len(models))
    width = 0.5

    ax.bar(x_pos, same_class, width, label='TP (Same class)', color='#2ca02c', zorder=BAR_ZORDER)
    ax.bar(x_pos, diff_class, width, bottom=same_class, label='Misclassified (Diff)', color='#ff7f0e',
           zorder=BAR_ZORDER)
    ax.bar(x_pos, no_overlap, width, bottom=same_class + diff_class, label='FP (No overlap)', color='#d62728',
           zorder=BAR_ZORDER)

    ax.set_ylabel('Number of Predicted Masks', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title('Prediction Confusion Breakdown\n(IoU $\geq$ 0.5)',
                 fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.grid(axis='y', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, zorder=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine multiple CSV plots into a dynamically sized image grid.")
    parser.add_argument("--csvs", nargs='+', required=True, help="List of paths to the CSV files to combine")
    parser.add_argument("--output", type=str, default="combined_plots.png", help="Path to save the combined image")

    args = parser.parse_args()

    n_plots = len(args.csvs)
    if n_plots == 0:
        print("No CSV files provided.")
        exit(0)

    cols = 2 if n_plots > 1 else 1
    rows = math.ceil(n_plots / cols)

    # Calculate dimensions based on original sizes
    sizes = [get_original_figsize(os.path.basename(csv)) for csv in args.csvs]
    print(sizes)
    # Pad sizes array for incomplete rows
    while len(sizes) < rows * cols:
        sizes.append((0, 0))

    grid_widths = []
    for c in range(cols):
        col_widths = [sizes[r * cols + c][0] for r in range(rows)]
        grid_widths.append(max(col_widths) if max(col_widths) > 0 else 10)

    grid_heights = []
    for r in range(rows):
        row_heights = [sizes[r * cols + c][1] for c in range(cols)]
        grid_heights.append(max(row_heights) if max(row_heights) > 0 else 6)

    total_width = sum(grid_widths)
    total_height = sum(grid_heights)

    # Build Grid layout
    fig = plt.figure(figsize=(total_width, total_height))
    gs = GridSpec(rows, cols, width_ratios=grid_widths, height_ratios=grid_heights, figure=fig)

    for i, csv_path in enumerate(args.csvs):
        r = i // cols
        c = i % cols
        ax = fig.add_subplot(gs[r, c])

        filename = os.path.basename(csv_path)

        try:
            if filename.startswith("splits_overall_") and filename.endswith("_plot.csv"):
                model = filename.replace("splits_overall_", "").replace("_plot.csv", "")
                plot_splits_overall(csv_path, ax, model)

            elif filename.startswith("splits_by_class_") and filename.endswith("_plot.csv"):
                model = filename.replace("splits_by_class_", "").replace("_plot.csv", "")
                plot_splits_by_class(csv_path, ax, model)

            elif filename.startswith("gt_splits_overall_distribution_") and filename.endswith("_plot.csv"):
                model = filename.replace("gt_splits_overall_distribution_", "").replace("_plot.csv", "")
                plot_gt_splits_distribution(csv_path, ax, model)

            elif filename.startswith("gt_splits_accuracy_") and filename.endswith("_plot.csv"):
                model = filename.replace("gt_splits_accuracy_", "").replace("_plot.csv", "")
                plot_gt_splits_accuracy(csv_path, ax, model)

            elif filename == "gt_samples_per_occlusion_category_AnyClass_plot.csv":
                plot_gt_samples_any_class(csv_path, ax)

            elif filename == "global_samples_per_occlusion_category_comparison_plot.csv":
                plot_global_samples_comparison(csv_path, ax)

            elif filename.startswith("combined_occlusion_impact_bars_") and filename.endswith("_plot.csv"):
                model_name = os.path.basename(os.path.dirname(csv_path))
                occ_type = filename.replace("combined_occlusion_impact_bars_", "").replace("_plot.csv", "")
                plot_combined_occlusion_impact(csv_path, ax, model_name, occ_type)

            elif filename.startswith("compare_all_models_occlusion_bars_") and filename.endswith("_plot.csv"):
                occ_type = filename.replace("compare_all_models_occlusion_bars_", "").replace("_plot.csv", "")
                plot_cross_model_occlusion(csv_path, ax, occ_type)

            elif filename == "compare_all_models_overall_mAP_bars_plot.csv":
                plot_overall_map(csv_path, ax)

            elif filename == "compare_all_models_overlap_confusion_plot.csv":
                plot_overlap_confusion(csv_path, ax)
            else:
                ax.text(0.5, 0.5, f"Unknown CSV Type:\n{filename}", ha='center', va='center', fontsize=TITLE_FONT_SIZE)
                ax.set_xticks([])
                ax.set_yticks([])

        except Exception as e:
            ax.text(0.5, 0.5, f"Error plotting:\n{filename}\n{str(e)}", ha='center', va='center', color='red',
                    fontsize=AXIS_LABEL_FONT_SIZE)
            print(f"Error processing {filename}: {e}")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Successfully combined {n_plots} plots into {args.output}")