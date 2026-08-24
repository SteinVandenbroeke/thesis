import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def extract_map_values(filepath):
    """
    Helper function to load a CSV and extract mAP values.
    It grabs the first 5 values (assuming the 5th is 'Overall')
    and moves 'Overall' to the front to match the new category order.
    """
    df = pd.read_csv(filepath)
    vals = df['mAP'].tolist()[:5]

    # If we successfully grabbed 5 items, move the last one ('Overall') to the front
    if len(vals) == 5:
        vals = [vals[-1]] + vals[:-1]

    return vals


def main():
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Combine occlusion mAP plots from CSV data.")
    parser.add_argument(
        '-d', '--dir',
        type=str,
        default='.',
        help='Path to the directory containing the CSV files. Defaults to current directory.'
    )
    args = parser.parse_args()

    data_dir = args.dir

    # Define the occlusion categories in the desired order ('Overall' moved to the left)
    categories = ['Overall', 'No Overlap', 'Low (0-25%)', 'Medium (25-50%)', 'High (>50%)']
    x = np.arange(len(categories))
    width = 0.25  # Width of the bars

    # Set up the combined figure with 2 subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=True)

    # --- Plot 1: Any Class ---
    ax1 = axes[0]
    try:
        # Use os.path.join to connect the directory with the file names
        any_075 = extract_map_values(os.path.join(data_dir, "occlusion_mAP_Any_Class_results_0_75.csv"))
        any_050 = extract_map_values(os.path.join(data_dir, "occlusion_mAP_Any_Class_results_0_5.csv"))
        any_030 = extract_map_values(os.path.join(data_dir, "occlusion_mAP_Any_Class_results_0_3.csv"))

        ax1.bar(x - width, any_075, width, label='mAP @ IoU 0.75')
        ax1.bar(x, any_050, width, label='mAP @ IoU 0.5')
        ax1.bar(x + width, any_030, width, label='mAP @ IoU 0.3')
    except FileNotFoundError as e:
        print(f"Error loading Any_Class files: {e}")

    # Text formatting
    ax1.set_title("mAP vs. True Mask Occlusion (BAS)", fontsize=22, pad=15)
    ax1.set_xlabel("Occlusion Category", fontsize=18, labelpad=10)
    ax1.set_ylabel("mAP", fontsize=18, labelpad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend(fontsize=16)

    # --- Plot 2: Same Class ---
    ax2 = axes[1]
    try:
        same_075 = extract_map_values(os.path.join(data_dir, "occlusion_mAP_Same_Class_results_0_75.csv"))
        same_050 = extract_map_values(os.path.join(data_dir, "occlusion_mAP_Same_Class_results_0_5.csv"))
        same_030 = extract_map_values(os.path.join(data_dir, "occlusion_mAP_Same_Class_results_0_3.csv"))

        ax2.bar(x - width, same_075, width, label='mAP @ IoU 0.75')
        ax2.bar(x, same_050, width, label='mAP @ IoU 0.5')
        ax2.bar(x + width, same_030, width, label='mAP @ IoU 0.3')
    except FileNotFoundError as e:
        print(f"Error loading Same_Class files: {e}")

    # Text formatting
    ax2.set_title("mAP vs. Same-Class Mask Occlusion (Overlapping Items Only) (BAS)", fontsize=22, pad=15)
    ax2.set_xlabel("Occlusion Category", fontsize=18, labelpad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.legend(fontsize=16)

    # Adjust layout and save the combined plot into the target directory
    plt.tight_layout()
    output_path = os.path.join(data_dir, "combined_occlusion_impact_bars_Side_By_Side.png")
    plt.savefig(output_path, dpi=300)
    print(f"Success! Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()