import matplotlib.pyplot as plt
import csv
import os

LOG_FILE = 'checkpoints/PAM/loss_log.csv'
OUT_IMAGE = 'loss_curve.png'

if not os.path.exists(LOG_FILE):
    print(f"Error: Could not find {LOG_FILE}. Have you finished at least one epoch?")
    exit()

epochs = []
losses = []

# Read the CSV
with open(LOG_FILE, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    for row in reader:
        if len(row) >= 2:
            epochs.append(int(row[0]))
            losses.append(float(row[1]))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b', linewidth=2, label='Training Loss')

# Formatting
plt.title('Model Training Loss over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Force the X-axis to show whole numbers for epochs
plt.xticks(epochs) 

# Save and close
plt.tight_layout()
plt.savefig(OUT_IMAGE, dpi=300)
print(f"Success! Saved loss plot to {OUT_IMAGE}")