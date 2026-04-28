import cv2
import os
import glob

# ==========================================
# CONFIGURATION
# ==========================================
IMG_DIR = '../data/VOCdevkit/VOC2012/JPEGImages'
PEAK_DIR = '../data/VOCdevkit/VOC2012/Peak_points'
OUT_DIR = 'visualizations'

os.makedirs(OUT_DIR, exist_ok=True)

# Find all the generated text files (let's just do the first 20 for a quick check)
txt_files = glob.glob(os.path.join(PEAK_DIR, '*.txt'))[:20]

print(f"Found {len(glob.glob(os.path.join(PEAK_DIR, '*.txt')))} total peak files. Visualizing a sample...")

count = 0
for txt_path in txt_files:
    filename = os.path.basename(txt_path)
    img_name = filename.replace('.txt', '.jpg')
    img_path = os.path.join(IMG_DIR, img_name)

    # Skip if the original image doesn't exist
    if not os.path.exists(img_path):
        continue

    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Read the peak points
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # Draw a circle for each peak
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
            
        try:
            # Correctly parse [y] [x] [class] [score]
            x, y = int(float(parts[0])), int(float(parts[1]))
            
            cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(img, (x, y), radius=8, color=(0, 255, 255), thickness=2)
            
        except ValueError:
            continue

    # Save the visualized image
    out_path = os.path.join(OUT_DIR, f"peak_{img_name}")
    cv2.imwrite(out_path, img)
    count += 1

print(f"Success! Saved {count} visualizations to the '{OUT_DIR}' folder.")