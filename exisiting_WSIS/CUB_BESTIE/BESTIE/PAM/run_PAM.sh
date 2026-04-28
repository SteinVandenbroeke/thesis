#!/bin/bash

# Detect the number of available NVIDIA GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# Create a comma-separated list (e.g., "0,1,2" if NUM_GPUS is 3)
# sequence -s joins numbers with a separator; -p handles the 0-indexing
GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))

echo "Detected $NUM_GPUS GPUs. Using devices: $GPU_IDS"

# Run Training
CUDA_VISIBLE_DEVICES=$GPU_IDS python train.py \
    --root_dir=../data/VOCdevkit/VOC2012 \
    --lr=0.01 \
    --epoch=50 \
    --decay_points='5,10' \
    --alpha=0.7 \
    --save_folder=checkpoints/PAM \
    --show_interval=50 \
    --num_classes=200

# Run Point Extraction
CUDA_VISIBLE_DEVICES=$GPU_IDS python point_extraction.py \
    --root_dir=../data/VOCdevkit/VOC2012 \
    --alpha=0.7 \
    --checkpoint=checkpoints/PAM/ckpt_50.pth \
    --save_dir=../data/VOCdevkit/VOC2012/Peak_points \
    --num_classes=200