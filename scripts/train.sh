#!/usr/bin/env bash
# Train RIRCI on a dataset laid out as described in the README.
set -e

DATASET_DIR=${DATASET_DIR:-data/RIRCI}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}
NAME=${NAME:-rirci}
GPU=${GPU:-0}

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --dataset rirci \
    --dataset_dir "${DATASET_DIR}" \
    --checkpoint "${CHECKPOINT_DIR}" \
    --name "${NAME}" \
    --nets rirci \
    --models rirci \
    --epochs 100 \
    --schedule 65 \
    --lr 1e-3 \
    --train-batch 8 \
    --test-batch 1 \
    --input-size 256 \
    --crop_size 256 \
    --lambda_l1 2 \
    --lambda_content 1 \
    --lambda_mask 3 \
    --use_refine
