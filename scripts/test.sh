#!/usr/bin/env bash
# Evaluate a trained RIRCI checkpoint on the validation / test split.
set -e

DATASET_DIR=${DATASET_DIR:-data/RIRCI}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}
NAME=${NAME:-rirci}
GPU=${GPU:-0}

CUDA_VISIBLE_DEVICES=${GPU} python test.py \
    --evaluate \
    --dataset rirci \
    --dataset_dir "${DATASET_DIR}" \
    --checkpoint "${CHECKPOINT_DIR}" \
    --name "${NAME}" \
    --nets rirci \
    --models rirci \
    --eval_split val \
    --test-batch 1 \
    --input-size 256 \
    --crop_size 256 \
    --use_refine \
    --resume "${CHECKPOINT_DIR}/${NAME}/model_best.pth.tar"
