#!/usr/bin/env bash

set -euo pipefail

GPU_ID=0
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

NUM_WORKERS=0
BATCH_SIZE=8
PRECISION="32"
IMG_ENCODER="facebook/dinov2-base"

CSV_PATH="/mnt/f/data/train_with_test_data.csv"
IMG_ROOT="/mnt/f/data/images_png"
PATH_PATTERN="{pid}/{iid}"

BASE_EXP_NAME="5fold_train_validation_test"

IMG_SIZE=336
CROP_SIZE=336
LLM_TYPE="bert"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"

python -u export_5fold_predictions.py \
  --base_exp_name "${BASE_EXP_NAME}" \
  --rsna_mammo \
  --img_cls_ft \
  --llm_type "${LLM_TYPE}" \
  --img_encoder "${IMG_ENCODER}" \
  --devices 1 \
  --precision "${PRECISION}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --k_fold 5 \
  --num_folds 5 \
  --num_classes 1 \
  --weighted_binary \
  --img_size "${IMG_SIZE}" \
  --crop_size "${CROP_SIZE}" \
  --rsna_csv_path "${CSV_PATH}" \
  --rsna_img_root "${IMG_ROOT}" \
  --rsna_path_pattern "${PATH_PATTERN}" \
  --rsna_patient_col "patient_id" \
  --rsna_image_col "image_id" \
  --rsna_label_col "cancer" \
  --rsna_split_col "split" \
  --rsna_train_split_value "training" \
  --rsna_test_split_value "test"
