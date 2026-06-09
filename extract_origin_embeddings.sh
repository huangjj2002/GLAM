#!/bin/bash
set -euo pipefail

GPU_ID=1

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"

# Use the pretrained encoder checkpoint by default. You can override it with
# the first argument or PRETRAINED_ENCODER env var if needed.
PRETRAINED_ENCODER="${1:-${PRETRAINED_ENCODER:-${ROOT_DIR}/pretrain_model/last.ckpt}}"

# Same data paths/settings as run.sh.
CSV_PATH="/home/dhao4/workspace/hjj_workspace/data/data.csv"
IMG_ROOT="/home/dhao4/workspace/hjj_workspace/data/images_png"
PATH_PATTERN="{pid}/{iid}"

PATIENT_COL="patient_id"
IMAGE_COL="image_id"
LABEL_COL="cancer"
SPLIT_COL="split"
COHORT_COL="cohert_num"
SPLIT_SOURCE="cohort"
TRAIN_COHORTS="1-8"
TEST_COHORTS="9-10"

IMG_ENCODER="dinov2_vitb14_reg"
LLM_TYPE="bert"
EMB_DIM=512
IMG_SIZE=336
CROP_SIZE=336
BATCH_SIZE=32
NUM_WORKERS=4
DTYPE="float16"

OUTPUT_DIR="outputs/origin_embeddings/origin_last_all"

# Set to a positive number for a quick smoke test, e.g. MAX_BATCHES=2.
MAX_BATCHES=0

export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ ! -f "${PRETRAINED_ENCODER}" ]]; then
  echo "[ERROR] Pretrained encoder checkpoint not found: ${PRETRAINED_ENCODER}"
  exit 1
fi
if [[ ! -f "${CSV_PATH}" ]]; then
  echo "[ERROR] CSV file not found: ${CSV_PATH}"
  exit 1
fi
if [[ ! -d "${IMG_ROOT}" ]]; then
  echo "[ERROR] IMG_ROOT directory not found: ${IMG_ROOT}"
  exit 1
fi

MAX_BATCHES_FLAG=()
if [[ "${MAX_BATCHES}" != "0" ]]; then
  MAX_BATCHES_FLAG+=(--max_batches "${MAX_BATCHES}")
fi

echo "Root dir           : ${ROOT_DIR}"
echo "GPU_ID             : ${GPU_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "PRETRAINED_ENCODER : ${PRETRAINED_ENCODER}"
echo "CSV_PATH           : ${CSV_PATH}"
echo "IMG_ROOT           : ${IMG_ROOT}"
echo "PATH_PATTERN       : ${PATH_PATTERN}"
echo "OUTPUT_DIR         : ${OUTPUT_DIR}"
echo "DTYPE              : ${DTYPE}"
echo "MAX_BATCHES        : ${MAX_BATCHES}"

python -u extract_origin_embeddings.py \
  --pretrained_encoder "${PRETRAINED_ENCODER}" \
  --no_save_logits \
  --csv_path "${CSV_PATH}" \
  --img_root "${IMG_ROOT}" \
  --path_pattern "${PATH_PATTERN}" \
  --patient_col "${PATIENT_COL}" \
  --image_col "${IMAGE_COL}" \
  --label_col "${LABEL_COL}" \
  --rsna_split_col "${SPLIT_COL}" \
  --rsna_cohort_col "${COHORT_COL}" \
  --split_source "${SPLIT_SOURCE}" \
  --train_cohorts "${TRAIN_COHORTS}" \
  --test_cohorts "${TEST_COHORTS}" \
  --split all \
  --img_encoder "${IMG_ENCODER}" \
  --llm_type "${LLM_TYPE}" \
  --emb_dim "${EMB_DIM}" \
  --linear_proj \
  --img_cls_ft \
  --weighted_binary \
  --num_classes 1 \
  --img_size "${IMG_SIZE}" \
  --crop_size "${CROP_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --dtype "${DTYPE}" \
  --device cuda \
  --output_dir "${OUTPUT_DIR}" \
  "${MAX_BATCHES_FLAG[@]}"

echo "Embedding export done."
echo "Saved to: ${ROOT_DIR}/${OUTPUT_DIR}"
