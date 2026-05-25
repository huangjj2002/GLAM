#!/bin/bash
set -euo pipefail

# ===========================================================================
# EDL Training & Evaluation Script
# Mirrors run.sh data pipeline but uses EDL model instead of vanilla GLAM
# ===========================================================================

GPU_ID=2
MAX_EPOCHS=25
WARM_UP=1
NUM_GPUS=1
BATCH_SIZE=16
NUM_WORKERS=4
# Windows下建议 num_workers=0
PRECISION="32"
LEARNING_RATE=1e-4

# 早停策略
EARLY_STOP=1
EARLY_STOP_PATIENCE=3
EARLY_STOP_MIN_EPOCHS=12
MONITOR_METRIC="val_AUROC"

OUTPUT_DIR="outputs"

IMG_ENCODER="dinov2_vitb14_reg"
EMB_DIM=512
USE_LINEAR_PROJ=1
LLM_TYPE="bert"
IMG_SIZE=336
CROP_SIZE=336

# EDL-specific
EVIDENCE_TYPE="softplus"
EDL_LOSS_TYPE="digamma"
ANNEALING_STEP=10
LAMBDA_KL=0.1
FREEZE_BACKBONE=0
# 0 keeps the original imbalanced distribution and uses pos_weight only.
# 1 uses WeightedRandomSampler; do not combine it with auto pos_weight unless
# you explicitly want very aggressive positive-class emphasis.
BALANCE_TRAINING=0
# Empty means auto-compute neg/pos from each fold's training split.
BCE_POS_WEIGHT=""

CSV_PATH="/home/dhao4/workspace/hjj_workspace/data/data.csv"
IMG_ROOT="/home/dhao4/workspace/hjj_workspace/data/images_png"
PATH_PATTERN="{pid}/{iid}"

PATIENT_COL="patient_id"
IMAGE_COL="image_id"
LABEL_COL="cancer"
SPLIT_COL="split"
COHORT_COL="cohert_num"
SPLIT_SOURCE="cohort"
TRAIN_SPLIT_VALUE="training"
TEST_SPLIT_VALUE="test"
TRAIN_COHORTS="1-8"
TEST_COHORTS="9-10"
OUTPUT_SPLIT_COL="split"
VAL_SPLIT=1
VAL_FRACTION=0.15
VAL_MAX_FRACTION=0.25
VAL_MIN_POSITIVE_PATIENTS=3
VAL_RANDOM_STATE=42

PRETRAINED_MODEL="$(cd "$(dirname "$0")" && pwd)/pretrain_model/last.ckpt"
NUM_FOLDS=0
FOLDS_TO_RUN=""
BASE_EXP_NAME="edl_kfold_ft"

export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"

if [[ "${OUTPUT_DIR}" = /* ]]; then
  OUTPUT_ROOT="${OUTPUT_DIR}"
else
  OUTPUT_ROOT="${ROOT_DIR}/${OUTPUT_DIR}"
fi
RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${BASE_EXP_NAME}_${RUN_TAG}"
mkdir -p "${RUN_OUTPUT_DIR}"

if [[ "${NUM_FOLDS}" == "0" ]]; then
  FOLDS_TO_RUN="0"
fi

LINEAR_PROJ_FLAG=()
if [[ "${USE_LINEAR_PROJ}" == "1" ]]; then
  LINEAR_PROJ_FLAG+=(--linear_proj)
fi

EARLY_STOP_FLAG=()
if [[ "${EARLY_STOP}" == "1" ]]; then
  EARLY_STOP_FLAG+=(--early_stop --early_stop_patience "${EARLY_STOP_PATIENCE}" --early_stop_min_epochs "${EARLY_STOP_MIN_EPOCHS}")
fi

FREEZE_BACKBONE_FLAG=()
if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
  FREEZE_BACKBONE_FLAG+=(--freeze_backbone)
fi

POS_WEIGHT_FLAG=()
if [[ -n "${BCE_POS_WEIGHT}" ]]; then
  POS_WEIGHT_FLAG+=(--pos_weight "${BCE_POS_WEIGHT}")
fi

BALANCE_TRAINING_FLAG=()
if [[ "${BALANCE_TRAINING}" == "1" ]]; then
  BALANCE_TRAINING_FLAG+=(--balance_training)
fi

VAL_SPLIT_FLAG=()
if [[ "${VAL_SPLIT}" == "1" ]]; then
  VAL_SPLIT_FLAG+=(--rsna_val_split)
fi

FOLDS_TO_RUN_FLAG=()
if [[ -n "${FOLDS_TO_RUN}" ]]; then
  FOLDS_TO_RUN_FLAG+=(--folds_to_run "${FOLDS_TO_RUN}")
fi

echo "Root dir                : ${ROOT_DIR}"
echo "GPU_ID                  : ${GPU_ID}"
echo "CUDA_VISIBLE_DEVICES    : ${CUDA_VISIBLE_DEVICES}"
echo "CSV_PATH                : ${CSV_PATH}"
echo "IMG_ROOT                : ${IMG_ROOT}"
echo "PATH_PATTERN            : ${PATH_PATTERN}"
echo "PRETRAINED_MODEL        : ${PRETRAINED_MODEL}"
echo "IMG_ENCODER             : ${IMG_ENCODER}"
echo "EMB_DIM                 : ${EMB_DIM}"
echo "USE_LINEAR_PROJ         : ${USE_LINEAR_PROJ}"
echo "MAX_EPOCHS              : ${MAX_EPOCHS}"
echo "WARM_UP (epochs)        : ${WARM_UP}"
echo "EARLY_STOP              : ${EARLY_STOP}"
echo "EARLY_STOP_PATIENCE     : ${EARLY_STOP_PATIENCE}"
echo "EARLY_STOP_MIN_EPOCHS   : ${EARLY_STOP_MIN_EPOCHS}"
echo "MONITOR_METRIC          : ${MONITOR_METRIC}"
echo "EDL_LOSS_TYPE           : ${EDL_LOSS_TYPE}"
echo "EVIDENCE_TYPE           : ${EVIDENCE_TYPE}"
echo "FREEZE_BACKBONE         : ${FREEZE_BACKBONE}"
echo "BALANCE_TRAINING       : ${BALANCE_TRAINING}"
echo "ANNEALING_STEP          : ${ANNEALING_STEP}"
echo "LAMBDA_KL               : ${LAMBDA_KL}"
echo "BCE_POS_WEIGHT          : ${BCE_POS_WEIGHT:-auto per fold}"
echo "VAL_SPLIT               : ${VAL_SPLIT}"
echo "VAL_FRACTION            : ${VAL_FRACTION}"
echo "VAL_MAX_FRACTION        : ${VAL_MAX_FRACTION}"
echo "VAL_MIN_POS_PATIENTS    : ${VAL_MIN_POSITIVE_PATIENTS}"
echo "VAL_RANDOM_STATE        : ${VAL_RANDOM_STATE}"
echo "K_FOLD                  : ${NUM_FOLDS}"
echo "FOLDS_TO_RUN            : ${FOLDS_TO_RUN:-all folds}"
echo "RUN_OUTPUT_DIR          : ${RUN_OUTPUT_DIR}"

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "[ERROR] CSV file not found: ${CSV_PATH}"
  exit 1
fi
if [[ ! -d "${IMG_ROOT}" ]]; then
  echo "[ERROR] IMG_ROOT directory not found: ${IMG_ROOT}"
  exit 1
fi
if [[ ! -f "${PRETRAINED_MODEL}" ]]; then
  echo "[ERROR] PRETRAINED_MODEL checkpoint not found: ${PRETRAINED_MODEL}"
  exit 1
fi

echo ""
echo "=============================="
echo "Running EDL Training Pipeline"
echo "=============================="

python -u edl_train.py \
  --csv_path "${CSV_PATH}" \
  --img_root "${IMG_ROOT}" \
  --path_pattern "${PATH_PATTERN}" \
  --pretrained_model "${PRETRAINED_MODEL}" \
  --patient_col "${PATIENT_COL}" \
  --image_col "${IMAGE_COL}" \
  --label_col "${LABEL_COL}" \
  --split_col "${SPLIT_COL}" \
  --split_source "${SPLIT_SOURCE}" \
  --cohort_col "${COHORT_COL}" \
  --train_split_value "${TRAIN_SPLIT_VALUE}" \
  --test_split_value "${TEST_SPLIT_VALUE}" \
  --train_cohorts "${TRAIN_COHORTS}" \
  --test_cohorts "${TEST_COHORTS}" \
  --output_split_col "${OUTPUT_SPLIT_COL}" \
  "${VAL_SPLIT_FLAG[@]}" \
  --rsna_val_fraction "${VAL_FRACTION}" \
  --rsna_val_max_fraction "${VAL_MAX_FRACTION}" \
  --rsna_val_min_positive_patients "${VAL_MIN_POSITIVE_PATIENTS}" \
  --rsna_val_random_state "${VAL_RANDOM_STATE}" \
  --img_encoder "${IMG_ENCODER}" \
  --emb_dim "${EMB_DIM}" \
  "${LINEAR_PROJ_FLAG[@]}" \
  --evidence_type "${EVIDENCE_TYPE}" \
  --edl_loss_type "${EDL_LOSS_TYPE}" \
  --annealing_step "${ANNEALING_STEP}" \
  --lambda_kl "${LAMBDA_KL}" \
  "${POS_WEIGHT_FLAG[@]}" \
  "${FREEZE_BACKBONE_FLAG[@]}" \
  --k_fold "${NUM_FOLDS}" \
  "${FOLDS_TO_RUN_FLAG[@]}" \
  --max_epochs "${MAX_EPOCHS}" \
  --warm_up "${WARM_UP}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --learning_rate "${LEARNING_RATE}" \
  --monitor_metric "${MONITOR_METRIC}" \
  --devices "${NUM_GPUS}" \
  --strategy auto \
  --precision "${PRECISION}" \
  "${BALANCE_TRAINING_FLAG[@]}" \
  --img_size "${IMG_SIZE}" \
  --crop_size "${CROP_SIZE}" \
  --output_dir "${RUN_OUTPUT_DIR}" \
  --experiment_name "${BASE_EXP_NAME}" \
  "${EARLY_STOP_FLAG[@]}"

echo "All done."
echo "Run output dir: ${RUN_OUTPUT_DIR}"
echo "Per-model CSV dir: ${RUN_OUTPUT_DIR}/per_model_predictions/"
echo "Ensemble CSV: ${RUN_OUTPUT_DIR}/per_model_predictions/ensemble_edl_predictions.csv"
echo "Report: ${RUN_OUTPUT_DIR}/per_model_predictions/edl_eval_report.txt"
