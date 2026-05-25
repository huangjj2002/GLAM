#!/bin/bash
set -euo pipefail

GPU_ID=1
#指定GPU
MAX_EPOCHS=25
#指定每一折训练的epoch
WARM_UP=1
#指定warm_up的epoch数
NUM_GPUS=1
BATCH_SIZE=32
#batch_size大小
NUM_WORKERS=4
PRECISION="32"
LEARNING_RATE=1e-4
SHOW_PROGRESS_BAR=1
#是否显示Lightning训练进度条（1=显示，0=隐藏）

# 早停策略配置
EARLY_STOP=1
#是否启用早停（1=启用，0=禁用）
EARLY_STOP_PATIENCE=3
#早停耐心轮数，即验证指标连续多少个epoch没有改善后停止训练
EARLY_STOP_MIN_EPOCHS=0
#早停最小训练epoch数；原模型默认不强制，EDL脚本默认会设置更高
MONITOR_METRIC="val_AUROC"
#checkpoint与早停监控指标
BALANCE_TRAINING=0
#是否启用训练集重采样（1=启用，0=禁用），默认与run_edl.sh保持一致
BCE_POS_WEIGHT=""
#空值表示自动按训练split的neg/pos计算

OUTPUT_DIR="outputs"

IMG_ENCODER="dinov2_vitb14_reg"
EMB_DIM=512
USE_LINEAR_PROJ=1
LLM_TYPE="bert"
IMG_SIZE=336
CROP_SIZE=336

CSV_PATH="/home/dhao4/workspace/hjj_workspace/data/data.csv"
#csv文件地址
IMG_ROOT="/home/dhao4/workspace/hjj_workspace/data/images_png"
#图片地址
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

PRETRAINED_ENCODER="$(cd "$(dirname "$0")" && pwd)/pretrain_model/last.ckpt"
NUM_FOLDS=0
BASE_EXP_NAME="glam_kfold_ft"

PRED_THRESHOLD=0.5


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
RESULTS_DIR="${RUN_OUTPUT_DIR}/logs"
PER_MODEL_CSV_DIR="${RUN_OUTPUT_DIR}/per_model_predictions"
FULL_REPORT_PATH="${RUN_OUTPUT_DIR}/full_eval_report.txt"
FULL_PRED_CSV="${RUN_OUTPUT_DIR}/ensemble_predictions.csv"
mkdir -p "${RESULTS_DIR}" "${PER_MODEL_CSV_DIR}"

LINEAR_PROJ_FLAG=()
if [[ "${USE_LINEAR_PROJ}" == "1" ]]; then
  LINEAR_PROJ_FLAG+=(--linear_proj)
fi

PROGRESS_BAR_FLAG=()
if [[ "${SHOW_PROGRESS_BAR}" != "1" ]]; then
  PROGRESS_BAR_FLAG+=(--no_progress_bar)
fi

EARLY_STOP_FLAG=()
if [[ "${EARLY_STOP}" == "1" ]]; then
  EARLY_STOP_FLAG+=(--early_stop --early_stop_patience "${EARLY_STOP_PATIENCE}" --early_stop_min_epochs "${EARLY_STOP_MIN_EPOCHS}")
fi

BALANCE_TRAINING_FLAG=()
if [[ "${BALANCE_TRAINING}" == "1" ]]; then
  BALANCE_TRAINING_FLAG+=(--balance_training)
fi

POS_WEIGHT_FLAG=()
if [[ -n "${BCE_POS_WEIGHT}" ]]; then
  POS_WEIGHT_FLAG+=(--pos_weight "${BCE_POS_WEIGHT}")
fi

VAL_SPLIT_FLAG=()
FULL_EVAL_VAL_SPLIT_FLAG=()
if [[ "${VAL_SPLIT}" == "1" ]]; then
  VAL_SPLIT_FLAG+=(--rsna_val_split)
  FULL_EVAL_VAL_SPLIT_FLAG+=(--val_split)
fi

echo "Root dir                : ${ROOT_DIR}"
echo "GPU_ID                  : ${GPU_ID}"
echo "CUDA_VISIBLE_DEVICES    : ${CUDA_VISIBLE_DEVICES}"
echo "CSV_PATH                : ${CSV_PATH}"
echo "IMG_ROOT                : ${IMG_ROOT}"
echo "PATH_PATTERN            : ${PATH_PATTERN}"
echo "PRETRAINED_ENCODER      : ${PRETRAINED_ENCODER}"
echo "IMG_ENCODER             : ${IMG_ENCODER}"
echo "EMB_DIM                 : ${EMB_DIM}"
echo "USE_LINEAR_PROJ         : ${USE_LINEAR_PROJ}"
echo "MAX_EPOCHS              : ${MAX_EPOCHS}"
echo "WARM_UP (epochs)        : ${WARM_UP}"
echo "SHOW_PROGRESS_BAR       : ${SHOW_PROGRESS_BAR}"
echo "EARLY_STOP              : ${EARLY_STOP}"
echo "EARLY_STOP_PATIENCE     : ${EARLY_STOP_PATIENCE}"
echo "EARLY_STOP_MIN_EPOCHS   : ${EARLY_STOP_MIN_EPOCHS}"
echo "MONITOR_METRIC          : ${MONITOR_METRIC}"
echo "BALANCE_TRAINING        : ${BALANCE_TRAINING}"
echo "BCE_POS_WEIGHT          : ${BCE_POS_WEIGHT:-auto per fold}"
echo "VAL_SPLIT               : ${VAL_SPLIT}"
echo "VAL_FRACTION            : ${VAL_FRACTION}"
echo "VAL_MAX_FRACTION        : ${VAL_MAX_FRACTION}"
echo "VAL_MIN_POS_PATIENTS    : ${VAL_MIN_POSITIVE_PATIENTS}"
echo "VAL_RANDOM_STATE        : ${VAL_RANDOM_STATE}"
echo "RUN_OUTPUT_DIR          : ${RUN_OUTPUT_DIR}"
echo "RESULTS_DIR             : ${RESULTS_DIR}"
echo "FULL_REPORT_PATH        : ${FULL_REPORT_PATH}"
echo "FULL_PRED_CSV           : ${FULL_PRED_CSV}"
echo "PER_MODEL_CSV_DIR       : ${PER_MODEL_CSV_DIR}"

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "[ERROR] CSV file not found: ${CSV_PATH}"
  exit 1
fi
if [[ ! -d "${IMG_ROOT}" ]]; then
  echo "[ERROR] IMG_ROOT directory not found: ${IMG_ROOT}"
  exit 1
fi
if [[ ! -f "${PRETRAINED_ENCODER}" ]]; then
  echo "[ERROR] PRETRAINED_ENCODER checkpoint not found: ${PRETRAINED_ENCODER}"
  exit 1
fi

echo "=============================="
if [[ "${NUM_FOLDS}" == "0" ]]; then
  echo "Stage 1/3: single-model training"
else
  echo "Stage 1/3: ${NUM_FOLDS}-fold training"
fi
echo "=============================="

TRAIN_FOLD_START=0
TRAIN_FOLD_END=${NUM_FOLDS}
if [[ "${NUM_FOLDS}" == "0" ]]; then
  TRAIN_FOLD_END=1
fi

for ((FOLD=TRAIN_FOLD_START; FOLD<TRAIN_FOLD_END; FOLD++)); do
  EXP_NAME="${BASE_EXP_NAME}_${RUN_TAG}_fold${FOLD}"
  TRAIN_LOG="${RESULTS_DIR}/fold${FOLD}_train.log"
  echo "---- Training fold ${FOLD} (${EXP_NAME}) ----"

  python -u train.py \
    --experiment_name "${EXP_NAME}" \
    "${PROGRESS_BAR_FLAG[@]}" \
    --train_by_epoch \
    --rsna_mammo \
    --img_cls_ft \
    --llm_type "${LLM_TYPE}" \
    --img_encoder "${IMG_ENCODER}" \
    --emb_dim "${EMB_DIM}" \
    "${LINEAR_PROJ_FLAG[@]}" \
    --pretrained_encoder "${PRETRAINED_ENCODER}" \
    --k_fold "${NUM_FOLDS}" \
    --fold "${FOLD}" \
    --devices "${NUM_GPUS}" \
    --strategy auto \
    --precision "${PRECISION}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --learning_rate "${LEARNING_RATE}" \
    --max_epochs "${MAX_EPOCHS}" \
    --warm_up "${WARM_UP}" \
    --monitor_metric "${MONITOR_METRIC}" \
    --output_dir "${RUN_OUTPUT_DIR}" \
    --img_size "${IMG_SIZE}" \
    --crop_size "${CROP_SIZE}" \
    --num_classes 1 \
    --weighted_binary \
    "${POS_WEIGHT_FLAG[@]}" \
    "${BALANCE_TRAINING_FLAG[@]}" \
    "${VAL_SPLIT_FLAG[@]}" \
    --rsna_val_fraction "${VAL_FRACTION}" \
    --rsna_val_max_fraction "${VAL_MAX_FRACTION}" \
    --rsna_val_min_positive_patients "${VAL_MIN_POSITIVE_PATIENTS}" \
    --rsna_val_random_state "${VAL_RANDOM_STATE}" \
    --split_source "${SPLIT_SOURCE}" \
    --rsna_cohort_col "${COHORT_COL}" \
    --train_cohorts "${TRAIN_COHORTS}" \
    --test_cohorts "${TEST_COHORTS}" \
    --output_split_col "${OUTPUT_SPLIT_COL}" \
    --rsna_csv_path "${CSV_PATH}" \
    --rsna_img_root "${IMG_ROOT}" \
    --rsna_path_pattern "${PATH_PATTERN}" \
    --rsna_patient_col "${PATIENT_COL}" \
    --rsna_image_col "${IMAGE_COL}" \
    --rsna_label_col "${LABEL_COL}" \
    --rsna_split_col "${SPLIT_COL}" \
    --rsna_train_split_value "${TRAIN_SPLIT_VALUE}" \
    --rsna_test_split_value "${TEST_SPLIT_VALUE}" \
    "${EARLY_STOP_FLAG[@]}" 2>&1 | tee "${TRAIN_LOG}"
done

echo "=============================="
if [[ "${NUM_FOLDS}" == "0" ]]; then
  echo "Stage 2/3: Full-data eval with single best classifier"
else
  echo "Stage 2/3: Full-data eval with ${NUM_FOLDS} best classifiers"
fi
echo "=============================="

BEST_CKPTS=()
for ((FOLD=TRAIN_FOLD_START; FOLD<TRAIN_FOLD_END; FOLD++)); do
  EXP_NAME="${BASE_EXP_NAME}_${RUN_TAG}_fold${FOLD}"
  CKPT_DIR="$(ls -dt logs/ckpts/GLAM/*"${EXP_NAME}" 2>/dev/null | head -n 1 || true)"
  if [[ -z "${CKPT_DIR}" ]]; then
    echo "[ERROR] No checkpoint dir found for ${EXP_NAME}"
    exit 1
  fi

  BEST_CKPT_YAML="${CKPT_DIR}/best_ckpts.yaml"
  CKPT_PATH=""
  if [[ -f "${BEST_CKPT_YAML}" && -s "${BEST_CKPT_YAML}" ]]; then
    CKPT_PATH="$(awk -F': ' 'NR==1{best=$2;path=$1} $2>best{best=$2;path=$1} END{print path}' "${BEST_CKPT_YAML}")"
  fi
  if [[ -z "${CKPT_PATH}" || ! -f "${CKPT_PATH}" ]]; then
    CKPT_PATH="${CKPT_DIR}/last.ckpt"
  fi
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[ERROR] No usable checkpoint found in ${CKPT_DIR}"
    exit 1
  fi
  BEST_CKPTS+=("${CKPT_PATH}")

  EVAL_LOG="${RESULTS_DIR}/fold${FOLD}_full_eval.log"
  echo "---- Full-data eval fold ${FOLD} ----"
  echo "CKPT: ${CKPT_PATH}"

  python -u train.py \
    --eval \
    "${PROGRESS_BAR_FLAG[@]}" \
    --save_prediction \
    --rsna_mammo \
    --img_cls_ft \
    --llm_type "${LLM_TYPE}" \
    --img_encoder "${IMG_ENCODER}" \
    --emb_dim "${EMB_DIM}" \
    "${LINEAR_PROJ_FLAG[@]}" \
    --devices 1 \
    --precision "${PRECISION}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --k_fold 0 \
    --fold 0 \
    --rsna_use_all_data \
    --pretrained_model "${CKPT_PATH}" \
    --output_dir "${RUN_OUTPUT_DIR}" \
    --num_classes 1 \
    --weighted_binary \
    "${POS_WEIGHT_FLAG[@]}" \
    --split_source "${SPLIT_SOURCE}" \
    --rsna_cohort_col "${COHORT_COL}" \
    --train_cohorts "${TRAIN_COHORTS}" \
    --test_cohorts "${TEST_COHORTS}" \
    --output_split_col "${OUTPUT_SPLIT_COL}" \
    --rsna_csv_path "${CSV_PATH}" \
    --rsna_img_root "${IMG_ROOT}" \
    --rsna_path_pattern "${PATH_PATTERN}" \
    --rsna_patient_col "${PATIENT_COL}" \
    --rsna_image_col "${IMAGE_COL}" \
    --rsna_label_col "${LABEL_COL}" \
    --rsna_split_col "${SPLIT_COL}" \
    --rsna_train_split_value "${TRAIN_SPLIT_VALUE}" \
    --rsna_test_split_value "${TEST_SPLIT_VALUE}" 2>&1 | tee "${EVAL_LOG}"
done

echo "=============================="
echo "Stage 3/3: Export per-model CSVs and ensemble CSV"
echo "=============================="

python -u full_eval_5fold.py \
  --csv_path "${CSV_PATH}" \
  --img_root "${IMG_ROOT}" \
  --path_pattern "${PATH_PATTERN}" \
  --patient_col "${PATIENT_COL}" \
  --image_col "${IMAGE_COL}" \
  --label_col "${LABEL_COL}" \
  --split_col "${SPLIT_COL}" \
  --cohort_col "${COHORT_COL}" \
  --split_source "${SPLIT_SOURCE}" \
  --test_split_value "${TEST_SPLIT_VALUE}" \
  --train_split_value "${TRAIN_SPLIT_VALUE}" \
  --train_cohorts "${TRAIN_COHORTS}" \
  --test_cohorts "${TEST_COHORTS}" \
  "${FULL_EVAL_VAL_SPLIT_FLAG[@]}" \
  --val_fraction "${VAL_FRACTION}" \
  --val_max_fraction "${VAL_MAX_FRACTION}" \
  --val_min_positive_patients "${VAL_MIN_POSITIVE_PATIENTS}" \
  --val_random_state "${VAL_RANDOM_STATE}" \
  --output_split_col "${OUTPUT_SPLIT_COL}" \
  --k_fold "${NUM_FOLDS}" \
  --threshold "${PRED_THRESHOLD}" \
  --ckpt_paths "${BEST_CKPTS[@]}" \
  --output_report "${FULL_REPORT_PATH}" \
  --output_csv "${FULL_PRED_CSV}" \
  --per_model_output_dir "${PER_MODEL_CSV_DIR}"

echo "All done."
echo "Run output dir: ${RUN_OUTPUT_DIR}"
echo "Report: ${FULL_REPORT_PATH}"
echo "Ensemble CSV: ${FULL_PRED_CSV}"
echo "Per-model CSV dir: ${PER_MODEL_CSV_DIR}"
