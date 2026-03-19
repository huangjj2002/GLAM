set -euo pipefail
WANDB_MODE="offline"
GPU_ID=0
# 指定使用的GPU
RUN_DEVICES=1
NUM_WORKERS=0
PRECISION="32"
CV_FOLDS=5
IMG_ENCODER="auto"
DEFAULT_IMG_ENCODER="dinov2_vitb14_reg"
LLM_TYPE="bert"
LEARNING_RATE=1e-4
MAX_EPOCHS=100 #指定运行的epoch数目
WARM_UP_EPOCHS=1
IMG_SIZE=336
CROP_SIZE=336
BATCH_SIZE=8
BASE_EXP_NAME="5fold_train_validation_test"
PRED_THRESHOLD=0.5


CSV_PATH="/mnt/f/data/train_with_test_data.csv"
#指定csv文件的路径
IMG_ROOT="/mnt/f/data/images_png"
#指定图片的路径
PATH_PATTERN="{pid}/{iid}"
RSNA_PATIENT_COL="patient_id"
RSNA_IMAGE_COL="image_id"
RSNA_LABEL_COL="cancer"
RSNA_SPLIT_COL="split"
RSNA_TRAIN_SPLIT_VALUE="training"
RSNA_TEST_SPLIT_VALUE="test"


PRETRAINED_MODEL_DIR="$(cd "$(dirname "$0")" && pwd)/pretrain_model"

PRETRAINED_ENCODER_CKPT=""


ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${ROOT_DIR}/logs/results/${BASE_EXP_NAME}_${RUN_TAG}"
SUMMARY_CSV="${ROOT_DIR}/results_summary_dinov2_${RUN_TAG}.csv"
PRED_EXPORT_DIR="${ROOT_DIR}/${BASE_EXP_NAME}_predictions_${RUN_TAG}"

mkdir -p "${RESULTS_DIR}"

export WANDB_MODE
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ "${CV_FOLDS}" -lt 2 ]]; then
  echo "[ERROR] CV_FOLDS must be >= 2."
  exit 1
fi
FOLDS=()
for ((i=0; i<CV_FOLDS; i++)); do
  FOLDS+=("${i}")
done

resolve_best_ckpt_from_yaml() {
  local best_yaml="$1"
  local fallback_dir="$2"
  local candidate=""

  if [[ -f "${best_yaml}" && -s "${best_yaml}" ]]; then
    candidate="$(awk -F': ' 'NR==1{best=$2;path=$1} $2>best{best=$2;path=$1} END{print path}' "${best_yaml}")"
    if [[ -n "${candidate}" ]]; then
      if [[ -f "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      candidate="${fallback_dir}/$(basename "${candidate}")"
      if [[ -f "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    fi
  fi
  return 1
}

resolve_pretrained_encoder_ckpt() {
  local ckpt=""
  if [[ -n "${PRETRAINED_MODEL_DIR}" ]]; then
    if [[ ! -d "${PRETRAINED_MODEL_DIR}" ]]; then
      return 1
    fi
    ckpt="$(resolve_best_ckpt_from_yaml "${PRETRAINED_MODEL_DIR}/best_ckpts.yaml" "${PRETRAINED_MODEL_DIR}" || true)"
    if [[ -z "${ckpt}" && -f "${PRETRAINED_MODEL_DIR}/last.ckpt" ]]; then
      ckpt="${PRETRAINED_MODEL_DIR}/last.ckpt"
    fi
  elif [[ -n "${PRETRAINED_ENCODER_CKPT}" && -f "${PRETRAINED_ENCODER_CKPT}" ]]; then
    ckpt="${PRETRAINED_ENCODER_CKPT}"
  fi

  if [[ -n "${ckpt}" ]]; then
    printf '%s\n' "${ckpt}"
    return 0
  fi
  return 1
}

resolve_img_encoder_from_ckpt() {
  local ckpt_path="$1"
  python -c "import sys, torch; ckpt=torch.load(sys.argv[1], map_location='cpu'); hp=ckpt.get('hyper_parameters', {}); print(hp.get('img_encoder',''))" "${ckpt_path}" 2>/dev/null | tr -d '\r' | tail -n 1
}

resolve_fold_ckpt() {
  local exp_name="$1"
  local ckpt_dir=""
  local ckpt=""

  ckpt_dir="$(ls -dt "${ROOT_DIR}"/logs/ckpts/GLAM/*"${exp_name}" 2>/dev/null | head -n 1 || true)"
  if [[ -z "${ckpt_dir}" ]]; then
    return 1
  fi

  ckpt="$(resolve_best_ckpt_from_yaml "${ckpt_dir}/best_ckpts.yaml" "${ckpt_dir}" || true)"
  if [[ -z "${ckpt}" && -f "${ckpt_dir}/last.ckpt" ]]; then
    ckpt="${ckpt_dir}/last.ckpt"
  fi

  if [[ -n "${ckpt}" && -f "${ckpt}" ]]; then
    printf '%s\n' "${ckpt}"
    return 0
  fi
  return 1
}

PRETRAINED_ARGS=()
PRETRAINED_CKPT="$(resolve_pretrained_encoder_ckpt || true)"
if [[ -n "${PRETRAINED_CKPT}" ]]; then
  PRETRAINED_ARGS+=(--pretrained_encoder "${PRETRAINED_CKPT}")
else
  echo "[ERROR] No usable pretrained encoder checkpoint found."
  echo "Checked PRETRAINED_MODEL_DIR='${PRETRAINED_MODEL_DIR}' and PRETRAINED_ENCODER_CKPT='${PRETRAINED_ENCODER_CKPT}'."
  exit 1
fi

if [[ "${IMG_ENCODER}" == "auto" || -z "${IMG_ENCODER}" ]]; then
  IMG_ENCODER="$(resolve_img_encoder_from_ckpt "${PRETRAINED_CKPT}" || true)"
fi
if [[ -z "${IMG_ENCODER}" ]]; then
  IMG_ENCODER="${DEFAULT_IMG_ENCODER}"
fi

COMMON_MODEL_ARGS=(
  --rsna_mammo
  --img_cls_ft
  --llm_type "${LLM_TYPE}"
  --img_encoder "${IMG_ENCODER}"
  --num_classes 1
  --weighted_binary
  --img_size "${IMG_SIZE}"
  --crop_size "${CROP_SIZE}"
)

COMMON_DATA_ARGS=(
  --rsna_csv_path "${CSV_PATH}"
  --rsna_img_root "${IMG_ROOT}"
  --rsna_path_pattern "${PATH_PATTERN}"
  --rsna_patient_col "${RSNA_PATIENT_COL}"
  --rsna_image_col "${RSNA_IMAGE_COL}"
  --rsna_label_col "${RSNA_LABEL_COL}"
  --rsna_split_col "${RSNA_SPLIT_COL}"
  --rsna_train_split_value "${RSNA_TRAIN_SPLIT_VALUE}"
  --rsna_test_split_value "${RSNA_TEST_SPLIT_VALUE}"
)

echo "Root dir              : ${ROOT_DIR}"
echo "CSV_PATH              : ${CSV_PATH}"
echo "IMG_ROOT              : ${IMG_ROOT}"
echo "PATH_PATTERN          : ${PATH_PATTERN}"
echo "IMG_ENCODER           : ${IMG_ENCODER}"
echo "LLM_TYPE              : ${LLM_TYPE}"
echo "CV_FOLDS              : ${CV_FOLDS}"
echo "GPU_ID                : ${GPU_ID}"
echo "RUN_DEVICES           : ${RUN_DEVICES}"
echo "WARM_UP_EPOCHS        : ${WARM_UP_EPOCHS}"
echo "PRED_THRESHOLD        : ${PRED_THRESHOLD}"
echo "CUDA_VISIBLE_DEVICES  : ${CUDA_VISIBLE_DEVICES}"
echo "RESULTS_DIR           : ${RESULTS_DIR}"
echo "SUMMARY_CSV           : ${SUMMARY_CSV}"
echo "PRED_EXPORT_DIR       : ${PRED_EXPORT_DIR}"
echo "PRETRAINED_ENCODER    : ${PRETRAINED_CKPT}"

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "[ERROR] CSV file not found: ${CSV_PATH}"
  exit 1
fi
if [[ ! -d "${IMG_ROOT}" ]]; then
  echo "[ERROR] IMG_ROOT directory not found: ${IMG_ROOT}"
  exit 1
fi

echo "=============================="
echo "Stage 1: ${CV_FOLDS}-fold cross-validation (train + test)"
echo "=============================="
for FOLD in "${FOLDS[@]}"; do
  EXP_NAME="${BASE_EXP_NAME}_fold${FOLD}"
  TRAIN_LOG="${RESULTS_DIR}/fold${FOLD}_train.log"
  TEST_LOG="${RESULTS_DIR}/fold${FOLD}_test.log"
  echo "---- Fold ${FOLD}: training (${EXP_NAME}) ----"

  python -u "${ROOT_DIR}/train.py" \
    --experiment_name "${EXP_NAME}" \
    --no_progress_bar \
    --train_by_epoch \
    --k_fold "${CV_FOLDS}" \
    --fold "${FOLD}" \
    --devices "${RUN_DEVICES}" \
    --strategy auto \
    --precision "${PRECISION}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --learning_rate "${LEARNING_RATE}" \
    --max_epochs "${MAX_EPOCHS}" \
    --warm_up_epochs "${WARM_UP_EPOCHS}" \
    --balance_training \
    "${COMMON_MODEL_ARGS[@]}" \
    "${COMMON_DATA_ARGS[@]}" \
    "${PRETRAINED_ARGS[@]}" 2>&1 | tee "${TRAIN_LOG}"

  CKPT_PATH="$(resolve_fold_ckpt "${EXP_NAME}" || true)"
  if [[ -z "${CKPT_PATH}" ]]; then
    echo "[WARN] No usable checkpoint found for ${EXP_NAME}, skip test for fold ${FOLD}."
    continue
  fi

  echo "---- Fold ${FOLD}: testing ----"
  echo "CKPT: ${CKPT_PATH}"

  python -u "${ROOT_DIR}/train.py" \
    --eval \
    --no_progress_bar \
    --devices "${RUN_DEVICES}" \
    --precision "${PRECISION}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --k_fold "${CV_FOLDS}" \
    --fold "${FOLD}" \
    --pretrained_model "${CKPT_PATH}" \
    "${PRETRAINED_ARGS[@]}" \
    "${COMMON_MODEL_ARGS[@]}" \
    "${COMMON_DATA_ARGS[@]}" 2>&1 | tee "${TEST_LOG}"
done

echo "=============================="
echo "Stage 2: ${CV_FOLDS}-fold full-data prediction export"
echo "=============================="
for FOLD in "${FOLDS[@]}"; do
  EXP_NAME="${BASE_EXP_NAME}_fold${FOLD}"
  CKPT_PATH="$(resolve_fold_ckpt "${EXP_NAME}" || true)"
  if [[ -z "${CKPT_PATH}" ]]; then
    echo "[WARN] No usable checkpoint found for ${EXP_NAME}, skip export for fold ${FOLD}."
    continue
  fi

  echo "---- Fold ${FOLD}: export train/valid/test predictions ----"
  python -u "${ROOT_DIR}/export_5fold_predictions.py" \
    --base_exp_name "${BASE_EXP_NAME}" \
    --num_folds "${CV_FOLDS}" \
    --fold_indices "${FOLD}" \
    --threshold "${PRED_THRESHOLD}" \
    --output_dir "${PRED_EXPORT_DIR}" \
    --no_progress_bar \
    --devices "${RUN_DEVICES}" \
    --precision "${PRECISION}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --k_fold "${CV_FOLDS}" \
    "${PRETRAINED_ARGS[@]}" \
    "${COMMON_MODEL_ARGS[@]}" \
    "${COMMON_DATA_ARGS[@]}"
done

python "${ROOT_DIR}/summarize_rsna_results.py" \
  --results_dir "${RESULTS_DIR}" \
  --output_csv "${SUMMARY_CSV}" \
  --num_folds "${CV_FOLDS}"

echo "All done."
echo "Summary CSV       : ${SUMMARY_CSV}"
echo "Prediction CSV dir: ${PRED_EXPORT_DIR}"

