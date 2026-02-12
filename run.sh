
set -euo pipefail
export WANDB_MODE="offline"
GPU_ID=0 #用来指定使用的GPU编号
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

NUM_GPUS=1
BATCH_SIZE=8
NUM_WORKERS=0
PRECISION="32"


IMG_ENCODER="facebook/dinov2-base"


CSV_PATH="/mnt/f/data/train_with_test_data.csv"
#指定数据的csv文件

IMG_ROOT="/mnt/f/data/images_png"
#指定png文件的根目录
PATH_PATTERN="{pid}/{iid}"


BASE_EXP_NAME="5fold_train_validation_test"


LEARNING_RATE=1e-4
MAX_EPOCHS=5
#最大训练轮数
WARM_UP=102
IMG_SIZE=336
CROP_SIZE=336
LLM_TYPE="bert"


ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${ROOT_DIR}/logs/results/${BASE_EXP_NAME}_${RUN_TAG}"
mkdir -p "${RESULTS_DIR}"
SUMMARY_CSV="${ROOT_DIR}/results_summary_dinov2_${RUN_TAG}.csv"

echo "Root dir       : ${ROOT_DIR}"
echo "CSV_PATH       : ${CSV_PATH}"
echo "IMG_ROOT       : ${IMG_ROOT}"
echo "PATH_PATTERN   : ${PATH_PATTERN}"
echo "IMG_ENCODER    : ${IMG_ENCODER}"
echo "LLM_TYPE       : ${LLM_TYPE}"
echo "WANDB_MODE     : ${WANDB_MODE}"
echo "GPU_ID         : ${GPU_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "RESULTS_DIR    : ${RESULTS_DIR}"

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "[ERROR] CSV file not found: ${CSV_PATH}"
  exit 1
fi

if [[ ! -d "${IMG_ROOT}" ]]; then
  echo "[ERROR] IMG_ROOT directory not found: ${IMG_ROOT}"
  echo "Please set IMG_ROOT to your image root, e.g.:"
  echo "  IMG_ROOT=/path/to/images bash run_dinov2_rsna_5fold.sh"
  exit 1
fi

echo "=============================="
echo "Stage 1: 5-fold training"
echo "split=training used for train/val pool"
echo "=============================="

for FOLD in 0 1 2 3 4; do
  EXP_NAME="${BASE_EXP_NAME}_fold${FOLD}"
  echo "---- Training fold ${FOLD} (${EXP_NAME}) ----"
  TRAIN_LOG="${RESULTS_DIR}/fold${FOLD}_train.log"

  python -u train.py \
    --experiment_name "${EXP_NAME}" \
    --no_progress_bar \
    --train_by_epoch \
    --rsna_mammo \
    --img_cls_ft \
    --llm_type "${LLM_TYPE}" \
    --img_encoder "${IMG_ENCODER}" \
    --k_fold 5 \
    --fold "${FOLD}" \
    --devices "${NUM_GPUS}" \
    --strategy auto \
    --precision "${PRECISION}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --learning_rate "${LEARNING_RATE}" \
    --max_epochs "${MAX_EPOCHS}" \
    --warm_up "${WARM_UP}" \
    --img_size "${IMG_SIZE}" \
    --crop_size "${CROP_SIZE}" \
    --num_classes 1 \
    --weighted_binary \
    --balance_training \
    --rsna_csv_path "${CSV_PATH}" \
    --rsna_img_root "${IMG_ROOT}" \
    --rsna_path_pattern "${PATH_PATTERN}" \
    --rsna_patient_col "patient_id" \
    --rsna_image_col "image_id" \
    --rsna_label_col "cancer" \
    --rsna_split_col "split" \
    --rsna_train_split_value "training" \
    --rsna_test_split_value "test" 2>&1 | tee "${TRAIN_LOG}"
done

echo "=============================="
echo "Stage 2: per-fold testing on split=test"
echo "=============================="

for FOLD in 0 1 2 3 4; do
  EXP_NAME="${BASE_EXP_NAME}_fold${FOLD}"
  CKPT_DIR="$(ls -dt logs/ckpts/GLAM/*"${EXP_NAME}" 2>/dev/null | head -n 1 || true)"

  if [[ -z "${CKPT_DIR}" ]]; then
    echo "[WARN] No checkpoint dir found for ${EXP_NAME}, skipping."
    continue
  fi


  BEST_CKPT_YAML="${CKPT_DIR}/best_ckpts.yaml"
  CKPT_PATH=""
  if [[ -f "${BEST_CKPT_YAML}" && -s "${BEST_CKPT_YAML}" ]]; then
    # best_ckpts.yaml now stores val_AUROC, so larger is better.
    CKPT_PATH="$(awk -F': ' 'NR==1{best=$2;path=$1} $2>best{best=$2;path=$1} END{print path}' "${BEST_CKPT_YAML}")"
  fi
  if [[ -z "${CKPT_PATH}" || ! -f "${CKPT_PATH}" ]]; then
    CKPT_PATH="${CKPT_DIR}/last.ckpt"
  fi
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[WARN] No usable checkpoint found in ${CKPT_DIR}, skipping."
    continue
  fi

  echo "---- Testing fold ${FOLD} ----"
  echo "CKPT: ${CKPT_PATH}"
  TEST_LOG="${RESULTS_DIR}/fold${FOLD}_test.log"

  python -u train.py \
    --eval \
    --no_progress_bar \
    --rsna_mammo \
    --img_cls_ft \
    --llm_type "${LLM_TYPE}" \
    --img_encoder "${IMG_ENCODER}" \
    --devices 1 \
    --precision "${PRECISION}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --k_fold 5 \
    --fold "${FOLD}" \
    --pretrained_model "${CKPT_PATH}" \
    --num_classes 1 \
    --weighted_binary \
    --rsna_csv_path "${CSV_PATH}" \
    --rsna_img_root "${IMG_ROOT}" \
    --rsna_path_pattern "${PATH_PATTERN}" \
    --rsna_patient_col "patient_id" \
    --rsna_image_col "image_id" \
    --rsna_label_col "cancer" \
    --rsna_split_col "split" \
    --rsna_train_split_value "training" \
    --rsna_test_split_value "test" 2>&1 | tee "${TEST_LOG}"
done

python "${ROOT_DIR}/summarize_rsna_results.py" \
  --results_dir "${RESULTS_DIR}" \
  --output_csv "${SUMMARY_CSV}"

echo "All done."
echo "Summary CSV: ${SUMMARY_CSV}"
