#!/bin/bash
set -euo pipefail

# ===========================================================================
# Round-2 EDL lambda_kl sweep
# Fixed pos_weight mode: auto neg/pos from edl_train.py (no --pos_weight flag).
# Runs three KL regularization strengths in parallel:
#   0.01 -> GPU 1
#   0.1  -> GPU 2
#   0.5  -> GPU 3
# ===========================================================================

# ---------------------------------------------------------------------------
# Editable configuration
# ---------------------------------------------------------------------------
GPU_ID=2
GPU_IDS=("1" "2" "3")
RUN_IN_PARALLEL=1
MAX_EPOCHS=25
WARM_UP=1
NUM_GPUS=1
BATCH_SIZE=48
NUM_WORKERS=4
PRECISION="32"
LEARNING_RATE=1e-4
SHOW_PROGRESS_BAR=1
PROGRESS_PRINT_INTERVAL=50

EARLY_STOP=1
EARLY_STOP_PATIENCE=3
EARLY_STOP_MIN_EPOCHS=12
MONITOR_METRIC="val_AUROC"

OUTPUT_DIR="outputs"
ROUND2_BASE_EXP_NAME="edl_round2_lambda"
LAMBDA_KL_VALUES=("0.01" "0.1" "0.5")
POS_WEIGHT_MODE="auto"

IMG_ENCODER="dinov2_vitb14_reg"
EMB_DIM=512
USE_LINEAR_PROJ=1
IMG_SIZE=336
CROP_SIZE=336

EVIDENCE_TYPE="softplus"
EDL_LOSS_TYPE="digamma"
ANNEALING_STEP=10
FREEZE_BACKBONE=0
# Keep this disabled for this sweep so only lambda_kl changes.
BALANCE_TRAINING=0

CSV_PATH="${CSV_PATH:-/home/dhao4/workspace/hjj_workspace/data/data.csv}"
IMG_ROOT="${IMG_ROOT:-/home/dhao4/workspace/hjj_workspace/data/images_png}"
DEFAULT_PATH_PATTERN='{pid}/{iid}'
PATH_PATTERN="${PATH_PATTERN:-${DEFAULT_PATH_PATTERN}}"

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

# Use the Python from the active environment by default. Override when needed:
#   PYTHON_BIN="/mnt/e/conda_env/torch/python.exe" bash run_edl_round2_lambda_sweep.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

# Set DRY_RUN=1 to create the output folders and print commands without training.
DRY_RUN="${DRY_RUN:-0}"

# Set SKIP_PATH_CHECKS=1 only for command-inspection dry runs on a different host.
SKIP_PATH_CHECKS="${SKIP_PATH_CHECKS:-0}"

# ---------------------------------------------------------------------------
# Derived flags and setup
# ---------------------------------------------------------------------------
export WANDB_MODE="offline"
# CUDA_VISIBLE_DEVICES is set per experiment so each lambda can use a different GPU.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"

if [[ "${OUTPUT_DIR}" = /* ]]; then
  OUTPUT_ROOT="${OUTPUT_DIR}"
else
  OUTPUT_ROOT="${ROOT_DIR}/${OUTPUT_DIR}"
fi
SWEEP_OUTPUT_DIR="${OUTPUT_ROOT}/${ROUND2_BASE_EXP_NAME}_${RUN_TAG}"
mkdir -p "${SWEEP_OUTPUT_DIR}"

if [[ "${NUM_FOLDS}" == "0" ]]; then
  FOLDS_TO_RUN="0"
fi

if [[ "${BALANCE_TRAINING}" != "0" ]]; then
  echo "[ERROR] BALANCE_TRAINING must be 0 for this lambda_kl sweep."
  exit 1
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

VAL_SPLIT_FLAG=()
if [[ "${VAL_SPLIT}" == "1" ]]; then
  VAL_SPLIT_FLAG+=(--rsna_val_split)
fi

PROGRESS_BAR_FLAG=()
if [[ "${SHOW_PROGRESS_BAR}" != "1" ]]; then
  PROGRESS_BAR_FLAG+=(--no_progress_bar)
fi

FOLDS_TO_RUN_FLAG=()
if [[ -n "${FOLDS_TO_RUN}" ]]; then
  FOLDS_TO_RUN_FLAG+=(--folds_to_run "${FOLDS_TO_RUN}")
fi

check_paths() {
  if [[ "${SKIP_PATH_CHECKS}" == "1" ]]; then
    echo "[WARN] SKIP_PATH_CHECKS=1; not validating CSV_PATH, IMG_ROOT, or PRETRAINED_MODEL."
    return
  fi
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
}

check_path_pattern() {
  PATH_PATTERN="${PATH_PATTERN}" "${PYTHON_BIN}" - <<'PY'
import os

pattern = os.environ["PATH_PATTERN"]
try:
    rendered = pattern.format(pid="sample_pid", iid="sample_iid")
except Exception as exc:
    raise SystemExit(f"[ERROR] Invalid PATH_PATTERN={pattern!r}: {exc}")
print(f"[OK] PATH_PATTERN={pattern!r} -> {rendered!r}")
PY
}

print_config() {
  echo "============================================================"
  echo "Round-2 EDL lambda_kl sweep"
  echo "============================================================"
  echo "Root dir                : ${ROOT_DIR}"
  echo "Python                  : ${PYTHON_BIN}"
  echo "GPU_ID (sequential)     : ${GPU_ID}"
  echo "GPU_IDS (parallel)      : ${GPU_IDS[*]}"
  echo "RUN_IN_PARALLEL         : ${RUN_IN_PARALLEL}"
  echo "CSV_PATH                : ${CSV_PATH}"
  echo "IMG_ROOT                : ${IMG_ROOT}"
  echo "PATH_PATTERN            : ${PATH_PATTERN}"
  echo "PRETRAINED_MODEL        : ${PRETRAINED_MODEL}"
  echo "IMG_ENCODER             : ${IMG_ENCODER}"
  echo "EMB_DIM                 : ${EMB_DIM}"
  echo "MAX_EPOCHS              : ${MAX_EPOCHS}"
  echo "BATCH_SIZE              : ${BATCH_SIZE}"
  echo "EARLY_STOP              : ${EARLY_STOP}"
  echo "EARLY_STOP_MIN_EPOCHS   : ${EARLY_STOP_MIN_EPOCHS}"
  echo "MONITOR_METRIC          : ${MONITOR_METRIC}"
  echo "EDL_LOSS_TYPE           : ${EDL_LOSS_TYPE}"
  echo "EVIDENCE_TYPE           : ${EVIDENCE_TYPE}"
  echo "ANNEALING_STEP          : ${ANNEALING_STEP}"
  echo "LAMBDA_KL_VALUES        : ${LAMBDA_KL_VALUES[*]}"
  echo "POS_WEIGHT_MODE         : ${POS_WEIGHT_MODE}"
  echo "BALANCE_TRAINING        : ${BALANCE_TRAINING}"
  echo "VAL_SPLIT               : ${VAL_SPLIT}"
  echo "K_FOLD                  : ${NUM_FOLDS}"
  echo "FOLDS_TO_RUN            : ${FOLDS_TO_RUN:-all folds}"
  echo "DRY_RUN                 : ${DRY_RUN}"
  echo "SWEEP_OUTPUT_DIR        : ${SWEEP_OUTPUT_DIR}"
  echo "============================================================"
}

build_and_run_command() {
  local lambda_kl="$1"
  local lambda_output_dir="$2"
  local exp_name="$3"
  local lambda_gpu_id="$4"

  local cmd=(
    "${PYTHON_BIN}" -u edl_train.py
    --csv_path "${CSV_PATH}"
    --img_root "${IMG_ROOT}"
    --path_pattern "${PATH_PATTERN}"
    --pretrained_model "${PRETRAINED_MODEL}"
    "${PROGRESS_BAR_FLAG[@]}"
    --progress_print_interval "${PROGRESS_PRINT_INTERVAL}"
    --patient_col "${PATIENT_COL}"
    --image_col "${IMAGE_COL}"
    --label_col "${LABEL_COL}"
    --split_col "${SPLIT_COL}"
    --split_source "${SPLIT_SOURCE}"
    --cohort_col "${COHORT_COL}"
    --train_split_value "${TRAIN_SPLIT_VALUE}"
    --test_split_value "${TEST_SPLIT_VALUE}"
    --train_cohorts "${TRAIN_COHORTS}"
    --test_cohorts "${TEST_COHORTS}"
    --output_split_col "${OUTPUT_SPLIT_COL}"
    "${VAL_SPLIT_FLAG[@]}"
    --rsna_val_fraction "${VAL_FRACTION}"
    --rsna_val_max_fraction "${VAL_MAX_FRACTION}"
    --rsna_val_min_positive_patients "${VAL_MIN_POSITIVE_PATIENTS}"
    --rsna_val_random_state "${VAL_RANDOM_STATE}"
    --img_encoder "${IMG_ENCODER}"
    --emb_dim "${EMB_DIM}"
    "${LINEAR_PROJ_FLAG[@]}"
    --evidence_type "${EVIDENCE_TYPE}"
    --edl_loss_type "${EDL_LOSS_TYPE}"
    --annealing_step "${ANNEALING_STEP}"
    --lambda_kl "${lambda_kl}"
    "${FREEZE_BACKBONE_FLAG[@]}"
    --k_fold "${NUM_FOLDS}"
    "${FOLDS_TO_RUN_FLAG[@]}"
    --max_epochs "${MAX_EPOCHS}"
    --warm_up "${WARM_UP}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --learning_rate "${LEARNING_RATE}"
    --monitor_metric "${MONITOR_METRIC}"
    --devices "${NUM_GPUS}"
    --strategy auto
    --precision "${PRECISION}"
    --img_size "${IMG_SIZE}"
    --crop_size "${CROP_SIZE}"
    --output_dir "${lambda_output_dir}"
    --experiment_name "${exp_name}"
    "${EARLY_STOP_FLAG[@]}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] lambda_kl=${lambda_kl} command:"
    printf ' CUDA_VISIBLE_DEVICES=%q' "${lambda_gpu_id}"
    printf ' %q' "${cmd[@]}"
    echo
  else
    CUDA_VISIBLE_DEVICES="${lambda_gpu_id}" "${cmd[@]}"
  fi
}

check_required_outputs() {
  local lambda_output_dir="$1"
  local required=(
    "${lambda_output_dir}/per_model_predictions/edl_eval_report.txt"
    "${lambda_output_dir}/per_model_predictions/fold0_edl_predictions.csv"
    "${lambda_output_dir}/per_model_predictions/ensemble_edl_predictions.csv"
  )
  for path in "${required[@]}"; do
    if [[ ! -f "${path}" ]]; then
      echo "[ERROR] Expected output missing: ${path}"
      exit 1
    fi
  done
}

lambda_dir_name() {
  local lambda_kl="$1"
  echo "kl_${lambda_kl}"
}

run_lambda() {
  local lambda_kl="$1"
  local lambda_gpu_id="$2"
  local dir_name
  dir_name="$(lambda_dir_name "${lambda_kl}")"
  local lambda_output_dir="${SWEEP_OUTPUT_DIR}/${dir_name}"
  local exp_name="${ROUND2_BASE_EXP_NAME}_${dir_name}"
  mkdir -p "${lambda_output_dir}"

  echo ""
  echo "------------------------------------------------------------"
  echo "Running lambda_kl        : ${lambda_kl}"
  echo "Experiment name          : ${exp_name}"
  echo "Resolved pos_weight      : auto neg/pos from edl_train.py"
  echo "BALANCE_TRAINING         : ${BALANCE_TRAINING}"
  echo "CUDA_VISIBLE_DEVICES     : ${lambda_gpu_id}"
  echo "Output dir               : ${lambda_output_dir}"
  echo "------------------------------------------------------------"

  build_and_run_command "${lambda_kl}" "${lambda_output_dir}" "${exp_name}" "${lambda_gpu_id}"

  if [[ "${DRY_RUN}" != "1" ]]; then
    check_required_outputs "${lambda_output_dir}"
    echo "[OK] Required outputs found for lambda_kl=${lambda_kl}"
  fi
}

check_paths
check_path_pattern
print_config

if [[ "${RUN_IN_PARALLEL}" == "1" && "${DRY_RUN}" != "1" ]]; then
  if [[ "${#GPU_IDS[@]}" -lt "${#LAMBDA_KL_VALUES[@]}" ]]; then
    echo "[ERROR] RUN_IN_PARALLEL=1 needs at least one GPU id per lambda_kl value."
    echo "        LAMBDA_KL_VALUES=${LAMBDA_KL_VALUES[*]}"
    echo "        GPU_IDS=${GPU_IDS[*]}"
    exit 1
  fi

  pids=()
  pid_lambdas=()
  pid_logs=()

  for idx in "${!LAMBDA_KL_VALUES[@]}"; do
    lambda_kl="${LAMBDA_KL_VALUES[$idx]}"
    lambda_gpu_id="${GPU_IDS[$idx]}"
    dir_name="$(lambda_dir_name "${lambda_kl}")"
    lambda_output_dir="${SWEEP_OUTPUT_DIR}/${dir_name}"
    mkdir -p "${lambda_output_dir}"
    shell_log="${lambda_output_dir}/round2_${dir_name}.log"

    echo ""
    echo "[LAUNCH] lambda_kl=${lambda_kl} gpu=${lambda_gpu_id} log=${shell_log}"
    (
      run_lambda "${lambda_kl}" "${lambda_gpu_id}"
    ) > "${shell_log}" 2>&1 &

    pids+=("$!")
    pid_lambdas+=("${lambda_kl}")
    pid_logs+=("${shell_log}")
  done

  failed=0
  for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    lambda_kl="${pid_lambdas[$idx]}"
    shell_log="${pid_logs[$idx]}"
    if wait "${pid}"; then
      echo "[OK] lambda_kl=${lambda_kl} finished. Log: ${shell_log}"
    else
      status="$?"
      echo "[ERROR] lambda_kl=${lambda_kl} failed with exit code ${status}. Log: ${shell_log}"
      failed=1
    fi
  done

  if [[ "${failed}" != "0" ]]; then
    echo "[ERROR] One or more round-2 experiments failed."
    exit 1
  fi
else
  for idx in "${!LAMBDA_KL_VALUES[@]}"; do
    lambda_kl="${LAMBDA_KL_VALUES[$idx]}"
    lambda_gpu_id="${GPU_IDS[$idx]:-${GPU_ID}}"
    run_lambda "${lambda_kl}" "${lambda_gpu_id}"
  done
fi

echo ""
echo "Round-2 lambda_kl sweep done."
echo "Sweep output dir: ${SWEEP_OUTPUT_DIR}"
