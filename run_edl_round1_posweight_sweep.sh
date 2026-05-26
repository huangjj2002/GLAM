#!/bin/bash
set -euo pipefail

# ===========================================================================
# Round-1 EDL positive-class-weight sweep
# Runs three EDL experiments without modifying run_edl.sh:
#   none -> --pos_weight 1.0
#   sqrt -> --pos_weight sqrt(neg / pos), computed from the effective train split
#   auto -> no --pos_weight flag, so edl_train.py auto-computes neg / pos
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
ROUND1_BASE_EXP_NAME="edl_round1_posweight"
POS_WEIGHT_MODES=("none" "sqrt" "auto")

IMG_ENCODER="dinov2_vitb14_reg"
EMB_DIM=512
USE_LINEAR_PROJ=1
IMG_SIZE=336
CROP_SIZE=336

EVIDENCE_TYPE="softplus"
EDL_LOSS_TYPE="digamma"
ANNEALING_STEP=10
LAMBDA_KL=0.1
FREEZE_BACKBONE=0
# Keep this disabled for this sweep so only pos_weight changes.
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
#   PYTHON_BIN="/mnt/e/conda_env/torch/python.exe" bash run_edl_round1_posweight_sweep.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

# Set DRY_RUN=1 to create the output folders and print commands without training.
DRY_RUN="${DRY_RUN:-0}"

# Set SKIP_PATH_CHECKS=1 only for command-inspection dry runs on a different host.
SKIP_PATH_CHECKS="${SKIP_PATH_CHECKS:-0}"

# ---------------------------------------------------------------------------
# Derived flags and setup
# ---------------------------------------------------------------------------
export WANDB_MODE="offline"
# CUDA_VISIBLE_DEVICES is set per experiment so each mode can use a different GPU.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"

if [[ "${OUTPUT_DIR}" = /* ]]; then
  OUTPUT_ROOT="${OUTPUT_DIR}"
else
  OUTPUT_ROOT="${ROOT_DIR}/${OUTPUT_DIR}"
fi
SWEEP_OUTPUT_DIR="${OUTPUT_ROOT}/${ROUND1_BASE_EXP_NAME}_${RUN_TAG}"
mkdir -p "${SWEEP_OUTPUT_DIR}"

if [[ "${NUM_FOLDS}" == "0" ]]; then
  FOLDS_TO_RUN="0"
fi

if [[ "${BALANCE_TRAINING}" != "0" ]]; then
  echo "[ERROR] BALANCE_TRAINING must be 0 for this pos_weight sweep."
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

compute_sqrt_pos_weight() {
  CSV_PATH="${CSV_PATH}" \
  PATIENT_COL="${PATIENT_COL}" \
  LABEL_COL="${LABEL_COL}" \
  SPLIT_SOURCE="${SPLIT_SOURCE}" \
  SPLIT_COL="${SPLIT_COL}" \
  TRAIN_SPLIT_VALUE="${TRAIN_SPLIT_VALUE}" \
  TEST_SPLIT_VALUE="${TEST_SPLIT_VALUE}" \
  COHORT_COL="${COHORT_COL}" \
  TRAIN_COHORTS="${TRAIN_COHORTS}" \
  TEST_COHORTS="${TEST_COHORTS}" \
  NUM_FOLDS="${NUM_FOLDS}" \
  FOLDS_TO_RUN="${FOLDS_TO_RUN}" \
  VAL_SPLIT="${VAL_SPLIT}" \
  VAL_FRACTION="${VAL_FRACTION}" \
  VAL_MAX_FRACTION="${VAL_MAX_FRACTION}" \
  VAL_MIN_POSITIVE_PATIENTS="${VAL_MIN_POSITIVE_PATIENTS}" \
  VAL_RANDOM_STATE="${VAL_RANDOM_STATE}" \
  "${PYTHON_BIN}" - <<'PY'
import math
import os
import sys

import pandas as pd

from split_utils import build_split_metadata

csv_path = os.environ["CSV_PATH"]
patient_col = os.environ["PATIENT_COL"]
label_col = os.environ["LABEL_COL"]
k_fold = int(os.environ["NUM_FOLDS"])
folds_to_run = os.environ.get("FOLDS_TO_RUN", "").strip()

df = pd.read_csv(csv_path)
metadata = build_split_metadata(
    df,
    patient_col=patient_col,
    label_col=label_col,
    split_source=os.environ["SPLIT_SOURCE"],
    split_col=os.environ["SPLIT_COL"],
    train_split_value=os.environ["TRAIN_SPLIT_VALUE"],
    test_split_value=os.environ["TEST_SPLIT_VALUE"],
    cohort_col=os.environ["COHORT_COL"],
    train_cohorts=os.environ["TRAIN_COHORTS"],
    test_cohorts=os.environ["TEST_COHORTS"],
    k_fold=k_fold,
    val_split=os.environ["VAL_SPLIT"] == "1",
    val_fraction=float(os.environ["VAL_FRACTION"]),
    val_max_fraction=float(os.environ["VAL_MAX_FRACTION"]),
    val_min_positive_patients=int(os.environ["VAL_MIN_POSITIVE_PATIENTS"]),
    val_random_state=int(os.environ["VAL_RANDOM_STATE"]),
)

train_mask = metadata["train_mask"].copy()
if k_fold > 1:
    fold_items = [item.strip() for item in folds_to_run.split(",") if item.strip()]
    if len(fold_items) != 1:
        raise SystemExit(
            "sqrt pos_weight supports NUM_FOLDS=0 or one explicit FOLDS_TO_RUN value. "
            "Run folds separately for exact per-fold sqrt weights."
        )
    fold = int(fold_items[0])
    train_mask = train_mask & (metadata["fold_assignment"] != fold)

labels = df.loc[train_mask, label_col].astype(int)
num_pos = int((labels == 1).sum())
num_neg = int((labels == 0).sum())
if num_pos <= 0:
    raise SystemExit("No positive samples in effective train split; cannot compute sqrt pos_weight.")
if num_neg <= 0:
    raise SystemExit("No negative samples in effective train split; cannot compute sqrt pos_weight.")

ratio = num_neg / num_pos
sqrt_weight = math.sqrt(ratio)
print(f"{num_neg},{num_pos},{ratio:.12g},{sqrt_weight:.12g}")
PY
}

print_config() {
  echo "============================================================"
  echo "Round-1 EDL pos_weight sweep"
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
  echo "LAMBDA_KL               : ${LAMBDA_KL}"
  echo "BALANCE_TRAINING        : ${BALANCE_TRAINING}"
  echo "VAL_SPLIT               : ${VAL_SPLIT}"
  echo "K_FOLD                  : ${NUM_FOLDS}"
  echo "FOLDS_TO_RUN            : ${FOLDS_TO_RUN:-all folds}"
  echo "POS_WEIGHT_MODES        : ${POS_WEIGHT_MODES[*]}"
  echo "DRY_RUN                 : ${DRY_RUN}"
  echo "SWEEP_OUTPUT_DIR        : ${SWEEP_OUTPUT_DIR}"
  echo "============================================================"
}

build_and_run_command() {
  local mode="$1"
  local mode_output_dir="$2"
  local exp_name="$3"
  local mode_gpu_id="$4"
  shift 4
  local pos_weight_flag=("$@")

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
    --lambda_kl "${LAMBDA_KL}"
    "${pos_weight_flag[@]}"
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
    --output_dir "${mode_output_dir}"
    --experiment_name "${exp_name}"
    "${EARLY_STOP_FLAG[@]}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] ${mode} command:"
    printf ' CUDA_VISIBLE_DEVICES=%q' "${mode_gpu_id}"
    printf ' %q' "${cmd[@]}"
    echo
  else
    CUDA_VISIBLE_DEVICES="${mode_gpu_id}" "${cmd[@]}"
  fi
}

check_required_outputs() {
  local mode_output_dir="$1"
  local required=(
    "${mode_output_dir}/per_model_predictions/edl_eval_report.txt"
    "${mode_output_dir}/per_model_predictions/fold0_edl_predictions.csv"
    "${mode_output_dir}/per_model_predictions/ensemble_edl_predictions.csv"
  )
  for path in "${required[@]}"; do
    if [[ ! -f "${path}" ]]; then
      echo "[ERROR] Expected output missing: ${path}"
      exit 1
    fi
  done
}

check_paths
check_path_pattern
print_config

SQRT_POS_WEIGHT=""
SQRT_NEG=""
SQRT_POS=""
SQRT_RATIO=""

run_mode() {
  local mode="$1"
  local mode_gpu_id="$2"
  mode_output_dir="${SWEEP_OUTPUT_DIR}/${mode}"
  mkdir -p "${mode_output_dir}"
  exp_name="${ROUND1_BASE_EXP_NAME}_${mode}"

  pos_weight_desc=""
  pos_weight_flag=()

  case "${mode}" in
    none)
      pos_weight_desc="1.0 (no extra positive emphasis)"
      pos_weight_flag=(--pos_weight "1.0")
      ;;
    sqrt)
      if [[ -z "${SQRT_POS_WEIGHT}" ]]; then
        IFS=',' read -r SQRT_NEG SQRT_POS SQRT_RATIO SQRT_POS_WEIGHT < <(compute_sqrt_pos_weight)
      fi
      pos_weight_desc="${SQRT_POS_WEIGHT} (sqrt neg/pos; neg=${SQRT_NEG}, pos=${SQRT_POS}, neg/pos=${SQRT_RATIO})"
      pos_weight_flag=(--pos_weight "${SQRT_POS_WEIGHT}")
      ;;
    auto)
      pos_weight_desc="auto neg/pos from edl_train.py"
      pos_weight_flag=()
      ;;
    *)
      echo "[ERROR] Unknown POS_WEIGHT mode: ${mode}"
      exit 1
      ;;
  esac

  echo ""
  echo "------------------------------------------------------------"
  echo "Running mode             : ${mode}"
  echo "Experiment name          : ${exp_name}"
  echo "Resolved pos_weight      : ${pos_weight_desc}"
  echo "BALANCE_TRAINING         : ${BALANCE_TRAINING}"
  echo "CUDA_VISIBLE_DEVICES     : ${mode_gpu_id}"
  echo "Output dir               : ${mode_output_dir}"
  echo "------------------------------------------------------------"

  build_and_run_command "${mode}" "${mode_output_dir}" "${exp_name}" "${mode_gpu_id}" "${pos_weight_flag[@]}"

  if [[ "${DRY_RUN}" != "1" ]]; then
    check_required_outputs "${mode_output_dir}"
    echo "[OK] Required outputs found for mode=${mode}"
  fi
}

if [[ "${RUN_IN_PARALLEL}" == "1" && "${DRY_RUN}" != "1" ]]; then
  if [[ "${#GPU_IDS[@]}" -lt "${#POS_WEIGHT_MODES[@]}" ]]; then
    echo "[ERROR] RUN_IN_PARALLEL=1 needs at least one GPU id per mode."
    echo "        POS_WEIGHT_MODES=${POS_WEIGHT_MODES[*]}"
    echo "        GPU_IDS=${GPU_IDS[*]}"
    exit 1
  fi

  pids=()
  pid_modes=()
  pid_logs=()

  for idx in "${!POS_WEIGHT_MODES[@]}"; do
    mode="${POS_WEIGHT_MODES[$idx]}"
    mode_gpu_id="${GPU_IDS[$idx]}"
    mode_output_dir="${SWEEP_OUTPUT_DIR}/${mode}"
    mkdir -p "${mode_output_dir}"
    shell_log="${mode_output_dir}/round1_${mode}.log"

    echo ""
    echo "[LAUNCH] mode=${mode} gpu=${mode_gpu_id} log=${shell_log}"
    (
      run_mode "${mode}" "${mode_gpu_id}"
    ) > "${shell_log}" 2>&1 &

    pids+=("$!")
    pid_modes+=("${mode}")
    pid_logs+=("${shell_log}")
  done

  failed=0
  for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    mode="${pid_modes[$idx]}"
    shell_log="${pid_logs[$idx]}"
    if wait "${pid}"; then
      echo "[OK] mode=${mode} finished. Log: ${shell_log}"
    else
      status="$?"
      echo "[ERROR] mode=${mode} failed with exit code ${status}. Log: ${shell_log}"
      failed=1
    fi
  done

  if [[ "${failed}" != "0" ]]; then
    echo "[ERROR] One or more round-1 experiments failed."
    exit 1
  fi
else
  for idx in "${!POS_WEIGHT_MODES[@]}"; do
    mode="${POS_WEIGHT_MODES[$idx]}"
    mode_gpu_id="${GPU_IDS[$idx]:-${GPU_ID}}"
    run_mode "${mode}" "${mode_gpu_id}"
  done
fi

echo ""
echo "Round-1 pos_weight sweep done."
echo "Sweep output dir: ${SWEEP_OUTPUT_DIR}"
