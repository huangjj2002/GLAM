#!/bin/bash
set -euo pipefail

# ===========================================================================
# Prototype + EDL 训练与评估入口
# 说明：这是第三个独立入口，不修改 run.sh 和 run_edl.sh。
#       数据读取、cohert_num 划分、验证集划分、输出逻辑与 EDL 保持一致。
# ===========================================================================

# ---------------------------------------------------------------------------
# 常用配置：通常只需要改这一段
# ---------------------------------------------------------------------------
GPU_ID=2
# 使用哪张 GPU
NUM_GPUS=1
# Lightning 使用的 GPU 数量
PYTHON_BIN="${PYTHON_BIN:-python}"
# Python 命令；需要指定环境时可在命令前覆盖 PYTHON_BIN

MAX_EPOCHS=25
# 每个 fold 的最大训练 epoch
WARM_UP=1
# cosine scheduler 的 warm-up epoch 数
BATCH_SIZE=32
# 训练和推理 batch size
NUM_WORKERS=4
# DataLoader worker 数；Windows 或 WSL 不稳定时可改为 0
PRECISION="32"
# Lightning precision
LEARNING_RATE=1e-4
# 初始学习率

SHOW_PROGRESS_BAR=1
# 是否显示 Lightning/tqdm 进度条：1=显示，0=隐藏
PROGRESS_PRINT_INTERVAL=50
# 每隔多少个 batch 打印一次文本进度

EARLY_STOP=1
# 是否启用早停：1=启用，0=禁用
EARLY_STOP_PATIENCE=3
# 早停耐心轮数
EARLY_STOP_MIN_EPOCHS=12
# 早停前至少训练多少 epoch
MONITOR_METRIC="val_AUROC"
# checkpoint 和早停监控指标

OUTPUT_DIR="outputs"
# 输出根目录；相对路径会解析到当前项目目录下
BASE_EXP_NAME="prototype_edl_kfold_ft"
# 本次实验名前缀

CSV_PATH="/home/dhao4/workspace/hjj_workspace/data/data.csv"
# 输入 CSV 文件
IMG_ROOT="/home/dhao4/workspace/hjj_workspace/data/images_png"
# 图像根目录
PATH_PATTERN="{pid}/{iid}"
# 图像相对路径模板，支持 {pid} 和 {iid}

PATIENT_COL="patient_id"
# 病人 ID 列
IMAGE_COL="image_id"
# 图像 ID 列
LABEL_COL="cancer"
# 二分类标签列，要求 0/1
SPLIT_COL="split"
# split 列；当 SPLIT_SOURCE=split 时使用
COHORT_COL="cohert_num"
# cohort 列；保持和现有 EDL 一致，默认使用 cohert_num
SPLIT_SOURCE="cohort"
# 数据划分来源：cohort 或 split
TRAIN_SPLIT_VALUE="training"
# split 模式下的训练值
TEST_SPLIT_VALUE="test"
# split 模式下的测试值
TRAIN_COHORTS="1-8"
# cohort 模式下作为训练池的 cohort
TEST_COHORTS="9-10"
# cohort 模式下作为测试集的 cohort
OUTPUT_SPLIT_COL="split"
# 输出 CSV 中的 split 列名

VAL_SPLIT=1
# NUM_FOLDS=0 时，是否从训练池再切 patient-level validation
VAL_FRACTION=0.15
# 目标验证集病人比例
VAL_MAX_FRACTION=0.25
# 验证集最大病人比例
VAL_MIN_POSITIVE_PATIENTS=3
# 验证集至少包含的阳性病人数
VAL_RANDOM_STATE=42
# 验证集划分随机种子

PRETRAINED_MODEL="$(cd "$(dirname "$0")" && pwd)/pretrain_model/last.ckpt"
# 预训练 GLAM encoder checkpoint
NUM_FOLDS=0
# 0=单模型训练；>=2 时在训练池内做 patient-level StratifiedGroupKFold
FOLDS_TO_RUN=""
# 只跑指定 fold，例如 "0" 或 "0,2,4"；空值表示全部 fold

# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------
IMG_ENCODER="dinov2_vitb14_reg"
# 图像 encoder 名称
EMB_DIM=512
# GLAM embedding 维度配置
USE_LINEAR_PROJ=1
# 是否沿用 GLAM linear projection：1=启用，0=禁用
IMG_SIZE=336
# resize 后图像尺寸
CROP_SIZE=336
# crop 尺寸

# ---------------------------------------------------------------------------
# EDL 配置
# ---------------------------------------------------------------------------
EVIDENCE_TYPE="softplus"
# evidence 激活：softplus/relu/exp
EDL_LOSS_TYPE="digamma"
# EDL loss：digamma/log/mse
ANNEALING_STEP=10
# KL annealing 步数
LAMBDA_KL=0.1
# KL 正则权重
FREEZE_BACKBONE=0
# 是否冻结 GLAM backbone，只训练 prototype head：1=冻结，0=微调
BALANCE_TRAINING=0
# 是否启用 WeightedRandomSampler；默认关闭，和 EDL 保持一致
BCE_POS_WEIGHT=""
# 空值表示每个 fold 自动按 neg/pos 计算

# ---------------------------------------------------------------------------
# Prototype 配置
# ---------------------------------------------------------------------------
EDL_PROTO_K=4
# 每个类别的 prototype 数量，二分类总数为 2*K
EDL_PROTO_TOPK=3
# 每个类别导出 top-k 个最相近 prototype
PROTOTYPE_TEMPERATURE=1.0
# 距离转相似度的温度：similarity = -distance / temperature
PROTOTYPE_INIT="kmeans"
# prototype 初始化方式：kmeans 或 random
PROTOTYPE_NORMALIZE=1
# 是否对 embedding 和 prototype 做 L2 normalize：1=启用，0=禁用
PROTOTYPE_INIT_MAX_SAMPLES_PER_CLASS=0
# KMeans 每类最多使用多少训练样本；0=使用当前 fold 训练集全部样本
PROTOTYPE_INIT_BATCH_SIZE=0
# 抽 embedding 初始化 prototype 的 batch size；0=复用 BATCH_SIZE
PROTOTYPE_INIT_NUM_WORKERS=0
# 抽 embedding 初始化 prototype 的 worker 数；0=复用 NUM_WORKERS

# ---------------------------------------------------------------------------
# 派生配置：一般不需要修改
# ---------------------------------------------------------------------------
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

PROGRESS_BAR_FLAG=()
if [[ "${SHOW_PROGRESS_BAR}" != "1" ]]; then
  PROGRESS_BAR_FLAG+=(--no_progress_bar)
fi

FOLDS_TO_RUN_FLAG=()
if [[ -n "${FOLDS_TO_RUN}" ]]; then
  FOLDS_TO_RUN_FLAG+=(--folds_to_run "${FOLDS_TO_RUN}")
fi

PROTOTYPE_NORMALIZE_FLAG=()
if [[ "${PROTOTYPE_NORMALIZE}" != "1" ]]; then
  PROTOTYPE_NORMALIZE_FLAG+=(--no_prototype_normalize)
fi

echo "Root dir                         : ${ROOT_DIR}"
echo "Python                           : ${PYTHON_BIN}"
echo "GPU_ID                           : ${GPU_ID}"
echo "CUDA_VISIBLE_DEVICES             : ${CUDA_VISIBLE_DEVICES}"
echo "CSV_PATH                         : ${CSV_PATH}"
echo "IMG_ROOT                         : ${IMG_ROOT}"
echo "PATH_PATTERN                     : ${PATH_PATTERN}"
echo "PRETRAINED_MODEL                 : ${PRETRAINED_MODEL}"
echo "SPLIT_SOURCE                     : ${SPLIT_SOURCE}"
echo "COHORT_COL                       : ${COHORT_COL}"
echo "TRAIN_COHORTS                    : ${TRAIN_COHORTS}"
echo "TEST_COHORTS                     : ${TEST_COHORTS}"
echo "VAL_SPLIT                        : ${VAL_SPLIT}"
echo "K_FOLD                           : ${NUM_FOLDS}"
echo "FOLDS_TO_RUN                     : ${FOLDS_TO_RUN:-all folds}"
echo "MAX_EPOCHS                       : ${MAX_EPOCHS}"
echo "BATCH_SIZE                       : ${BATCH_SIZE}"
echo "LEARNING_RATE                    : ${LEARNING_RATE}"
echo "EVIDENCE_TYPE                    : ${EVIDENCE_TYPE}"
echo "EDL_LOSS_TYPE                    : ${EDL_LOSS_TYPE}"
echo "LAMBDA_KL                        : ${LAMBDA_KL}"
echo "FREEZE_BACKBONE                  : ${FREEZE_BACKBONE}"
echo "BALANCE_TRAINING                 : ${BALANCE_TRAINING}"
echo "BCE_POS_WEIGHT                   : ${BCE_POS_WEIGHT:-auto per fold}"
echo "EDL_PROTO_K                      : ${EDL_PROTO_K}"
echo "EDL_PROTO_TOPK                   : ${EDL_PROTO_TOPK}"
echo "PROTOTYPE_TEMPERATURE            : ${PROTOTYPE_TEMPERATURE}"
echo "PROTOTYPE_INIT                   : ${PROTOTYPE_INIT}"
echo "PROTOTYPE_NORMALIZE              : ${PROTOTYPE_NORMALIZE}"
echo "PROTOTYPE_INIT_MAX_SAMPLES_CLASS : ${PROTOTYPE_INIT_MAX_SAMPLES_PER_CLASS}"
echo "RUN_OUTPUT_DIR                   : ${RUN_OUTPUT_DIR}"

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
echo "Running Prototype EDL Pipeline"
echo "=============================="

"${PYTHON_BIN}" -u prototype_edl_train.py \
  --csv_path "${CSV_PATH}" \
  --img_root "${IMG_ROOT}" \
  --path_pattern "${PATH_PATTERN}" \
  --pretrained_model "${PRETRAINED_MODEL}" \
  "${PROGRESS_BAR_FLAG[@]}" \
  --progress_print_interval "${PROGRESS_PRINT_INTERVAL}" \
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
  --edl_proto_k "${EDL_PROTO_K}" \
  --edl_proto_topk "${EDL_PROTO_TOPK}" \
  --prototype_temperature "${PROTOTYPE_TEMPERATURE}" \
  --prototype_init "${PROTOTYPE_INIT}" \
  "${PROTOTYPE_NORMALIZE_FLAG[@]}" \
  --prototype_init_max_samples_per_class "${PROTOTYPE_INIT_MAX_SAMPLES_PER_CLASS}" \
  --prototype_init_batch_size "${PROTOTYPE_INIT_BATCH_SIZE}" \
  --prototype_init_num_workers "${PROTOTYPE_INIT_NUM_WORKERS}" \
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
