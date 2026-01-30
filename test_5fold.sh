
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
CUSTOM_CSV="/mnt/f/new_embed_data/embed_data.csv"
IMG_ROOT="/mnt/f/new_embed_data/images_jpg/" 
PATH_PATTERN="{pid}/{iid}_resized.jpg"
BASE_EXP_NAME="kfold"
for FOLD in 0 1 2 3 4; do
  EXP_NAME="${BASE_EXP_NAME}_fold${FOLD}"

  CKPT_DIR=$(ls -dt logs/ckpts/GLAM/*"${EXP_NAME}" | head -n 1)
  CKPT_PATH="${CKPT_DIR}/last.ckpt"

  echo "=============================="
  echo "Testing fold ${FOLD}"
  echo "CKPT: ${CKPT_PATH}"
  echo "=============================="

  python train.py \
    --eval \
    --rsna_mammo \
    --devices 1 \
    --batch_size 8 \
    --num_workers 0 \
    --k_fold 5 \
    --fold ${FOLD} \
    --pretrained_model "${CKPT_PATH}" \
    --rsna_csv_path "${CUSTOM_CSV}" \
    --rsna_img_root "${IMG_ROOT}" \
    --rsna_path_pattern "${PATH_PATTERN}" \
    --rsna_split_col "split" \
    --rsna_train_split_value "training" \
    --rsna_test_split_value "test"\
    --img_cls_ft \
    --num_classes 1 \
    --weighted_binary 
done