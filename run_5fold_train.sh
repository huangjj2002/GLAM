export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
BATCH_SIZE=8
BASE_EXP_NAME="kfold"
CUSTOM_CSV="/mnt/f/new_embed_data/embed_data.csv"
IMG_ROOT="/mnt/f/new_embed_data/images_jpg/" 
PATH_PATTERN='{pid}/{iid}_resized.jpg'  
for FOLD in 0 1 2 3 4; do
  EXP_NAME="${BASE_EXP_NAME}_fold${FOLD}"
  python train.py \
    --experiment_name "${EXP_NAME}" \
    --rsna_mammo \
    --k_fold 5 \
    --fold "${FOLD}" \
    --devices "${NUM_GPUS}" \
    --strategy auto \
    --precision bf16-mixed \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate 1e-4 \
    --max_steps 100000 \
    --warm_up 50 \
    --emb_dim 512 \
    --img_size 336 \
    --crop_size 336 \
    --num_workers 0 \
    --img_cls_ft \
    --num_classes 1 \
    --weighted_binary \
    --rsna_csv_path "${CUSTOM_CSV}" \
    --rsna_img_root "${IMG_ROOT}" \
    --rsna_path_pattern "${PATH_PATTERN}" \
    --rsna_patient_col "patient_id" \
    --rsna_image_col "image_id" \
    --rsna_label_col "cancer" \
    --rsna_split_col "split" \
    --rsna_train_split_value "training" \
    --rsna_test_split_value "test"
done
