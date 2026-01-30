export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

NUM_GPUS=1
BATCH_SIZE=8
BASE_EXP_NAME="embed_as_rsna_kfold_wbce"

for FOLD in 0 1 2 3 4
do
  EXP_NAME="${BASE_EXP_NAME}_fold${FOLD}"

  echo "======================================"
  echo "Running K-Fold training (Weighted-BCE): FOLD = ${FOLD}"
  echo "Experiment name: ${EXP_NAME}"
  echo "======================================"

  python train.py \
    --experiment_name ${EXP_NAME} \
    --rsna_mammo \
    --k_fold 5 \
    --fold ${FOLD} \
    --devices ${NUM_GPUS} \
    --strategy auto \
    --precision bf16-mixed \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 1e-4 \
    --max_steps 1000 \
    --warm_up 50 \
    --emb_dim 512 \
    --num_classes 1 \
    --weighted_binary \
    --img_size 336 \
    --crop_size 336 \
    --num_workers 8 \
    --img_cls_ft

  echo "Finished FOLD = ${FOLD}"
done