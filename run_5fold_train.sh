export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

NUM_GPUS=1
BATCH_SIZE=8
BASE_EXP_NAME="embed_as_rsna_kfold_wbce_pretrain_init"
PRETRAINED_CKPT="/path/to/pretrain_model.ckpt"

for FOLD in 0 1 2 3 4
do
  EXP_NAME="${BASE_EXP_NAME}_fold${FOLD}"

  echo "======================================"
  echo "Experiment name: ${EXP_NAME}"
  echo "Pretrained ckpt: ${PRETRAINED_CKPT}"
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
    --max_steps 100000 \
    --warm_up 50 \
    --emb_dim 512 \
    --img_size 336 \
    --crop_size 336 \
    --num_workers 8 \
    --img_cls_ft \
    --num_classes 1 \
    --binary_loss \
    --weighted_binary \
    --pretrained_model "${PRETRAINED_CKPT}"
  echo "Finished FOLD = ${FOLD}"
done
