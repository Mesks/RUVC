#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
GPU_index="0,1"
prefix="demotest_RUVC"

start_epoch=0
epoch_num=70
batch_size=4
eval_batch_size=5
sample_size=144
training_CRF=dynamic
log_dir="$SCRIPT_DIR/log/$prefix"
log_path="$log_dir/train_log.txt"
dataset_path="./dataset/UVC46k_trainingset/"
evalset_path="./dataset/UVC46k_valset/"
if [ -d $log_dir ]; then
  if [ -f $log_path ]; then
    rm $log_path
  fi
else
  mkdir -p "$log_dir"
fi

### From A Empty Model
python -u train_RUVC.py --epoch $epoch_num --GPU_index $GPU_index --dataset $dataset_path --batch_size $batch_size --eval_batch_size $eval_batch_size --use_surrogate 0 --training_CRF $training_CRF --prefix $prefix --external_evalutionset $evalset_path --sample_size $sample_size | tee -a $log_path

### From A Pretrained Model
# init_model="./model/20250530_anchor/RUVC_model_28.pth"
# init_optimizer="./model/20250530_anchor/RUVC_optimizer_28.pt"
# python -u train_RUVC.py --epoch $epoch_num --GPU_index $GPU_index --dataset $dataset_path --batch_size $batch_size --eval_batch_size $eval_batch_size --use_surrogate 0 --training_CRF $training_CRF --prefix $prefix --start_epoch $start_epoch --init_model $init_model --init_optimizer  --external_evalutionset $evalset_path --sample_size $sample_size | tee -a $log_path
