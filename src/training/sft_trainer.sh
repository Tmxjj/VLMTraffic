
###
 # @Author: yufei Ji
 # @Date: 2026-01-14 16:45:42
 # @LastEditTime: 2026-01-14 16:46:19
 # @Description: this script is used to train VLM model with SFT method
 # @FilePath: /VLMTraffic/src/training/sft_trainer.sh
### 
DEBUG_MODE="-m debugpy --listen 127.0.0.1:5679 --wait-for-client"

model_name=Qwen2.5-7B-Instruct
task=webshop

beta=0.1
num_train_epochs=4
per_device_train_batch_size=4
gradient_accumulation_steps=1
learning_rate=2e-5
model_path=/mnt/nas1/yufei_bk/project/IPR-main/models/ # path to the original LLM
save_dir=/mnt/nas1/yufei_bk/project/IPR-main/results/checkpoints_${task}/    # checkpoint save path
save_path=/mnt/nas1/yufei_bk/project/IPR-main/results/experiments/sft-${model_name}-${task}-beta-0.1-lr2e-5/  # output save path
data_file=data/webshop_sft.json 
batch_size=$((per_device_train_batch_size * gradient_accumulation_steps * 8))

method=sft

accelerate launch --config_file accelerate_config.yaml src/training/trainer.py \
    --method ${method} \
    --model_path ${model_path}${model_name} \
    --dataset_path ${data_file} \
    --output_dir /home/jiyufei.jyf/results/checktpoints_${task}/sft_${model_name}-beta-${beta}-epoch-${num_train_epochs}-batch-${batch_size}-per_bs-${per_device_train_batch_size}-${learning_rate} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --beta ${beta} \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${num_train_epochs} 