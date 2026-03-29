###
 # @Author: yufei Ji
 # @Date: 2026-03-29
 # @Description: this script is used to train VLM model with DPO method
 # @FilePath: /VLMTraffic/src/training/dpo_trainer.sh
### 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
model_name=qwen3_vl_8b_dpo # 用于拼接输出文件夹名称
method=dpo


beta=0.1
num_train_epochs=4 # DPO 通常比 SFT 容易过拟合，建议 1-2 个 epoch 即可
per_device_train_batch_size=2
gradient_accumulation_steps=16 # 通过累积梯度实现等效的全局 Batch Size，适合显存受限的情况
learning_rate=3e-6 # DPO 的学习率通常比 SFT 低一个数量级
num_gpus=2 # 设定为 2 张 A100

# 路径配置
model_path=/root/autodl-tmp/model/qwen3_vl_8b_sft
data_file=/root/autodl-tmp/dpo_dataset.jsonl
output_base_dir=/root/autodl-tmp/results/checkpoints_${method}

# 修正了全局 Batch Size 计算逻辑（原脚本硬编码了 * 8）
batch_size=$((per_device_train_batch_size * gradient_accumulation_steps * num_gpus))

# 确保输出目录存在
mkdir -p ${output_base_dir}

# 显式添加 --num_processes ${num_gpus} 确保 accelerate 调度两张卡
# ENABLE_DEBUGPY=true accelerate launch --config_file /root/code/VLMTraffic/configs/accelerate_config.yaml --num_processes ${num_gpus} src/training/trainer.py \
accelerate launch --config_file /root/code/VLMTraffic/configs/accelerate_config.yaml --num_processes ${num_gpus} src/training/trainer.py \
    --method ${method} \
    --model_path ${model_path} \
    --dataset_path ${data_file} \
    --output_dir ${output_base_dir}/${method}_qwen3-beta-${beta}-epoch-${num_train_epochs}-batch-${batch_size}-per_bs-${per_device_train_batch_size}-lr-${learning_rate} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --beta ${beta} \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${num_train_epochs}