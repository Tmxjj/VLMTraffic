###
 # @Author: yufei Ji
 # @Date: 2026-04-11
 # @Description: NCDPO（数值距离连续惩罚 DPO）训练脚本
 #               在 RPO 框架基础上引入逐样本动态 β，惩罚计数幻觉严重的负样本
 #
 # 使用前准备：
 #   1. 在本机运行数据集增强脚本：
 #      conda activate VLMTraffic
 #      python src/dataset/DPO_data_construct/augment_ncdpo_weights.py
 #   2. 将生成的 dpo_dataset_ncdpo.jsonl 上传至远程服务器：
 #      scp src/dataset/DPO_data_construct/dpo_dataset_ncdpo.jsonl \
 #          <user>@<server>:/root/autodl-tmp/dpo_dataset_ncdpo.jsonl
 #   3. 在远程服务器执行本脚本
 #
 # 核心超参数说明：
 #   --ncdpo_alpha     : 动态 β 幅值系数 α (默认 1.0，使 β 最大翻倍)
 #   --ncdpo_eps_scale : tanh 归一化尺度 ε_scale (默认 2.0)
 #     β_i = β × (1 + α × tanh(ε_i / ε_scale))
 #     当 α=1.0, ε_scale=2.0 时：
 #       ε=0.0 (计数完全一致) → β_i = β        (无额外惩罚)
 #       ε=0.34 (均值 MAE)    → β_i ≈ 1.17 × β (+17%)
 #       ε=2.0               → β_i ≈ 1.76 × β (+76%)
 #       ε→∞                → β_i → 2.0 × β  (最大 2×)
 # @FilePath: /VLMTraffic/src/training/ncdpo_trainer.sh
###

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# 训练方法配置
# ============================================================
method=ncdpo

# ============================================================
# 模型与数据路径
# ============================================================
model_path=/root/autodl-tmp/model/qwen3_vl_8b_sft    # SFT 微调后的模型作为初始化
data_file=/root/autodl-tmp/dpo_dataset_ncdpo.jsonl   # 包含 count_error_weight 的增强数据集
output_base_dir=/root/autodl-tmp/results/checkpoints_${method}

# ============================================================
# 基础训练超参数
# ============================================================
beta=0.1                        # 基准 KL 惩罚系数（NCDPO 会逐样本放大它）
num_train_epochs=3
per_device_train_batch_size=4
gradient_accumulation_steps=4
learning_rate=3e-6              # NCDPO 学习率与 RPO 保持一致
num_gpus=4

# ============================================================
# 参数冻结策略（与 RPO 保持一致）
# ============================================================
freeze_vit="true"
freeze_merger="true"

# ============================================================
# NCDPO 专属超参数
# ============================================================
ncdpo_alpha=1.0       # α: 动态 β 幅值系数
ncdpo_eps_scale=2.0   # ε_scale: tanh 归一化尺度（与数据集均值 MAE ≈ 0.34 对应）

# ============================================================
# 计算全局 Batch Size（仅用于命名输出目录）
# ============================================================
batch_size=$((per_device_train_batch_size * gradient_accumulation_steps * num_gpus))

# 确保输出目录存在
mkdir -p ${output_base_dir}

# ============================================================
# 启动训练
# ============================================================
accelerate launch \
    --config_file /root/code/VLMTraffic/configs/accelerate_config.yaml \
    --num_processes ${num_gpus} \
    src/training/trainer.py \
    --method          ${method} \
    --model_path      ${model_path} \
    --dataset_path    ${data_file} \
    --output_dir      ${output_base_dir}/${method}_qwen3-beta-${beta}-alpha-${ncdpo_alpha}-eps-${ncdpo_eps_scale}-epoch-${num_train_epochs}-batch-${batch_size}-lr-${learning_rate} \
    --per_device_train_batch_size  ${per_device_train_batch_size} \
    --gradient_accumulation_steps  ${gradient_accumulation_steps} \
    --beta            ${beta} \
    --learning_rate   ${learning_rate} \
    --num_train_epochs ${num_train_epochs} \
    --freeze_vit      ${freeze_vit} \
    --freeze_merger   ${freeze_merger} \
    --ncdpo_alpha     ${ncdpo_alpha} \
    --ncdpo_eps_scale ${ncdpo_eps_scale}
