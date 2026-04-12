###
 # @Author: yufei Ji
 # @Date: 2026-04-11
 # @Description: Simulation-Grounded Dual-Verifiable RLVR 训练脚本
 #               使用 GRPO 范式对 Qwen3-VL 进行在线强化学习微调
 #
 # 使用前准备：
 #   1. 在本机（或仿真服务器）运行数据收集脚本：
 #      conda activate VLMTraffic
 #      python src/training/rlvr/rlvr_data_collector.py \
 #          --scenario JiNan \
 #          --route_file anon_3_4_jinan_real.rou.xml \
 #          --max_steps 120 \
 #          --output_dir data/rlvr_dataset
 #
 #   2. 将生成的 JSONL 数据集上传至训练服务器：
 #      scp data/rlvr_dataset/rlvr_train_*.jsonl \
 #          <user>@<server>:/root/autodl-tmp/rlvr_dataset/
 #
 #   3. 在训练服务器上安装额外依赖（基于 VLMTraffic 虚拟环境）：
 #      pip install "peft>=0.14.0"
 #      # trl >= 0.29.1 / transformers >= 4.56.2 已满足，无需重装
 #
 #   4. 在训练服务器执行本脚本：
 #      bash src/training/rlvr_trainer.sh
 #
 # 奖励值域（默认 α=0.7, β=0.3）：
 #   r_perc ∈ [-1.0,  0.0]  感知奖励（计数幻觉惩罚）
 #   r_env  ∈ [ 0.0,  1.0]  效率奖励（MaxPressure 代理）
 #   r      ∈ [-0.7,  1.0]  联合奖励
 #
 # @FilePath: /VLMTraffic/src/training/rlvr_trainer.sh
###

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# 训练方法标识
# ============================================================
method=rlvr

# ============================================================
# 模型与数据路径
# ============================================================
# 建议从 SFT 或 DPO checkpoint 初始化，充分利用已有的指令跟随能力
model_path=/root/autodl-tmp/model/qwen3_vl_8b_sft
# RLVR 训练数据集（由 rlvr_data_collector.py 生成的 JSONL）
data_file=/root/autodl-tmp/rlvr_dataset/rlvr_train.jsonl
output_base_dir=/root/autodl-tmp/results/checkpoints_${method}

# ============================================================
# GRPO 核心超参数
# ============================================================
num_generations=8         # 每个 prompt 采样 G=8 个响应（GRPO 组内相对优势）
max_new_tokens=512        # CoT 推理链最大生成长度
temperature=0.9           # 采样温度（高温鼓励探索多样化推理）
top_p=0.95
kl_coef=0.04              # KL 惩罚系数（防止策略严重偏离参考模型）

# ============================================================
# 双路可验证奖励权重
# ============================================================
alpha=0.7          # r_perc 感知奖励权重（优先修复计数幻觉）
beta_reward=0.3    # r_env  效率奖励权重（MaxPressure 代理）

# ============================================================
# 基础训练超参数
# ============================================================
num_train_epochs=3
# GRPO 每步需要前向推理 G=8 次，显存压力远高于 DPO
# 建议 per_device_train_batch_size=1，靠梯度累积扩大有效 batch
per_device_train_batch_size=1
gradient_accumulation_steps=8
# GRPO 在线策略更新，学习率比 DPO 更小，防止快速遗忘
learning_rate=1e-6
num_gpus=4
max_prompt_length=2048

# ============================================================
# LoRA 超参数
# ============================================================
lora_r=16
lora_alpha=32
lora_dropout=0.05

# ============================================================
# 参数冻结策略（与 DPO 保持一致）
# ============================================================
freeze_vit="true"
freeze_merger="true"

# ============================================================
# 计算全局 Batch Size（仅用于目录命名）
# ============================================================
batch_size=$((per_device_train_batch_size * gradient_accumulation_steps * num_gpus))

# 确保输出目录存在
mkdir -p ${output_base_dir}

# ============================================================
# 启动 GRPO 训练
# ============================================================
accelerate launch \
    --config_file /root/code/VLMTraffic/configs/accelerate_config.yaml \
    --num_processes ${num_gpus} \
    src/training/rlvr/rlvr_grpo_trainer.py \
    --model_path      ${model_path} \
    --dataset_path    ${data_file} \
    --output_dir      ${output_base_dir}/${method}_qwen3-G${num_generations}-alpha${alpha}-beta${beta_reward}-kl${kl_coef}-epoch${num_train_epochs}-batch${batch_size}-lr${learning_rate} \
    --num_generations       ${num_generations} \
    --max_new_tokens        ${max_new_tokens} \
    --temperature           ${temperature} \
    --top_p                 ${top_p} \
    --kl_coef               ${kl_coef} \
    --alpha                 ${alpha} \
    --beta_reward           ${beta_reward} \
    --num_train_epochs      ${num_train_epochs} \
    --per_device_train_batch_size  ${per_device_train_batch_size} \
    --gradient_accumulation_steps  ${gradient_accumulation_steps} \
    --learning_rate         ${learning_rate} \
    --max_prompt_length     ${max_prompt_length} \
    --lora_r                ${lora_r} \
    --lora_alpha            ${lora_alpha} \
    --lora_dropout          ${lora_dropout} \
    --freeze_vit            ${freeze_vit} \
    --freeze_merger         ${freeze_merger} \
    --bf16
