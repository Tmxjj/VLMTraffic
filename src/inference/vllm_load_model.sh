python -m vllm.entrypoints.openai.api_server \
    --model models/base_models/Qwen3-VL-4B-Instruct \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.65 \
    --max-model-len 8192 \
    --max-num-seqs 2 \
    --served-model-name qwen3-vl-4b \
    --port 8000

    # --speculative-model [DRAFT_MODEL_PATH_OR_KEYWORD] \
    # --num-speculative-tokens 5 \


# ——————多GPU测评————————————
# 启动第 1 个实例 (GPU 0, 端口 8000)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/results/checkpoints_dpo/dpo_qwen3-beta-0.1-epoch-3-batch-64-per_bs-4-lr-3e-6/checkpoint-96 \
    --trust-remote-code --dtype bfloat16 --gpu-memory-utilization 0.8 \
    --max-model-len 8192 --max-num-seqs 16 \
    --served-model-name dpo_qwen3—8b-beta-0.1-epoch-3-batch-64-per_bs-4-lr-3e-6-checkpoint-96 --port 8000 &

# 启动第 2 个实例 (GPU 1, 端口 8001)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/results/checkpoints_dpo/dpo_qwen3-beta-0.1-epoch-3-batch-64-per_bs-4-lr-3e-6/checkpoint-144 \
    --trust-remote-code --dtype bfloat16 --gpu-memory-utilization 0.8 \
    --max-model-len 8192 --max-num-seqs 16 \
    --served-model-name dpo_qwen3—8b-beta-0.1-epoch-3-batch-64-per_bs-4-lr-3e-6-checkpoint-144 --port 8001 &

# 启动第 3 个实例 (GPU 2, 端口 8002)
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/results/checkpoints_rpo/rpo_qwen3-beta-0.1-epoch-3-batch-64-per_bs-4-lr-3e-6/checkpoint-96 \
    --trust-remote-code --dtype bfloat16 --gpu-memory-utilization 0.8 \
    --max-model-len 8192 --max-num-seqs 16 \
    --served-model-name rpo_qwen3—8b-beta-0.1-epoch-3-batch-64-per_bs-4-lr-3e-6-checkpoint-96 --port 8002 &

# 启动第 4 个实例 (GPU 3, 端口 8003)
0-
