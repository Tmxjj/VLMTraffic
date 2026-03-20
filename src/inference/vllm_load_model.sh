python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/base_model/qwen3-vl-8b \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --max-num-seqs 2 \
    --enforce-eager \
    --disable-log-stats \
    --swap-space 16