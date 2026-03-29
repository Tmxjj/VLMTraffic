python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/model/base_model/qwen3-vl-8b \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 16192 \
    --max-num-seqs 16 \
    --served-model-name qwen3-vl-8b \
    --port 8000

    # --speculative-model [DRAFT_MODEL_PATH_OR_KEYWORD] \
    # --num-speculative-tokens 5 \
