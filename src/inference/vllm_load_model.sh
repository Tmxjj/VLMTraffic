python -m vllm.entrypoints.openai.api_server \
    --model models/base_models/Qwen3-VL-4B-Instruct \
    --trust-remote-code \
    --served-model-name qwen3-vl-4b \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.8 \
    --port 8000

    # --speculative-model [DRAFT_MODEL_PATH_OR_KEYWORD] \
    # --num-speculative-tokens 5 \