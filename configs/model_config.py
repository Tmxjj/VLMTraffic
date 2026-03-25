'''
Author: yufei Ji
Date: 2026-01-12 17:09:21
LastEditTime: 2026-01-26 21:45:40
Description: this script is used to 
FilePath: /VLMTraffic/configs/model_config.py
'''
"""
Model configuration for VLMAgent.
"""

MODEL_CONFIG = {
    # Default API type to use: "local_model", "openai_sdk", "gemini_sdk", "requests"
    "api_type": "requests", 
    
    # Global generation settings
    "temperature": 0, # 0.0 to 1.0
    "max_new_tokens": 8192, # Max tokens to generate

    # Configuration for specific backends
    # "local_model": {
    #     "model_path": "root/autodl-tmp/results/Qwen3-VL-8B-SFT-Merged",
    #     "device": "cuda"
    # },
    "openai_sdk": {
        "api_key": "YOUR_OPENAI_API_KEY",
        "base_url": None, # Optional: "https://api.openai.com/v1"
        "model_name": "gpt-4-vision-preview"
    },
    
    "gemini_sdk": {
        "api_key": "****",  # the api don't need to upload to github
        "model_name": "gemini-3-pro-preview" 
    },
    
    "requests": {
        # 基于 VLLM 的本地服务地址，其中 model_name 应与启动脚本中 --served-model-name 保持一致
        "url": "http://localhost:8000/v1/chat/completions",
        "headers": {"Content-Type": "application/json"},
        "model_name": "qwen3-vl-8b"
    }
}
