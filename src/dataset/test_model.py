'''
Author: yufei Ji
Date: 2026-03-03 16:27:31
LastEditTime: 2026-03-03 20:32:00
Description: this script is used to 短时间内测试模型的内容理解能力
FilePath: /VLMTraffic/src/dataset/test_model.py
'''
from configs.model_config import MODEL_CONFIG
from configs.prompt_builder import PromptBuilder
from src.inference.vlm_agent import VLMAgent
scenario_key = "JiNan"
agent = VLMAgent() 

test_data_list = [
    {
        "img_path": "data/sft_dataset/JiNan/step_4/intersection_3_1_bev_watermarked.jpg",
        "phase_id": 3
    },
    {
        "img_path": "data/sft_dataset/JiNan/step_3/intersection_4_1_bev_watermarked.jpg",
        "phase_id": 2
    },
    {
        "img_path": "data/sft_dataset/JiNan/step_3/intersection_3_1_bev_watermarked.jpg",
        "phase_id": 3
    }
]
api_type = MODEL_CONFIG["api_type"] 
model_name = MODEL_CONFIG.get(api_type, {}).get("model_name", "N/A")
for data in test_data_list:
    img_path = data["img_path"]
    phase_id = data["phase_id"]
    prompt = PromptBuilder.build_decision_prompt(current_phase_id=phase_id, scenario_name=scenario_key)
    # prompt = "请你分析当前交通场景中处于绿灯状态的车道数量，并给出一个简短的回答"
    vlm_response, _, vlm_action_idx, native_thought = agent.get_decision(img_path, prompt)

# 将测试结果保存到本地文件
    with open(f"src/dataset/test_results_watermarked_{api_type}_{model_name}.txt", "a") as f:
        f.write(f"Image: {img_path}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"VLM Response: {vlm_response}\n")
        f.write(f"VLM Action Index: {vlm_action_idx}\n")
        f.write(f"Native Thought: {native_thought}\n")
        f.write("-" * 50 + "\n")

