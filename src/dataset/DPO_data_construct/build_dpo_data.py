import re
import json
import os
import random
import glob
from pathlib import Path
import sys
'''
--- 数据量统计 ---
文件夹 [JiNan-anon_3_4_jinan_real_2500.rou]: 1139 条有效 DPO 数据
文件夹 [JiNan-anon_3_4_jinan_real_2000.rou]: 59 条有效 DPO 数据
文件夹 [JiNan-anon_3_4_jinan_real]: 238 条有效 DPO 数据
文件夹 [JiNan-anon_3_4_jinan_real_emergency.rou]: 536 条有效 DPO 数据
文件夹 [Hangzhou-anon_4_4_hangzhou_real.rou]: 485 条有效 DPO 数据
文件夹 [Hangzhou-anon_4_4_hangzhou_real_emergy.rou]: 132 条有效 DPO 数据
文件夹 [Hangzhou-anon_4_4_hangzhou_real_5816.rou]: 984 条有效 DPO 数据
总计: 3573 条有效 DPO 数据
--- 数据量统计 ---
'''
def format_and_check_response(response: str) -> str:
    """
    检查并格式化模型生成的 response，抹平各种不规范换行、多余缩进和 Markdown 符号
    需要遵循以下基本结构：
    Thought: [
    ...
    ]
    Action: <num>
    """
    if not response:
        return ""
    
    # 移除 markdown 代码块符号
    response = re.sub(r'```[a-zA-Z]*\n?', '', response)
    response = re.sub(r'```', '', response)
    
    # 清理多余的空白字符和缩进
    # 每行去掉首尾空白
    lines = [line.strip() for line in response.split('\n')]
    # 过滤掉空行，重新组合
    response_clean = '\n'.join([line for line in lines if line])
    
    # 使用正则提取 Thought 块和 Action 块
    # 匹配 Thought: [ ... ] 和 Action: ...
    # 为了避免截断或者部分不匹配，采用宽容模式匹配大块
    match = re.search(r'(?i)Thought:\s*\[(.*?)\].*?Action:\s*([^\n]+)', response_clean, re.DOTALL)
    if not match:
        return ""
        
    thought_content = match.group(1).strip()
    action_content = match.group(2).strip()
    
    # 进一步规范化 Thought 内部的内容，确立主键换行
    thought_content = re.sub(r'(?i)(Scene Understanding:)', r'\n\1\n', thought_content)
    thought_content = re.sub(r'(?i)(- Lane Analysis.*?:)', r'\1\n', thought_content)
    thought_content = re.sub(r'(?i)(- Phase Mapping:)', r'\n\1\n', thought_content)
    thought_content = re.sub(r'(?i)(Scene Analysis:)', r'\n\1\n', thought_content)
    thought_content = re.sub(r'(?i)(- Emergency Check:)', r'\1 ', thought_content)
    thought_content = re.sub(r'(?i)(- Final Condition:)', r'\1 ', thought_content)
    thought_content = re.sub(r'(?i)(Selection Logic:)', r'\n\1\n', thought_content)
    thought_content = re.sub(r'(?i)(- Rule Identification:)', r'\1 ', thought_content)
    thought_content = re.sub(r'(?i)(- Reasoning:)', r'\1 ', thought_content)
    thought_content = re.sub(r'(?i)(- Conclusion:)', r'\1 ', thought_content)
    
    # 清理连续换行
    thought_content = re.sub(r'\n{2,}', '\n', thought_content).strip()
    
    formatted_response = f"Thought: [\n{thought_content}\n]\n\nAction: {action_content}"
    return formatted_response

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from configs.prompt_builder import PromptBuilder

def process_dpo_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/sft_dataset'))
    output_file = os.path.join(os.path.dirname(__file__), 'dpo_dataset.jsonl')
    
    # 查找所有的 04_dataset_final.jsonl 文件
    pattern = os.path.join(base_dir, '**', '04_dataset_final.jsonl')
    jsonl_files = glob.glob(pattern, recursive=True)
    
    all_dpo_data = []
    folder_stats = {}
    
    for file_path in jsonl_files:
        try:
            # 提取 folder 信息，例如 JiNan/anon_3_4_jinan_real
            rel_path = os.path.relpath(file_path, base_dir)
            parts = rel_path.split(os.sep)
            if len(parts) >= 3:
                scenario = parts[0]
                route_file = parts[1]
                folder_name = f"{scenario}-{route_file}"
            else:
                folder_name = "unknown"
                scenario = "unknown"
                route_file = "unknown"
                
            if folder_name not in folder_stats:
                folder_stats[folder_name] = 0
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            decoder = json.JSONDecoder()
            pos = 0
            content = content.lstrip()
            
            while pos < len(content):
                try:
                    data, idx = decoder.raw_decode(content[pos:])
                    pos += idx
                    while pos < len(content) and content[pos].isspace():
                        pos += 1
                        
                    # 1. 提取 chosen 和 rejected 的 response
                    chosen_response = data.get('corrected_response', '')
                    
                    # 负样本优先挑选 step3, 然后 step2, 最后 vlm_response_raw
                    # 选择step3的前提在于"human_label": "无误"的情况下
                    human_label = data.get('human_label', '')
                    if 'step3_vlm_response_raw' in data and data['step3_vlm_response_raw'] and human_label != "无误":
                        rejected_response = data['step3_vlm_response_raw']
                    elif 'step_2_vlm_response_raw' in data and data['step_2_vlm_response_raw']:
                        rejected_response = data['step_2_vlm_response_raw']
                    else:
                        rejected_response = data.get('vlm_response_raw', '')
                    
                    # 过滤空字段或包含 error 的数据
                    if not chosen_response or not rejected_response:
                        continue
                        
                    # 排除 error
                    if 'error' in chosen_response.lower() or 'error' in rejected_response.lower():
                        continue
                        
                    # 正则化及格式检查
                    formatted_chosen = format_and_check_response(chosen_response)
                    formatted_rejected = format_and_check_response(rejected_response)
                    if not formatted_chosen or not formatted_rejected:
                        continue
                    
                    chosen_response = formatted_chosen
                    rejected_response = formatted_rejected
                        
                    # 2. 从 prompt_builder 构建 prompt
                    current_phase = data.get('current_phase', 0)
                    data_scenario = data.get('scenario', scenario)
                    new_prompt_text = PromptBuilder.build_decision_prompt(current_phase, data_scenario)
                    
                    # 3. 命名一个 id: {scenario}_{该数据所在的route file名}_{step_id}_{junction_id}
                    step_id = data.get('step', 'unknown')
                    junction_id = data.get('junction_id', 'unknown')
                    sample_id = f"{data_scenario}_{route_file}_{step_id}_{junction_id}"
                    
                    # 5. 替换图片路径名
                    image_path = data.get('image_path', '')
                    old_prefix = "/home/jyf/code/trafficVLM/code/VLMTraffic/data/sft_dataset/"
                    new_prefix = "/root/autodl-tmp/golden_data/"
                    if image_path.startswith(old_prefix):
                        image_path = image_path.replace(old_prefix, new_prefix)
                        
                    # 构建 DPO 格式数据
                    dpo_sample = {
                        "id": sample_id,
                        "prompt": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image_path},
                                    {"type": "text", "text": new_prompt_text}
                                ]
                            }
                        ],
                        "chosen": [{"role": "assistant", "content": chosen_response}],
                        "rejected": [{"role": "assistant", "content": rejected_response}]
                    }
                    
                    all_dpo_data.append(dpo_sample)
                    folder_stats[folder_name] += 1
                    
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file {file_path} at position {pos}")
                    break
                        
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            
    # 4. 对生成的数据进行随机打乱，随机种子为42
    random.seed(42)
    random.shuffle(all_dpo_data)
    
    # 写入最终的 jsonl 文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for item in all_dpo_data:
                # 写入标准 jsonl 格式（人类可读的方式）
                # f_out.write(json.dumps(item, indent=4, ensure_ascii=False) + '\n')
                # 写入标准 jsonl 格式（一行一个 json 对象）
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Successfully saved {len(all_dpo_data)} DPO samples to {output_file}")
    except Exception as e:
        print(f"Error saving output file: {str(e)}")
        
    # 6. 统计各个文件夹下的数据量并打印
    print("\n--- 数据量统计 ---")
    sum = 0
    for folder, count in folder_stats.items():
        sum += count
        print(f"文件夹 [{folder}]: {count} 条有效 DPO 数据")
    print(f"总计: {sum} 条有效 DPO 数据")
    print("--- 数据量统计 ---")

        
if __name__ == "__main__":
    process_dpo_data()
