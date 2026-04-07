import json
import os
import sys
import re
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.inference.vlm_agent import VLMAgent
from configs.model_config import MODEL_CONFIG

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
    lines = [line.strip() for line in response.split('\n')]
    # 过滤掉空行，重新组合
    response_clean = '\n'.join([line for line in lines if line])
    
    # 使用正则提取 Thought 块和 Action 块
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

def main():
    input_file = os.path.join(os.path.dirname(__file__), 'dpo_dataset.jsonl')
    output_file = os.path.join(os.path.dirname(__file__), 'dpo_dataset_sft_model_gener.jsonl')
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return

    # 初始化 VLMAgent
    # 假设使用 batch inference，这里配置 batch_size
    batch_size = 8
    print("Initializing VLMAgent...")
    agent = VLMAgent(batch_size=batch_size)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    all_data = []
    # Parse all data
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            all_data.append(data)
        except json.JSONDecodeError:
            pass

    print(f"Loaded {len(all_data)} samples.")
    
    batch_image_paths = []
    batch_prompts = []
    batch_indices = []
    
    updated_data = []
    
    def process_batch(curr_image_paths, curr_prompts, curr_indices):
        results = agent.get_batch_decision(curr_image_paths, curr_prompts)
        for i, res in enumerate(results):
            idx = curr_indices[i]
            # Handle getting response object correctly 
            if isinstance(res, tuple):
                response_str = res[0]
            else:
                response_str = str(res)
            
            formatted_res = format_and_check_response(response_str)
            if not formatted_res:
                formatted_res = response_str.strip() if response_str.strip() else "ERROR"
            
            all_data[idx]['rejected'] = [{"role": "assistant", "content": formatted_res}]
            
    print("Generating rejected samples...")
    for idx, data in tqdm(enumerate(all_data), total=len(all_data)):
        try:
            # 提取 prompt 中的图片路径和文本
            prompt_content = data['prompt'][0]['content']
            image_path = ""
            prompt_text = ""
            for item in prompt_content:
                if item['type'] == 'image':
                    # 将DPO里的远程替换地址重新回退到本地绝对路径
                    image_path_raw = item['image']
                    old_prefix = "/root/autodl-tmp/golden_data/"
                    new_prefix = "/home/jyf/code/trafficVLM/code/VLMTraffic/data/sft_dataset/"
                    if image_path_raw.startswith(old_prefix):
                        image_path = image_path_raw.replace(old_prefix, new_prefix)
                    else:
                        image_path = image_path_raw
                elif item['type'] == 'text':
                    prompt_text = item['text']
            
            batch_image_paths.append(image_path)
            batch_prompts.append(prompt_text)
            batch_indices.append(idx)
            
            if len(batch_image_paths) >= batch_size:
                process_batch(batch_image_paths, batch_prompts, batch_indices)
                batch_image_paths.clear()
                batch_prompts.clear()
                batch_indices.clear()
                
        except Exception as e:
            print(f"Error preparing data for {data.get('id', 'unknown')}: {e}")
            
    # Process remaining
    if batch_image_paths:
        process_batch(batch_image_paths, batch_prompts, batch_indices)
        
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in all_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
    print("Done!")

if __name__ == "__main__":
    main()
