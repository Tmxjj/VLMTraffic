'''
Author: yufei Ji
Date: 2026-04-07 23:48:43
LastEditTime: 2026-04-08 10:00:00
Description: 这个脚本用于对 DPO 数据集进行格式清洗和数据对齐。主要功能包括去除 Markdown 语法(如**)、修正常见拼写错误(如noraml)并筛除包含无效结构(Unknown)的数据样本。
FilePath: /VLMTraffic/src/dataset/DPO_data_construct/clean_dpo_data.py
'''
import json
import os
import re

def extract_field(text, field_name):
    """
    提取文本中对应字段名后的值。
    对于类似 "- Final Condition: Normal" 或换行的 "- Final Condition: \n - Normal" 都能匹配。
    """
    match = re.search(rf"- {field_name}:\s*(?:-\s*)?([^\n]+)", text, re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        if val:
            return val
    return "Unknown"

def normalize_final_cond(cond, text, expected_id):
    """
    规范化 Final Condition 的拼写。
    将常见的拼写错误 (如 noraml, speical) 映射到标准分类，
    如果确实不满足，则打印问题发生的片段方便人工调试。
    """
    c = cond.lower().strip()
    if c in ["normal", "noraml", "normal."]:
        return "Normal"
    elif c in ["special", "speical", "special."]:
        return "Special"
    else:
        print(f"\n[Warning] Final Condition parsing issue for ID {expected_id}:")
        print(f"  Extracted string: '{cond}'")
        return "Unknown"

def normalize_rule(rule, text, expected_id):
    """
    规范化 Rule Identification。
    去除标点后进行关键字匹配，将杂乱的格式转换为标准规则标签。
    """
    r = rule.lower().replace(".", "").replace(",", "").strip()
    if "emergency_priority" in r:
        return "Emergency_Priority"
    elif "incident_avoidance" in r:
        return "Incident_Avoidance"
    elif "fallback_static" in r or "fallback_cycle" in r:
        return "Fallback_Static"
    elif "contextual_adaptation" in r:
        return "Contextual_Adaptation"
    elif "tie" in r or "ie_breaker" in r:
        if "index" in r or "(c)" in r:
            return "Tie_Breaker (Index_Order)"
        elif "straight" in r or "(a)" in r:
            return "Tie_Breaker (Straight > Left)"
        elif "max single" in r or "(b)" in r:
            return "Tie_Breaker (Max Single Lane)"
        return "Tie_Breaker"
    elif "bottleneck" in r:
        return "Bottleneck_Rule"
    else:
        print(f"\n[Warning] Rule Identification parsing issue for ID {expected_id}:")
        print(f"  Extracted string: '{rule}'")
        return "Unknown"

def normalize_conclusion(conc, text, expected_id):
    """
    规范化 Conclusion。
    由于模型可能会多生成一些多余内容，这里正则表达式仅提取 `Phase X` (其中 X 是0-3)。
    """
    c = conc.lower().strip()
    match = re.search(r'phase\s*(\d)', c, re.IGNORECASE)
    if match:
        return f"Phase {match.group(1)}"
    else:
        print(f"\n[Warning] Conclusion parsing issue for ID {expected_id}:")
        print(f"  Extracted string: '{conc}'")
        return "Unknown"

def main():
    base_dir = os.path.dirname(__file__)
    # 待清洗输入文件
    raw_file_path = os.path.join(base_dir, 'dpo_dataset_sft_model_gener_cleaned.jsonl')
    gener_file_path = os.path.join(base_dir, 'dpo_dataset_sft_model_gener.jsonl')
    
    # 清洗后的输出文件
    cleaned_file_path = os.path.join(base_dir, 'dpo_dataset_cleaned.jsonl')
    gener_cleaned_file_path = os.path.join(base_dir, 'dpo_dataset_sft_model_gener_cleaned.jsonl')
    
    if not os.path.exists(raw_file_path):
        print(f"找不到基准文件: {raw_file_path}")
        return

    print("====== 阶段 1: 清洗原始的 DPO 数据集 (dpo_dataset.jsonl) ======")
    total_raw_samples = 0
    deleted_raw_samples = 0
    valid_data_list = []
    valid_ids = set()

    with open(raw_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_raw_samples += 1
            data = json.loads(line)
            # 全局替换掉可能由大模型生成的加粗 markdown 语法，防止干扰正则表达式
            chosen = data['chosen'][0]['content'].replace('**', '')

            sample_id = data.get("id", "Unknown_ID")
            chosen_cond_raw = extract_field(chosen, "Final Condition")
            chosen_rule_raw = extract_field(chosen, "Rule Identification")
            chosen_conc_raw = extract_field(chosen, "Conclusion")
            
            # 经过业务逻辑将其标准化
            chosen_cond = normalize_final_cond(chosen_cond_raw, chosen, sample_id)
            chosen_rule = normalize_rule(chosen_rule_raw, chosen, sample_id)
            chosen_conc = normalize_conclusion(chosen_conc_raw, chosen, sample_id)

            # 剔除因为幻觉/截断导致任一关键字段缺失(Unknown)的样本
            if chosen_cond == "Unknown" or chosen_rule == "Unknown" or chosen_conc == "Unknown":
                deleted_raw_samples += 1
                continue
            
            # 若所有字段正常，则记录此正样本为有效，将其 ID 保留作第二阶段基准
            valid_ids.add(sample_id)
            valid_data_list.append(data)

    print(f"--- 原始样本总数: {total_raw_samples} ---")
    print(f"--- 剔除无效格式样本数: {deleted_raw_samples} ---")
    print(f"--- 剩余有效样本数: {len(valid_data_list)} ---")
    
    if len(valid_data_list) > 0:
        with open(cleaned_file_path, 'w', encoding='utf-8') as f_out:
            for d in valid_data_list:
                f_out.write(json.dumps(d, ensure_ascii=False) + '\n')
        print(f"基准数据集已清洗保存至: {cleaned_file_path}\n")

    if not os.path.exists(gener_file_path):
        print(f"找不到生成的SFT数据文件: {gener_file_path}，第二阶段清洗跳过。")
        return

    # print("====== 阶段 2: 清洗 SFT 生成的负样本数据并与基准对齐 ======")
    # total_gener_samples = 0
    # deleted_not_in_cleaned = 0
    # deleted_invalid_rejected = 0
    # valid_gener_data_list = []

    # with open(gener_file_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         total_gener_samples += 1
    #         data = json.loads(line)
    #         sample_id = data.get("id", "Unknown_ID")
            
    #         # 逻辑说明：由于基准数据中删除了部分结构无效的样本，这里必须保证两端一致
    #         # 如果这里的 ID 在上一步并没有被归为 valid_ids，就说明基准中已丢弃，这里直接舍弃
    #         if sample_id not in valid_ids:
    #             deleted_not_in_cleaned += 1
    #             continue
            
    #         rejected = data['rejected'][0]['content'].replace('**', '')

    #         rej_cond_raw = extract_field(rejected, "Final Condition")
    #         rej_rule_raw = extract_field(rejected, "Rule Identification")
    #         rej_conc_raw = extract_field(rejected, "Conclusion")
            
    #         rej_cond = normalize_final_cond(rej_cond_raw, rejected, sample_id + "_rejected")
    #         rej_rule = normalize_rule(rej_rule_raw, rejected, sample_id + "_rejected")
    #         rej_conc = normalize_conclusion(rej_conc_raw, rejected, sample_id + "_rejected")

    #         # 同样剔除因为模型幻觉导致格式严重残缺(Unknown)的负样本
    #         if rej_cond == "Unknown" or rej_rule == "Unknown" or rej_conc == "Unknown":
    #             deleted_invalid_rejected += 1
    #             continue
            
    #         valid_gener_data_list.append(data)

    # print(f"--- SFT生成的原始总样本数: {total_gener_samples} ---")
    # print(f"--- 剔除 (因ID不合规与基准断配): {deleted_not_in_cleaned} ---")
    # print(f"--- 剔除 (因负样本格式不完整/Unknown): {deleted_invalid_rejected} ---")
    # print(f"--- 最终并对齐后的高质量SFT样本数: {len(valid_gener_data_list)} ---")
    
    # if len(valid_gener_data_list) > 0:
    #     with open(gener_cleaned_file_path, 'w', encoding='utf-8') as f_out:
    #         for d in valid_gener_data_list:
    #             f_out.write(json.dumps(d, ensure_ascii=False) + '\n')
    #     print(f"负样本对齐及清洗后的数据集保存至: {gener_cleaned_file_path}\n")
    # print("====== 清洗结束 ======")

if __name__ == "__main__":
    main()
