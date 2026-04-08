'''
Author: yufei Ji
Date: 2026-04-08 10:30:00
Description: 此脚本专门分析清洗后的正负样本对齐数据 (dpo_dataset_sft_model_gener_cleaned.jsonl)。
提取正负样本在排队长度、相位拥挤水平等方面的具体特征并做分布展示和对照差异分析。
分析报告直接写入当前目录下的 dpo_dataset_analysis_report.txt。
FilePath: /VLMTraffic/src/dataset/DPO_data_construct/analyze_dpo_data.py
'''
import json
import os
import re
from collections import defaultdict
import numpy as np

SCENARIOS = ["JiNan_test", "JiNan", "Hangzhou", "Hongkong_YMT", "NewYork", "SouthKorea_Songdo", "France_Massy"]

def extract_lane_counts(text):
    """提取排队长度字典"""
    lane_counts = {}
    matches = re.finditer(r'(North|South|East|West) Approach:.*?Lane 1[^:]*:(\d+),?\s*Lane 2[^:]*:(\d+),?\s*Lane 3[^:]*:(\d+)', text)
    for match in matches:
        approach = match.group(1)
        lane_counts[f"{approach}_L1"] = int(match.group(2))
        lane_counts[f"{approach}_L2"] = int(match.group(3))
        lane_counts[f"{approach}_L3"] = int(match.group(4))
    return lane_counts

def extract_congestion_levels(text):
    """
    提取 各Phase 的拥堵水平。
    例如 Phase 0 (ETWT): High 
    返回 dict: {'Phase 0': 'High', 'Phase 1': 'Low', ...}
    """
    congestion = {}
    matches = re.finditer(r'Phase (\d) .*?:\s*(Low|Medium|High|Gridlock)', text, re.IGNORECASE)
    for match in matches:
        congestion[f"Phase {match.group(1)}"] = match.group(2).capitalize()
    return congestion

def extract_field(text, field_name):
    """提取文本中对应字段名后的值"""
    match = re.search(rf"- {field_name}:\s*(?:-\s*)?([^\n]+)", text, re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        if val:
            return val
    return "Unknown"

def normalize_final_cond(cond):
    c = cond.lower().strip()
    if c in ["normal", "noraml", "normal."]: return "Normal"
    elif c in ["special", "speical", "special."]: return "Special"
    return "Unknown"

def normalize_rule(rule):
    r = rule.lower().replace(".", "").replace(",", "").strip()
    if "emergency_priority" in r: return "Emergency_Priority"
    elif "incident_avoidance" in r: return "Incident_Avoidance"
    elif "fallback_static" in r or "fallback_cycle" in r: return "Fallback_Static"
    elif "contextual_adaptation" in r: return "Contextual_Adaptation"
    elif "tie" in r or "ie_breaker" in r:
        if "index" in r or "(c)" in r: return "Tie_Breaker (Index_Order)"
        elif "straight" in r or "(a)" in r: return "Tie_Breaker (Straight > Left)"
        elif "max single" in r or "(b)" in r: return "Tie_Breaker (Max Single Lane)"
        return "Tie_Breaker"
    elif "bottleneck" in r: return "Bottleneck_Rule"
    return "Unknown"

def normalize_conclusion(conc):
    c = conc.lower().strip()
    match = re.search(r'phase\s*(\d)', c, re.IGNORECASE)
    if match: return f"Phase {match.group(1)}"
    return "Unknown"

def parse_folder_name(sample_id):
    """
    从 sample_id 中解析出数据来源文件夹名。
    例如 ID: JiNan_anon_3_4_jinan_real_2500.rou_15_intersection_2_3
    提取出文件夹: [JiNan-anon_3_4_jinan_real_2500.rou]
    """
    scenario = "unknown"
    for s in SCENARIOS:
        if sample_id.startswith(s + "_"):
            scenario = s
            break
            
    if scenario != "unknown":
        remainder = sample_id[len(scenario)+1:]
        # remainder 即为 anon_3_4_jinan_real_2500.rou_15_intersection_2_3
        # 使用正则把后面 step 的 _数字_intersection... 切掉
        m = re.search(r'^(.*?)_(\d+)_intersection_.*$', remainder)
        if m:
            route_file = m.group(1)
            return f"[{scenario}-{route_file}]"
            
    return f"[{sample_id.split('_')[0]}-unknown]"

def main():
    base_dir = os.path.dirname(__file__)
    data_file = os.path.join(base_dir, 'dpo_dataset_sft_model_gener_cleaned.jsonl')
    report_file = os.path.join(base_dir, 'dpo_dataset_analysis_report.txt')
    
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        return

    # ----------- 数据结构初始化 -----------
    folder_counts = defaultdict(int)
    
    # 特征分布 (正负两套)
    chosen_lanes = defaultdict(list)
    rejected_lanes = defaultdict(list)
    
    chosen_congestion = defaultdict(lambda: defaultdict(int))
    rejected_congestion = defaultdict(lambda: defaultdict(int))

    chosen_cond_dist = defaultdict(int)
    rejected_cond_dist = defaultdict(int)

    chosen_rule_dist = defaultdict(int)
    rejected_rule_dist = defaultdict(int)

    chosen_conc_dist = defaultdict(int)
    rejected_conc_dist = defaultdict(int)
    
    # 差异对比
    diff_queue_len = []       # 每个样本正负全部车道差值的 L1 平均
    diff_congestion_count = 0 # 记录拥堵发生不同断判的样本数
    diff_rule_count = 0       # 记录规则选用产生分歧的样本数

    total_samples = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_samples += 1
            data = json.loads(line)
            sample_id = data.get("id", "Unknown_ID")
            
            # --- 1. 统计数据来源 ---
            folder_name = parse_folder_name(sample_id)
            folder_counts[folder_name] += 1
            
            # 提取清洗后的文本
            chosen_text = data['chosen'][0]['content'].replace('**', '')
            rejected_text = data['rejected'][0]['content'].replace('**', '')

            # --- 2. 正向样本 (Chosen) 提取与累积 ---
            c_lanes = extract_lane_counts(chosen_text)
            for k, v in c_lanes.items(): chosen_lanes[k].append(v)
            
            c_cong = extract_congestion_levels(chosen_text)
            for phase, lvl in c_cong.items(): chosen_congestion[phase][lvl] += 1
            
            c_cond = normalize_final_cond(extract_field(chosen_text, "Final Condition"))
            chosen_cond_dist[c_cond] += 1
            
            c_rule = normalize_rule(extract_field(chosen_text, "Rule Identification"))
            chosen_rule_dist[c_rule] += 1
            
            c_conc = normalize_conclusion(extract_field(chosen_text, "Conclusion"))
            chosen_conc_dist[c_conc] += 1
            
            # --- 3. 负向样本 (Rejected) 提取与累积 ---
            r_lanes = extract_lane_counts(rejected_text)
            for k, v in r_lanes.items(): rejected_lanes[k].append(v)
            
            r_cong = extract_congestion_levels(rejected_text)
            for phase, lvl in r_cong.items(): rejected_congestion[phase][lvl] += 1
            
            r_cond = normalize_final_cond(extract_field(rejected_text, "Final Condition"))
            rejected_cond_dist[r_cond] += 1
            
            r_rule = normalize_rule(extract_field(rejected_text, "Rule Identification"))
            rejected_rule_dist[r_rule] += 1
            
            r_conc = normalize_conclusion(extract_field(rejected_text, "Conclusion"))
            rejected_conc_dist[r_conc] += 1
            
            # --- 4. 差异分析对比 ---
            # 1. 排队识别差异
            lane_keys = set(c_lanes.keys()).union(set(r_lanes.keys()))
            avg_diff = np.mean([abs(c_lanes.get(k, 0) - r_lanes.get(k, 0)) for k in lane_keys]) if lane_keys else 0
            diff_queue_len.append(avg_diff)
            
            # 2. 拥堵分级差异 (只要有其中一个 Phase 不一致，就算对拥堵判断分歧)
            is_cong_diff = False
            for phase in ["Phase 0", "Phase 1", "Phase 2", "Phase 3"]:
                if c_cong.get(phase) != r_cong.get(phase):
                    is_cong_diff = True
                    break
            if is_cong_diff:
                diff_congestion_count += 1
                
            # 3. 规则判断差异
            if c_rule != r_rule:
                diff_rule_count += 1

    # -------------------- 生成输出报告 --------------------
    report_lines = []
    def log(msg=""): 
        report_lines.append(msg)
    
    log("=========================================")
    log("    DPO 正负样本对比及统计分析报告 (清洗后)")
    log("=========================================\n")
    
    log(f"--- 样本总计: {total_samples} 条 ---")
    log("\n[1] 数据来源文件夹分布统计：")
    # 按数量降序
    for folder, count in sorted(folder_counts.items(), key=lambda x: x[1], reverse=True):
        log(f"文件夹 {folder}: {count} 条有效 DPO 数据")
    log("-" * 30)

    log("\n[2] 排队长度 (车道平均值) 对比：")
    log(f"{'车道 (Lane)':<15} | {'正样本平均':<15} | {'负样本平均':<15}")
    lane_keys_sorted = sorted(list(chosen_lanes.keys()))
    for k in lane_keys_sorted:
        c_mean = np.mean(chosen_lanes[k]) if chosen_lanes[k] else 0
        r_mean = np.mean(rejected_lanes[k]) if rejected_lanes[k] else 0
        log(f"{k:<15} | {c_mean:<12.2f} | {r_mean:<12.2f}")
        
    log("\n[3] 相位拥堵水平 (Congestion Level) 分布：")
    for phase in sorted(chosen_congestion.keys()):
        log(f"  {phase}:")
        c_lvls = chosen_congestion[phase]
        r_lvls = rejected_congestion[phase]
        all_lvls = set(c_lvls.keys()).union(set(r_lvls.keys()))
        for lvl in sorted(all_lvls):
            log(f"    - {lvl:<10} => 正样本: {c_lvls.get(lvl, 0):<5} | 负样本: {r_lvls.get(lvl, 0):<5}")
            
            
    log("\n[4] Final Condition 是否紧急情况对比：")
    all_conds = set(chosen_cond_dist.keys()).union(set(rejected_cond_dist.keys()))
    for c in sorted(all_conds):
        log(f"  {c:<10} => 正样本: {chosen_cond_dist.get(c, 0):<5} | 负样本: {rejected_cond_dist.get(c, 0):<5}")

    log("\n[5] Rule Identification 规则使用对比：")
    # 按名称排列
    all_rules = sorted(set(chosen_rule_dist.keys()).union(set(rejected_rule_dist.keys())))
    for r in all_rules:
        log(f"  {r:<30} => 正样本: {chosen_rule_dist.get(r, 0):<5} | 负样本: {rejected_rule_dist.get(r, 0):<5}")

    log("\n[6] Conclusion 最终决策选择对比：")
    all_concs = sorted(set(chosen_conc_dist.keys()).union(set(rejected_conc_dist.keys())))
    for c in all_concs:
        log(f"  {c:<10} => 正样本: {chosen_conc_dist.get(c, 0):<5} | 负样本: {rejected_conc_dist.get(c, 0):<5}")

    log("\n=========================================")
    log("    正负样本核心差异量化评估 (Difference)")
    log("=========================================")
    mean_q_diff = np.mean(diff_queue_len) if diff_queue_len else 0
    log(f"1. 平均排队长度识别差异 (MAE): 每样本每车道平均相差 {mean_q_diff:.3f} 辆车")
    log(f"2. 相位拥挤情况断判分歧率:    在 {diff_congestion_count} 个样本上存在差异 (占比 {diff_congestion_count/total_samples*100:.2f}%)")
    log(f"3. 核心决策规则选用变更率:    在 {diff_rule_count} 个样本上发生规则偏差 (占比 {diff_rule_count/total_samples*100:.2f}%)")
    
    report_content = "\n".join(report_lines)
    
    # 打印到控制台
    print(report_content)
    
    # 写入到输出文件，不需要使用原调用的外挂脚本
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n[执行完毕] 此详细分析报告已自动保存至:\n{report_file}")

if __name__ == "__main__":
    main()
