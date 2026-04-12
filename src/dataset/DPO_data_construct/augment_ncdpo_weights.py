'''
Author: yufei Ji
Date: 2026-04-11
Description: 为 DPO 数据集增加计数误差权重字段（count_error_weight），
             用于 NCDPO（数值距离连续惩罚 DPO）训练。

功能说明：
    遍历已清洗的 DPO 数据集 (dpo_dataset_sft_model_gener_cleaned.jsonl)，
    对每条样本：
      1. 从正样本 (chosen) 和负样本 (rejected) 文本中提取各车道车辆计数
      2. 计算两者之间的绝对差均值 (MAE) 作为"计数误差"
      3. 将该误差值作为 count_error_weight 字段写入新的 JSONL 文件

输出文件：
    dpo_dataset_ncdpo.jsonl —— 在原始字段基础上新增 count_error_weight (float)

使用方法：
    conda activate VLMTraffic
    python src/dataset/DPO_data_construct/augment_ncdpo_weights.py
FilePath: /VLMTraffic/src/dataset/DPO_data_construct/augment_ncdpo_weights.py
'''
import json
import re
import os
import numpy as np
from collections import defaultdict


def extract_lane_counts(text: str) -> dict:
    """
    从 Thought 文本的 Scene Understanding 部分提取各车道排队车辆数。

    匹配形如：
        North Approach: Lane 1(Left-Turn):2, Lane 2(Straight):3, Lane 3(Right-Turn):0

    Returns:
        dict: {'North_L1': 2, 'North_L2': 3, ...}，提取失败时返回空字典。
    """
    lane_counts = {}
    # 正则：方向 + 三条车道的数字
    matches = re.finditer(
        r'(North|South|East|West)\s+Approach:.*?'
        r'Lane\s+1[^:]*:\s*(\d+),?\s*'
        r'Lane\s+2[^:]*:\s*(\d+),?\s*'
        r'Lane\s+3[^:]*:\s*(\d+)',
        text,
        re.IGNORECASE
    )
    for match in matches:
        approach = match.group(1).capitalize()
        lane_counts[f"{approach}_L1"] = int(match.group(2))
        lane_counts[f"{approach}_L2"] = int(match.group(3))
        lane_counts[f"{approach}_L3"] = int(match.group(4))
    return lane_counts


def compute_count_error_weight(chosen_text: str, rejected_text: str) -> float:
    """
    计算正负样本之间的计数误差权重（MAE，跨所有可配对车道）。

    数学定义：
        ε = (1/K) * Σ_k |count_chosen_k - count_rejected_k|
    其中 K 为正负样本共同识别到的车道数量。

    Args:
        chosen_text:   正样本（ground-truth）的文本内容
        rejected_text: 负样本（模型生成）的文本内容

    Returns:
        float: 平均绝对误差（MAE），0.0 表示计数完全一致或无法解析。
    """
    chosen_counts  = extract_lane_counts(chosen_text)
    rejected_counts = extract_lane_counts(rejected_text)

    # 仅对两侧都成功解析的车道计算误差
    common_lanes = set(chosen_counts.keys()) & set(rejected_counts.keys())
    if not common_lanes:
        return 0.0

    mae = float(np.mean([
        abs(chosen_counts[k] - rejected_counts[k])
        for k in common_lanes
    ]))
    return mae


def print_distribution(errors: list, total: int):
    """打印计数误差的分布直方图（分桶统计）"""
    bins   = [0, 0.25, 0.5, 1.0, 2.0, 3.0, float('inf')]
    labels = ["[0,0.25)", "[0.25,0.5)", "[0.5,1.0)", "[1.0,2.0)", "[2.0,3.0)", "[3.0,∞)"]
    print("\n[MAE 分布 (count_error_weight 原始值)]")
    for lo, hi, label in zip(bins[:-1], bins[1:], labels):
        cnt = sum(1 for e in errors if lo <= e < hi)
        bar = '█' * int(cnt / total * 50)
        print(f"  {label:12s}: {cnt:5d} 个 ({cnt/total*100:5.1f}%) {bar}")


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    input_file  = os.path.join(base_dir, "dpo_dataset_sft_model_gener_cleaned.jsonl")
    output_file = os.path.join(base_dir, "dpo_dataset_ncdpo.jsonl")

    if not os.path.exists(input_file):
        print(f"[Error] 找不到输入文件: {input_file}")
        print("请先运行 clean_dpo_data.py 生成清洗后的数据集。")
        return

    print(f"[Info] 读取输入文件: {input_file}")

    # -----------------------------------------------
    # 遍历数据集，计算并写入 count_error_weight
    # -----------------------------------------------
    total          = 0
    zero_error_cnt = 0
    parse_fail_cnt = 0  # 正负双侧均无法解析出车道数的样本
    errors         = []
    per_phase_diff = defaultdict(list)  # 各方向的误差统计

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            data = json.loads(line.strip())
            total += 1

            # 去除 Markdown 加粗符号，避免干扰正则
            chosen_text   = data['chosen'][0]['content'].replace('**', '')
            rejected_text = data['rejected'][0]['content'].replace('**', '')

            # 计算计数误差
            weight = compute_count_error_weight(chosen_text, rejected_text)

            # 统计解析失败的情况（正负侧均未提取到车道数）
            c_counts = extract_lane_counts(chosen_text)
            r_counts = extract_lane_counts(rejected_text)
            if not c_counts and not r_counts:
                parse_fail_cnt += 1

            if weight == 0.0:
                zero_error_cnt += 1
            errors.append(weight)

            # 统计各车道方向的误差
            common = set(c_counts.keys()) & set(r_counts.keys())
            for lane_key in common:
                direction = lane_key.split('_')[0]  # North / South / East / West
                per_phase_diff[direction].append(abs(c_counts[lane_key] - r_counts[lane_key]))

            # 写入新字段
            data['count_error_weight'] = weight
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    # -----------------------------------------------
    # 打印统计报告
    # -----------------------------------------------
    print(f"\n{'='*55}")
    print(f"  NCDPO 数据集权重增强报告")
    print(f"{'='*55}")
    print(f"  处理总样本数:             {total}")
    print(f"  车道解析失败 (双侧为空):   {parse_fail_cnt} ({parse_fail_cnt/total*100:.1f}%)")
    print(f"  计数误差 = 0 (完全一致):   {zero_error_cnt} ({zero_error_cnt/total*100:.1f}%)")
    print(f"\n  count_error_weight 统计:")
    print(f"    均值  (Mean):   {np.mean(errors):.4f}")
    print(f"    中位数(Median): {np.median(errors):.4f}")
    print(f"    标准差(Std):    {np.std(errors):.4f}")
    print(f"    最大值(Max):    {np.max(errors):.4f}")
    print(f"    P90:            {np.percentile(errors, 90):.4f}")
    print(f"    P95:            {np.percentile(errors, 95):.4f}")

    print(f"\n  各方向车道平均计数误差:")
    for direction in ['North', 'South', 'East', 'West']:
        diffs = per_phase_diff.get(direction, [])
        if diffs:
            print(f"    {direction:5s}: {np.mean(diffs):.4f} 辆/车道")

    print_distribution(errors, total)

    print(f"\n[Info] tanh 归一化预览 (eps_scale=2.0, alpha=1.0, beta_base=0.1):")
    print(f"  {'ε (MAE)':>10s} → {'w=tanh(ε/2)':>12s} → {'β_eff':>10s}")
    for eps_val in [0.0, 0.25, 0.5, 1.0, float(np.mean(errors)), 2.0, 3.0, 5.0]:
        w = np.tanh(eps_val / 2.0)
        beta_eff = 0.1 * (1.0 + 1.0 * w)
        print(f"  {eps_val:>10.3f} → {w:>12.4f} → {beta_eff:>10.4f}")

    print(f"\n[输出] 已写入增强数据集: {output_file}")
    print(f"  (包含原始 DPO 字段 + count_error_weight 新字段)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
