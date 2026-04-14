'''
Author: yufei Ji
Date: 2026-04-13
Description: 紧急车辆场景指标收集脚本。
             扫描 data/eval/{Scenario}/{emergy_route}/{method}/ 下的 SUMO 输出文件，
             计算 ATT/AWT/AQL（所有车辆）和 EATT/EAWT（紧急车辆）并写入
             results/emergency_result.csv。

             紧急车辆指标定义：
               EATT (Emergency ATT): Average Emergency Vehicle Travel Time
               EAWT (Emergency AWT): Average Emergency Vehicle Waiting Time

使用方式：
    python src/evaluation/emergency_metrics.py

路由文件命名规则（_emergy 后缀）：
    JiNan:             anon_3_4_jinan_real_emergy.rou.xml
    Hangzhou:          anon_4_4_hangzhou_real_emergy.rou.xml
    SouthKorea_Songdo: songdo_emergy.rou.xml
    France_Massy:      massy_emergy.rou.xml
    Hongkong_YMT:      YMT_emergy.rou.xml
    NewYork:           anon_28_7_newyork_real_double_emergy.rou.xml
'''
import csv
import os
import sys

sys.path.append(".")
from src.evaluation.metrics import MetricsCalculator

# ─────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = "."
EMERGENCY_CSV_PATH = os.path.join(PROJECT_ROOT, "results", "emergency_result.csv")

# ─────────────────────────────────────────────────────────────────
# CSV 列映射（0-indexed）
#
# CSV 列结构：
#   col0=Category, col1=Method,
#   col2-6   = JiNan_emergy          (ATT, AWT, AQL, EATT, EAWT)
#   col7-11  = Hangzhou_emergy        (ATT, AWT, AQL, EATT, EAWT)
#   col12-16 = SouthKorea_Songdo_emergy (ATT, AWT, AQL, EATT, EAWT)
#   col17-21 = France_Massy_emergy    (ATT, AWT, AQL, EATT, EAWT)
#   col22-26 = Hongkong_YMT_emergy    (ATT, AWT, AQL, EATT, EAWT)
#   col27-31 = NewYork_double_emergy  (ATT, AWT, AQL, EATT, EAWT)
# ─────────────────────────────────────────────────────────────────
ROUTE_TO_CSV_COLS = {
    ("JiNan",             "anon_3_4_jinan_real_emergy.rou"):              (2,  3,  4,  5,  6),
    ("Hangzhou",          "anon_4_4_hangzhou_real_emergy.rou"):           (7,  8,  9,  10, 11),
    ("SouthKorea_Songdo", "songdo_emergy.rou"):                           (12, 13, 14, 15, 16),
    ("France_Massy",      "massy_emergy.rou"):                            (17, 18, 19, 20, 21),
    ("Hongkong_YMT",      "YMT_emergy.rou"):                              (22, 23, 24, 25, 26),
    ("NewYork",           "anon_28_7_newyork_real_double_emergy.rou"):    (27, 28, 29, 30, 31),
}

# ─────────────────────────────────────────────────────────────────
# 评测计划：(场景, 路由文件, 方法目录名, CSV行名)
#
# 方法目录名由 vlm_decision.py --model_name 参数决定（或 fixed_time / max_pressure）。
# CSV 行名必须与 emergency_result.csv 中 Method 列完全一致。
# ─────────────────────────────────────────────────────────────────
EVAL_PLAN = [
    # ── 基线方法 ─────────────────────────────────────────────────
    ("JiNan",             "anon_3_4_jinan_real_emergy.rou",              "fixed_time",   "FixedTime"),
    ("JiNan",             "anon_3_4_jinan_real_emergy.rou",              "max_pressure", "MaxPressure"),
    ("Hangzhou",          "anon_4_4_hangzhou_real_emergy.rou",           "fixed_time",   "FixedTime"),
    ("Hangzhou",          "anon_4_4_hangzhou_real_emergy.rou",           "max_pressure", "MaxPressure"),
    ("SouthKorea_Songdo", "songdo_emergy.rou",                           "fixed_time",   "FixedTime"),
    ("SouthKorea_Songdo", "songdo_emergy.rou",                           "max_pressure", "MaxPressure"),
    ("France_Massy",      "massy_emergy.rou",                            "fixed_time",   "FixedTime"),
    ("France_Massy",      "massy_emergy.rou",                            "max_pressure", "MaxPressure"),
    ("Hongkong_YMT",      "YMT_emergy.rou",                              "fixed_time",   "FixedTime"),
    ("Hongkong_YMT",      "YMT_emergy.rou",                              "max_pressure", "MaxPressure"),
    ("NewYork",           "anon_28_7_newyork_real_double_emergy.rou",    "fixed_time",   "FixedTime"),
    ("NewYork",           "anon_28_7_newyork_real_double_emergy.rou",    "max_pressure", "MaxPressure"),

    # ── VLM 方法（按实际 --model_name 目录名填写）────────────────
    # 示例（取消注释并按需修改）：
    # ("JiNan",             "anon_3_4_jinan_real_emergy.rou",   "qwen3-vl-8b-zero-shot", "qwen3-8b-vl Zero-Shot"),
    # ("JiNan",             "anon_3_4_jinan_real_emergy.rou",   "qwen3-vl-8b-sft",       "qwen3-8b-vl SFT"),
    # ("JiNan",             "anon_3_4_jinan_real_emergy.rou",   "qwen3-vl-8b-sft-dpo",   "qwen3-8b-vl SFT+DPO"),
    # ("Hangzhou",          "anon_4_4_hangzhou_real_emergy.rou","qwen3-vl-8b-zero-shot", "qwen3-8b-vl Zero-Shot"),
    # ("Hangzhou",          "anon_4_4_hangzhou_real_emergy.rou","qwen3-vl-8b-sft",       "qwen3-8b-vl SFT"),
    # ("Hangzhou",          "anon_4_4_hangzhou_real_emergy.rou","qwen3-vl-8b-sft-dpo",   "qwen3-8b-vl SFT+DPO"),
    # ... 以此类推补全其余场景
]


def update_emergency_csv(metrics: dict, scenario: str, route_file: str, row_key: str):
    """
    将 ATT/AWT/AQL/Special_ATT/Special_AWT 写入 results/emergency_result.csv。

    Args:
        metrics    : MetricsCalculator.calculate_from_files() 返回的指标字典
        scenario   : 场景名称
        route_file : 路由文件名（带 _emergy 后缀）
        row_key    : CSV Method 列的字符串
    """
    col_indices = ROUTE_TO_CSV_COLS.get((scenario, route_file))
    if col_indices is None:
        print(f"⚠️  [CSV] ({scenario}, {route_file}) 未在 ROUTE_TO_CSV_COLS 中配置，跳过。")
        return

    if not os.path.exists(EMERGENCY_CSV_PATH):
        print(f"❌ [CSV] 紧急场景结果 CSV 不存在: {EMERGENCY_CSV_PATH}")
        return

    att_col, awt_col, aql_col, eatt_col, eawt_col = col_indices

    try:
        with open(EMERGENCY_CSV_PATH, 'r', newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))

        target_row_idx = None
        for idx, row in enumerate(rows):
            if len(row) > 1 and row[1].strip() == row_key:
                target_row_idx = idx
                break

        if target_row_idx is None:
            print(f"⚠️  [CSV] 未找到 Method='{row_key}' 的行，跳过写入。")
            return

        target_row = rows[target_row_idx]
        max_col = max(att_col, awt_col, aql_col, eatt_col, eawt_col)
        while len(target_row) <= max_col:
            target_row.append('')

        target_row[att_col]  = f"{metrics.get('ATT', 0.0):.2f}"
        target_row[awt_col]  = f"{metrics.get('AWT', 0.0):.2f}"
        target_row[aql_col]  = f"{metrics.get('AQL', 0.0):.6f}"
        target_row[eatt_col] = f"{metrics.get('Special_ATT', 0.0):.2f}"
        target_row[eawt_col] = f"{metrics.get('Special_AWT', 0.0):.2f}"
        rows[target_row_idx] = target_row

        with open(EMERGENCY_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"✅ [CSV] {row_key} | {scenario}/{route_file} | "
              f"ATT={metrics.get('ATT',0):.2f}, AWT={metrics.get('AWT',0):.2f}, "
              f"AQL={metrics.get('AQL',0):.4f}, "
              f"EATT={metrics.get('Special_ATT',0):.2f}, "
              f"EAWT={metrics.get('Special_AWT',0):.2f}")

    except Exception as e:
        print(f"❌ [CSV] 写入失败: {e}")


def collect_all():
    """扫描 EVAL_PLAN，计算每个条目的指标并写入 emergency_result.csv。"""
    eval_root = os.path.join(PROJECT_ROOT, "data", "eval")
    total, success, skipped = 0, 0, 0

    for scenario, route_file, model_dir, row_key in EVAL_PLAN:
        total += 1
        base_path     = os.path.join(eval_root, scenario, route_file, model_dir)
        stat_path     = os.path.join(base_path, "statistic_output.xml")
        queue_path    = os.path.join(base_path, "queue_output.xml")
        tripinfo_path = os.path.join(base_path, "tripinfo.out.xml")

        if not os.path.exists(stat_path) or not os.path.exists(queue_path):
            print(f"⚠️  [Skip] 文件缺失: {base_path}")
            skipped += 1
            continue

        try:
            calc = MetricsCalculator()
            metrics = calc.calculate_from_files(
                stat_path, queue_path, scenario,
                tripinfo_file=tripinfo_path  # 提供 tripinfo 以计算紧急车辆指标
            )
        except Exception as e:
            print(f"❌ 指标计算失败 {base_path}: {e}")
            skipped += 1
            continue

        update_emergency_csv(metrics, scenario, route_file, row_key)
        success += 1

    print(f"\n{'='*60}")
    print(f"紧急场景指标收集完成：共 {total} 条，成功 {success} 条，跳过 {skipped} 条")
    print(f"结果已写入：{EMERGENCY_CSV_PATH}")


if __name__ == "__main__":
    collect_all()
