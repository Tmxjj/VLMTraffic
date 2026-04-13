'''
Author: yufei Ji
Date: 2026-04-13
Description: 泛化场景指标收集脚本。
             扫描 data/eval/{Scenario}/{route_name}/{model_name}/ 下的 SUMO 输出文件，
             计算 ATT/AWT/AQL 并写入 results/generalization_result.csv。

使用方式：
    python src/evaluation/generalization_metrics.py

适用场景（均不参与训练，用于泛化验证）：
    - SouthKorea_Songdo  (songdo.rou.xml)
    - France_Massy       (massy.rou.xml)
    - Hongkong_YMT       (YMT.rou.xml)
    - NewYork            (anon_28_7_newyork_real_double.rou.xml)
    - NewYork            (anon_28_7_newyork_real_triple.rou.xml)
'''
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.evaluation.metrics import MetricsCalculator

# ─────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENERALIZATION_CSV_PATH = os.path.join(PROJECT_ROOT, "results", "generalization_result.csv")

# ─────────────────────────────────────────────────────────────────
# CSV 列映射
#
# CSV 列结构（0-indexed）：
#   col0=Category, col1=Method,
#   col2-4  = SouthKorea_Songdo (ATT, AWT, AQL)
#   col5-7  = France_Massy      (ATT, AWT, AQL)
#   col8-10 = Hongkong_YMT      (ATT, AWT, AQL)
#   col11-13= NewYork_double    (ATT, AWT, AQL)
#   col14-16= NewYork_triple    (ATT, AWT, AQL)
# ─────────────────────────────────────────────────────────────────
ROUTE_TO_CSV_COLS = {
    ("SouthKorea_Songdo", "songdo.rou.xml"):                           (2,  3,  4),
    ("France_Massy",      "massy.rou.xml"):                            (5,  6,  7),
    ("Hongkong_YMT",      "YMT.rou.xml"):                              (8,  9,  10),
    ("NewYork",           "anon_28_7_newyork_real_double.rou.xml"):    (11, 12, 13),
    ("NewYork",           "anon_28_7_newyork_real_triple.rou.xml"):    (14, 15, 16),
}

# 目录名 → CSV Method 列字符串（与 generalization_result.csv 中的行名对应）
METHOD_TO_CSV_ROW_KEY = {
    "fixed_time":   "FixedTime",
    "max_pressure": "MaxPressure",
    # VLM 方法：目录名与模型检查点相关，在下方 EVAL_PLAN 中直接指定 row_key
}

# ─────────────────────────────────────────────────────────────────
# 评测计划：列举需要收集的 (场景, 路由, 方法目录名, CSV行名) 四元组
#
# 方法目录名来自 vlm_decision.py --model_name 参数（或 fixed_time / max_pressure）。
# CSV 行名必须与 generalization_result.csv 中 Method 列完全一致。
# ─────────────────────────────────────────────────────────────────
EVAL_PLAN = [
    # ── 基线方法 ─────────────────────────────────────────────────
    ("SouthKorea_Songdo", "songdo.rou.xml",                        "fixed_time",               "FixedTime"),
    ("SouthKorea_Songdo", "songdo.rou.xml",                        "max_pressure",             "MaxPressure"),
    ("France_Massy",      "massy.rou.xml",                         "fixed_time",               "FixedTime"),
    ("France_Massy",      "massy.rou.xml",                         "max_pressure",             "MaxPressure"),
    ("Hongkong_YMT",      "YMT.rou.xml",                           "fixed_time",               "FixedTime"),
    ("Hongkong_YMT",      "YMT.rou.xml",                           "max_pressure",             "MaxPressure"),
    ("NewYork",           "anon_28_7_newyork_real_double.rou.xml", "fixed_time",               "FixedTime"),
    ("NewYork",           "anon_28_7_newyork_real_double.rou.xml", "max_pressure",             "MaxPressure"),
    ("NewYork",           "anon_28_7_newyork_real_triple.rou.xml", "fixed_time",               "FixedTime"),
    ("NewYork",           "anon_28_7_newyork_real_triple.rou.xml", "max_pressure",             "MaxPressure"),

    # ── VLM 方法（--model_name 参数决定目录名；按实际目录名修改下方字段）──
    # 格式：(场景, 路由文件, 模型目录名, CSV行名)
    # 示例：
    # ("SouthKorea_Songdo", "songdo.rou.xml", "qwen3-vl-8b-zero-shot", "qwen3-8b-vl Zero-Shot"),
    # ("SouthKorea_Songdo", "songdo.rou.xml", "qwen3-vl-8b-sft",       "qwen3-8b-vl SFT"),
    # ("SouthKorea_Songdo", "songdo.rou.xml", "qwen3-vl-8b-sft-dpo",   "qwen3-8b-vl SFT+DPO"),
    # ... 依此类推，补全其余场景/路由的 VLM 行
]


def update_generalization_csv(metrics: dict, scenario: str, route_file: str, row_key: str):
    """
    将 ATT/AWT/AQL 写入 results/generalization_result.csv 对应的行列位置。

    Args:
        metrics   : 包含 ATT, AWT, AQL 的指标字典
        scenario  : 场景名称（如 "SouthKorea_Songdo"）
        route_file: 路由文件名（如 "songdo.rou.xml"）
        row_key   : CSV Method 列的字符串（如 "MaxPressure"）
    """
    col_indices = ROUTE_TO_CSV_COLS.get((scenario, route_file))
    if col_indices is None:
        print(f"⚠️  [CSV] ({scenario}, {route_file}) 未在 ROUTE_TO_CSV_COLS 中配置，跳过。")
        return

    if not os.path.exists(GENERALIZATION_CSV_PATH):
        print(f"❌ [CSV] 泛化结果 CSV 不存在: {GENERALIZATION_CSV_PATH}")
        return

    att_col, awt_col, aql_col = col_indices

    try:
        with open(GENERALIZATION_CSV_PATH, 'r', newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))

        # 找到目标行（Method 列完全匹配 row_key）
        target_row_idx = None
        for idx, row in enumerate(rows):
            if len(row) > 1 and row[1].strip() == row_key:
                target_row_idx = idx
                break

        if target_row_idx is None:
            print(f"⚠️  [CSV] 未找到 Method='{row_key}' 的行，跳过写入。")
            return

        target_row = rows[target_row_idx]
        max_col = max(att_col, awt_col, aql_col)
        while len(target_row) <= max_col:
            target_row.append('')

        target_row[att_col] = f"{metrics['ATT']:.2f}"
        target_row[awt_col] = f"{metrics['AWT']:.2f}"
        target_row[aql_col] = f"{metrics['AQL']:.6f}"
        rows[target_row_idx] = target_row

        with open(GENERALIZATION_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"✅ [CSV] {row_key} | {scenario}/{route_file} | "
              f"ATT={metrics['ATT']:.2f}, AWT={metrics['AWT']:.2f}, AQL={metrics['AQL']:.6f}")

    except Exception as e:
        print(f"❌ [CSV] 写入失败: {e}")


def collect_all():
    """扫描 EVAL_PLAN，计算每个条目的指标并写入 CSV。"""
    eval_root = os.path.join(PROJECT_ROOT, "data", "eval")
    total, success, skipped = 0, 0, 0

    for scenario, route_file, model_dir, row_key in EVAL_PLAN:
        total += 1
        base_path  = os.path.join(eval_root, scenario, route_file, model_dir)
        stat_path  = os.path.join(base_path, "statistic_output.xml")
        queue_path = os.path.join(base_path, "queue_output.xml")

        if not os.path.exists(stat_path) or not os.path.exists(queue_path):
            print(f"⚠️  [Skip] 文件缺失: {base_path}")
            skipped += 1
            continue

        try:
            calc = MetricsCalculator()
            metrics = calc.calculate_from_files(stat_path, queue_path, scenario)
        except Exception as e:
            print(f"❌ 指标计算失败 {base_path}: {e}")
            skipped += 1
            continue

        update_generalization_csv(metrics, scenario, route_file, row_key)
        success += 1

    print(f"\n{'='*60}")
    print(f"泛化指标收集完成：共 {total} 条，成功 {success} 条，跳过 {skipped} 条")
    print(f"结果已写入：{GENERALIZATION_CSV_PATH}")


if __name__ == "__main__":
    collect_all()
