'''
Author: yufei Ji
Date: 2026-04-20
Description: 统一评估指标收集入口。
             替代原来的 6 个独立 *_metrics.py 脚本，通过 --type 参数指定事件类型。
             所有配置（CSV 路径、列映射、评测计划、计算参数）集中于本文件的 _CONFIGS 字典。

             输出路径规范（与 src/evaluation/run_eval.py 一致）：
               data/eval/{dataset}/{route_file_name}/{method}/
             scene_type 对应：
               normal       — 无事件（泛化场景 / JiNan / Hangzhou 基础评测）
               normal_triple— NewYork 高密度流量变体
               emergency    — 紧急车辆注入
               bus          — 公交/校车注入
               accident     — 交通事故
               debris       — 路面碎片/路障
               pedestrian   — 行人过街

             指标差异对照：
               generalization : ATT / AWT / AQL（无特殊车辆过滤）
                             comparison     : ATT / AWT / AQL（多数据集横向对比）
                             ablation       : ATT / AWT / AQL（Zero-Shot / SFT / SFT+RLVR）
               emergency      : ATT / AWT / AQL / Special_ATT(EATT) / Special_AWT(EAWT)
                                special_vtypes = {emergency, police, fire_engine}
               bus            : ATT / AWT / AQL / Special_ATT(BATT) / Special_AWT(BAWT)
                                special_vtypes = {bus, school_bus}
               accident       : ATT / AWT / AQL / MaxQL / TPT
                                event_id_prefixes = [accident_]
               debris         : ATT / AWT / AQL / MaxQL / TPT
                                event_id_prefixes = [debris_]
               pedestrian     : ATT / AWT / AQL / MaxQL / TPT
                                event_id_prefixes = [ped_]

使用方式：
    python src/evaluation/collect_metrics.py --type generalization
    python src/evaluation/collect_metrics.py --type comparison
    python src/evaluation/collect_metrics.py --type ablation
    python src/evaluation/collect_metrics.py --type emergency
    python src/evaluation/collect_metrics.py --type bus
    python src/evaluation/collect_metrics.py --type accident
    python src/evaluation/collect_metrics.py --type debris
    python src/evaluation/collect_metrics.py --type pedestrian
    python src/evaluation/collect_metrics.py --type all
'''
import argparse
import csv
import os
import sys

sys.path.append(".")
from src.evaluation.metrics import MetricsCalculator

PROJECT_ROOT = "."

# ── 各指标 key 的格式化函数 ───────────────────────────────────────────────────
_FMT = {
    'ATT':         lambda v: f"{float(v):.2f}",
    'AWT':         lambda v: f"{float(v):.2f}",
    'AQL':         lambda v: f"{float(v):.6f}",
    'MaxQL':       lambda v: f"{float(v):.2f}",
    'TPT':         lambda v: f"{int(v)}",
    'Special_ATT': lambda v: f"{float(v):.2f}",
    'Special_AWT': lambda v: f"{float(v):.2f}",
}

# ── 各事件类型完整配置 ─────────────────────────────────────────────────────────
#
# 每个配置项包含：
#   csv_path    : 结果 CSV 的相对路径
#   metric_keys : 按顺序写入的指标 key 列表
#   scene_to_cols: {(scenario, scene_type): (col_att, col_awt, ...)} 列映射
#   eval_plan   : [(scenario, scene_type, method, csv_row_key), ...] 评测计划
#                 - scenario  : 数据集键名（如 "JiNan"）
#                 - scene_type: 场景类型（如 "emergency"），决定 data/eval 下的目录名
#                 - method    : 方法名（fixed_time / max_pressure / <model_name>）
#                 - csv_row_key: 结果 CSV 中的行标识
#   calc_kwargs : 传递给 MetricsCalculator.calculate_from_files() 的额外参数
#
_CONFIGS: dict = {

    # ── 1. 泛化场景（无事件注入，3 项指标）────────────────────────────────────
    'generalization': {
        'csv_path': 'results/generalization_result.csv',
        'metric_keys': ['ATT', 'AWT', 'AQL'],
        'scene_to_cols': {
            ("SouthKorea_Songdo", "songdo"):         (2,  3,  4),
            ("France_Massy",      "massy"):         (5,  6,  7),
            ("Hongkong_YMT",      "YMT"):         (8,  9,  10),
            ("NewYork",           "anon_28_7_newyork_real_double"):         (11, 12, 13),
            ("NewYork",           "anon_28_7_newyork_real_triple"):  (14, 15, 16),
        },
        'eval_plan': [
            ("SouthKorea_Songdo", "songdo",         "fixed_time",   "FixedTime"),
            ("SouthKorea_Songdo", "songdo",         "max_pressure", "MaxPressure"),
            ("SouthKorea_Songdo", "songdo",         "qwen3-vl-8b",   "qwen3-vl-8b"),
            ("SouthKorea_Songdo", "songdo",         "qwen3-vl-8b-SFT",   "qwen3-vl-8b-SFT"),
            ("SouthKorea_Songdo", "songdo",         "qwen3-vl-8b-SFT-RLVR",   "qwen3-vl-8b-SFT-RLVR"),

            ("France_Massy",      "massy",         "fixed_time",   "FixedTime"),
            ("France_Massy",      "massy",         "max_pressure", "MaxPressure"),
            ("France_Massy",      "massy",         "qwen3-vl-8b", "qwen3-vl-8b"),
            ("France_Massy",      "massy",         "qwen3-vl-8b-SFT", "qwen3-vl-8b-SFT"),
            ("France_Massy",      "massy",         "qwen3-vl-8b-SFT-RLVR", "qwen3-vl-8b-SFT-RLVR"),


            ("Hongkong_YMT",      "YMT",         "fixed_time",   "FixedTime"),
            ("Hongkong_YMT",      "YMT",         "max_pressure", "MaxPressure"),
            ("Hongkong_YMT",      "YMT",         "qwen3-vl-8b", "qwen3-vl-8b"),
            ("Hongkong_YMT",      "YMT",         "qwen3-vl-8b-SFT", "qwen3-vl-8b-SFT"),
            ("Hongkong_YMT",      "YMT",         "qwen3-vl-8b-SFT-RLVR", "qwen3-vl-8b-SFT-RLVR"),

            ("NewYork",           "anon_28_7_newyork_real_double",         "fixed_time",   "FixedTime"),
            ("NewYork",           "anon_28_7_newyork_real_double",         "max_pressure", "MaxPressure"),
            ("NewYork",           "anon_28_7_newyork_real_double",         "qwen3-vl-8b", "qwen3-vl-8b"),
            ("NewYork",           "anon_28_7_newyork_real_double",         "qwen3-vl-8b-SFT", "qwen3-vl-8b-SFT"),
            ("NewYork",           "anon_28_7_newyork_real_double",         "qwen3-vl-8b-SFT-RLVR", "qwen3-vl-8b-SFT-RLVR"),

        ],
        'calc_kwargs': {},
    },

    # ── 1.1 对比总表（7 个常规路由，3 项指标）────────────────────────────────
    'comparison': {
        'csv_path': 'results/comparsion_result.csv',
        'metric_keys': ['ATT', 'AWT', 'AQL'],
        'scene_to_cols': {
            ("JiNan",    "anon_3_4_jinan_real"):                     (2,  3,  4),
            ("JiNan",    "anon_3_4_jinan_real_2000"):                (5,  6,  7),
            ("JiNan",    "anon_3_4_jinan_real_2500"):                (8,  9,  10),
            ("JiNan",    "anon_3_4_jinan_synthetic_24000_60min"):    (11, 12, 13),
            ("Hangzhou", "anon_4_4_hangzhou_real"):                  (14, 15, 16),
            ("Hangzhou", "anon_4_4_hangzhou_real_5816"):             (17, 18, 19),
            ("Hangzhou", "anon_4_4_hangzhou_synthetic_24000_60min"): (20, 21, 22),
        },
        'eval_plan': [
            # Traditional
            ("JiNan",    "anon_3_4_jinan_real",                     "fixed_time",   "FixedTime"),
            ("JiNan",    "anon_3_4_jinan_real",                     "max_pressure", "MaxPressure"),
            ("JiNan",    "anon_3_4_jinan_real_2000",                "fixed_time",   "FixedTime"),
            ("JiNan",    "anon_3_4_jinan_real_2000",                "max_pressure", "MaxPressure"),
            ("JiNan",    "anon_3_4_jinan_real_2500",                "fixed_time",   "FixedTime"),
            ("JiNan",    "anon_3_4_jinan_real_2500",                "max_pressure", "MaxPressure"),
            ("JiNan",    "anon_3_4_jinan_synthetic_24000_60min",    "fixed_time",   "FixedTime"),
            ("JiNan",    "anon_3_4_jinan_synthetic_24000_60min",    "max_pressure", "MaxPressure"),
            ("Hangzhou", "anon_4_4_hangzhou_real",                  "fixed_time",   "FixedTime"),
            ("Hangzhou", "anon_4_4_hangzhou_real",                  "max_pressure", "MaxPressure"),
            ("Hangzhou", "anon_4_4_hangzhou_real_5816",             "fixed_time",   "FixedTime"),
            ("Hangzhou", "anon_4_4_hangzhou_real_5816",             "max_pressure", "MaxPressure"),
            ("Hangzhou", "anon_4_4_hangzhou_synthetic_24000_60min", "fixed_time",   "FixedTime"),
            ("Hangzhou", "anon_4_4_hangzhou_synthetic_24000_60min", "max_pressure", "MaxPressure"),

            # E2ELight (ours)
            ("JiNan",    "anon_3_4_jinan_real",                     "qwen3-vl-4b",  "Qwen-3-vl-4b"),
            ("JiNan",    "anon_3_4_jinan_real",                     "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("JiNan",    "anon_3_4_jinan_real",                     ["sft", "qwen3-vl-8b-sft", "qwen3_vl_8b_sft", "Qwen3-VL-8B-SFT-Merged"], "SFT"),
            ("JiNan",    "anon_3_4_jinan_real",                     ["sft+dpo", "sft_dpo", "qwen3-vl-8b-sft-dpo", "qwen3_vl_8b_sft_dpo", "Qwen3-VL-8B-SFT-DPO-Merged"], "SFT+RLVR"),
            ("JiNan",    "anon_3_4_jinan_real_2000",                "qwen3-vl-4b",  "Qwen-3-vl-4b"),
            ("JiNan",    "anon_3_4_jinan_real_2000",                "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("JiNan",    "anon_3_4_jinan_real_2000",                ["sft", "qwen3-vl-8b-sft", "qwen3_vl_8b_sft", "Qwen3-VL-8B-SFT-Merged"], "SFT"),
            ("JiNan",    "anon_3_4_jinan_real_2000",                ["sft+dpo", "sft_dpo", "qwen3-vl-8b-sft-dpo", "qwen3_vl_8b_sft_dpo", "Qwen3-VL-8B-SFT-DPO-Merged"], "SFT+RLVR"),
            ("JiNan",    "anon_3_4_jinan_real_2500",                "qwen3-vl-4b",  "Qwen-3-vl-4b"),
            ("JiNan",    "anon_3_4_jinan_real_2500",                "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("JiNan",    "anon_3_4_jinan_real_2500",                ["sft", "qwen3-vl-8b-sft", "qwen3_vl_8b_sft", "Qwen3-VL-8B-SFT-Merged"], "SFT"),
            ("JiNan",    "anon_3_4_jinan_real_2500",                ["sft+dpo", "sft_dpo", "qwen3-vl-8b-sft-dpo", "qwen3_vl_8b_sft_dpo", "Qwen3-VL-8B-SFT-DPO-Merged"], "SFT+RLVR"),
            ("JiNan",    "anon_3_4_jinan_synthetic_24000_60min",    "qwen3-vl-4b",  "Qwen-3-vl-4b"),
            ("JiNan",    "anon_3_4_jinan_synthetic_24000_60min",    "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("JiNan",    "anon_3_4_jinan_synthetic_24000_60min",    ["sft", "qwen3-vl-8b-sft", "qwen3_vl_8b_sft", "Qwen3-VL-8B-SFT-Merged"], "SFT"),
            ("JiNan",    "anon_3_4_jinan_synthetic_24000_60min",    ["sft+dpo", "sft_dpo", "qwen3-vl-8b-sft-dpo", "qwen3_vl_8b_sft_dpo", "Qwen3-VL-8B-SFT-DPO-Merged"], "SFT+RLVR"),
            ("Hangzhou", "anon_4_4_hangzhou_real",                  "qwen3-vl-4b",  "Qwen-3-vl-4b"),
            ("Hangzhou", "anon_4_4_hangzhou_real",                  "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("Hangzhou", "anon_4_4_hangzhou_real",                  ["sft", "qwen3-vl-8b-sft", "qwen3_vl_8b_sft", "Qwen3-VL-8B-SFT-Merged"], "SFT"),
            ("Hangzhou", "anon_4_4_hangzhou_real",                  ["sft+dpo", "sft_dpo", "qwen3-vl-8b-sft-dpo", "qwen3_vl_8b_sft_dpo", "Qwen3-VL-8B-SFT-DPO-Merged"], "SFT+RLVR"),
            ("Hangzhou", "anon_4_4_hangzhou_real_5816",             "qwen3-vl-4b",  "Qwen-3-vl-4b"),
            ("Hangzhou", "anon_4_4_hangzhou_real_5816",             "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("Hangzhou", "anon_4_4_hangzhou_real_5816",             ["sft", "qwen3-vl-8b-sft", "qwen3_vl_8b_sft", "Qwen3-VL-8B-SFT-Merged"], "SFT"),
            ("Hangzhou", "anon_4_4_hangzhou_real_5816",             ["sft+dpo", "sft_dpo", "qwen3-vl-8b-sft-dpo", "qwen3_vl_8b_sft_dpo", "Qwen3-VL-8B-SFT-DPO-Merged"], "SFT+RLVR"),
            ("Hangzhou", "anon_4_4_hangzhou_synthetic_24000_60min", "qwen3-vl-4b",  "Qwen-3-vl-4b"),
            ("Hangzhou", "anon_4_4_hangzhou_synthetic_24000_60min", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("Hangzhou", "anon_4_4_hangzhou_synthetic_24000_60min", ["sft", "qwen3-vl-8b-sft", "qwen3_vl_8b_sft", "Qwen3-VL-8B-SFT-Merged"], "SFT"),
            ("Hangzhou", "anon_4_4_hangzhou_synthetic_24000_60min", ["sft+dpo", "sft_dpo", "qwen3-vl-8b-sft-dpo", "qwen3_vl_8b_sft_dpo", "Qwen3-VL-8B-SFT-DPO-Merged"], "SFT+RLVR"),
        ],
        'calc_kwargs': {},
    },

    # ── 1.2 消融总表（3 个方法变体，3 项指标）────────────────────────────────
    'ablation': {
        'csv_path': 'results/ablation_result.csv',
        'metric_keys': ['ATT', 'AWT', 'AQL'],
        'scene_to_cols': {
            ("JiNan",    "anon_3_4_jinan_real"):                     (2,  3,  4),
            ("JiNan",    "+RLVR"):                (5,  6,  7),
            ("JiNan",    "anon_3_4_jinan_real_2500"):                (8,  9,  10),
            ("JiNan",    "anon_3_4_jinan_synthetic_24000_60min"):    (11, 12, 13),
            ("Hangzhou", "anon_4_4_hangzhou_real"):                  (14, 15, 16),
            ("Hangzhou", "anon_4_4_hangzhou_real_5816"):             (17, 18, 19),
            ("Hangzhou", "anon_4_4_hangzhou_synthetic_24000_60min"): (20, 21, 22),
        },
        'eval_plan': [
            ("JiNan",    "anon_3_4_jinan_real",                     "qwen3-vl-8b",  "Zero-Shot"),
            ("JiNan",    "anon_3_4_jinan_real",                     ["sft", "qwen3-vl-8b-sft", "qwen3_vl_8b_sft", "Qwen3-VL-8B-SFT-Merged"], "SFT"),
            ("JiNan",    "anon_3_4_jinan_real",                     ["sft+dpo", "sft_dpo", "qwen3-vl-8b-sft-dpo", "qwen3_vl_8b_sft_dpo", "Qwen3-VL-8B-SFT-DPO-Merged"], "SFT+RLVR"),
            ("JiNan",    "anon_3_4_jinan_real_2000",                "qwen3-vl-8b",  "Zero-Shot"),
            ("JiNan",    "anon_3_4_jinan_real_2000",                ["sft", "qwen3-vl-8b-sft", "qwen3_vl_8b_sft", "Qwen3-VL-8B-SFT-Merged"], "SFT"),
            ("JiNan",    "anon_3_4_jinan_real_2000",                ["sft+dpo", "sft_dpo", "qwen3-vl-8b-sft-dpo", "qwen3_vl_8b_sft_dpo", "Qwen3-VL-8B-SFT-DPO-Merged"], "SFT+RLVR"),
            ("JiNan",    "anon_3_4_jinan_real_2500",                "qwen3-vl-8b",  "Zero-Shot"),
            ("JiNan",    "anon_3_4_jinan_real_2500",                ["sft", "qwen3-vl-8b-sft", "qwen3_vl_8b_sft", "Qwen3-VL-8B-SFT-Merged"], "SFT"),
            ("JiNan",    "anon_3_4_jinan_real_2500",                ["sft+dpo", "sft_dpo", "qwen3-vl-8b-sft-dpo", "qwen3_vl_8b_sft_dpo", "Qwen3-VL-8B-SFT-DPO-Merged"], "SFT+RLVR"),
            ("JiNan",    "anon_3_4_jinan_synthetic_24000_60min",    "qwen3-vl-8b",  "Zero-Shot"),
            ("JiNan",    "anon_3_4_jinan_synthetic_24000_60min",    ["sft", "qwen3-vl-8b-sft", "qwen3_vl_8b_sft", "Qwen3-VL-8B-SFT-Merged"], "SFT"),
            ("JiNan",    "anon_3_4_jinan_synthetic_24000_60min",    ["sft+dpo", "sft_dpo", "qwen3-vl-8b-sft-dpo", "qwen3_vl_8b_sft_dpo", "Qwen3-VL-8B-SFT-DPO-Merged"], "SFT+RLVR"),
            ("Hangzhou", "anon_4_4_hangzhou_real",                  "qwen3-vl-8b",  "Zero-Shot"),
            ("Hangzhou", "anon_4_4_hangzhou_real",                  ["sft", "qwen3-vl-8b-sft", "qwen3_vl_8b_sft", "Qwen3-VL-8B-SFT-Merged"], "SFT"),
            ("Hangzhou", "anon_4_4_hangzhou_real",                  ["sft+dpo", "sft_dpo", "qwen3-vl-8b-sft-dpo", "qwen3_vl_8b_sft_dpo", "Qwen3-VL-8B-SFT-DPO-Merged"], "SFT+RLVR"),
            ("Hangzhou", "anon_4_4_hangzhou_real_5816",             "qwen3-vl-8b",  "Zero-Shot"),
            ("Hangzhou", "anon_4_4_hangzhou_real_5816",             ["sft", "qwen3-vl-8b-sft", "qwen3_vl_8b_sft", "Qwen3-VL-8B-SFT-Merged"], "SFT"),
            ("Hangzhou", "anon_4_4_hangzhou_real_5816",             ["sft+dpo", "sft_dpo", "qwen3-vl-8b-sft-dpo", "qwen3_vl_8b_sft_dpo", "Qwen3-VL-8B-SFT-DPO-Merged"], "SFT+RLVR"),
            ("Hangzhou", "anon_4_4_hangzhou_synthetic_24000_60min", "qwen3-vl-8b",  "Zero-Shot"),
            ("Hangzhou", "anon_4_4_hangzhou_synthetic_24000_60min", ["sft", "qwen3-vl-8b-sft", "qwen3_vl_8b_sft", "Qwen3-VL-8B-SFT-Merged"], "SFT"),
            ("Hangzhou", "anon_4_4_hangzhou_synthetic_24000_60min", ["sft+dpo", "sft_dpo", "qwen3-vl-8b-sft-dpo", "qwen3_vl_8b_sft_dpo", "Qwen3-VL-8B-SFT-DPO-Merged"], "SFT+RLVR"),
        ],
        'calc_kwargs': {},
    },

    # ── 2. 紧急车辆（5 项：ATT/AWT/AQL + EATT/EAWT）──────────────────────────
    # vType: emergency, police, fire_engine（由 add_emergency_vehicles.py 注入）
    'emergency': {
        'csv_path': 'results/emergency_result.csv',
        'metric_keys': ['ATT', 'AWT', 'AQL', 'Special_ATT', 'Special_AWT'],
        'scene_to_cols': {
            ("JiNan",             "anon_3_4_jinan_real_emergy"):  (2,  3,  4,  5,  6),
            ("Hangzhou",          "anon_4_4_hangzhou_real_emergy"):  (7,  8,  9,  10, 11),
            ("SouthKorea_Songdo", "songdo_emergy"):  (12, 13, 14, 15, 16),
            ("France_Massy",      "massy_emergy"):  (17, 18, 19, 20, 21),
            ("Hongkong_YMT",      "YMT_emergy"):  (22, 23, 24, 25, 26),
            ("NewYork",           "anon_28_7_newyork_real_double_emergy"):  (27, 28, 29, 30, 31),
        },
        'eval_plan': [
            ("JiNan",             "anon_3_4_jinan_real_emergy", "fixed_time",   "FixedTime"),
            ("JiNan",             "anon_3_4_jinan_real_emergy", "max_pressure", "MaxPressure"),
            ("JiNan",             "anon_3_4_jinan_real_emergy", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("Hangzhou",          "anon_4_4_hangzhou_real_emergy", "fixed_time",   "FixedTime"),
            ("Hangzhou",          "anon_4_4_hangzhou_real_emergy", "max_pressure", "MaxPressure"),
            ("Hangzhou",          "anon_4_4_hangzhou_real_emergy", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("SouthKorea_Songdo", "songdo_emergy", "fixed_time",   "FixedTime"),
            ("SouthKorea_Songdo", "songdo_emergy", "max_pressure", "MaxPressure"),
            ("SouthKorea_Songdo", "songdo_emergy", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("France_Massy",      "massy_emergy", "fixed_time",   "FixedTime"),
            ("France_Massy",      "massy_emergy", "max_pressure", "MaxPressure"),
            ("France_Massy",      "massy_emergy", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("Hongkong_YMT",      "YMT_emergy", "fixed_time",   "FixedTime"),
            ("Hongkong_YMT",      "YMT_emergy", "max_pressure", "MaxPressure"),
            ("Hongkong_YMT",      "YMT_emergy", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("NewYork",           "anon_28_7_newyork_real_double_emergy", "fixed_time",   "FixedTime"),
            ("NewYork",           "anon_28_7_newyork_real_double_emergy", "max_pressure", "MaxPressure"),
            ("NewYork",           "anon_28_7_newyork_real_double_emergy", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
        ],
        'calc_kwargs': {
            'special_vtypes': {"emergency", "police", "fire_engine"},
        },
    },

    # ── 3. 公交/校车（5 项：ATT/AWT/AQL + BATT/BAWT）────────────────────────
    # vType: bus, school_bus（由 add_bus_vehicles.py 注入）
    'bus': {
        'csv_path': 'results/bus_result.csv',
        'metric_keys': ['ATT', 'AWT', 'AQL', 'Special_ATT', 'Special_AWT'],
        'scene_to_cols': {
            ("JiNan",             "anon_3_4_jinan_real_bus"):  (2,  3,  4,  5,  6),
            ("Hangzhou",          "anon_4_4_hangzhou_real_bus"):  (7,  8,  9,  10, 11),
            ("SouthKorea_Songdo", "songdo_bus"):  (12, 13, 14, 15, 16),
            ("France_Massy",      "massy_bus"):  (17, 18, 19, 20, 21),
            ("Hongkong_YMT",      "YMT_bus"):  (22, 23, 24, 25, 26),
            ("NewYork",           "anon_28_7_newyork_real_double_bus"):  (27, 28, 29, 30, 31),
        },
        'eval_plan': [
            ("JiNan",             "anon_3_4_jinan_real_bus", "fixed_time",   "FixedTime"),
            ("JiNan",             "anon_3_4_jinan_real_bus", "max_pressure", "MaxPressure"),
            ("JiNan",             "anon_3_4_jinan_real_bus", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("Hangzhou",          "anon_4_4_hangzhou_real_bus", "fixed_time",   "FixedTime"),
            ("Hangzhou",          "anon_4_4_hangzhou_real_bus", "max_pressure", "MaxPressure"),
            ("Hangzhou",          "anon_4_4_hangzhou_real_bus", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("SouthKorea_Songdo", "songdo_bus", "fixed_time",   "FixedTime"),
            ("SouthKorea_Songdo", "songdo_bus", "max_pressure", "MaxPressure"),
            ("SouthKorea_Songdo", "songdo_bus", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("France_Massy",      "massy_bus", "fixed_time",   "FixedTime"),
            ("France_Massy",      "massy_bus", "max_pressure", "MaxPressure"),
            ("France_Massy",      "massy_bus", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("Hongkong_YMT",      "YMT_bus", "fixed_time",   "FixedTime"),
            ("Hongkong_YMT",      "YMT_bus", "max_pressure", "MaxPressure"),
            ("Hongkong_YMT",      "YMT_bus", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("NewYork",           "anon_28_7_newyork_real_double_bus", "fixed_time",   "FixedTime"),
            ("NewYork",           "anon_28_7_newyork_real_double_bus", "max_pressure", "MaxPressure"),
            ("NewYork",           "anon_28_7_newyork_real_double_bus", "qwen3-vl-8b",  "Qwen-3-vl-8b"),     
            ],
        'calc_kwargs': {
            'special_vtypes': {"bus", "school_bus"},
        },
    },

    # ── 4. 交通事故（5 项：ATT/AWT/AQL + MaxQL/TPT）──────────────────────────
    # trip id 前缀: accident_*（由 generate_traffic_accident.py 生成）
    'accident': {
        'csv_path': 'results/accident_result.csv',
        'metric_keys': ['ATT', 'AWT', 'AQL', 'MaxQL', 'TPT'],
        'scene_to_cols': {
            ("JiNan",             "anon_3_4_jinan_real_accident"):  (2,  3,  4,  5,  6),
            ("Hangzhou",          "anon_4_4_hangzhou_real_accident"):  (7,  8,  9,  10, 11),
            ("SouthKorea_Songdo", "songdo_accident"):  (12, 13, 14, 15, 16),
            ("France_Massy",      "massy_accident"):  (17, 18, 19, 20, 21),
            ("Hongkong_YMT",      "YMT_accident"):  (22, 23, 24, 25, 26),
            ("NewYork",           "anon_28_7_newyork_real_double_accident"):  (27, 28, 29, 30, 31),
        },
        'eval_plan': [
            ("JiNan",             "anon_3_4_jinan_real_accident", "fixed_time",   "FixedTime"),
            ("JiNan",             "anon_3_4_jinan_real_accident", "max_pressure", "MaxPressure"),
            ("JiNan",             "anon_3_4_jinan_real_accident", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("Hangzhou",          "anon_4_4_hangzhou_real_accident", "fixed_time",   "FixedTime"),
            ("Hangzhou",          "anon_4_4_hangzhou_real_accident", "max_pressure", "MaxPressure"),
            ("Hangzhou",          "anon_4_4_hangzhou_real_accident", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("SouthKorea_Songdo", "songdo_accident", "fixed_time",   "FixedTime"),
            ("SouthKorea_Songdo", "songdo_accident", "max_pressure", "MaxPressure"),
            ("SouthKorea_Songdo", "songdo_accident", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("France_Massy",      "massy_accident", "fixed_time",   "FixedTime"),
            ("France_Massy",      "massy_accident", "max_pressure", "MaxPressure"),
            ("France_Massy",      "massy_accident", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("Hongkong_YMT",      "YMT_accident", "fixed_time",   "FixedTime"),
            ("Hongkong_YMT",      "YMT_accident", "max_pressure", "MaxPressure"),
            ("Hongkong_YMT",      "YMT_accident", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("NewYork",           "anon_28_7_newyork_real_double_accident", "fixed_time",   "FixedTime"),
            ("NewYork",           "anon_28_7_newyork_real_double_accident", "max_pressure", "MaxPressure"),
            ("NewYork",           "anon_28_7_newyork_real_double_accident", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
        ],
        'calc_kwargs': {
            'event_id_prefixes': ["accident_"],
        },
    },

    # ── 5. 路面碎片/路障（5 项：ATT/AWT/AQL + MaxQL/TPT）────────────────────
    # trip id 前缀: debris_*（由 generate_road_debris.py 生成）
    'debris': {
        'csv_path': 'results/debris_result.csv',
        'metric_keys': ['ATT', 'AWT', 'AQL', 'MaxQL', 'TPT'],
        'scene_to_cols': {
            ("JiNan",             "anon_3_4_jinan_real_debris"):  (2,  3,  4,  5,  6),
            ("Hangzhou",          "anon_4_4_hangzhou_real_debris"):  (7,  8,  9,  10, 11),
            ("SouthKorea_Songdo", "songdo_debris"):  (12, 13, 14, 15, 16),
            ("France_Massy",      "massy_debris"):  (17, 18, 19, 20, 21),
            ("Hongkong_YMT",      "YMT_debris"):  (22, 23, 24, 25, 26),
            ("NewYork",           "anon_28_7_newyork_real_double_debris"):  (27, 28, 29, 30, 31),
        },
        'eval_plan': [
            ("JiNan",             "anon_3_4_jinan_real_debris", "fixed_time",   "FixedTime"),
            ("JiNan",             "anon_3_4_jinan_real_debris", "max_pressure", "MaxPressure"),
            ("JiNan",             "anon_3_4_jinan_real_debris", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("Hangzhou",          "anon_4_4_hangzhou_real_debris", "fixed_time",   "FixedTime"),
            ("Hangzhou",          "anon_4_4_hangzhou_real_debris", "max_pressure", "MaxPressure"),
            ("Hangzhou",          "anon_4_4_hangzhou_real_debris", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("SouthKorea_Songdo", "songdo_debris", "fixed_time",   "FixedTime"),
            ("SouthKorea_Songdo", "songdo_debris", "max_pressure", "MaxPressure"),
            ("SouthKorea_Songdo", "songdo_debris", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("France_Massy",      "massy_debris", "fixed_time",   "FixedTime"),
            ("France_Massy",      "massy_debris", "max_pressure", "MaxPressure"),
            ("France_Massy",      "massy_debris", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("Hongkong_YMT",      "YMT_debris", "fixed_time",   "FixedTime"),
            ("Hongkong_YMT",      "YMT_debris", "max_pressure", "MaxPressure"),
            ("Hongkong_YMT",      "YMT_debris", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
            ("NewYork",           "anon_28_7_newyork_real_double_debris", "fixed_time",   "FixedTime"),
            ("NewYork",           "anon_28_7_newyork_real_double_debris", "max_pressure", "MaxPressure"),
            ("NewYork",           "anon_28_7_newyork_real_double_debris", "qwen3-vl-8b",  "Qwen-3-vl-8b"),
        ],
        'calc_kwargs': {
            'event_id_prefixes': ["debris_"],
        },
    },

    # ── 6. 行人过街（5 项：ATT/AWT/AQL + MaxQL/TPT）──────────────────────────
    # trip id 前缀: ped_*（由 generate_pedestrian_crossing.py 生成）
    # 'pedestrian': {
    #     'csv_path': 'results/pedestrian_result.csv',
    #     'metric_keys': ['ATT', 'AWT', 'AQL', 'MaxQL', 'TPT'],
    #     'scene_to_cols': {
    #         ("JiNan",             "anon_3_4_jinan_real_pedestrian"):  (2,  3,  4,  5,  6),
    #         ("Hangzhou",          "anon_4_4_hangzhou_real_pedestrian"):  (7,  8,  9,  10, 11),
    #         ("SouthKorea_Songdo", "songdo_pedestrian"):  (12, 13, 14, 15, 16),
    #         ("France_Massy",      "massy_pedestrian"):  (17, 18, 19, 20, 21),
    #         ("Hongkong_YMT",      "YMT_pedestrian"):  (22, 23, 24, 25, 26),
    #         ("NewYork",           "anon_28_7_newyork_real_double_pedestrian"):  (27, 28, 29, 30, 31),
    #     },
    #     'eval_plan': [
    #         ("JiNan",             "anon_3_4_jinan_real_pedestrian", "fixed_time",   "FixedTime"),
    #         ("JiNan",             "anon_3_4_jinan_real_pedestrian", "max_pressure", "MaxPressure"),
    #         ("Hangzhou",          "anon_4_4_hangzhou_real_pedestrian", "fixed_time",   "FixedTime"),
    #         ("Hangzhou",          "anon_4_4_hangzhou_real_pedestrian", "max_pressure", "MaxPressure"),
    #         ("SouthKorea_Songdo", "songdo_pedestrian", "fixed_time",   "FixedTime"),
    #         ("SouthKorea_Songdo", "songdo_pedestrian", "max_pressure", "MaxPressure"),
    #         ("France_Massy",      "massy_pedestrian", "fixed_time",   "FixedTime"),
    #         ("France_Massy",      "massy_pedestrian", "max_pressure", "MaxPressure"),
    #         ("Hongkong_YMT",      "YMT_pedestrian", "fixed_time",   "FixedTime"),
    #         ("Hongkong_YMT",      "YMT_pedestrian", "max_pressure", "MaxPressure"),
    #         ("NewYork",           "anon_28_7_newyork_real_double_pedestrian", "fixed_time",   "FixedTime"),
    #         ("NewYork",           "anon_28_7_newyork_real_double_pedestrian", "max_pressure", "MaxPressure"),
    #     ],
    #     'calc_kwargs': {
    #         'event_id_prefixes': ["ped_"],
    #     },
    # },
}


def _dedup_keep_order(items):
    """去重并保持原顺序。"""
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _resolve_base_path(eval_root: str, scenario: str, scene_type: str, model_dir) -> tuple:
    """解析 data/eval 目录，兼容 scene_type 与 model_dir 的命名差异。"""
    scene_candidates = [scene_type]
    if scene_type.endswith('.rou'):
        scene_candidates.append(scene_type[:-4])
    else:
        scene_candidates.append(f"{scene_type}.rou")
    scene_candidates = _dedup_keep_order(scene_candidates)

    if isinstance(model_dir, (list, tuple)):
        model_candidates = list(model_dir)
    else:
        model_candidates = [model_dir]
    model_candidates = _dedup_keep_order(model_candidates)

    for scene_name in scene_candidates:
        for model_name in model_candidates:
            base_path = os.path.join(eval_root, scenario, scene_name, model_name)
            stat_path = os.path.join(base_path, "statistic_output.xml")
            queue_path = os.path.join(base_path, "queue_output.xml")
            if os.path.exists(stat_path) and os.path.exists(queue_path):
                return base_path, scene_name, model_name

    # 回退：用于打印更准确的缺失路径
    return os.path.join(eval_root, scenario, scene_candidates[0], model_candidates[0]), scene_candidates[0], model_candidates[0]


# ─────────────────────────────────────────────────────────────────────────────
# CSV 自动创建
# ─────────────────────────────────────────────────────────────────────────────

def _build_empty_csv(cfg: dict) -> list:
    """根据 scene_to_cols 和 eval_plan 构造空白 CSV 的行列表。

    第 0 行：表头（Category, Method, 场景列组...）
    第 1 行：指标行（空, 空, ATT↓, AWT↓, ...）
    后续行：每个唯一 row_key 对应一行，值全空。
    """
    metric_keys = cfg['metric_keys']
    scene_to_cols = cfg['scene_to_cols']

    # 计算总列数
    max_col = max(max(cols) for cols in scene_to_cols.values())
    total_cols = max_col + 1

    # 第 0 行：场景名表头
    header0 = [''] * total_cols
    header0[0] = 'Category'
    header0[1] = 'Method'
    for (scenario, _scene_type), cols in scene_to_cols.items():
        header0[cols[0]] = scenario

    # 第 1 行：指标名表头（带箭头方向提示）
    _arrow = {'ATT': 'ATT↓', 'AWT': 'AWT↓', 'AQL': 'AQL↓',
              'MaxQL': 'MaxQL↓', 'TPT': 'TPT↑',
              'Special_ATT': 'Special_ATT↓', 'Special_AWT': 'Special_AWT↓'}
    header1 = [''] * total_cols
    for (_scenario, _scene_type), cols in scene_to_cols.items():
        for key, col in zip(metric_keys, cols):
            header1[col] = _arrow.get(key, key)

    # 收集所有唯一 row_key（保持顺序）
    row_keys = _dedup_keep_order(rk for _, _, _, rk in cfg['eval_plan'])

    # 为每个 row_key 生成空数据行
    data_rows = []
    for rk in row_keys:
        row = [''] * total_cols
        row[1] = rk
        data_rows.append(row)

    return [header0, header1] + data_rows


# ─────────────────────────────────────────────────────────────────────────────
# 通用 CSV 写入
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(cfg: dict, metrics: dict, scenario: str, scene_type: str, row_key: str) -> None:
    """将 metric_keys 按 scene_to_cols 指定的列写入对应 CSV。

    若 CSV 文件不存在则自动创建空模板。
    """
    col_indices = cfg['scene_to_cols'].get((scenario, scene_type))
    if col_indices is None:
        print(f"⚠️  [CSV] ({scenario}, {scene_type}) 未在 scene_to_cols 中配置，跳过。")
        return

    csv_path = os.path.join(PROJECT_ROOT, cfg['csv_path'])

    # CSV 不存在时自动创建
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        empty_rows = _build_empty_csv(cfg)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(empty_rows)
        print(f"ℹ️  [CSV] 自动创建空模板: {csv_path}")

    metric_keys = cfg['metric_keys']
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))

        target_row_idx = None
        for idx, row in enumerate(rows):
            if len(row) > 1 and row[1].strip() == row_key:
                target_row_idx = idx
                break

        # row_key 不存在时追加新行
        if target_row_idx is None:
            new_row = [''] * (max(col_indices) + 1)
            new_row[1] = row_key
            rows.append(new_row)
            target_row_idx = len(rows) - 1
            print(f"ℹ️  [CSV] 新增行: Method='{row_key}'")

        target_row = rows[target_row_idx]
        while len(target_row) <= max(col_indices):
            target_row.append('')

        for key, col in zip(metric_keys, col_indices):
            target_row[col] = _FMT[key](metrics.get(key, 0))
        rows[target_row_idx] = target_row

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)

        kv = ', '.join(f"{k}={_FMT[k](metrics.get(k, 0))}" for k in metric_keys)
        print(f"✅  {row_key} | {scenario}/{scene_type} | {kv}")

    except Exception as e:
        print(f"❌ [CSV] 写入失败: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 核心收集函数
# ─────────────────────────────────────────────────────────────────────────────

def collect(event_type: str) -> None:
    """收集指定事件类型的全部评估指标并写入对应结果 CSV。

    路径格式：data/eval/{scenario}/{scene_type}/{method}/
    """
    cfg = _CONFIGS.get(event_type)
    if cfg is None:
        print(f"❌ 未知事件类型: '{event_type}'，可选: {list(_CONFIGS)}")
        return

    eval_root = os.path.join(PROJECT_ROOT, "data", "eval")
    total, success, skipped = 0, 0, 0

    for scenario, scene_type, model_dir, row_key in cfg['eval_plan']:
        total += 1
        base_path, scene_type_resolved, model_dir_resolved = _resolve_base_path(
            eval_root, scenario, scene_type, model_dir
        )
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
                tripinfo_file=tripinfo_path,
                **cfg['calc_kwargs'],
            )
        except Exception as e:
            print(f"❌ 指标计算失败 {base_path}: {e}")
            skipped += 1
            continue

        _write_csv(cfg, metrics, scenario, scene_type, row_key)
        if scene_type_resolved != scene_type or model_dir_resolved != model_dir:
            print(f"ℹ️  [Resolve] {scenario}/{scene_type}/{model_dir} -> {scenario}/{scene_type_resolved}/{model_dir_resolved}")
        success += 1

    csv_abs = os.path.join(PROJECT_ROOT, cfg['csv_path'])
    print(f"\n{'='*60}")
    print(f"[{event_type}] 完成：{total} 条，成功 {success}，跳过 {skipped}")
    print(f"结果已写入：{csv_abs}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="统一评估指标收集入口（替代各 *_metrics.py 独立脚本）"
    )
    parser.add_argument(
        "--type", "--type_-t",
        type=str,
        required=True,
        choices=list(_CONFIGS) + ["all"],
        metavar="TYPE",
        help="事件类型：" + " | ".join(list(_CONFIGS)) + " | all（全部运行）",
    )
    args = parser.parse_args()

    if args.type == "all":
        for et in _CONFIGS:
            print(f"\n{'#'*60}\n# {et}\n{'#'*60}")
            collect(et)
    else:
        collect(args.type)
