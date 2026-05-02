'''
Author: yufei Ji
Date: 2026-04-20
Description: 统一评估指标收集入口。

按论文实验设计分为四大模块：
  ┌─────────────────────────────────────────────────────────────────────┐
  │  模块一（main）       主实验：常规场景性能对比                       │
  │                       Jinan×3 + Hangzhou×2，13 种方法              │
  │                       输出：results/main_result.csv                 │
  ├─────────────────────────────────────────────────────────────────────┤
  │  模块二（generalize） 泛化性实验，三个子任务：                       │
  │    gen_topology       拓扑迁移：Songdo / Massy / YMT               │
  │    gen_scale          规模迁移：NewYork（196路口）                  │
  │    gen_event          事件迁移：Jinan×3+Hangzhou×2+拓扑3场景，      │
  │                       每场景 2 个混合事件路由文件                   │
  │                       输出：results/gen_topology_result.csv         │
  │                             results/gen_scale_result.csv            │
  │                             results/gen_event_emergy_bus_result.csv  │
  │                             results/gen_event_accident_debris_result.csv │
  ├─────────────────────────────────────────────────────────────────────┤
  │  模块三（ablation）   消融实验，四个子任务：                         │
  │    abl_train          训练阶段消融：Zero-shot / SFT / E2ELight      │
  │    abl_action         动作空间消融：Phase-only / Phase+Duration     │
  │    abl_cot            快慢思考消融：Fast-only / Slow-only / Adaptive│
  │    abl_bulletin       广播机制消融：No / All / Directed             │
  │                       输出：results/abl_train_result.csv            │
  │                             results/abl_action_result.csv           │
  │                             results/abl_cot_result.csv              │
  │                             results/abl_bulletin_result.csv         │
  ├─────────────────────────────────────────────────────────────────────┤
  │  模块四（supplement） 补充实验（推理效率/幻觉/事件识别分析）        │
  │                       无自动 CSV 写入，仅打印汇总路径供人工整理     │
  └─────────────────────────────────────────────────────────────────────┘

指标说明：
  普通场景：ATT / AWT / AQL
  紧急车辆事件：ATT / AWT / AQL / EATT / EAWT（vType=emergency/police/fire_engine）
  公交/校车事件：ATT / AWT / AQL / BATT / BAWT（vType=bus/school_bus）
  事故/占道事件：ATT / AWT / AQL / MaxQL / TPT
  emergy_bus 混合路由：同时输出紧急车辆指标 + 公交指标（共 7 项）
  accident_debris 混合路由：ATT / AWT / AQL / MaxQL / TPT（共 5 项）

使用方式：
  # 运行单个模块
  python src/evaluation/collect_metrics.py --type main
  python src/evaluation/collect_metrics.py --type gen_topology
  python src/evaluation/collect_metrics.py --type gen_scale
  python src/evaluation/collect_metrics.py --type gen_event
  python src/evaluation/collect_metrics.py --type abl_train
  python src/evaluation/collect_metrics.py --type abl_action
  python src/evaluation/collect_metrics.py --type abl_cot
  python src/evaluation/collect_metrics.py --type abl_bulletin
  # 运行全部（按顺序）
  python src/evaluation/collect_metrics.py --type all
'''

import argparse
import csv
import os
import sys

sys.path.append(".")
from src.evaluation.metrics import MetricsCalculator

# 项目根目录（collect_metrics.py 从项目根运行）
PROJECT_ROOT = "."

# ─────────────────────────────────────────────────────────────────────────────
# 全局方法名常量
# ─────────────────────────────────────────────────────────────────────────────

# 规则基线方法名
FIXED_TIME   = "fixed_time"
MAX_PRESSURE = "max_pressure"

# LLM/VLM 增强方法（在 data/eval 目录下与训练后模型目录同级）
LLMLIGHT  = "llmlight"
VLMLIGHT  = "vlmlight"

# E2ELight 变体（实际目录名可能有别名，用列表兼容多种命名）
ZERO_SHOT = "qwen3-vl-8b"                                      # 零样本预训练模型
SFT_ONLY  = ["sft", "qwen3-vl-8b-sft", "Qwen3-VL-8B-SFT-Merged"]   # 仅 SFT
E2ELIGHT  = ["sft-rlvr", "qwen3-vl-8b-sft-rlvr",
             "Qwen3-VL-8B-SFT-RLVR-Merged"]                   # SFT + RLVR（完整模型）

# 消融专用变体
PHASE_ONLY     = "phase-only"          # 仅选相位，时长固定 27s
FAST_ONLY      = "fast-only"           # 强制短路径推理
SLOW_ONLY      = "slow-only"           # 强制长路径推理
NO_BULLETIN    = "no-bulletin"         # 关闭事件广播
ALL_BULLETIN   = "all-bulletin"        # 广播所有邻居（无定向）
DIR_BULLETIN   = E2ELIGHT              # 定向广播（与完整模型复用）

# 主实验全量对比方法（13 种）
MAIN_METHODS = [
    (FIXED_TIME,   "FixedTime"),
    (MAX_PRESSURE, "MaxPressure"),
    # 经典 RL 方法（结果从离线数据读取，目录名与方法名一致）
    ("intellilight", "IntelliLight"),
    ("frap",         "FRAP"),
    ("presslight",   "PressLight"),
    ("metalight",    "MetaLight"),
    ("mplight",      "MPLight"),
    ("dynamiclight", "DynamicLight"),
    ("unitsa",       "UniTSA"),
    # LLM/VLM 增强对比方法
    (LLMLIGHT, "LLMLight"),
    (VLMLIGHT, "VLMLight"),
    # E2ELight（ours）
    (E2ELIGHT, "E2ELight"),
]

# 泛化/消融通用对比方法（5 种）
LIGHT_METHODS = [
    (FIXED_TIME,   "FixedTime"),
    (MAX_PRESSURE, "MaxPressure"),
    (LLMLIGHT,     "LLMLight"),
    (VLMLIGHT,     "VLMLight"),
    (E2ELIGHT,     "E2ELight"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 指标格式化（统一保留位数）
# ─────────────────────────────────────────────────────────────────────────────

_FMT = {
    "ATT":         lambda v: f"{float(v):.2f}",
    "AWT":         lambda v: f"{float(v):.2f}",
    "AQL":         lambda v: f"{float(v):.6f}",
    "MaxQL":       lambda v: f"{float(v):.2f}",
    "TPT":         lambda v: f"{int(v)}",
    "Special_ATT": lambda v: f"{float(v):.2f}",   # EATT 或 BATT
    "Special_AWT": lambda v: f"{float(v):.2f}",   # EAWT 或 BAWT
}

# CSV 表头箭头方向（↓ 越小越好，↑ 越大越好）
_ARROW = {
    "ATT":         "ATT↓",
    "AWT":         "AWT↓",
    "AQL":         "AQL↓",
    "MaxQL":       "MaxQL↓",
    "TPT":         "TPT↑",
    "Special_ATT": "Special_ATT↓",
    "Special_AWT": "Special_AWT↓",
}

# ─────────────────────────────────────────────────────────────────────────────
# 场景路由文件名定义（所有数据集的 scene_type 名称与 data/eval 目录对应）
# ─────────────────────────────────────────────────────────────────────────────

# 主实验：Jinan×3 + Hangzhou×2
JINAN_SCENES = [
    "anon_3_4_jinan_real",
    "anon_3_4_jinan_real_2000",
    "anon_3_4_jinan_real_2500",
]
HANGZHOU_SCENES = [
    "anon_4_4_hangzhou_real",
    "anon_4_4_hangzhou_real_5816",
]

# 拓扑迁移场景
TOPOLOGY_SCENES = [
    ("SouthKorea_Songdo", "songdo"),
    ("France_Massy",      "massy"),
    ("Hongkong_YMT",      "YMT"),
]

# 规模迁移场景（NewYork 全量 196 路口）
NEWYORK_SCENE = ("NewYork", "anon_28_7_newyork_real_double")

# 事件路由文件后缀（两类混合文件）
EVENT_EMERGY_BUS      = "_emergy_bus"      # 紧急车辆 + 公交/校车
EVENT_ACCIDENT_DEBRIS = "_accident_debris" # 交通事故 + 路面占道

# 消融实验使用的事件基础场景（Jinan×3 + Hangzhou×2）
ABLATION_EVENT_BASE_SCENES = [
    ("JiNan",    s) for s in JINAN_SCENES
] + [
    ("Hangzhou", s) for s in HANGZHOU_SCENES
]

# ─────────────────────────────────────────────────────────────────────────────
# 配置字典：每个实验类型的完整定义
# ─────────────────────────────────────────────────────────────────────────────
# 结构说明：
#   csv_path     : 结果 CSV 相对路径
#   metric_keys  : 按列顺序写入的指标 key 列表
#   scene_to_cols: {(scenario, scene_type): (col_att, col_awt, ...)}
#                  描述每个场景对应的 CSV 列号（0-indexed，col 0=Category, 1=Method）
#   eval_plan    : [(scenario, scene_type, method_dir, row_label), ...]
#                  描述每一行评测任务
#   calc_kwargs  : 传给 MetricsCalculator.calculate_from_files() 的额外参数

def _build_scene_to_cols(scene_list, metric_count):
    """
    根据场景列表和每场景指标数，自动计算 scene_to_cols 字典。
    col 0 = Category，col 1 = Method，从 col 2 开始按场景顺序排列。
    """
    result = {}
    start = 2
    for key in scene_list:
        result[key] = tuple(range(start, start + metric_count))
        start += metric_count
    return result


def _build_eval_plan(scene_list, methods):
    """
    笛卡尔积构造评测计划：所有场景 × 所有方法。
    scene_list: [(scenario, scene_type), ...]
    methods:    [(method_dir, row_label), ...]
    返回: [(scenario, scene_type, method_dir, row_label), ...]
    """
    plan = []
    for scenario, scene_type in scene_list:
        for method_dir, row_label in methods:
            plan.append((scenario, scene_type, method_dir, row_label))
    return plan


# ─── 模块一：主实验 ────────────────────────────────────────────────────────

# 主实验场景列表（Jinan×3 + Hangzhou×2）
_MAIN_SCENES = (
    [("JiNan",    s) for s in JINAN_SCENES] +
    [("Hangzhou", s) for s in HANGZHOU_SCENES]
)

_CONFIGS = {

    # =========================================================================
    # 模块一：主实验 ——  常规场景 vs 13 种方法
    # 指标：ATT / AWT / AQL（每场景 3 列，共 5 场景 = 15 数据列）
    # =========================================================================
    "main": {
        "csv_path":      "results/main_result.csv",
        "metric_keys":   ["ATT", "AWT", "AQL"],
        "scene_to_cols": _build_scene_to_cols(_MAIN_SCENES, 3),
        "eval_plan":     _build_eval_plan(_MAIN_SCENES, MAIN_METHODS),
        "calc_kwargs":   {},
    },

    # =========================================================================
    # 模块二：泛化性实验
    # =========================================================================

    # ── 2a. 拓扑迁移：Songdo / Massy / YMT，5 种方法 ────────────────────────
    "gen_topology": {
        "csv_path":      "results/gen_topology_result.csv",
        "metric_keys":   ["ATT", "AWT", "AQL"],
        "scene_to_cols": _build_scene_to_cols(TOPOLOGY_SCENES, 3),
        "eval_plan":     _build_eval_plan(TOPOLOGY_SCENES, LIGHT_METHODS),
        "calc_kwargs":   {},
    },

    # ── 2b. 规模迁移：NewYork 196 路口，5 种方法 ─────────────────────────────
    "gen_scale": {
        "csv_path":      "results/gen_scale_result.csv",
        "metric_keys":   ["ATT", "AWT", "AQL"],
        "scene_to_cols": _build_scene_to_cols([NEWYORK_SCENE], 3),
        "eval_plan":     _build_eval_plan([NEWYORK_SCENE], LIGHT_METHODS),
        "calc_kwargs":   {},
    },

    # ── 2c. 事件迁移（emergy_bus）：紧急车辆 + 公交校车混合路由 ──────────────
    # 场景：Jinan×3 + Hangzhou×2 + Songdo×1 + Massy×1 + YMT×1 = 7 个
    # 每场景路由文件名 = base_scene + EVENT_EMERGY_BUS
    # 指标：ATT / AWT / AQL / EATT / EAWT / BATT / BAWT（7 项）
    "gen_event_emergy_bus": {
        "csv_path":      "results/gen_event_emergy_bus_result.csv",
        "metric_keys":   ["ATT", "AWT", "AQL", "Special_ATT", "Special_AWT"],
        "scene_to_cols": _build_scene_to_cols(
            # 为每个基础场景拼接事件后缀，生成事件路由场景键
            [("JiNan",              s + EVENT_EMERGY_BUS) for s in JINAN_SCENES] +
            [("Hangzhou",           s + EVENT_EMERGY_BUS) for s in HANGZHOU_SCENES] +
            [(ds, ss + EVENT_EMERGY_BUS) for ds, ss in TOPOLOGY_SCENES],
            5
        ),
        "eval_plan": _build_eval_plan(
            [("JiNan",              s + EVENT_EMERGY_BUS) for s in JINAN_SCENES] +
            [("Hangzhou",           s + EVENT_EMERGY_BUS) for s in HANGZHOU_SCENES] +
            [(ds, ss + EVENT_EMERGY_BUS) for ds, ss in TOPOLOGY_SCENES],
            LIGHT_METHODS
        ),
        # 紧急车辆专项指标：按 vType 过滤 tripinfo
        "calc_kwargs": {
            "special_vtypes": {"emergency", "police", "fire_engine"},
        },
    },

    # ── 2d. 事件迁移（accident_debris）：事故 + 路面占道混合路由 ─────────────
    # 指标：ATT / AWT / AQL / MaxQL / TPT（5 项）
    "gen_event_accident_debris": {
        "csv_path":      "results/gen_event_accident_debris_result.csv",
        "metric_keys":   ["ATT", "AWT", "AQL", "MaxQL", "TPT"],
        "scene_to_cols": _build_scene_to_cols(
            [("JiNan",              s + EVENT_ACCIDENT_DEBRIS) for s in JINAN_SCENES] +
            [("Hangzhou",           s + EVENT_ACCIDENT_DEBRIS) for s in HANGZHOU_SCENES] +
            [(ds, ss + EVENT_ACCIDENT_DEBRIS) for ds, ss in TOPOLOGY_SCENES],
            5
        ),
        "eval_plan": _build_eval_plan(
            [("JiNan",              s + EVENT_ACCIDENT_DEBRIS) for s in JINAN_SCENES] +
            [("Hangzhou",           s + EVENT_ACCIDENT_DEBRIS) for s in HANGZHOU_SCENES] +
            [(ds, ss + EVENT_ACCIDENT_DEBRIS) for ds, ss in TOPOLOGY_SCENES],
            LIGHT_METHODS
        ),
        # accident_* 和 debris_* 前缀车辆为占道假车，需从普通车辆统计中排除
        "calc_kwargs": {
            "event_id_prefixes": ["accident_", "debris_"],
        },
    },

    # =========================================================================
    # 模块三：消融实验
    # =========================================================================

    # ── 3a. 训练阶段消融：Zero-shot / SFT-only / E2ELight ────────────────────
    # 数据集：Jinan×3 + Hangzhou×2（常规路由）
    "abl_train": {
        "csv_path":      "results/abl_train_result.csv",
        "metric_keys":   ["ATT", "AWT", "AQL"],
        "scene_to_cols": _build_scene_to_cols(_MAIN_SCENES, 3),
        "eval_plan":     _build_eval_plan(
            _MAIN_SCENES,
            [
                (ZERO_SHOT, "Zero-shot"),
                (SFT_ONLY,  "SFT-only"),
                (E2ELIGHT,  "E2ELight"),
            ]
        ),
        "calc_kwargs": {},
    },

    # ── 3b. 动作空间消融：Phase-only vs Phase+Duration ────────────────────────
    # 数据集：Jinan×3 + Hangzhou×2（常规路由）
    "abl_action": {
        "csv_path":      "results/abl_action_result.csv",
        "metric_keys":   ["ATT", "AWT", "AQL"],
        "scene_to_cols": _build_scene_to_cols(_MAIN_SCENES, 3),
        "eval_plan":     _build_eval_plan(
            _MAIN_SCENES,
            [
                (PHASE_ONLY, "Phase-only"),
                (E2ELIGHT,   "Phase+Duration"),
            ]
        ),
        "calc_kwargs": {},
    },

    # ── 3c. 快慢思考 CoT 消融：Fast-only / Slow-only / Adaptive ──────────────
    # 数据集：Jinan×3 + Hangzhou×2，emergy_bus 事件路由（测试事件识别能力）
    "abl_cot": {
        "csv_path":      "results/abl_cot_result.csv",
        "metric_keys":   ["ATT", "AWT", "AQL", "Special_ATT", "Special_AWT"],
        "scene_to_cols": _build_scene_to_cols(
            [(ds, ss + EVENT_EMERGY_BUS) for ds, ss in ABLATION_EVENT_BASE_SCENES],
            5
        ),
        "eval_plan": _build_eval_plan(
            [(ds, ss + EVENT_EMERGY_BUS) for ds, ss in ABLATION_EVENT_BASE_SCENES],
            [
                (FAST_ONLY, "Fast-only"),
                (SLOW_ONLY, "Slow-only"),
                (E2ELIGHT,  "Adaptive (E2ELight)"),
            ]
        ),
        "calc_kwargs": {
            "special_vtypes": {"emergency", "police", "fire_engine"},
        },
    },

    # ── 3d. EventBulletin 广播机制消融 ───────────────────────────────────────
    # 数据集：Jinan×3 + Hangzhou×2，两类事件路由各跑一次
    # emergy_bus：ATT/AWT/AQL + EATT/EAWT（5 项）
    "abl_bulletin_emergy": {
        "csv_path":      "results/abl_bulletin_result.csv",
        "metric_keys":   ["ATT", "AWT", "AQL", "Special_ATT", "Special_AWT"],
        "scene_to_cols": _build_scene_to_cols(
            [(ds, ss + EVENT_EMERGY_BUS) for ds, ss in ABLATION_EVENT_BASE_SCENES],
            5
        ),
        "eval_plan": _build_eval_plan(
            [(ds, ss + EVENT_EMERGY_BUS) for ds, ss in ABLATION_EVENT_BASE_SCENES],
            [
                (NO_BULLETIN,  "No-Bulletin"),
                (ALL_BULLETIN, "Broadcast-All"),
                (DIR_BULLETIN, "Directed (E2ELight)"),
            ]
        ),
        "calc_kwargs": {
            "special_vtypes": {"emergency", "police", "fire_engine"},
        },
    },

    # accident_debris：ATT/AWT/AQL + MaxQL/TPT（5 项）
    # 共用同一 CSV，但写入不同列组（accident_debris 后缀区分）
    "abl_bulletin_accident": {
        "csv_path":      "results/abl_bulletin_result.csv",
        "metric_keys":   ["ATT", "AWT", "AQL", "MaxQL", "TPT"],
        "scene_to_cols": _build_scene_to_cols(
            [(ds, ss + EVENT_ACCIDENT_DEBRIS) for ds, ss in ABLATION_EVENT_BASE_SCENES],
            5
        ),
        "eval_plan": _build_eval_plan(
            [(ds, ss + EVENT_ACCIDENT_DEBRIS) for ds, ss in ABLATION_EVENT_BASE_SCENES],
            [
                (NO_BULLETIN,  "No-Bulletin"),
                (ALL_BULLETIN, "Broadcast-All"),
                (DIR_BULLETIN, "Directed (E2ELight)"),
            ]
        ),
        "calc_kwargs": {
            "event_id_prefixes": ["accident_", "debris_"],
        },
    },
}

# 对外暴露的 --type 合法值（含 "all"）
_ALL_TYPES = list(_CONFIGS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _dedup_keep_order(items):
    """去重并保持原顺序（用于路径候选列表去重）。"""
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _resolve_base_path(eval_root: str, scenario: str, scene_type: str,
                       model_dir) -> tuple:
    """
    解析 data/eval 目录，兼容 scene_type 与 model_dir 的命名差异。

    逻辑：
      1. scene_type 可能带/不带 .rou 后缀，各尝试一次
      2. model_dir 可能是字符串或列表，按候选顺序逐一尝试
      3. 找到同时包含 statistic_output.xml 和 queue_output.xml 的目录即返回

    返回：(base_path, scene_name_resolved, model_name_resolved)
    """
    # 场景名候选（带/不带 .rou 后缀）
    if scene_type.endswith(".rou"):
        scene_candidates = [scene_type, scene_type[:-4]]
    else:
        scene_candidates = [scene_type, f"{scene_type}.rou"]
    scene_candidates = _dedup_keep_order(scene_candidates)

    # 模型目录候选
    if isinstance(model_dir, (list, tuple)):
        model_candidates = list(model_dir)
    else:
        model_candidates = [model_dir]
    model_candidates = _dedup_keep_order(model_candidates)

    for scene_name in scene_candidates:
        for model_name in model_candidates:
            base_path  = os.path.join(eval_root, scenario, scene_name, model_name)
            stat_path  = os.path.join(base_path, "statistic_output.xml")
            queue_path = os.path.join(base_path, "queue_output.xml")
            if os.path.exists(stat_path) and os.path.exists(queue_path):
                return base_path, scene_name, model_name

    # 未找到时返回第一候选（用于打印缺失路径）
    return (
        os.path.join(eval_root, scenario, scene_candidates[0], model_candidates[0]),
        scene_candidates[0],
        model_candidates[0],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV 自动创建与写入
# ─────────────────────────────────────────────────────────────────────────────

def _build_empty_csv(cfg: dict) -> list:
    """
    根据 scene_to_cols 和 eval_plan 构造空白 CSV 的行列表。

    行结构：
      第 0 行：场景名表头（Category, Method, 场景A, , , 场景B, , , ...）
      第 1 行：指标行（空, 空, ATT↓, AWT↓, AQL↓, ...）
      第 2+行：每个唯一 row_key 一行，值全空
    """
    metric_keys   = cfg["metric_keys"]
    scene_to_cols = cfg["scene_to_cols"]

    # 计算 CSV 总列数
    max_col    = max(max(cols) for cols in scene_to_cols.values())
    total_cols = max_col + 1

    # 第 0 行：场景名表头
    header0    = [""] * total_cols
    header0[0] = "Category"
    header0[1] = "Method"
    for (scenario, scene_type), cols in scene_to_cols.items():
        # 只在每组列的第一列写场景标签
        header0[cols[0]] = f"{scenario}/{scene_type}"

    # 第 1 行：指标名表头（带方向箭头）
    header1 = [""] * total_cols
    for (_scenario, _scene_type), cols in scene_to_cols.items():
        for key, col in zip(metric_keys, cols):
            header1[col] = _ARROW.get(key, key)

    # 收集唯一 row_key（按 eval_plan 顺序去重）
    row_keys  = _dedup_keep_order(rk for _, _, _, rk in cfg["eval_plan"])

    # 为每个 row_key 生成空数据行
    data_rows = []
    for rk in row_keys:
        row    = [""] * total_cols
        row[1] = rk
        data_rows.append(row)

    return [header0, header1] + data_rows


def _write_csv(cfg: dict, metrics: dict, scenario: str,
               scene_type: str, row_key: str) -> None:
    """
    将 metric_keys 按 scene_to_cols 指定的列写入对应 CSV。

    若 CSV 文件不存在则自动创建空模板；
    若对应 row_key 行不存在则在末尾追加。
    """
    col_indices = cfg["scene_to_cols"].get((scenario, scene_type))
    if col_indices is None:
        print(f"⚠️  [CSV] ({scenario}, {scene_type}) 未在 scene_to_cols 中配置，跳过。")
        return

    csv_path = os.path.join(PROJECT_ROOT, cfg["csv_path"])

    # CSV 不存在时自动创建空模板
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        empty_rows = _build_empty_csv(cfg)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(empty_rows)
        print(f"ℹ️  [CSV] 自动创建空模板: {csv_path}")

    metric_keys = cfg["metric_keys"]
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        # 查找目标行（按 Method 列匹配 row_key）
        target_row_idx = None
        for idx, row in enumerate(rows):
            if len(row) > 1 and row[1].strip() == row_key:
                target_row_idx = idx
                break

        # row_key 不存在时追加新行
        if target_row_idx is None:
            new_row    = [""] * (max(col_indices) + 1)
            new_row[1] = row_key
            rows.append(new_row)
            target_row_idx = len(rows) - 1
            print(f"ℹ️  [CSV] 新增行: Method='{row_key}'")

        # 确保行足够长
        target_row = rows[target_row_idx]
        while len(target_row) <= max(col_indices):
            target_row.append("")

        # 写入各指标值
        for key, col in zip(metric_keys, col_indices):
            target_row[col] = _FMT[key](metrics.get(key, 0))
        rows[target_row_idx] = target_row

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

        kv = ", ".join(f"{k}={_FMT[k](metrics.get(k, 0))}" for k in metric_keys)
        print(f"✅  {row_key} | {scenario}/{scene_type} | {kv}")

    except Exception as e:
        print(f"❌ [CSV] 写入失败: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 核心收集函数
# ─────────────────────────────────────────────────────────────────────────────

def collect(exp_type: str) -> None:
    """
    收集指定实验类型的全部评测结果并写入对应结果 CSV。

    路径约定：data/eval/{scenario}/{scene_type}/{method}/
              包含 statistic_output.xml、queue_output.xml、tripinfo.out.xml

    参数：
        exp_type: 实验类型，须为 _CONFIGS 中的 key 之一
    """
    cfg = _CONFIGS.get(exp_type)
    if cfg is None:
        print(f"❌ 未知实验类型: '{exp_type}'，可选: {_ALL_TYPES}")
        return

    eval_root            = os.path.join(PROJECT_ROOT, "data", "eval")
    total, success, skip = 0, 0, 0

    print(f"\n{'='*60}")
    print(f"[{exp_type}] 开始收集 → {cfg['csv_path']}")
    print(f"{'='*60}")

    for scenario, scene_type, model_dir, row_key in cfg["eval_plan"]:
        total += 1
        base_path, scene_resolved, model_resolved = _resolve_base_path(
            eval_root, scenario, scene_type, model_dir
        )
        stat_path     = os.path.join(base_path, "statistic_output.xml")
        queue_path    = os.path.join(base_path, "queue_output.xml")
        tripinfo_path = os.path.join(base_path, "tripinfo.out.xml")

        # 核心文件不存在则跳过（tripinfo 为可选文件，缺失时部分指标返回 0）
        if not os.path.exists(stat_path) or not os.path.exists(queue_path):
            print(f"⚠️  [Skip] 文件缺失: {base_path}")
            skip += 1
            continue

        try:
            calc    = MetricsCalculator()
            metrics = calc.calculate_from_files(
                stat_path, queue_path, scenario,
                tripinfo_file=tripinfo_path,
                **cfg["calc_kwargs"],
            )
        except Exception as e:
            print(f"❌ 指标计算失败 {base_path}: {e}")
            skip += 1
            continue

        # 写入 CSV（使用 scene_type 原始键，而非解析后的名称）
        _write_csv(cfg, metrics, scenario, scene_type, row_key)

        # 如果发生了路径解析（别名命中），打印提示
        if scene_resolved != scene_type or model_resolved != (
            model_dir[0] if isinstance(model_dir, list) else model_dir
        ):
            print(
                f"ℹ️  [Resolve] {scenario}/{scene_type}/{model_dir} "
                f"→ {scenario}/{scene_resolved}/{model_resolved}"
            )
        success += 1

    csv_abs = os.path.join(PROJECT_ROOT, cfg["csv_path"])
    print(f"\n{'='*60}")
    print(f"[{exp_type}] 完成：{total} 条，成功 {success}，跳过 {skip}")
    print(f"结果文件：{csv_abs}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E2ELight 统一评估指标收集（按实验模块分类）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _type_help = "\n".join([
        "实验类型（--type 参数）：",
        "  main                   主实验（Jinan×3 + Hangzhou×2，13种方法）",
        "  gen_topology           泛化：拓扑迁移（Songdo/Massy/YMT）",
        "  gen_scale              泛化：规模迁移（NewYork 196路口）",
        "  gen_event_emergy_bus   泛化：事件迁移（紧急车辆+公交）",
        "  gen_event_accident_debris  泛化：事件迁移（事故+占道）",
        "  abl_train              消融：训练阶段（Zero-shot/SFT/E2ELight）",
        "  abl_action             消融：动作空间（Phase-only/Phase+Duration）",
        "  abl_cot                消融：快慢思考（Fast/Slow/Adaptive）",
        "  abl_bulletin_emergy    消融：广播机制（emergy_bus场景）",
        "  abl_bulletin_accident  消融：广播机制（accident_debris场景）",
        "  all                    依次运行所有类型",
    ])
    parser.add_argument(
        "--type", "-t",
        type=str,
        required=True,
        choices=_ALL_TYPES + ["all"],
        metavar="TYPE",
        help=_type_help,
    )
    args = parser.parse_args()

    if args.type == "all":
        for et in _ALL_TYPES:
            print(f"\n{'#'*60}\n# {et}\n{'#'*60}")
            collect(et)
    else:
        collect(args.type)
