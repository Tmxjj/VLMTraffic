#!/bin/bash
# =============================================================================
# batch_generate_mixed_events.sh
# 批量为所有数据集（含 Syn-Train）的每条路由文件生成两类混合事件路由文件：
#
#   {BASE}_emergy_bus.rou.xml      紧急车辆（emergency/police/fire_engine）
#                                  + 公交/校车（bus/school_bus）
#                                  ← 替换普通车辆 type，时间均匀分布
#
#   {BASE}_accident_debris.rou.xml  交通事故（crash_vehicle_a/b）
#                                  + 路面碎片（barrier_A~E / tree_branch）
#                                  ← trip+stop 静态障碍物，时空均匀分布
#
# 均匀性保证（由 generate_mixed_events.py 实现）：
#   时间均匀：将 [0, max_depart] 等分为 N 个桶，每桶随机取一个出发时间
#   空间均匀：路口按 ID 排序后 round-robin 轮转分配事件（每路口接受等量事件）
#
# 参数：沿用 batch_generate_all_scenes.sh 中已有默认值，不做更改。
#
# 用法：
#   cd /path/to/VLMTraffic
#   bash scripts/event_scene_generation/batch_generate_mixed_events.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo " E2ELight Mixed Event Generation Pipeline"
echo " Project root: ${PROJECT_ROOT}"
echo "=========================================="

# ── 通用参数（与 batch_generate_all_scenes.sh 保持一致）────────────────────
SEED=42
EMERGENCY_RATIO=0.015
BUS_RATIO=0.015
ACCIDENT_RATE=0.8
DEBRIS_RATE=0.8
MAX_RANGE=80.0
EVENT_DURATION=600.0
DEBRIS_MIN_GAP=5.0
DEBRIS_MAX_GAP=50.0
DEBRIS_MIN_DUR=200.0
DEBRIS_MAX_DUR=600.0
BARRIER_RATIO=0.8

PYTHON_SCRIPT="scripts/event_scene_generation/generate_mixed_events.py"

# ── 场景配置 ──────────────────────────────────────────────────────────────────
# 格式：SCENARIO_ID|SCENE_NAME|NET_FILE|BASE_ROU_FILE
#   SCENARIO_ID  : Python --scenario 参数（用于从 SCENARIO_CONFIGS 读取路口列表）
#   SCENE_NAME   : 日志中的可读名称
#   NET_FILE     : .net.xml 路径（accident_debris 场景需要）
#   BASE_ROU_FILE: 基础路由文件路径（可有多条，用空格分隔后在数组中处理）

# 每条路由文件单独一行：SCENE|NET|BASE_ROU|SCENARIO_ID（可选，空则自动探测路网路口）
declare -a ENTRIES

# JiNan × 3
ENTRIES+=("JiNan|data/raw/JiNan/env/jinan.net.xml|data/raw/JiNan/env/anon_3_4_jinan_real.rou.xml|JiNan")
ENTRIES+=("JiNan|data/raw/JiNan/env/jinan.net.xml|data/raw/JiNan/env/anon_3_4_jinan_real_2000.rou.xml|JiNan")
ENTRIES+=("JiNan|data/raw/JiNan/env/jinan.net.xml|data/raw/JiNan/env/anon_3_4_jinan_real_2500.rou.xml|JiNan")

# Hangzhou × 2
ENTRIES+=("Hangzhou|data/raw/Hangzhou/env/Hangzhou.net.xml|data/raw/Hangzhou/env/anon_4_4_hangzhou_real.rou.xml|Hangzhou")
ENTRIES+=("Hangzhou|data/raw/Hangzhou/env/Hangzhou.net.xml|data/raw/Hangzhou/env/anon_4_4_hangzhou_real_5816.rou.xml|Hangzhou")

# SouthKorea_Songdo × 1
ENTRIES+=("SouthKorea_Songdo|data/raw/SouthKorea_Songdo/env/songdo.net.xml|data/raw/SouthKorea_Songdo/env/songdo.rou.xml|SouthKorea_Songdo")

# France_Massy × 1
ENTRIES+=("France_Massy|data/raw/France_Massy/env/massy.net.xml|data/raw/France_Massy/env/massy.rou.xml|France_Massy")

# Hongkong_YMT × 1
ENTRIES+=("Hongkong_YMT|data/raw/Hongkong_YMT/env/YMT.net.xml|data/raw/Hongkong_YMT/env/YMT.rou.xml|Hongkong_YMT")

# NewYork × 1
ENTRIES+=("NewYork|data/raw/NewYork/env/NewYork.net.xml|data/raw/NewYork/env/anon_28_7_newyork_real_double.rou.xml|NewYork")

# Syn-Train × 1
ENTRIES+=("Syn-Train|data/raw/Syn-Train/env/roadnet_4_4.net.xml|data/raw/Syn-Train/env/anon_4_4_synthetic_8000.rou.xml|")

# ── 主循环 ────────────────────────────────────────────────────────────────────
total=${#ENTRIES[@]}
done_count=0

for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r SCENE NET BASE_ROU SCENARIO_ID <<< "${entry}"

    ENV_DIR="$(dirname "${BASE_ROU}")"
    BASE_NAME="$(basename "${BASE_ROU}" .rou.xml)"
    OUT_EB="${ENV_DIR}/${BASE_NAME}_emergy_bus.rou.xml"
    OUT_AD="${ENV_DIR}/${BASE_NAME}_accident_debris.rou.xml"

    echo ""
    echo "──────────────────────────────────────────────────────"
    echo " 场景: ${SCENE}  |  路由: ${BASE_NAME}"
    echo "──────────────────────────────────────────────────────"

    # 校验输入文件
    if [ ! -f "${BASE_ROU}" ]; then
        echo "[WARN] 基础路由文件不存在，跳过: ${BASE_ROU}"
        continue
    fi
    if [ ! -f "${NET}" ]; then
        echo "[WARN] 路网文件不存在，跳过: ${NET}"
        continue
    fi

    # ── 1. 生成 emergy_bus ────────────────────────────────────────────────
    echo "[1/2] 生成 emergy_bus → ${OUT_EB}"
    python "${PYTHON_SCRIPT}" emergy_bus \
        --input            "${BASE_ROU}" \
        --output           "${OUT_EB}" \
        --emergency_ratio  "${EMERGENCY_RATIO}" \
        --bus_ratio        "${BUS_RATIO}" \
        --seed             "${SEED}"

    # ── 2. 生成 accident_debris ──────────────────────────────────────────
    echo "[2/2] 生成 accident_debris → ${OUT_AD}"
    AD_SCENARIO_OPT=""
    if [ -n "${SCENARIO_ID}" ]; then
        AD_SCENARIO_OPT="--scenario ${SCENARIO_ID}"
    fi

    python "${PYTHON_SCRIPT}" accident_debris \
        --net              "${NET}" \
        --base_rou         "${BASE_ROU}" \
        --output           "${OUT_AD}" \
        ${AD_SCENARIO_OPT} \
        --accident_rate    "${ACCIDENT_RATE}" \
        --debris_rate      "${DEBRIS_RATE}" \
        --max_range        "${MAX_RANGE}" \
        --event_duration   "${EVENT_DURATION}" \
        --debris_min_gap   "${DEBRIS_MIN_GAP}" \
        --debris_max_gap   "${DEBRIS_MAX_GAP}" \
        --debris_min_dur   "${DEBRIS_MIN_DUR}" \
        --debris_max_dur   "${DEBRIS_MAX_DUR}" \
        --barrier_ratio    "${BARRIER_RATIO}" \
        --seed             "${SEED}"

    done_count=$((done_count + 1))
    echo "[Done] ${SCENE}/${BASE_NAME} 完成（${done_count}/${total}）"
    echo "  EmgBus:    ${OUT_EB}"
    echo "  AccDebris: ${OUT_AD}"
done

echo ""
echo "=========================================="
echo " All ${done_count}/${total} entries processed!"
echo "=========================================="
