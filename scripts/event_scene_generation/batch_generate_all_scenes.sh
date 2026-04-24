#!/bin/bash
# =============================================================================
# batch_generate_all_scenes.sh
# 批量为所有数据集生成五类交通事件场景路由文件，并生成路网可视化图。
# 事件类型：Emergency Vehicles、School/City Bus、Traffic Accident、Road Debris、Pedestrian Crossing
#
# 用法：
#   cd /path/to/VLMTraffic
#   bash scripts/event_scene_generation/batch_generate_all_scenes.sh
#
# 输出文件（按数据集）：
#   data/raw/{DATASET}/env/{BASE}_emergy.rou.xml
#   data/raw/{DATASET}/env/{BASE}_bus.rou.xml
#   data/raw/{DATASET}/env/{BASE}_accident.rou.xml
#   data/raw/{DATASET}/env/{BASE}_debris.rou.xml
#   data/raw/{DATASET}/env/{BASE}_pedestrian.rou.xml
#   data/raw/{DATASET}/env/{DATASET}_event_network.png
#
# ── 各事件场景生成的 vType / trip ID 汇总 ────────────────────────────────────
#
# 1. Emergency Vehicles  (add_emergency_vehicles.py → {BASE}_emergy.rou.xml)
#    vType id : emergency, police, fire_engine
#    载体    : <vehicle> 元素，将普通车辆 type 替换（真实运动车辆）
#    metrics : ATT/AWT/AQL + EATT/EAWT（special_vtypes 过滤 vType）
#
# 2. School/City Bus     (add_bus_vehicles.py → {BASE}_bus.rou.xml)
#    vType id : bus, school_bus
#    载体    : <vehicle> 元素，将普通车辆 type 替换（真实运动车辆）
#    metrics : ATT/AWT/AQL + BATT/BAWT（special_vtypes 过滤 vType）
#
# 3. Traffic Accident    (generate_traffic_accident.py → {BASE}_accident.rou.xml)
#    vType id : crash_vehicle_a, crash_vehicle_b
#    载体    : <trip>+<stop> 占位假车（静态障碍物，不渲染为背景车）
#    trip id  : accident_{junction}_{idx}
#              accident_{junction}_{idx}_blocker1 / _blocker2（3车道封路）
#    metrics : ATT/AWT/AQL（过滤 accident_* 前缀）+ MaxQL/TPT
#
# 4. Road Debris         (generate_road_debris.py → {BASE}_debris.rou.xml)
#    vType id : barrier_A_<len>, barrier_B_<len>, ..., barrier_E_<len>（路障，含长度后缀）
#              tree_branch_1lane（树枝，单车道）
#    载体    : <trip>+<stop> 占位假车（静态障碍物，EmergencyManager3D 渲染）
#    trip id  : debris_{junction}_{idx}
#    metrics : ATT/AWT/AQL（过滤 debris_* 前缀）+ MaxQL/TPT
#
# 5. Pedestrian Crossing (generate_pedestrian_crossing.py → {BASE}_pedestrian.rou.xml)
#    vType id : pedestrian_crossing
#    载体    : <trip>+<stop> 占位假车（静态障碍物，EmergencyManager3D 渲染）
#    trip id  : ped_{junction}_{grp}_{ped}
#    metrics : ATT/AWT/AQL（过滤 ped_* 前缀）+ MaxQL/TPT
#
# =============================================================================

set -e

# ── 项目根目录（自动检测）────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo " E2ELight Event Scene Generation Pipeline"
echo " Project root: ${PROJECT_ROOT}"
echo "=========================================="

# ── 通用参数 ─────────────────────────────────────────────────────────────────
SEED=42
EMERGENCY_RATIO=0.02  # 紧急车辆注入比例（4%）

BUS_RATIO=0.03        # 公交/校车注入比例（5%）

ACCIDENT_RATE=0.8    # 每小时每交叉口事故发生数
ACCIDENT_PED_RATIO=0.5 # 到地行人生成概率

DEBRIS_RATE=0.8     # 每小时每交叉口路障生成组数
DEBRIS_MIN_GAP=5.0    # 最小前后路障间距（m）
DEBRIS_MAX_GAP=50.0    # 最大前后路障间距（m）
DEBRIS_MIN_DUR=200.0  # 最小持续时间（s）
DEBRIS_MAX_DUR=600.0  # 最大持续时间（s）

PEDESTRIAN_RATE=1.0   # 每小时每交叉口行人过街组数
PEDESTRIAN_PER_GROUP=4 # 每组行人数

MAX_RANGE=80.0       # 距路口最大距离（m）针对路障、事故的范围限制

# ── 各数据集配置（路口 ID 由各 Python 脚本自动从 configs/scenairo_config.py 读取）───
declare -A NET_FILES
declare -A BASE_ROUX
declare -A BASE_NAMES

NET_FILES[JiNan]="data/raw/JiNan/env/jinan.net.xml"
BASE_ROUX[JiNan]="data/raw/JiNan/env/anon_3_4_jinan_real.rou.xml"
BASE_NAMES[JiNan]="anon_3_4_jinan_real"

NET_FILES[Hangzhou]="data/raw/Hangzhou/env/Hangzhou.net.xml"
BASE_ROUX[Hangzhou]="data/raw/Hangzhou/env/anon_4_4_hangzhou_real.rou.xml"
BASE_NAMES[Hangzhou]="anon_4_4_hangzhou_real"

NET_FILES[SouthKorea_Songdo]="data/raw/SouthKorea_Songdo/env/songdo.net.xml"
BASE_ROUX[SouthKorea_Songdo]="data/raw/SouthKorea_Songdo/env/songdo.rou.xml"
BASE_NAMES[SouthKorea_Songdo]="songdo"

NET_FILES[France_Massy]="data/raw/France_Massy/env/massy.net.xml"
BASE_ROUX[France_Massy]="data/raw/France_Massy/env/massy.rou.xml"
BASE_NAMES[France_Massy]="massy"

NET_FILES[Hongkong_YMT]="data/raw/Hongkong_YMT/env/YMT.net.xml"
BASE_ROUX[Hongkong_YMT]="data/raw/Hongkong_YMT/env/YMT.rou.xml"
BASE_NAMES[Hongkong_YMT]="YMT"

NET_FILES[NewYork]="data/raw/NewYork/env/NewYork.net.xml"
BASE_ROUX[NewYork]="data/raw/NewYork/env/anon_28_7_newyork_real_double.rou.xml"
BASE_NAMES[NewYork]="anon_28_7_newyork_real_double"

SCENARIOS=("JiNan" "Hangzhou" "SouthKorea_Songdo" "France_Massy" "Hongkong_YMT" "NewYork")
# （“JiNan” "Hangzhou" "SouthKorea_Songdo" "France_Massy" "Hongkong_YMT" "NewYork")

# ── 主循环 ────────────────────────────────────────────────────────────────────
for SCENE in "${SCENARIOS[@]}"; do
    NET="${NET_FILES[$SCENE]}"
    BASE_ROU="${BASE_ROUX[$SCENE]}"
    BASE="${BASE_NAMES[$SCENE]}"
    ENV_DIR="$(dirname "${BASE_ROU}")"

    echo ""
    echo "──────────────────────────────────────────"
    echo " Processing: ${SCENE}"
    echo "──────────────────────────────────────────"

    # 校验文件是否存在
    if [ ! -f "${NET}" ]; then
        echo "[WARN] 网络文件不存在，跳过: ${NET}"
        continue
    fi
    if [ ! -f "${BASE_ROU}" ]; then
        echo "[WARN] 基础路由文件不存在，跳过: ${BASE_ROU}"
        continue
    fi

    OUT_EMERGY="${ENV_DIR}/${BASE}_emergy.rou.xml"
    OUT_BUS="${ENV_DIR}/${BASE}_bus.rou.xml"
    OUT_ACCIDENT="${ENV_DIR}/${BASE}_accident.rou.xml"
    OUT_DEBRIS="${ENV_DIR}/${BASE}_debris.rou.xml"
    OUT_PEDESTRIAN="${ENV_DIR}/${BASE}_pedestrian.rou.xml"
    OUT_VIZ="${ENV_DIR}/${SCENE}_event_network.png"
    OUT_SUMMARY="${ENV_DIR}/${SCENE}_generation_summary.txt"

    echo "=== Generation Summary for ${SCENE} ===" > "${OUT_SUMMARY}"

    # ── 1. Emergency Vehicles ─────────────────────────────────────────────────
    echo "[1/6] Generating Emergency Vehicle scene..." | tee -a "${OUT_SUMMARY}"
    python scripts/event_scene_generation/add_emergency_vehicles.py \
        --input  "${BASE_ROU}" \
        --output "${OUT_EMERGY}" \
        --ratio  "${EMERGENCY_RATIO}" \
        --seed   "${SEED}" 2>&1 | tee -a "${OUT_SUMMARY}"

    # ── 2. School/City Bus ────────────────────────────────────────────────────
    echo "[2/6] Generating School/City Bus scene..." | tee -a "${OUT_SUMMARY}"
    python scripts/event_scene_generation/add_bus_vehicles.py \
        --input  "${BASE_ROU}" \
        --output "${OUT_BUS}" \
        --ratio  "${BUS_RATIO}" \
        --seed   "${SEED}" 2>&1 | tee -a "${OUT_SUMMARY}"

    # ── 3. Traffic Accident ───────────────────────────────────────────────────
    echo "[3/6] Generating Traffic Accident scene..." | tee -a "${OUT_SUMMARY}"
    python scripts/event_scene_generation/generate_traffic_accident.py \
        --scenario "${SCENE}" \
        --net      "${NET}" \
        --base_rou "${BASE_ROU}" \
        --output   "${OUT_ACCIDENT}" \
        --rate     "${ACCIDENT_RATE}" \
        --ped_ratio "${ACCIDENT_PED_RATIO}" \
        --range    "${MAX_RANGE}" \
        --seed     "${SEED}" 2>&1 | tee -a "${OUT_SUMMARY}"

    # ── 4. Road Debris ────────────────────────────────────────────────────────
    echo "[4/6] Generating Road Debris scene..." | tee -a "${OUT_SUMMARY}"
    python scripts/event_scene_generation/generate_road_debris.py \
        --scenario "${SCENE}" \
        --net      "${NET}" \
        --base_rou "${BASE_ROU}" \
        --output   "${OUT_DEBRIS}" \
        --rate     "${DEBRIS_RATE}" \
        --min_gap  "${DEBRIS_MIN_GAP}" \
        --max_gap  "${DEBRIS_MAX_GAP}" \
        --min_duration "${DEBRIS_MIN_DUR}" \
        --max_duration "${DEBRIS_MAX_DUR}" \
        --range    "${MAX_RANGE}" \
        --seed     "${SEED}" 2>&1 | tee -a "${OUT_SUMMARY}"

    # ── 5. Pedestrian Crossing ────────────────────────────────────────────────
    echo "[5/6] Generating Pedestrian Crossing scene..." | tee -a "${OUT_SUMMARY}"
    python scripts/event_scene_generation/generate_pedestrian_crossing.py \
        --scenario "${SCENE}" \
        --net      "${NET}" \
        --base_rou "${BASE_ROU}" \
        --output   "${OUT_PEDESTRIAN}" \
        --rate     "${PEDESTRIAN_RATE}" \
        --per_group "${PEDESTRIAN_PER_GROUP}" \
        --seed     "${SEED}" 2>&1 | tee -a "${OUT_SUMMARY}"

    # ── 6. 路网可视化 ──────────────────────────────────────────────────────────
    echo "[6/6] Generating event network visualization..." | tee -a "${OUT_SUMMARY}"
    python scripts/event_scene_generation/visualize_event_network.py \
        --net        "${NET}" \
        --output     "${OUT_VIZ}" \
        --scenario   "${SCENE}" \
        --emergency  "${OUT_EMERGY}" \
        --bus        "${OUT_BUS}" \
        --accident   "${OUT_ACCIDENT}" \
        --debris     "${OUT_DEBRIS}" \
        --pedestrian "${OUT_PEDESTRIAN}" \
        --dpi 150 2>&1 | tee -a "${OUT_SUMMARY}"

    echo "[Done] ${SCENE} 完成。输出文件:" | tee -a "${OUT_SUMMARY}"
    echo "  Emergency:  ${OUT_EMERGY}" | tee -a "${OUT_SUMMARY}"
    echo "  Bus:        ${OUT_BUS}" | tee -a "${OUT_SUMMARY}"
    echo "  Accident:   ${OUT_ACCIDENT}" | tee -a "${OUT_SUMMARY}"
    echo "  Debris:     ${OUT_DEBRIS}" | tee -a "${OUT_SUMMARY}"
    echo "  Pedestrian: ${OUT_PEDESTRIAN}" | tee -a "${OUT_SUMMARY}"
    echo "  Map:        ${OUT_VIZ}" | tee -a "${OUT_SUMMARY}"
    echo "  Summary:    ${OUT_SUMMARY}" | tee -a "${OUT_SUMMARY}"
done

echo ""
echo "=========================================="
echo " All scenarios processed successfully!"
echo " 路网可视化图保存在各个数据集的 env 目录下"
echo "=========================================="
