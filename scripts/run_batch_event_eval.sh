#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-04-20
 # @Description: normal + 五类交通事件场景批量评测脚本（基线 + VLM，合并版）
 #               在全部 6 个数据集 × 6 类场景上运行评测。
 #
 #               场景类型：
 #                 normal     — 常规车流（无事件注入）
 #                 emergency  — 紧急车辆（_emergy.rou.xml）
 #                 bus        — 公交/校车（_bus.rou.xml）
 #                 accident   — 交通事故（_accident.rou.xml）
 #                 debris     — 路面碎片/路障（_debris.rou.xml）
 #                 pedestrian — 行人过街（_pedestrian.rou.xml）
 #
 #               数据集：
 #                 JiNan / Hangzhou / SouthKorea_Songdo / France_Massy / Hongkong_YMT / NewYork
 #
 #               输出路径（与 src/evaluation/run_eval.py 一致）：
 #                 data/eval/{dataset}/{route_file_name}/{method}/
 #               例：data/eval/JiNan/anon_3_4_jinan_real_emergy/fixed_time/
 #
 #               参数已由最大决策步(max_steps)彻底换为总仿真时间秒数(max_sumo_seconds)。
 #               max_sumo_seconds 自动从对应路由文件的最大 depart 时间动态计算：
 #                 max_sumo_seconds = max_depart + BUFFER_S，上下限 [300, 6000]
 #
 #               使用说明：
 #                 # 仅跑基线（本地，无需 GPU) MaxPressure + FixedTim
 #                 bash scripts/run_batch_event_eval.sh --baseline-only
 #
 #                 # 仅跑特定场景类型的基线（支持 normal/emergency/bus/accident/debris/pedestrian）：
 #                 bash scripts/run_batch_event_eval.sh --baseline-only --event emergency
 #
 #                 # 仅跑 VLM（远程服务器，需提前启动 vLLM）：
 #                 bash scripts/run_batch_event_eval.sh --port 8000 --model_name qwen3-vl-8b-sft --temperature 0 --max_new_tokens 4096
 #
 #                 # 同时跑基线 + VLM：
 #                 bash scripts/run_batch_event_eval.sh --port 8000 --model_name qwen3-vl-8b-sft --with-baseline
 #
 #               评测完成后收集指标：
 #                 python src/evaluation/collect_metrics.py --type emergency
 #                 python src/evaluation/collect_metrics.py --type all
 #
 # @FilePath: /VLMTraffic/scripts/run_batch_event_eval.sh
###

# ─── 默认配置 ──────────────────────────────────────────────────
BUFFER_S=180     # 最后一辆车出发后的额外仿真缓冲
MIN_SUMO_SECONDS=300
MAX_SUMO_SECONDS_CAP=3600
LOG_DIR="./log/eval_results"

API_PORT=""
MODEL_NAME=""
TEMPERATURE=""
MAX_NEW_TOKENS=""
BASELINE_ONLY=false
WITH_BASELINE=false
# 可选：仅运行单类场景（不指定则跑全部 6 类：normal + 5 类事件）
EVENT_FILTER=""

# 解析参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port)           API_PORT="$2";      shift ;;
        --model_name)     MODEL_NAME="$2";    shift ;;
        --temperature)    TEMPERATURE="$2";   shift ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift ;;
        --baseline-only)  BASELINE_ONLY=true  ;;
        --with-baseline)  WITH_BASELINE=true  ;;
        --event)          EVENT_FILTER="$2";  shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

mkdir -p "$LOG_DIR"

# ─── 各数据集路由文件根路径 ─────────────────────────────────────
declare -A ENV_DIRS
ENV_DIRS[JiNan]="data/raw/JiNan/env"
ENV_DIRS[Hangzhou]="data/raw/Hangzhou/env"
ENV_DIRS[SouthKorea_Songdo]="data/raw/SouthKorea_Songdo/env"
ENV_DIRS[France_Massy]="data/raw/France_Massy/env"
ENV_DIRS[Hongkong_YMT]="data/raw/Hongkong_YMT/env"
ENV_DIRS[NewYork]="data/raw/NewYork/env"

# 各数据集对应的事件路由文件 basename（{DATASET}_{EVENT_SUFFIX}.rou.xml）
# 格式：declare -A {SUFFIX}_{DATASET}
# 为避免 Bash 关联数组的嵌套限制，使用扁平变量名

# ── normal（无事件注入）──────────────────────────────────────────
# ROUTE_normal_JiNan="anon_3_4_jinan_real.rou.xml"
ROUTE_normal_JiNan_1="anon_3_4_jinan_real_2000.rou.xml"
ROUTE_normal_JiNan_2="anon_3_4_jinan_real_2500.rou.xml"
ROUTE_normal_JiNan_3="anon_3_4_jinan_synthetic_24000_60min.rou.xml"
# ROUTE_normal_Hangzhou="anon_4_4_hangzhou_real.rou.xml"
ROUTE_normal_Hangzhou_1="anon_4_4_hangzhou_real_5816.rou.xml"
ROUTE_normal_Hangzhou_2="anon_4_4_hangzhou_synthetic_24000_60min.rou.xml"
# ROUTE_normal_SouthKorea_Songdo="songdo.rou.xml"
# ROUTE_normal_France_Massy="massy.rou.xml"
# ROUTE_normal_Hongkong_YMT="YMT.rou.xml"
# ROUTE_normal_NewYork="anon_28_7_newyork_real_double.rou.xml"
ROUTE_normal_NewYork_1="anon_28_7_newyork_real_triple.rou.xml"

# ── emergency（suffix: emergy）───────────────────────────────────
ROUTE_emergency_JiNan="anon_3_4_jinan_real_emergy.rou.xml"
ROUTE_emergency_Hangzhou="anon_4_4_hangzhou_real_emergy.rou.xml"
ROUTE_emergency_SouthKorea_Songdo="songdo_emergy.rou.xml"
ROUTE_emergency_France_Massy="massy_emergy.rou.xml"
ROUTE_emergency_Hongkong_YMT="YMT_emergy.rou.xml"
ROUTE_emergency_NewYork="anon_28_7_newyork_real_double_emergy.rou.xml"

# ── bus ─────────────────────────────────────────────────────────
ROUTE_bus_JiNan="anon_3_4_jinan_real_bus.rou.xml"
ROUTE_bus_Hangzhou="anon_4_4_hangzhou_real_bus.rou.xml"
ROUTE_bus_SouthKorea_Songdo="songdo_bus.rou.xml"
ROUTE_bus_France_Massy="massy_bus.rou.xml"
ROUTE_bus_Hongkong_YMT="YMT_bus.rou.xml"
ROUTE_bus_NewYork="anon_28_7_newyork_real_double_bus.rou.xml"

# ── accident ─────────────────────────────────────────────────────
ROUTE_accident_JiNan="anon_3_4_jinan_real_accident.rou.xml"
ROUTE_accident_Hangzhou="anon_4_4_hangzhou_real_accident.rou.xml"
ROUTE_accident_SouthKorea_Songdo="songdo_accident.rou.xml"
ROUTE_accident_France_Massy="massy_accident.rou.xml"
ROUTE_accident_Hongkong_YMT="YMT_accident.rou.xml"
ROUTE_accident_NewYork="anon_28_7_newyork_real_double_accident.rou.xml"

# ── debris ────────────────────────────────────────────────────────
ROUTE_debris_JiNan="anon_3_4_jinan_real_debris.rou.xml"
ROUTE_debris_Hangzhou="anon_4_4_hangzhou_real_debris.rou.xml"
ROUTE_debris_SouthKorea_Songdo="songdo_debris.rou.xml"
ROUTE_debris_France_Massy="massy_debris.rou.xml"
ROUTE_debris_Hongkong_YMT="YMT_debris.rou.xml"
ROUTE_debris_NewYork="anon_28_7_newyork_real_double_debris.rou.xml"

# ── pedestrian ────────────────────────────────────────────────────
ROUTE_pedestrian_JiNan="anon_3_4_jinan_real_pedestrian.rou.xml"
ROUTE_pedestrian_Hangzhou="anon_4_4_hangzhou_real_pedestrian.rou.xml"
ROUTE_pedestrian_SouthKorea_Songdo="songdo_pedestrian.rou.xml"
ROUTE_pedestrian_France_Massy="massy_pedestrian.rou.xml"
ROUTE_pedestrian_Hongkong_YMT="YMT_pedestrian.rou.xml"
ROUTE_pedestrian_NewYork="anon_28_7_newyork_real_double_pedestrian.rou.xml"

DATASETS=("JiNan" "Hangzhou" "SouthKorea_Songdo" "France_Massy" "Hongkong_YMT" "NewYork")
EVENT_TYPES=("normal" "emergency" "bus" "accident" "debris" "pedestrian")

# ─── 辅助函数 ──────────────────────────────────────────────────

get_route_files_for_scene() {
    local event="$1"
    local dataset="$2"
    local prefix="ROUTE_${event}_${dataset}"
    local var_name

    # 支持同一 (event, dataset) 下定义多个路由变量：
    # ROUTE_normal_JiNan / ROUTE_normal_JiNan_1 / ROUTE_normal_JiNan_2 ...
    while IFS= read -r var_name; do
        local route_file="${!var_name}"
        [ -n "$route_file" ] && echo "$route_file"
    done < <(compgen -A variable "$prefix" | sort -V)
}

get_max_sumo_seconds() {
    local rou_path="$1"
    python3 -c "
import xml.etree.ElementTree as ET, sys
try:
    root = ET.parse('${rou_path}').getroot()
    vals = []
    for tag in ('vehicle', 'trip', 'flow'):
        for v in root.findall(tag):
            d = v.get('depart') or v.get('begin')
            try: vals.append(float(d))
            except: pass
    if not vals:
        print(${MIN_SUMO_SECONDS}); sys.exit()
    # 直接计算总的 SUMO 秒数
    seconds = int(max(vals) + ${BUFFER_S})
    seconds = max(seconds, ${MIN_SUMO_SECONDS})
    seconds = min(seconds, ${MAX_SUMO_SECONDS_CAP})
    print(seconds)
except Exception as e:
    print(3600)
" 2>/dev/null
}

cleanup_sumo() { sleep 5; }

# 基线评测（无 GPU）
run_baseline() {
    local SCENARIO="$1"
    local ROUTE_FILE="$2"
    local SCENE_TYPE="$3"
    local METHOD_FLAG="$4"
    local METHOD_NAME="$5"
    local MAX_SUMO_SECONDS="$6"

    cleanup_sumo
    echo "  [${METHOD_NAME}] ${SCENARIO}/${SCENE_TYPE}  route=${ROUTE_FILE}  sumo_seconds=${MAX_SUMO_SECONDS}"
    python src/evaluation/run_eval.py \
        --scenario   "$SCENARIO" \
        --log_dir    "$LOG_DIR" \
        --route_file "$ROUTE_FILE" \
        --scene_type "$SCENE_TYPE" \
        --max_sumo_seconds  "$MAX_SUMO_SECONDS" \
        "$METHOD_FLAG"
    [ $? -ne 0 ] && echo "  [WARNING] ${METHOD_NAME} failed: ${SCENARIO}/${SCENE_TYPE}"
}

# VLM 评测（需 vLLM 服务 + GPU 渲染）
run_vlm() {
    local SCENARIO="$1"
    local ROUTE_FILE="$2"
    local SCENE_TYPE="$3"
    local MAX_SUMO_SECONDS="$4"
    local EXTRA="$5"

    cleanup_sumo
    echo "  [VLM] ${SCENARIO}/${SCENE_TYPE}  route=${ROUTE_FILE}  sumo_seconds=${MAX_SUMO_SECONDS}"
    ./vgl_python.sh src/evaluation/run_eval.py \
        --scenario   "$SCENARIO" \
        --log_dir    "$LOG_DIR" \
        --route_file "$ROUTE_FILE" \
        --scene_type "$SCENE_TYPE" \
        --max_sumo_seconds  "$MAX_SUMO_SECONDS" \
        $EXTRA
    [ $? -ne 0 ] && echo "  [WARNING] VLM failed: ${SCENARIO}/${SCENE_TYPE}"
}

# ─── 构建 VLM 额外参数 ─────────────────────────────────────────
EXTRA_VLM_ARGS=""
if [ -n "$API_PORT" ] && [ -n "$MODEL_NAME" ]; then
    EXTRA_VLM_ARGS="--api_url http://localhost:${API_PORT}/v1/chat/completions --model_name ${MODEL_NAME}"
fi
if [ -n "$TEMPERATURE" ]; then
    EXTRA_VLM_ARGS="${EXTRA_VLM_ARGS} --temperature ${TEMPERATURE}"
fi
if [ -n "$MAX_NEW_TOKENS" ]; then
    EXTRA_VLM_ARGS="${EXTRA_VLM_ARGS} --max_new_tokens ${MAX_NEW_TOKENS}"
fi

# ─── 打印头部信息 ───────────────────────────────────────────────
echo "============================================================"
echo "  E2ELight 交通事件场景批量评测"
echo "  LOG_DIR: $LOG_DIR"
if [ "$BASELINE_ONLY" = true ]; then
    echo "  模式   : 仅基线 (MaxPressure + FixedTime)"
elif [ -n "$MODEL_NAME" ]; then
    echo "  模式   : VLM [${MODEL_NAME}]$([ "$WITH_BASELINE" = true ] && echo " + 基线")"
fi
[ -n "$TEMPERATURE" ] && echo "  temperature: ${TEMPERATURE}"
[ -n "$MAX_NEW_TOKENS" ] && echo "  max_new_tokens: ${MAX_NEW_TOKENS}"
[ -n "$EVENT_FILTER" ] && echo "  场景过滤: ${EVENT_FILTER}" || echo "  场景类型: 全部 6 类（normal + 5 类事件）"
echo "============================================================"

# ─── 核心评测函数 ───────────────────────────────────────────────

run_all_baselines() {
    echo ""
    echo "=== [基线] MaxPressure + FixedTime ==="

    for EVENT in "${EVENT_TYPES[@]}"; do
        # 如果指定了 --event 过滤，跳过不匹配的场景类型
        [ -n "$EVENT_FILTER" ] && [ "$EVENT" != "$EVENT_FILTER" ] && continue

        echo ""
        echo "── Scene: ${EVENT} ─────────────────────────────────"
        for DS in "${DATASETS[@]}"; do
            mapfile -t ROUTE_FILES < <(get_route_files_for_scene "$EVENT" "$DS")
            if [ ${#ROUTE_FILES[@]} -eq 0 ]; then
                echo "  [SKIP] 未配置路由: ${DS}/${EVENT}"
                continue
            fi

            ENV_DIR="${ENV_DIRS[$DS]}"
            for ROUTE_FILE in "${ROUTE_FILES[@]}"; do
                FULL_ROUTE="${ENV_DIR}/${ROUTE_FILE}"
                if [ ! -f "$FULL_ROUTE" ]; then
                    echo "  [SKIP] 路由文件不存在: ${FULL_ROUTE}"
                    continue
                fi
                MAX_SUMO_SECONDS=$(get_max_sumo_seconds "$FULL_ROUTE")
                run_baseline "$DS" "$ROUTE_FILE" "$EVENT" "--max_pressure" "MaxPressure" "$MAX_SUMO_SECONDS"
                run_baseline "$DS" "$ROUTE_FILE" "$EVENT" "--fixed_time"   "FixedTime"   "$MAX_SUMO_SECONDS"
            done
        done
    done
}

run_all_vlm() {
    echo ""
    echo "=== [VLM] ${MODEL_NAME} ==="

    for EVENT in "${EVENT_TYPES[@]}"; do
        [ -n "$EVENT_FILTER" ] && [ "$EVENT" != "$EVENT_FILTER" ] && continue

        echo ""
        echo "── Scene: ${EVENT} ─────────────────────────────────"
        for DS in "${DATASETS[@]}"; do
            mapfile -t ROUTE_FILES < <(get_route_files_for_scene "$EVENT" "$DS")
            if [ ${#ROUTE_FILES[@]} -eq 0 ]; then
                echo "  [SKIP] 未配置路由: ${DS}/${EVENT}"
                continue
            fi

            ENV_DIR="${ENV_DIRS[$DS]}"
            for ROUTE_FILE in "${ROUTE_FILES[@]}"; do
                FULL_ROUTE="${ENV_DIR}/${ROUTE_FILE}"
                if [ ! -f "$FULL_ROUTE" ]; then
                    echo "  [SKIP] 路由文件不存在: ${FULL_ROUTE}"
                    continue
                fi
                MAX_SUMO_SECONDS=$(get_max_sumo_seconds "$FULL_ROUTE")
                run_vlm "$DS" "$ROUTE_FILE" "$EVENT" "$MAX_SUMO_SECONDS" "$EXTRA_VLM_ARGS"
            done
        done
    done
}

# ─── 执行 ──────────────────────────────────────────────────────
if [ "$BASELINE_ONLY" = true ]; then
    run_all_baselines
elif [ -n "$MODEL_NAME" ]; then
    [ "$WITH_BASELINE" = true ] && run_all_baselines
    run_all_vlm
else
    echo "❌ 请指定运行模式："
    echo "   --baseline-only                                          # 仅基线"
    echo "   --port 8000 --model_name <name>                          # 仅 VLM"
    echo "   --port 8000 --model_name <name> --with-baseline          # 两者都跑"
    echo "   --port 8000 --model_name <name> --temperature 0 --max_new_tokens 2048"
    echo "   --baseline-only --event emergency                        # 仅跑 emergency 基线"
    exit 1
fi

echo ""
echo "============================================================"
echo "  事件场景评测完成。"
echo "  结果保存在 data/eval/{dataset}/{route_file_name}/{method}/"
echo "  收集指标：python src/evaluation/collect_metrics.py --type all"
echo "============================================================"
