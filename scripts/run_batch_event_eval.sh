#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-05-02
 # @Description: 常规 + 事件迁移场景批量评测脚本（基线 + VLM）
 #
 #               路由文件类型：
 #                 normal          — 常规车流（*.rou.xml）
 #                 emergy_bus      — 紧急车辆 + 公交/校车（*_emergy_bus.rou.xml）
 #                 accident_debris — 交通事故 + 路面碎片（*_accident_debris.rou.xml）
 #
 #               评测数据集：
 #                 JiNan × 3 路由 + Hangzhou × 2 路由
 #                 + SouthKorea_Songdo / France_Massy / Hongkong_YMT × 1 路由各一
 #                 共 8 个数据集 × 3 类路由 = 24 条评测任务
 #
 #               输出路径（与 src/evaluation/run_eval.py 一致）：
 #                 data/eval/{dataset}/{route_stem}/{method}/
 #               例：data/eval/JiNan/anon_3_4_jinan_real_emergy_bus/fixed_time/
 #
 #               max_sumo_seconds 自动从路由文件最大 depart 时间动态计算：
 #                 max_sumo_seconds = max_depart + BUFFER_S，上下限 [MIN, CAP]
 #
 #               使用说明：
 #                 # 仅跑基线（本地，无需 GPU）
 #                 bash scripts/run_batch_event_eval.sh --baseline-only
 #
 #                 # 仅跑某类场景基线
 #                 bash scripts/run_batch_event_eval.sh --baseline-only --event normal
 #                 bash scripts/run_batch_event_eval.sh --baseline-only --event emergy_bus
 #                 bash scripts/run_batch_event_eval.sh --baseline-only --event accident_debris
 #
 #                 # 仅跑 VLM（远程服务器，需提前启动 vLLM）
 #                 bash scripts/run_batch_event_eval.sh --port 8000 --model_name qwen3-vl-8b-sft-rlvr
 #
 #                 # 同时跑基线 + VLM
 #                 bash scripts/run_batch_event_eval.sh --port 8000 --model_name qwen3-vl-8b-sft-rlvr --with-baseline
 #
 #               评测完成后收集指标：
 #                 python src/evaluation/collect_metrics.py --type main
 #                 python src/evaluation/collect_metrics.py --type gen_topology
 #                 python src/evaluation/collect_metrics.py --type gen_event_emergy_bus
 #                 python src/evaluation/collect_metrics.py --type gen_event_accident_debris
 #
 # @FilePath: /VLMTraffic/scripts/run_batch_event_eval.sh
###

# ─── 默认配置 ──────────────────────────────────────────────────────────────────
BUFFER_S=180
MIN_SUMO_SECONDS=300
MAX_SUMO_SECONDS_CAP=3600
LOG_DIR="./log/eval_results"

API_PORT=""
MODEL_NAME=""
TEMPERATURE=""
MAX_NEW_TOKENS=""
BASELINE_ONLY=false
WITH_BASELINE=false
EVENT_FILTER=""   # 可选：normal / emergy_bus / accident_debris（不指定则三类都跑）

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port)           API_PORT="$2";       shift ;;
        --model_name)     MODEL_NAME="$2";     shift ;;
        --temperature)    TEMPERATURE="$2";    shift ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift ;;
        --baseline-only)  BASELINE_ONLY=true   ;;
        --with-baseline)  WITH_BASELINE=true   ;;
        --event)          EVENT_FILTER="$2";   shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"
mkdir -p "$LOG_DIR"

# ─── 路由文件条目 ──────────────────────────────────────────────────────────────
# 格式：DATASET|ENV_DIR|BASE_STEM
# BASE_STEM 是不含 .rou.xml 的基础文件名；
# normal 使用 ${BASE_STEM}.rou.xml；
# 事件场景会自动在其后追加 _emergy_bus.rou.xml 或 _accident_debris.rou.xml。
declare -a ENTRIES

# JiNan × 3
ENTRIES+=("JiNan|data/raw/JiNan/env|anon_3_4_jinan_real")
ENTRIES+=("JiNan|data/raw/JiNan/env|anon_3_4_jinan_real_2000")
ENTRIES+=("JiNan|data/raw/JiNan/env|anon_3_4_jinan_real_2500")

# Hangzhou × 2
ENTRIES+=("Hangzhou|data/raw/Hangzhou/env|anon_4_4_hangzhou_real")
ENTRIES+=("Hangzhou|data/raw/Hangzhou/env|anon_4_4_hangzhou_real_5816")

# SouthKorea_Songdo × 1
ENTRIES+=("SouthKorea_Songdo|data/raw/SouthKorea_Songdo/env|songdo")

# France_Massy × 1
ENTRIES+=("France_Massy|data/raw/France_Massy/env|massy")

# Hongkong_YMT × 1
ENTRIES+=("Hongkong_YMT|data/raw/Hongkong_YMT/env|YMT")

# 场景类型列表
SCENE_TYPES=("normal" "emergy_bus" "accident_debris")

# ─── 工具函数 ──────────────────────────────────────────────────────────────────

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
    seconds = int(max(vals) + ${BUFFER_S})
    seconds = max(seconds, ${MIN_SUMO_SECONDS})
    seconds = min(seconds, ${MAX_SUMO_SECONDS_CAP})
    print(seconds)
except Exception:
    print(3600)
" 2>/dev/null
}

cleanup_sumo() { sleep 5; }

get_route_file() {
    local BASE_STEM="$1"
    local SCENE_TYPE="$2"
    if [ "$SCENE_TYPE" = "normal" ]; then
        echo "${BASE_STEM}.rou.xml"
    else
        echo "${BASE_STEM}_${SCENE_TYPE}.rou.xml"
    fi
}

run_baseline() {
    local DATASET="$1"
    local ROUTE_FILE="$2"    # 仅文件名（不含路径）
    local SCENE_TYPE="$3"
    local METHOD_FLAG="$4"
    local METHOD_NAME="$5"
    local MAX_SUMO_S="$6"

    cleanup_sumo
    echo "  [${METHOD_NAME}] ${DATASET}/${ROUTE_FILE}  sumo_s=${MAX_SUMO_S}"
    python src/evaluation/run_eval.py \
        --scenario         "$DATASET" \
        --log_dir          "$LOG_DIR" \
        --route_file       "$ROUTE_FILE" \
        --scene_type       "$SCENE_TYPE" \
        --max_sumo_seconds "$MAX_SUMO_S" \
        "$METHOD_FLAG"
    [ $? -ne 0 ] && echo "  [WARNING] ${METHOD_NAME} failed: ${DATASET}/${ROUTE_FILE}"
}

run_vlm() {
    local DATASET="$1"
    local ROUTE_FILE="$2"
    local SCENE_TYPE="$3"
    local MAX_SUMO_S="$4"
    local EXTRA="$5"

    cleanup_sumo
    echo "  [VLM] ${DATASET}/${ROUTE_FILE}  sumo_s=${MAX_SUMO_S}"
    ./vgl_python.sh src/evaluation/run_eval.py \
        --scenario         "$DATASET" \
        --log_dir          "$LOG_DIR" \
        --route_file       "$ROUTE_FILE" \
        --scene_type       "$SCENE_TYPE" \
        --max_sumo_seconds "$MAX_SUMO_S" \
        $EXTRA
    [ $? -ne 0 ] && echo "  [WARNING] VLM failed: ${DATASET}/${ROUTE_FILE}"
}

# ─── 构建 VLM 额外参数 ─────────────────────────────────────────────────────────
EXTRA_VLM_ARGS=""
if [ -n "$API_PORT" ] && [ -n "$MODEL_NAME" ]; then
    EXTRA_VLM_ARGS="--api_url http://localhost:${API_PORT}/v1/chat/completions --model_name ${MODEL_NAME}"
fi
[ -n "$TEMPERATURE"    ] && EXTRA_VLM_ARGS="${EXTRA_VLM_ARGS} --temperature ${TEMPERATURE}"
[ -n "$MAX_NEW_TOKENS" ] && EXTRA_VLM_ARGS="${EXTRA_VLM_ARGS} --max_new_tokens ${MAX_NEW_TOKENS}"

# ─── 打印头部信息 ──────────────────────────────────────────────────────────────
echo "================================================================"
echo "  E2ELight 常规 + 事件迁移场景批量评测"
echo "  LOG_DIR: $LOG_DIR"
if [ "$BASELINE_ONLY" = true ]; then
    echo "  模式   : 仅基线 (FixedTime + MaxPressure)"
elif [ -n "$MODEL_NAME" ]; then
    echo "  模式   : VLM [${MODEL_NAME}]$([ "$WITH_BASELINE" = true ] && echo " + 基线")"
fi
[ -n "$TEMPERATURE"    ] && echo "  temperature:    ${TEMPERATURE}"
[ -n "$MAX_NEW_TOKENS" ] && echo "  max_new_tokens: ${MAX_NEW_TOKENS}"
[ -n "$EVENT_FILTER"   ] && echo "  场景过滤: ${EVENT_FILTER}" || echo "  场景类型: normal + emergy_bus + accident_debris"
echo "================================================================"

# ─── 核心评测循环 ──────────────────────────────────────────────────────────────

run_all_baselines() {
    echo ""
    echo "=== [基线] FixedTime + MaxPressure ==="

    for EVENT in "${SCENE_TYPES[@]}"; do
        [ -n "$EVENT_FILTER" ] && [ "$EVENT" != "$EVENT_FILTER" ] && continue

        echo ""
        echo "── Scene: ${EVENT} ──────────────────────────────────────"
        for entry in "${ENTRIES[@]}"; do
            IFS='|' read -r DATASET ENV_DIR BASE_STEM <<< "$entry"
            ROUTE_FILE="$(get_route_file "$BASE_STEM" "$EVENT")"
            FULL_PATH="${ENV_DIR}/${ROUTE_FILE}"

            if [ ! -f "$FULL_PATH" ]; then
                echo "  [SKIP] 路由文件不存在: ${FULL_PATH}"
                continue
            fi

            MAX_S=$(get_max_sumo_seconds "$FULL_PATH")
            run_baseline "$DATASET" "$ROUTE_FILE" "$EVENT" "--fixed_time"   "FixedTime"   "$MAX_S"
            run_baseline "$DATASET" "$ROUTE_FILE" "$EVENT" "--max_pressure" "MaxPressure" "$MAX_S"
        done
    done
}

run_all_vlm() {
    echo ""
    echo "=== [VLM] ${MODEL_NAME} ==="

    for EVENT in "${SCENE_TYPES[@]}"; do
        [ -n "$EVENT_FILTER" ] && [ "$EVENT" != "$EVENT_FILTER" ] && continue

        echo ""
        echo "── Scene: ${EVENT} ──────────────────────────────────────"
        for entry in "${ENTRIES[@]}"; do
            IFS='|' read -r DATASET ENV_DIR BASE_STEM <<< "$entry"
            ROUTE_FILE="$(get_route_file "$BASE_STEM" "$EVENT")"
            FULL_PATH="${ENV_DIR}/${ROUTE_FILE}"

            if [ ! -f "$FULL_PATH" ]; then
                echo "  [SKIP] 路由文件不存在: ${FULL_PATH}"
                continue
            fi

            MAX_S=$(get_max_sumo_seconds "$FULL_PATH")
            run_vlm "$DATASET" "$ROUTE_FILE" "$EVENT" "$MAX_S" "$EXTRA_VLM_ARGS"
        done
    done
}

# ─── 执行 ──────────────────────────────────────────────────────────────────────
if [ "$BASELINE_ONLY" = true ]; then
    run_all_baselines
elif [ -n "$MODEL_NAME" ]; then
    [ "$WITH_BASELINE" = true ] && run_all_baselines
    run_all_vlm
else
    echo "请指定运行模式："
    echo "  --baseline-only                                        # 仅基线"
    echo "  --port 8000 --model_name <name>                        # 仅 VLM"
    echo "  --port 8000 --model_name <name> --with-baseline        # 两者都跑"
    echo "  --baseline-only --event normal                         # 仅跑常规场景基线"
    echo "  --baseline-only --event emergy_bus                     # 仅跑 emergy_bus 基线"
    echo "  --baseline-only --event accident_debris                # 仅跑 accident_debris 基线"
    exit 1
fi

echo ""
echo "================================================================"
echo "  常规 + 事件迁移评测完成。"
echo "  结果目录: data/eval/{dataset}/{route_stem}/{method}/"
echo "  收集指标:"
echo "    python src/evaluation/collect_metrics.py --type main"
echo "    python src/evaluation/collect_metrics.py --type gen_topology"
echo "    python src/evaluation/collect_metrics.py --type gen_event_emergy_bus"
echo "    python src/evaluation/collect_metrics.py --type gen_event_accident_debris"
echo "================================================================"
