#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-04-13
 # @Description: JiNan / Hangzhou 场景批量评测脚本（基线 + VLM，合并版）
 #               原 run_batch_eval.sh（VLM）与 run_batch_max_pressure.sh（基线）合并。
 #
 #               场景列表：
 #                 JiNan    — real / real_2000 / real_2500 / synthetic_24000_60min
 #                 Hangzhou — real / real_5816 / synthetic_24000_60min
 #
 #               max_steps 自动从对应路由文件的最大 depart 时间动态计算：
 #                 max_steps = ceil((max_depart + 300s 缓冲) / 30s 每步)
 #                 JiNan / Hangzhou 各路由约 130 步
 #
 #               使用说明：
 #                 # 仅跑基线（本地，无需 GPU）：
 #                 bash run_batch_eval.sh --baseline-only
 #
 #                 # 仅跑 VLM（远程服务器，需提前启动 vLLM）：
 #                 bash run_batch_eval.sh --port 8000 --model_name qwen3-vl-8b-sft-dpo
 #
 #                 # 同时跑基线 + VLM：
 #                 bash run_batch_eval.sh --port 8000 --model_name qwen3-vl-8b-sft-dpo --with-baseline
 #
 #               评测完成后收集指标：
 #                 python src/evaluation/metrics.py
 #
 # @FilePath: /VLMTraffic/run_batch_eval.sh
###

# ─── 默认配置 ──────────────────────────────────────────────────
LOG_DIR="./log/eval_results"
# 每个决策步时长 = delta_time(27s) + yellow_time(3s) = 30s
STEP_DUR=30
# 最后一辆车出发后的额外仿真缓冲时间（让已入场车辆完成行程）
BUFFER_S=300
# 最小/最大步数保护
MIN_STEPS=10
MAX_STEPS_CAP=200

API_PORT=""
MODEL_NAME=""
BASELINE_ONLY=false
WITH_BASELINE=false

# 解析参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port)           API_PORT="$2";    shift ;;
        --model_name)     MODEL_NAME="$2";  shift ;;
        --baseline-only)  BASELINE_ONLY=true ;;
        --with-baseline)  WITH_BASELINE=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

mkdir -p "$LOG_DIR"

# ─── 路由配置 ──────────────────────────────────────────────────
JINAN_ROUTES=(
    "anon_3_4_jinan_real_2000.rou.xml"
    "anon_3_4_jinan_real.rou.xml"
    "anon_3_4_jinan_real_2500.rou.xml"
    "anon_3_4_jinan_synthetic_24000_60min.rou.xml"
    # "anon_3_4_jinan_synthetic_24h_6000.rou.xml"
)

HANGZHOU_ROUTES=(
    "anon_4_4_hangzhou_real.rou.xml"
    "anon_4_4_hangzhou_real_5816.rou.xml"
    "anon_4_4_hangzhou_synthetic_24000_60min.rou.xml"
)

# ─── 辅助函数 ──────────────────────────────────────────────────

# 从路由文件中计算合适的 max_steps
get_max_steps() {
    local rou_path="$1"
    python3 -c "
import xml.etree.ElementTree as ET, math, sys
try:
    root = ET.parse('${rou_path}').getroot()
    departs = [float(v.get('depart', 0)) for v in root.findall('vehicle')]
    if not departs:
        print(${MIN_STEPS}); sys.exit()
    steps = int(math.ceil((max(departs) + ${BUFFER_S}) / ${STEP_DUR}))
    steps = max(steps, ${MIN_STEPS})
    steps = min(steps, ${MAX_STEPS_CAP})
    print(steps)
except:
    print(120)   # fallback
" 2>/dev/null
}

cleanup_sumo() { sleep 10; }

# 基线（无 GPU，直接 python）
run_baseline() {
    local SCENARIO="$1"
    local ROUTE_FILE="$2"
    local METHOD_FLAG="$3"
    local METHOD_NAME="$4"
    local MAX_STEPS="$5"

    cleanup_sumo
    echo "[${METHOD_NAME}] ${SCENARIO} / ${ROUTE_FILE}  (max_steps=${MAX_STEPS})"
    python vlm_decision.py \
        --scenario   "$SCENARIO" \
        --log_dir    "$LOG_DIR" \
        --route_file "$ROUTE_FILE" \
        --max_steps  "$MAX_STEPS" \
        "$METHOD_FLAG"
    [ $? -ne 0 ] && echo "[WARNING] ${METHOD_NAME} failed: ${SCENARIO}/${ROUTE_FILE}"
}

# VLM（需要 vLLM 服务 + GPU 渲染）
run_vlm() {
    local SCENARIO="$1"
    local ROUTE_FILE="$2"
    local MAX_STEPS="$3"
    local EXTRA="$4"

    cleanup_sumo
    echo "[VLM] ${SCENARIO} / ${ROUTE_FILE}  (max_steps=${MAX_STEPS})"
    ./vgl_python.sh vlm_decision.py \
        --scenario   "$SCENARIO" \
        --log_dir    "$LOG_DIR" \
        --route_file "$ROUTE_FILE" \
        --max_steps  "$MAX_STEPS" \
        $EXTRA
    [ $? -ne 0 ] && echo "[WARNING] VLM failed: ${SCENARIO}/${ROUTE_FILE}"
}

# ─── 构建 VLM 额外参数 ─────────────────────────────────────────
EXTRA_VLM_ARGS=""
if [ -n "$API_PORT" ] && [ -n "$MODEL_NAME" ]; then
    EXTRA_VLM_ARGS="--api_url http://localhost:${API_PORT}/v1/chat/completions --model_name ${MODEL_NAME}"
fi

# ─── 打印头部信息 ───────────────────────────────────────────────
echo "============================================================"
echo "  JiNan / Hangzhou 批量评测（合并版）"
echo "  LOG_DIR: $LOG_DIR"
if [ "$BASELINE_ONLY" = true ]; then
    echo "  模式   : 仅基线 (MaxPressure + FixedTime)"
elif [ -n "$MODEL_NAME" ]; then
    echo "  模式   : VLM [${MODEL_NAME}]$([ "$WITH_BASELINE" = true ] && echo " + 基线")"
fi
echo "============================================================"

# ─── 基线评测函数 ───────────────────────────────────────────────
run_all_baselines() {
    echo ""
    echo "=== [基线] MaxPressure + FixedTime ==="

    echo "--- JiNan ---"
    for route in "${JINAN_ROUTES[@]}"; do
        steps=$(get_max_steps "data/raw/JiNan/env/${route}")
        run_baseline "JiNan" "$route" "--max_pressure" "MaxPressure" "$steps"
        run_baseline "JiNan" "$route" "--fixed_time"   "FixedTime"   "$steps"
    done

    echo "--- Hangzhou ---"
    for route in "${HANGZHOU_ROUTES[@]}"; do
        steps=$(get_max_steps "data/raw/Hangzhou/env/${route}")
        run_baseline "Hangzhou" "$route" "--max_pressure" "MaxPressure" "$steps"
        run_baseline "Hangzhou" "$route" "--fixed_time"   "FixedTime"   "$steps"
    done
}

# ─── VLM 评测函数 ───────────────────────────────────────────────
run_all_vlm() {
    echo ""
    echo "=== [VLM] ${MODEL_NAME} ==="

    echo "--- JiNan ---"
    for route in "${JINAN_ROUTES[@]}"; do
        steps=$(get_max_steps "data/raw/JiNan/env/${route}")
        run_vlm "JiNan" "$route" "$steps" "$EXTRA_VLM_ARGS"
    done

    echo "--- Hangzhou ---"
    for route in "${HANGZHOU_ROUTES[@]}"; do
        steps=$(get_max_steps "data/raw/Hangzhou/env/${route}")
        run_vlm "Hangzhou" "$route" "$steps" "$EXTRA_VLM_ARGS"
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
    echo "   --baseline-only                          # 仅基线"
    echo "   --port 8000 --model_name <name>          # 仅 VLM"
    echo "   --port 8000 --model_name <name> --with-baseline  # 两者都跑"
    exit 1
fi

echo ""
echo "评测完成。结果保存在 data/eval/{JiNan,Hangzhou}/{route}/{method}/"
echo "收集指标：python src/evaluation/metrics.py"
