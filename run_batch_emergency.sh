#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-04-13
 # @Description: 紧急车辆场景批量评测脚本（基线 + VLM）
 #
 #               场景列表（_emergy 路由文件）：
 #                 JiNan             - anon_3_4_jinan_real_emergy.rou.xml
 #                 Hangzhou          - anon_4_4_hangzhou_real_emergy.rou.xml
 #                 SouthKorea_Songdo - songdo_emergy.rou.xml
 #                 France_Massy      - massy_emergy.rou.xml
 #                 Hongkong_YMT      - YMT_emergy.rou.xml
 #                 NewYork           - anon_28_7_newyork_real_double_emergy.rou.xml
 #
 #               max_steps 自动从对应路由文件的最大 depart 时间动态计算：
 #                 max_steps = ceil((max_depart + 600s 缓冲) / 30s 每步)
 #                 Songdo / Massy / YMT ≈ 34 步；JiNan / Hangzhou / NewYork ≈ 140-149 步
 #
 #               使用说明：
 #                 # 仅跑基线（本地，无需 GPU）：
 #                 bash run_batch_emergency.sh --baseline-only
 #
 #                 # 仅跑 VLM（远程服务器，需提前启动 vLLM）：
 #                 bash run_batch_emergency.sh --port 8000 --model_name qwen3-vl-8b-sft-dpo
 #
 #                 # 同时跑基线 + VLM：
 #                 bash run_batch_emergency.sh --port 8000 --model_name qwen3-vl-8b-sft-dpo --with-baseline
 #
 #               评测完成后收集指标：
 #                 python src/evaluation/emergency_metrics.py
 #
 # @FilePath: /VLMTraffic/run_batch_emergency.sh
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
except Exception as e:
    print(120)   # fallback
" 2>/dev/null
}

cleanup_sumo() { sleep 5; }

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

# ─── 预计算各场景 max_steps ────────────────────────────────────
STEPS_JINAN=$(get_max_steps    "data/raw/JiNan/env/anon_3_4_jinan_real_emergy.rou.xml")
STEPS_HANGZHOU=$(get_max_steps "data/raw/Hangzhou/env/anon_4_4_hangzhou_real_emergy.rou.xml")
STEPS_SONGDO=$(get_max_steps   "data/raw/SouthKorea_Songdo/env/songdo_emergy.rou.xml")
STEPS_MASSY=$(get_max_steps    "data/raw/France_Massy/env/massy_emergy.rou.xml")
STEPS_YMT=$(get_max_steps      "data/raw/Hongkong_YMT/env/YMT_emergy.rou.xml")
STEPS_NY=$(get_max_steps       "data/raw/NewYork/env/anon_28_7_newyork_real_double_emergy.rou.xml")

echo "============================================================"
echo "  紧急车辆场景批量评测"
echo "  LOG_DIR   : $LOG_DIR"
if [ "$BASELINE_ONLY" = true ]; then
    echo "  模式      : 仅基线 (MaxPressure + FixedTime)"
elif [ -n "$MODEL_NAME" ]; then
    echo "  模式      : VLM [${MODEL_NAME}]$([ "$WITH_BASELINE" = true ] && echo " + 基线")"
fi
echo "  max_steps : JiNan=${STEPS_JINAN} | Hangzhou=${STEPS_HANGZHOU} | Songdo=${STEPS_SONGDO}"
echo "              Massy=${STEPS_MASSY} | YMT=${STEPS_YMT} | NewYork=${STEPS_NY}"
echo "============================================================"

# ─── 基线评测函数 ───────────────────────────────────────────────
run_all_baselines() {
    echo ""
    echo "=== [基线] MaxPressure + FixedTime ==="

    echo "--- [1/6] JiNan ---"
    run_baseline "JiNan" "anon_3_4_jinan_real_emergy.rou.xml" "--max_pressure" "MaxPressure" "$STEPS_JINAN"
    run_baseline "JiNan" "anon_3_4_jinan_real_emergy.rou.xml" "--fixed_time"   "FixedTime"   "$STEPS_JINAN"

    echo "--- [2/6] Hangzhou ---"
    run_baseline "Hangzhou" "anon_4_4_hangzhou_real_emergy.rou.xml" "--max_pressure" "MaxPressure" "$STEPS_HANGZHOU"
    run_baseline "Hangzhou" "anon_4_4_hangzhou_real_emergy.rou.xml" "--fixed_time"   "FixedTime"   "$STEPS_HANGZHOU"

    echo "--- [3/6] SouthKorea_Songdo ---"
    run_baseline "SouthKorea_Songdo" "songdo_emergy.rou.xml" "--max_pressure" "MaxPressure" "$STEPS_SONGDO"
    run_baseline "SouthKorea_Songdo" "songdo_emergy.rou.xml" "--fixed_time"   "FixedTime"   "$STEPS_SONGDO"

    echo "--- [4/6] France_Massy ---"
    run_baseline "France_Massy" "massy_emergy.rou.xml" "--max_pressure" "MaxPressure" "$STEPS_MASSY"
    run_baseline "France_Massy" "massy_emergy.rou.xml" "--fixed_time"   "FixedTime"   "$STEPS_MASSY"

    echo "--- [5/6] Hongkong_YMT ---"
    run_baseline "Hongkong_YMT" "YMT_emergy.rou.xml" "--max_pressure" "MaxPressure" "$STEPS_YMT"
    run_baseline "Hongkong_YMT" "YMT_emergy.rou.xml" "--fixed_time"   "FixedTime"   "$STEPS_YMT"

    echo "--- [6/6] NewYork ---"
    run_baseline "NewYork" "anon_28_7_newyork_real_double_emergy.rou.xml" "--max_pressure" "MaxPressure" "$STEPS_NY"
    run_baseline "NewYork" "anon_28_7_newyork_real_double_emergy.rou.xml" "--fixed_time"   "FixedTime"   "$STEPS_NY"
}

# ─── VLM 评测函数 ───────────────────────────────────────────────
run_all_vlm() {
    echo ""
    echo "=== [VLM] ${MODEL_NAME} ==="

    echo "--- [1/6] JiNan ---"
    run_vlm "JiNan" "anon_3_4_jinan_real_emergy.rou.xml" "$STEPS_JINAN" "$EXTRA_VLM_ARGS"

    echo "--- [2/6] Hangzhou ---"
    run_vlm "Hangzhou" "anon_4_4_hangzhou_real_emergy.rou.xml" "$STEPS_HANGZHOU" "$EXTRA_VLM_ARGS"

    echo "--- [3/6] SouthKorea_Songdo ---"
    run_vlm "SouthKorea_Songdo" "songdo_emergy.rou.xml" "$STEPS_SONGDO" "$EXTRA_VLM_ARGS"

    echo "--- [4/6] France_Massy ---"
    run_vlm "France_Massy" "massy_emergy.rou.xml" "$STEPS_MASSY" "$EXTRA_VLM_ARGS"

    echo "--- [5/6] Hongkong_YMT ---"
    run_vlm "Hongkong_YMT" "YMT_emergy.rou.xml" "$STEPS_YMT" "$EXTRA_VLM_ARGS"

    echo "--- [6/6] NewYork ---"
    run_vlm "NewYork" "anon_28_7_newyork_real_double_emergy.rou.xml" "$STEPS_NY" "$EXTRA_VLM_ARGS"
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
echo "紧急场景评测完成。"
echo "结果保存在 data/eval/{Scenario}/{route}_emergy/{method}/"
echo "收集指标：python src/evaluation/emergency_metrics.py"
