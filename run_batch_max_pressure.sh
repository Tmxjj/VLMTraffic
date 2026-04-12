#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-04-11
 # @Description: 批量运行 MaxPressure 基线评测，自动将结果写入 results/comparsion_result.csv
 # @FilePath: /VLMTraffic/run_batch_max_pressure.sh
###

# --- 配置 ---
LOG_DIR="./log/eval_results"
MAX_STEPS=120  # 1h=120, 24h=2880

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

PYTHON_CMD="python"

# --- 场景与路由配置（与 run_batch_eval.sh 保持一致）---
SCENARIO_JINAN="JiNan"
JINAN_ROUTES=(
    "anon_3_4_jinan_real_2000.rou.xml"
    "anon_3_4_jinan_real.rou.xml"
    "anon_3_4_jinan_real_2500.rou.xml"
    "anon_3_4_jinan_synthetic_24000_60min.rou.xml"
)

SCENARIO_HANGZHOU="Hangzhou"
HANGZHOU_ROUTES=(
    "anon_4_4_hangzhou_real.rou.xml"
    "anon_4_4_hangzhou_real_5816.rou.xml"
    "anon_4_4_hangzhou_synthetic_24000_60min.rou.xml"
)

# --- 清理僵尸进程 ---
cleanup_sumo() {
    echo "Cleaning up potential zombie processes..."
    sleep 10
}

# --- 执行 ---
echo "=========================================="
echo " MaxPressure Baseline Batch Evaluation"
echo " LOG_DIR  : $LOG_DIR"
echo " MAX_STEPS: $MAX_STEPS"
echo " 结果将自动写入: results/comparsion_result.csv"
echo "=========================================="

mkdir -p "$LOG_DIR"

# 运行 JiNan 场景
echo "--- Running Scenario: $SCENARIO_JINAN ---"
for route in "${JINAN_ROUTES[@]}"; do
    cleanup_sumo
    echo "[MaxPressure] Running: $SCENARIO_JINAN / $route"

    (
        $PYTHON_CMD vlm_decision.py \
            --scenario "$SCENARIO_JINAN" \
            --log_dir  "$LOG_DIR" \
            --route_file "$route" \
            --max_steps "$MAX_STEPS" \
            --max_pressure
    )

    RET_CODE=$?
    if [ $RET_CODE -ne 0 ]; then
        echo "[WARNING] $SCENARIO_JINAN / $route failed (Code: $RET_CODE)."
    else
        echo "[SUCCESS] Finished: $route"
    fi
done

# 运行 Hangzhou 场景
echo "--- Running Scenario: $SCENARIO_HANGZHOU ---"
for route in "${HANGZHOU_ROUTES[@]}"; do
    cleanup_sumo
    echo "[MaxPressure] Running: $SCENARIO_HANGZHOU / $route"

    (
        $PYTHON_CMD vlm_decision.py \
            --scenario "$SCENARIO_HANGZHOU" \
            --log_dir  "$LOG_DIR" \
            --route_file "$route" \
            --max_steps "$MAX_STEPS" \
            --max_pressure
    )

    RET_CODE=$?
    if [ $RET_CODE -ne 0 ]; then
        echo "[WARNING] $SCENARIO_HANGZHOU / $route failed (Code: $RET_CODE)."
    else
        echo "[SUCCESS] Finished: $route"
    fi
done

echo ""
echo "MaxPressure Batch Evaluation Completed."
echo "请检查 results/comparsion_result.csv 中的 MaxPressure 行。"
