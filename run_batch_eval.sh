#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-03-24 22:41:35
 # @LastEditTime: 2026-03-25 11:04:42
 # @Description: this script is used to 
 # @FilePath: /VLMTraffic/run_batch_eval.sh
### 

# --- Configuration ---
LOG_DIR="./log/eval_results"
MAX_STEPS=120 # 1h为120，24h为2880

# 默认情况下不使用覆盖
API_PORT=""
MODEL_NAME=""

# 解析可选参数（允许外部传入模型端口和名称，覆盖 config）
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) API_PORT="$2"; shift ;;
        --model_name) MODEL_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

PYTHON_CMD="./vgl_python.sh" 

# --- Scenarios and Routes ---
SCENARIO_JINAN="JiNan"
JINAN_ROUTES=(
    "anon_3_4_jinan_real_2000.rou.xml"
    "anon_3_4_jinan_real.rou.xml"
    "anon_3_4_jinan_real_2500.rou.xml"
    "anon_3_4_jinan_synthetic_24000_60min.rou.xml"
    # "anon_3_4_jinan_synthetic_24h_6000.rou.xml"
)

SCENARIO_HANGZHOU="Hangzhou"
HANGZHOU_ROUTES=(
    "anon_4_4_hangzhou_real.rou.xml"
    "anon_4_4_hangzhou_real_5816.rou.xml"
    "anon_4_4_hangzhou_synthetic_24000_60min.rou.xml"
)

# --- Helper Function: Cleanup ---
cleanup_sumo() {
    echo "Cleaning up potential zombie processes..."
    # pkill -9 -f "sumo" 2>/dev/null
    # pkill -9 -f "vlm_decision.py" 2>/dev/null
    sleep 10  # 稍微多等一会儿，让显存回收
}

# --- Execution ---
echo "Starting Batch Evaluation with VirtualGL (GPU Acceleration)..."
if [ -n "$API_PORT" ] && [ -n "$MODEL_NAME" ]; then
    echo "Using custom API at port $API_PORT with model $MODEL_NAME"
    EXTRA_ARGS="--api_url http://localhost:${API_PORT}/v1/chat/completions --model_name ${MODEL_NAME}"
else
    EXTRA_ARGS=""
fi

mkdir -p "$LOG_DIR"

# Run for JiNan
echo "--- Running Scenario: $SCENARIO_JINAN ---"
for route in "${JINAN_ROUTES[@]}"; do
    cleanup_sumo
    echo "Running with route: $route"
    
    # 核心修复：不要在反斜杠后面直接接注释
    # 如果需要 --fixed_time，请加在上面一行末尾并补上反斜杠
    (
        $PYTHON_CMD vlm_decision.py \
            --scenario "$SCENARIO_JINAN" \
            --log_dir "$LOG_DIR" \
            --route_file "$route" \
            --max_steps "$MAX_STEPS" \
            $EXTRA_ARGS
            
    )
    
    RET_CODE=$?
    if [ $RET_CODE -ne 0 ]; then
        echo "[WARNING] Evaluation for $SCENARIO_JINAN with $route failed (Code: $RET_CODE)."
    else
        echo "[SUCCESS] Finished $route"
    fi
done

# Run for Hangzhou
echo "--- Running Scenario: $SCENARIO_HANGZHOU ---"
for route in "${HANGZHOU_ROUTES[@]}"; do
    cleanup_sumo
    echo "Running with route: $route"
    
    (
        $PYTHON_CMD vlm_decision.py \
            --scenario "$SCENARIO_HANGZHOU" \
            --log_dir "$LOG_DIR" \
            --route_file "$route" \
            --max_steps "$MAX_STEPS" \
            $EXTRA_ARGS
    )

    RET_CODE=$?
    if [ $RET_CODE -ne 0 ]; then
        echo "[WARNING] Evaluation for $SCENARIO_HANGZHOU with $route failed (Code: $RET_CODE)."
    else
        echo "[SUCCESS] Finished $route"
    fi
done

echo "Batch Evaluation Completed."