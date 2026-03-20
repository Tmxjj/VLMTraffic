#!/bin/bash

# This script runs batch evaluations for different scenarios and route files.

# --- Configuration ---
LOG_DIR="./log/eval_results"
MAX_STEPS=120 # Set a default number of decision steps, e.g., 120 for a 1-hour simulation with 10s steps

# 注入 launch.json 中的环境变量
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

# 使用 launch.json 中指定的 VirtualGL 包装脚本替代原生 python
# 确保 vgl_python.sh 在当前运行目录下，并且有可执行权限 (chmod +x vgl_python.sh)
PYTHON_CMD="./vgl_python.sh" 


# --- Scenarios and Routes ---

# JiNan Scenario
SCENARIO_JINAN="JiNan"
JINAN_ROUTES=(
    "anon_3_4_jinan_real.rou.xml"
    "anon_3_4_jinan_real_2000.rou.xml"
    "anon_3_4_jinan_real_2500.rou.xml"
    "anon_3_4_jinan_synthetic_24000_60min.rou.xml"
    "anon_3_4_jinan_synthetic_24h_6000.rou.xml"
)

# Hangzhou Scenario
SCENARIO_HANGZHOU="Hangzhou"
HANGZHOU_ROUTES=(
    "anon_4_4_hangzhou_real.rou.xml"
    "anon_4_4_hangzhou_real_5816.rou.xml"
    "anon_4_4_hangzhou_synthetic_24000_60min.rou.xml"
)


# --- Execution ---

echo "Starting Batch Evaluation with VirtualGL (GPU Acceleration)..."
mkdir -p "$LOG_DIR"

# Run for JiNan
echo "--- Running Scenario: $SCENARIO_JINAN ---"
for route in "${JINAN_ROUTES[@]}"; do
    echo "Running with route: $route"
    # 👇 修改这里：使用 PYTHON_CMD 替代 python
    $PYTHON_CMD vlm_decision.py \
        --scenario "$SCENARIO_JINAN" \
        --log_dir "$LOG_DIR" \
        --route_file "$route" \
        --max_steps "$MAX_STEPS"
    
    # Check exit code
    if [ $? -ne 0 ]; then
        echo "Error running evaluation for $SCENARIO_JINAN with $route. Exiting."
        exit 1
    fi
done
echo "--- Finished Scenario: $SCENARIO_JINAN ---"


# Run for Hangzhou
echo "--- Running Scenario: $SCENARIO_HANGZHOU ---"
for route in "${HANGZHOU_ROUTES[@]}"; do
    echo "Running with route: $route"
    # 👇 修改这里：使用 PYTHON_CMD 替代 python
    $PYTHON_CMD vlm_decision.py \
        --scenario "$SCENARIO_HANGZHOU" \
        --log_dir "$LOG_DIR" \
        --route_file "$route" \
        --max_steps "$MAX_STEPS"

    if [ $? -ne 0 ]; then
        echo "Error running evaluation for $SCENARIO_HANGZHOU with $route. It's possible the scenario config is missing. Exiting."
        exit 1
    fi
done
echo "--- Finished Scenario: $SCENARIO_HANGZHOU ---"

echo "Batch Evaluation Completed."