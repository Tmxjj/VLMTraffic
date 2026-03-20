#!/bin/bash

# This script runs batch evaluations for different scenarios and route files.

# --- Configuration ---
LOG_DIR="./log/eval_results"
MAX_STEPS=120 # Set a default number of decision steps, e.g., 120 for a 1-hour simulation with 10s steps

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
# NOTE: The config for "Hangzhou" needs to exist in configs/scenairo_config.py
# Assuming a "Hangzhou" config exists.
SCENARIO_HANGZHOU="Hangzhou"
HANGZHOU_ROUTES=(
    "anon_4_4_hangzhou_real.rou.xml"
    "anon_4_4_hangzhou_real_5816.rou.xml"
    "anon_4_4_hangzhou_synthetic_24000_60min.rou.xml"
)


# --- Execution ---

echo "Starting Batch Evaluation..."
mkdir -p $LOG_DIR

# Run for JiNan
echo "--- Running Scenario: $SCENARIO_JINAN ---"
for route in "${JINAN_ROUTES[@]}"; do
    echo "Running with route: $route"
    python vlm_decision.py \
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
# First, check if a config for Hangzhou exists. We can't do this in bash easily,
# so we rely on the python script to fail if it doesn't.
echo "--- Running Scenario: $SCENARIO_HANGZHOU ---"
for route in "${HANGZHOU_ROUTES[@]}"; do
    echo "Running with route: $route"
    python vlm_decision.py \
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
