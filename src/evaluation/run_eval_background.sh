#!/bin/bash

# Set PYTHONPATH to include the project root if needed
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# Configuration
SCENARIO="JiNan"
ROUTE_FILE="anon_3_4_jinan_real.rou.xml"
MAX_STEPS=120
LOG_DIR="./log/eval_results"

# Log file name
LOG_FILE="eval_run_${SCENARIO}_$(date +%Y%m%d_%H%M%S).log"

echo "Using Scenario: $SCENARIO"
echo "Using Route: $ROUTE_FILE"
echo "Log file: $LOG_FILE"

# Run the evaluation script in background
nohup python vlm_decision.py \
    --scenario "$SCENARIO" \
    --log_dir "$LOG_DIR" \
    --route_file "$ROUTE_FILE" \
    --max_steps "$MAX_STEPS" > "$LOG_FILE" 2>&1 &

echo "Evaluation started in background. Check log: $LOG_FILE"