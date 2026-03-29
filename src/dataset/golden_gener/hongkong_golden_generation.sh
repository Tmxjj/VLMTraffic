#!/bin/bash

# Set PYTHONPATH to include the project root if needed
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# Run the generation script
nohup python src/dataset/golden_generation.py \
    --scenario "Hongkong_YMT" \
    --max_steps 20 \
    --log_dir "./log/golden_dataset" > hongkong_golden_run.log 2>&1 &

echo "Golden generation started in background. Log: hongkong_golden_run.log"