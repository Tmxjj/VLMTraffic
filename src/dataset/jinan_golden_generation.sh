#!/bin/bash


# Set PYTHONPATH to include the project root if needed
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# Run the generation script in background
nohup python src/dataset/golden_generation.py \
    --scenario "JiNan" \
    --max_steps 20 \
    --log_dir "./log/golden_dataset" > jinan_golden_run.log 2>&1 &

echo "Golden generation started in background. Log: jinan_golden_run.log"
