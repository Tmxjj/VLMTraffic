#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-01-29 16:56:42
 # @LastEditTime: 2026-03-31 15:45:44
 # @Description: this script is used to 
 # @FilePath: /VLMTraffic/src/dataset/golden_gener/hangzhou_golden_generation.sh
### 


# Set PYTHONPATH to include the project root if needed
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# Run the generation script in background
nohup python src/dataset/golden_gener/1_golden_generation.py \
    --scenario "Hangzhou" \
    --max_steps 80 \
    --route_file "anon_4_4_hangzhou_real.rou.xml" \
    --log_dir "./log/golden_dataset" > hangzhou_golden_run.log 2>&1 &

echo "Golden generation started in background. Log: hangzhou_golden_run.log"
