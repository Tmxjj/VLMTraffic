#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-01-29 16:56:42
 # @LastEditTime: 2026-03-08 16:54:55
 # @Description: this script is used to 
 # @FilePath: /VLMTraffic/src/dataset/jinan_golden_generation_emergy.sh
### 


# Set PYTHONPATH to include the project root if needed
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# Run the generation script in background
nohup python src/dataset/1_golden_generation.py \
    --scenario "JiNan" \
    --max_steps 120 \
    --route_file "anon_3_4_jinan_real_emergency.rou.xml" \
    --log_dir "./log/golden_dataset" > jinan_golden_run_emergency.log 2>&1 &

echo "Golden generation started in background. Log: jinan_golden_run_emergency.log"