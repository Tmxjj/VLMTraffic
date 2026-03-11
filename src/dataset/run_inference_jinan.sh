#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-03-10 23:24:40
 # @LastEditTime: 2026-03-10 23:26:19
 # @Description: this script is used to 
 # @FilePath: /VLMTraffic/src/dataset/run_inference_jinan.sh
### 

# This script runs the 3_inference.py script in the background
# to process the auto-annotated dataset.
#
# Usage:
# ./scripts/run_inference.sh

# The output will be logged to inference_step3.log

# Activate your python environment if needed, e.g.:
# source /home/jyf/anaconda3/etc/profile.d/conda.sh
# conda activate VLMTraffic

echo "Starting inference process in the background..."
echo "Output will be saved to inference_step3.log"

nohup python src/dataset/3_inference.py \
    --scenario JiNan \
    --route_file anon_3_4_jinan_real_2500.rou \
    --jsonl data/sft_dataset/JiNan/anon_3_4_jinan_real_2500.rou/dataset_auto_annotated.jsonl \
    > inference_step3.log 2>&1 &

echo "Process started with PID $!"
echo "You can check the progress with: tail -f inference_step3.log"
