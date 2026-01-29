#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-01-29 21:57:10
 # @LastEditTime: 2026-01-29 22:02:26
 # @Description: this script is used to 
 # @FilePath: /VLMTraffic/scripts/run_viewer.sh
### 

# Run streamer viewer in background with output to console
streamlit run scripts/viewer.py -- --path data/sft_dataset/JiNan/dataset.jsonl &

