#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-01-29 21:57:10
 # @LastEditTime: 2026-02-06 17:54:26
 # @Description: this script is used to 
 # @FilePath: /VLMTraffic/scripts/run_viewer.sh
### 

# Run streamer viewer in background with output to console

streamlit run scripts/viewer.py -- --path data/sft_dataset/JiNan/dataset.jsonl &
# streamlit run scripts/viewer.py -- --path data/sft_dataset/Hongkong_YMT/dataset.jsonl &

