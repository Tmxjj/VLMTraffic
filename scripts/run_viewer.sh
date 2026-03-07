#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-01-29 21:57:10
 # @LastEditTime: 2026-03-07 18:13:44
 # @Description: this script is used to 
 # @FilePath: /VLMTraffic/scripts/run_viewer.sh
### 

# Run streamer viewer in background with output to console

streamlit run scripts/1_viewer.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real/dataset.jsonl &
# streamlit run scripts/1_viewer.py -- --path data/sft_dataset/Hongkong_YMT/dataset.jsonl &
