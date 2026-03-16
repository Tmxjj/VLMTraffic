#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-01-29 21:57:10
 # @LastEditTime: 2026-03-13 00:00:18
 # @Description: this script is used to 
 # @FilePath: /VLMTraffic/scripts/run_viewer.sh
### 

# Run streamer viewer in background with output to console

# streamlit run scripts/1_viewer.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real/dataset.jsonl &
# streamlit run scripts/1_viewer.py -- --path data/sft_dataset/Hongkong_YMT/dataset.jsonl &
# streamlit run scripts/2_viewer.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real_2500.rou/dataset_auto_annotated.jsonl &
# streamlit run scripts/2_viewer.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real_emergency.rou/dataset_auto_annotated.jsonl &
streamlit run scripts/3_viewer.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real_2500.rou/dataset_auto_annotated_final.jsonl &
# streamlit run scripts/3_viewer.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real_emergency.rou/dataset_auto_annotated_final.jsonl &