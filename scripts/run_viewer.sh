#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-01-29 21:57:10
 # @LastEditTime: 2026-03-17 22:15:29
 # @Description: this script is used to 
 # @FilePath: /VLMTraffic/scripts/run_viewer.sh
### 

# Run streamer viewer in background with output to console

# streamlit run scripts/1_viewer.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real/dataset.jsonl &
# streamlit run scripts/1_viewer.py -- --path data/sft_dataset/Hongkong_YMT/dataset.jsonl &
# streamlit run scripts/2_viewer.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real_2500.rou/dataset_auto_annotated.jsonl &
# streamlit run scripts/2_viewer.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real_emergency.rou/dataset_auto_annotated.jsonl &
# streamlit run scripts/3_viewer.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real_2500.rou/dataset_auto_annotated_final.jsonl &
# streamlit run scripts/3_viewer.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real_emergency.rou/03_dataset_reviewed.jsonl &
# streamlit run scripts/4_viewer.py -- --path data/sft_dataset/Hangzhou/anon_4_4_hangzhou_real_5816.rou/dataset_auto_annotated_final.jsonl &
streamlit run scripts/4_viewer.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real_2000.rou/03_dataset_reviewed.jsonl &
# streamlit run scripts/4_viewer.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real_2500.rou/temp.jsonl &