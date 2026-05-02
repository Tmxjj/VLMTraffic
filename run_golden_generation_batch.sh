#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-04-28 00:11:03
 # @LastEditTime: 2026-05-01 12:16:35
 # @Description: this script is used to 
 # @FilePath: /VLMTraffic/run_golden_generation_batch.sh
### 

echo "🚀 开始串行执行 9 个仿真任务..."

# bash scripts/run_golden_generation.sh --scenario France_Massy --route_file massy_accident.rou.xml --warmup_seconds 30 --max_sumo_seconds 800 --rollout_follow_steps 1
# bash scripts/run_golden_generation.sh --scenario Hongkong_YMT --route_file YMT_bus.rou.xml --warmup_seconds 30 --max_sumo_seconds 600 --rollout_follow_steps 1
# bash scripts/run_golden_generation.sh --scenario Hongkong_YMT --route_file YMT_debris.rou.xml --warmup_seconds 30 --max_sumo_seconds 800 --rollout_follow_steps 1
# bash scripts/run_golden_generation.sh --scenario Hongkong_YMT --route_file YMT_emergy.rou.xml --warmup_seconds 30 --max_sumo_seconds 800 --rollout_follow_steps 1
# bash scripts/run_golden_generation.sh --scenario Hongkong_YMT --route_file YMT_accident.rou.xml --warmup_seconds 30 --max_sumo_seconds 800 --rollout_follow_steps 1
# bash scripts/run_golden_generation.sh --scenario SouthKorea_Songdo --route_file songdo_debris.rou.xml --warmup_seconds 30 --max_sumo_seconds 800 --rollout_follow_steps 1
bash scripts/run_golden_generation.sh --scenario SouthKorea_Songdo --route_file songdo_bus.rou.xml --warmup_seconds 30 --max_sumo_seconds 800 --rollout_follow_steps 1
bash scripts/run_golden_generation.sh --scenario SouthKorea_Songdo --route_file songdo_accident.rou.xml --warmup_seconds 30 --max_sumo_seconds 800 --rollout_follow_steps 1
bash scripts/run_golden_generation.sh --scenario SouthKorea_Songdo --route_file songdo_emergy.rou.xml --warmup_seconds 30 --max_sumo_seconds 800 --rollout_follow_steps 1
bash scripts/run_golden_generation.sh --scenario JiNan --route_file anon_3_4_jinan_real.rou.xml --warmup_seconds 1000 --max_sumo_seconds 3700 --rollout_follow_steps 1
bash scripts/run_golden_generation.sh --scenario JiNan --route_file anon_3_4_jinan_real_debris_专为产数据.rou.xml --warmup_seconds 500 --max_sumo_seconds 1000 --rollout_follow_steps 1
echo "🎉 所有 9 个仿真任务已全部按顺序执行完毕！"