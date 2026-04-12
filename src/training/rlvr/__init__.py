'''
Author: yufei Ji
Date: 2026-04-11
Description: RLVR（Reinforcement Learning with Verifiable Rewards）训练模块
             实现 E2ELight 的 Simulation-Grounded Dual-Verifiable RLVR 框架：
               - r_perc：感知可验证奖励（SUMO GT计数 vs 模型预测计数）
               - r_env ：效率可验证奖励（相位压力代理 / SUMO rollout ΔATT）
FilePath: /VLMTraffic/src/training/rlvr/__init__.py
'''
