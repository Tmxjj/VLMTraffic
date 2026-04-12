# E2ELight 短期工作 TODO List

> 更新日期：2026-04-11  
> 目标：完成核心贡献验证，达到 KDD / AAAI 2026 投稿条件  
> 数据来源：`results/ablation_result.csv`（7 场景 DPO 全面提升）、`results/comparsion_result.csv`（FixedTime 已完成，其余待补）

---

## P0：紧急（本周内）

- [ ] **【对比基线】跑 MaxPressure 基线**
  - 场景：全部 7 个（JiNan_real/2000/2500、Jinan_synthetic、Hangzhou/5816/synthetic）
  - 重要性：MaxPressure 是最强无学习基线，若 SFT+DPO 无法超越，论文立论崩塌
  - 参考脚本：`scripts/run_eval_gpu*.sh`，填入 `results/comparsion_result.csv`

- [ ] **【对比基线】跑 CoLight（SOTA RL）基线**
  - 场景：同上
  - 重要性：KDD/AAAI 必须与 SOTA RL 方法对比，CoLight 是 LLM for TSC 论文的标准基线

---

## P1：高优先级（2 周内）

- [ ] **【方向 3 实现】数值距离连续惩罚 DPO**
  - 背景：标准 DPO 对 JiNan_real ATT 仅改善 0.4%，Jinan_synthetic ATT 仍高达 577（远高于 FixedTime 878 的 34% 压缩空间仍有限）；分级惩罚可进一步利用误差幅度信号
  - 具体步骤：
    - [ ] 解析 `src/dataset/DPO_data_construct/dpo_dataset_sft_model_gener_cleaned.jsonl`，提取每条负样本的 `|N_pred - N_gt|`，统计误差分布
    - [ ] 修改 `src/training/` 中的 DPO trainer，支持样本级权重 `w_i = 1 + α·|ΔN|`
    - [ ] 构造"空间规则违反"负样本（越停止线/逆向计数），赋予 2-3× 权重
    - [ ] 训练并评测，对比标准 DPO，填入 `results/ablation_result.csv`
  - 预期改善场景：Jinan_synthetic（最难、标准 DPO 增益最小）、JiNan_real（ATT 改善仅 0.4% 的场景）

- [ ] **【对比基线】补充 LLMLight（KDD 2025）结果**
  - 优先级：与本文最相关的 LLM for TSC 工作，必须直接对比
  - 填入 `results/comparsion_result.csv`

- [ ] **【方向 1 数据构建】Causal Step-DPO 的阶段性偏好对构造脚本**
  - 构造类型 A：Scene Understanding 注入计数错误 + Selection Logic 保持逻辑正确（前提错但推导自洽）
  - 构造类型 B：Scene Understanding 正确 + Selection Logic 选择与分析矛盾的相位
  - 技术路径：基于 SFT 数据集脚本化替换字段，人工审核 20-30 条后批量扩展

---

## P2：中优先级（4 周内）

- [ ] **【对比基线】补充 VLMLight 结果**
  - 重要性：与本文最直接重叠的竞争工作，顶会审稿人必然追问
  - 若无法复现，至少提供推理延迟对比：E2ELight 单次前向 vs. VLMLight VLM+LLM 串联

- [ ] **【对比基线】补充其余 RL 方法**（MPLight、Advanced-CoLight 等）
  - 填满 `results/comparsion_result.csv` 中的 RL 区块

- [ ] **【方向 1 实现】Causal Step-DPO 训练**
  - 基于方向 1 数据构建结果，实现阶段性 Mask 机制
  - 对 Scene Understanding / Scene Analysis / Selection Logic 三段分别计算 loss，差异化惩罚权重

- [ ] **【延迟分析】统计推理延迟**
  - 测量各版本（Zero-Shot / SFT / SFT+DPO）在单张 BEV 图像上的平均推理时间（ms）
  - 与 VLMLight 两阶段延迟对比

- [ ] **【可视化】幻觉错误 Case Study**
  - 对比 SFT vs SFT+DPO 在 BEV 图像上的计数/位置误差
  - 重点场景：Jinan_synthetic（模型最难判断的高流量场景）

---

## P3：中长期规划（6-12 周）

- [ ] **【方向 2 重设计】Simulation-Guided GRPO**
  - 参考 DeepSeek-R1 GRPO 范式：同场景多序列采样 → SUMO 并行仿真 → 组间相对奖励
  - 技术依赖：CityFlow/SUMO 批量并行接口

- [ ] **【泛化集评测】在零样本泛化集上补充完整结果**
  - 场景：Songdo、Yau Ma Tei (HK)、Massy (France)、New York (196路口)、Emergency
  - 包含 MaxPressure、CoLight 对比

- [ ] **【Sim-to-Real】数据增强**
  - 对 BEV 图像加入噪声/色彩抖动/视角扰动，验证模型鲁棒性

- [ ] **【大模型对比】补充 Generalist LLM 结果**
  - Gemini-3-pro、ChatGPT-5、Qwen-3.5 的零样本 TSC 性能
  - 体现专项微调 vs. 通用大模型的价值

---

## 投稿时间线参考

| 里程碑 | 目标时间 | 对应 TODO |
|:---|:---:|:---|
| MaxPressure + CoLight 基线 | Week 1 | P0 全部 |
| 方向 3 实现与评测 | Week 2-3 | P1 第一项 |
| LLMLight 基线 + 方向 1 数据构建 | Week 3 | P1 第二、三项 |
| VLMLight 基线 + 方向 1 训练 | Week 4-5 | P2 前两项 |
| 延迟分析 + Case Study | Week 5 | P2 后两项 |
| **论文初稿完成** | **Week 6** | — |
| KDD 2026 / AAAI 2026 投稿 | 参照官网 DDL | — |
