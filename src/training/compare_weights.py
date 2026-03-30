'''
Author: yufei Ji
Date: 2026-03-30 05:19:26
LastEditTime: 2026-03-30 05:19:26
Description: this script is used to compare the weights of two VLM models (before and after DPO training) layer by layer to verify which parameters were frozen and which were updated.
FilePath: /VLMTraffic/src/training/compare_weights.py
'''

import os
import torch
from transformers import AutoConfig, Qwen3VLForConditionalGeneration

# ================= 1. 配置路径 =================
# DPO 训练前的 SFT 模型路径
model_before_path = "/root/autodl-tmp/model/qwen3_vl_8b_sft"
# DPO 训练后的 Checkpoint 路径 (请替换为实际的评测 checkpoint 路径)
# model_after_path = "/root/autodl-tmp/results/checkpoints_rpo/rpo_qwen3-beta-0.1-epoch-3-batch-64-per_bs-4-lr-3e-6/checkpoint-96"
model_after_path = "/root/autodl-fs/model/base_model/qwen3-vl-8b"
# 输出比对结果的文件
output_txt_path = "weight_comparison_result.txt"

# ================= 2. 加载模型 (纯 CPU) =================
print("⏳ 正在将【训练前】模型加载到 CPU (节省显存)...")
config_before = AutoConfig.from_pretrained(model_before_path)
if hasattr(config_before.text_config, "rope_parameters") and getattr(config_before.text_config, "rope_scaling", None) is None:
    config_before.text_config.rope_scaling = config_before.text_config.rope_parameters

model_before = Qwen3VLForConditionalGeneration.from_pretrained(
    model_before_path,
    config=config_before,
    dtype=torch.bfloat16,
    device_map="cpu" # 强制在 CPU 加载
)

print("⏳ 正在将【训练后】模型加载到 CPU...")
config_after = AutoConfig.from_pretrained(model_after_path)
if hasattr(config_after.text_config, "rope_parameters") and getattr(config_after.text_config, "rope_scaling", None) is None:
    config_after.text_config.rope_scaling = config_after.text_config.rope_parameters

model_after = Qwen3VLForConditionalGeneration.from_pretrained(
    model_after_path,
    config=config_after,
    dtype=torch.bfloat16,
    device_map="cpu"
)
print("✅ 两个模型加载完成！开始逐层比对...\n")

# ================= 3. 获取状态字典并比对 =================
dict_before = model_before.state_dict()
dict_after = model_after.state_dict()

changed_layers = []
unchanged_layers = []
missing_layers = []

print(f"📝 正在将对比结果写入文件: {output_txt_path} ...")

with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write("="*30 + " 模型参数对比结果 " + "="*30 + "\n")
    f.write(f"模型 A (Before): {model_before_path}\n")
    f.write(f"模型 B (After):  {model_after_path}\n")
    f.write("="*78 + "\n\n")
    
    # 定义表头
    f.write(f"{'参数名称':<60} | {'状态':<15} | {'最大绝对差异 (Max Diff)'}\n")
    f.write("-" * 105 + "\n")

    for name in dict_before.keys():
        if name not in dict_after:
            missing_layers.append(name)
            f.write(f"{name:<60} | ❌ 缺失 (Missing) | N/A\n")
            continue
        
        tensor_before = dict_before[name].float() # 转换为 float32 计算差异，防止 bfloat16 溢出
        tensor_after = dict_after[name].float()

        # 计算最大绝对差异
        max_diff = torch.max(torch.abs(tensor_before - tensor_after)).item()

        # 只要存在大于 1e-4 的差异，就认为发生了更新
        if max_diff > 1e-4:
            changed_layers.append(name)
            status = "🔄 已更新"
        else:
            unchanged_layers.append(name)
            status = "🔒 未变化"
            max_diff = 0.0 # 抹平极小误差

        f.write(f"{name:<60} | {status:<15} | {max_diff:.8f}\n")

    # 汇总信息写入
    f.write("\n" + "="*30 + " 统计汇总 " + "="*30 + "\n")
    f.write(f"检查总层数:   {len(dict_before.keys())}\n")
    f.write(f"🔄 发生更新的层数: {len(changed_layers)}\n")
    f.write(f"🔒 保持原样的层数: {len(unchanged_layers)}\n")
    if missing_layers:
        f.write(f"❌ 训练后缺失的层: {len(missing_layers)}\n")
    f.write("="*70 + "\n")

# ================= 4. 终端打印精简汇总 =================
print("="*22 + " 统计汇总 " + "="*22)
print(f"🔄 发生更新的层数: {len(changed_layers)}")
print(f"🔒 保持原样的层数: {len(unchanged_layers)}")

# 分类统计一下哪些模块变了
visual_changed = sum(1 for name in changed_layers if "visual" in name)
llm_changed = sum(1 for name in changed_layers if "model.layers" in name)

print("\n🔍 更新模块分布:")
print(f"   - 视觉模块 (Visual): {visual_changed} 层更新")
print(f"   - 语言模型 (LLM):    {llm_changed} 层更新")
print("="*54)
print(f"✅ 详细的逐层对比已保存至: {output_txt_path}")