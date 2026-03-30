# 该脚本用于查看模型参数的冻结状态，主要查看viusal encoder是否参与参数更新，并将详细信息保存到 TXT 文件中，方便后续分析和记录。
import os
# 限制 PyTorch 只能看到 GPU 0，防止占用其他卡资源
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoConfig, Qwen3VLForConditionalGeneration
import torch

model_path = "/root/autodl-tmp/model/qwen3_vl_8b_sft"
output_txt_path = "model_parameters_detail.txt"

config = AutoConfig.from_pretrained(model_path)

# 补丁：处理 transformers 5.x 到 4.x 的版本兼容问题
if hasattr(config.text_config, "rope_parameters") and getattr(config.text_config, "rope_scaling", None) is None:
    print("⚠️ 检测到 transformers 版本配置冲突，正在将 rope_parameters 转换为 rope_scaling...")
    config.text_config.rope_scaling = config.text_config.rope_parameters

print("⏳ 正在将模型加载到 GPU 0，请稍候...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    config=config,
    dtype=torch.bfloat16,
    device_map="cuda:0"  
)
print("✅ 模型加载完成！")

# --- 全模型参数冻结状态检查逻辑并写入 TXT ---
print(f"📝 正在将参数详情写入文件: {output_txt_path} ...")

trainable_params_count = 0
frozen_params_count = 0
total_layers = 0

with open(output_txt_path, "w", encoding="utf-8") as f:
    # 写入文件头
    f.write("="*30 + " 模型参数详细信息 " + "="*30 + "\n")
    f.write(f"模型路径: {model_path}\n")
    f.write("="*78 + "\n\n")
    f.write(f"{'参数名称':<60} | {'形状 (Shape)':<25} | {'参数量':<12} | {'可训练?'}\n")
    f.write("-" * 115 + "\n")

    # 遍历并写入每一层的信息
    for name, param in model.named_parameters():
        total_layers += 1
        num_params = param.numel()
        shape_str = str(list(param.shape))
        req_grad = "✅ True" if param.requires_grad else "❌ False"
        
        # 格式化写入单行参数信息
        f.write(f"{name:<60} | {shape_str:<25} | {num_params:<12} | {req_grad}\n")
        
        # 统计数量
        if param.requires_grad:
            trainable_params_count += num_params
        else:
            frozen_params_count += num_params

    total_params_count = trainable_params_count + frozen_params_count

    # 写入文件尾部的汇总信息
    f.write("\n" + "="*30 + " 统计汇总 " + "="*30 + "\n")
    f.write(f"总层数:       {total_layers}\n")
    f.write(f"总参数量:     {total_params_count / 1e9:.4f} B\n")
    f.write(f"可训练参数:   {trainable_params_count / 1e9:.4f} B (占比 {trainable_params_count/total_params_count*100:.2f}%)\n")
    f.write(f"冻结参数:     {frozen_params_count / 1e9:.4f} B (占比 {frozen_params_count/total_params_count*100:.2f}%)\n")
    f.write("="*70 + "\n")

print(f"✅ 参数详情已成功保存至 {output_txt_path}")

# 终端仅输出精简汇总
print("\n" + "="*22 + " 统计汇总 " + "="*22)
print(f"📊 总参数量:   {total_params_count / 1e9:.4f} B")
print(f"🔓 可训练参数: {trainable_params_count / 1e9:.4f} B (占比 {trainable_params_count/total_params_count*100:.2f}%)")
print(f"🔒 冻结参数:   {frozen_params_count / 1e9:.4f} B (占比 {frozen_params_count/total_params_count*100:.2f}%)")
print("="*56 + "\n")

allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
print(f"💾 GPU 0 当前已分配显存: {allocated_memory:.2f} GB")