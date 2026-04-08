import os
import torch
import copy
from transformers import AutoConfig, Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

def main():
    # 1. 设置路径 (替换为你实际的基础模型路径)
    model_path = "/root/autodl-tmp/model/qwen3_vl_8b_sft"
    
    print("⏳ 正在加载模型配置...")
    config = AutoConfig.from_pretrained(model_path)
    if hasattr(config.text_config, "rope_parameters") and getattr(config.text_config, "rope_scaling", None) is None:
        config.text_config.rope_scaling = config.text_config.rope_parameters

    print("⏳ 正在加载模型 (这可能需要一些时间)...")
    # 为了防止 OOM，这里加载为 bfloat16 类型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map="cuda:0" # 加载到单卡即可验证
    )
    
    # DPO 还需要一个参考模型 (Reference Model)
    ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    
    processor = AutoProcessor.from_pretrained(model_path)

    # 2. 检查 ViT 模块的冻结状态
    print("\n🔍 检查 Visual Encoder (ViT) 参数的 requires_grad 属性:")
    vit_name = "visual" # 根据具体模型的参数名称，通常包含 visual 或者是 vision_tower
    has_frozen_vit = False
    for name, param in model.named_parameters():
        if vit_name in name:
            if not param.requires_grad:
                has_frozen_vit = True
                print(f"⚠️ 发现被冻结的视觉层: {name}")
                break
    
    if not has_frozen_vit:
        print("✅ 视觉编码器 (ViT) 的参数没有被冻结， requires_grad=True")

    # 3. 记录训练前视觉模块某层的初始权重
    # 取视觉模块的第一个具有权重的层作为观察对象
    target_layer_name = None
    initial_weight = None
    for name, param in model.named_parameters():
        if vit_name in name and "weight" in name and param.requires_grad:
            target_layer_name = name
            # clone 复制一份权重，以防被 inplace 操作覆盖
            initial_weight = param.clone().detach().cpu()
            break
            
    print(f"\n📸 选取观察的视觉层: {target_layer_name}")

    # 4. 伪造极少量的 DPO 数据
    # 为了跑通训练，我们需要符合格式的 chosen 和 rejected
    dummy_data = {
        "prompt": [
            [{"role": "user", "content": [{"type": "text", "text": "Describe the traffic."}]}],
            [{"role": "user", "content": [{"type": "text", "text": "What is in the image?"}]}]
        ],
        "chosen": [
            [{"role": "assistant", "content": [{"type": "text", "text": "There are cars."}]}],
            [{"role": "assistant", "content": [{"type": "text", "text": "A red car."}]}]
        ],
        "rejected": [
            [{"role": "assistant", "content": [{"type": "text", "text": "I don't know."}]}],
            [{"role": "assistant", "content": [{"type": "text", "text": "Nothing."}]}]
        ],
        "images": [[], []] # 简单起见，这里不放置真实图片，如果有需要可以填充 dummy images
    }
    train_dataset = Dataset.from_dict(dummy_data)

    # 5. 配置并启动极简 DPO 训练
    print("\n🚀 开始仅迭代几个 step 的 DPO 训练...")
    training_args = DPOConfig(
        output_dir="./tmp_dpo_verify",
        per_device_train_batch_size=1,  # 极小的 BS
        max_prompt_length=128,
        max_length=256,
        max_completion_length=128,
        max_steps=2,                    # 只跑 2 步用于验证
        learning_rate=1e-5,
        remove_unused_columns=False,
        gradient_checkpointing=True,    # 开启重算节省显存
        gradient_checkpointing_kwargs={'use_reentrant': False},
        bf16=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor
    )

    # 关闭 Wandb 避免因为没有 key 报错
    os.environ["WANDB_DISABLED"] = "true"
    
    trainer.train()

    # 6. 验证训练后权重是否发生变化
    print("\n⚖️ 训练结束，对比权重变化...")
    for name, param in model.named_parameters():
        if name == target_layer_name:
            final_weight = param.detach().cpu()
            # 计算绝对差异的最大值
            max_diff = torch.max(torch.abs(initial_weight.float() - final_weight.float())).item()
            
            print(f"[{target_layer_name}] 训练前后权重的最大差异: {max_diff:.8f}")
            if max_diff > 0:
                print("🎉 结论: ViT 模块的参数发生了变化，在 DPO 阶段**并未被冻结**，参与了反向传播和更新！")
            else:
                print("⚠️ 结论: ViT 模块的参数没有发生变化。可能是被框架某个角落冻结了，或者 lr 实在太小导致误差被忽略。")
            break

if __name__ == "__main__":
    main()
