"""
验证 DPO 训练中 Qwen3-VL 的 ViT (Visual Encoder) 是否真正参与参数更新。
增加了针对 ViT 和 Merger(Projector) 的精准冻结控制。

依赖版本要求（在远程服务器上执行）:
    pip install trl==0.29.1

运行方式:
    CUDA_VISIBLE_DEVICES=0 python src/training/verify_vit_update.py
"""

import os
import torch
from transformers import AutoConfig, Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from PIL import Image


def freeze_multimodal_components(model, freeze_vit=True, freeze_merger=True):
    """
    根据 Qwen3-VL 的具体参数架构，精准冻结多模态组件。
    """
    print("\n================ 开始配置参数冻结 ================")
    # ViT 相关的参数前缀
    vit_prefixes = [
        "visual.patch_embed", 
        "visual.pos_embed", 
        "visual.blocks"
    ]
    # Merger/Projector 相关的参数前缀 (包含深层 merger 列表)
    merger_prefixes = [
        "visual.merger", 
        "visual.deepstack_merger_list"
    ]

    frozen_vit_params = 0
    frozen_merger_params = 0
    vit_layer_num = 0
    merger_later_num = 0

    for name, param in model.named_parameters():
        # 1. 冻结 ViT
        if freeze_vit and any(name.startswith(f"model.{p}") for p in vit_prefixes):
            param.requires_grad = False
            frozen_vit_params += param.numel()
            vit_layer_num +=1
            
        # 2. 冻结 Merger
        elif freeze_merger and any(name.startswith(f"model.{p}") for p in merger_prefixes):
            param.requires_grad = False
            frozen_merger_params += param.numel()
            merger_later_num +=1

    print(f"✅ 冻结配置完成:")
    print(f" - 冻结 ViT: {freeze_vit} (锁定了{vit_layer_num}层，{frozen_vit_params / 1e6:.2f} M 个参数)")
    print(f" - 冻结 Merger: {freeze_merger} (锁定了{merger_later_num}层 {frozen_merger_params / 1e6:.2f} M 个参数)")
    print("==================================================\n")
    return model


def main():
    model_path = "/root/autodl-tmp/model/qwen3-vl-4b"

    print("正在加载模型配置...")
    config = AutoConfig.from_pretrained(model_path)
    if hasattr(config.text_config, "rope_parameters") and getattr(config.text_config, "rope_scaling", None) is None:
        config.text_config.rope_scaling = config.text_config.rope_parameters

    print("正在加载模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, config=config, dtype=torch.bfloat16,
    ).to("cuda")
    ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, config=config, dtype=torch.bfloat16,
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model_path)

    # ==========================================
    # ---- 核心修改点：调用冻结函数 ----
    # 在这里自由控制是否冻结 ViT 和 Merger
    # ==========================================
    model = freeze_multimodal_components(
        model, 
        freeze_vit=False,     # 设为 True 冻结视觉主干
        freeze_merger=False   # 设为 True 冻结投影融合层
    )

    # ---- 记录初始权重 (用于最终对比验证) ----
    # 无论是否被冻结，我们都强行抓取 ViT 的第一层投影权重和 Merger 的第一层权重进行观察
    target_vit_layer = "model.visual.patch_embed.proj.weight"
    target_merger_layer = "model.visual.merger.linear_fc1.weight"
    
    initial_vit_weight = None
    initial_merger_weight = None
    
    for name, param in model.named_parameters():
        if name == target_vit_layer:
            initial_vit_weight = param.clone().detach().cpu()
        elif name == target_merger_layer:
            initial_merger_weight = param.clone().detach().cpu()

    # ---- 构造数据集 ----
    img_path_1 = "/root/autodl-tmp/golden_data/JiNan/anon_3_4_jinan_real_2500.rou/step_8/intersection_3_1_bev_watermarked.png"
    img_path_2 = "/root/autodl-tmp/golden_data/JiNan/anon_3_4_jinan_real_2500.rou/step_9/intersection_3_1_bev_watermarked.png"

    train_dataset = Dataset.from_dict({
        "prompt": [
            [{"role": "user", "content": "Describe the traffic situation in the image."}],
            [{"role": "user", "content": "What traffic elements are visible in the image?"}],
        ],
        "chosen": [
            [{"role": "assistant", "content": "There are several vehicles at the intersection."}],
            [{"role": "assistant", "content": "I can see traffic lights and multiple lanes."}],
        ],
        "rejected": [
            [{"role": "assistant", "content": "I cannot determine the traffic situation."}],
            [{"role": "assistant", "content": "The image is unclear."}],
        ],
        "images": [
            [Image.open(img_path_1).convert("RGB")],
            [Image.open(img_path_2).convert("RGB")],
        ],
    })

    # ---- 训练配置 ----
    print("\n开始 DPO 训练...")
    training_args = DPOConfig(
        output_dir="/root/autodl-tmp/model/tmp_dpo_verify",
        per_device_train_batch_size=1,
        max_length=None,
        max_steps=2,
        learning_rate=1e-5,
        precompute_ref_log_probs=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        save_strategy="no",  # 测试脚本不保存权重，防止爆磁盘
        logging_steps=1,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
    )

    os.environ["WANDB_DISABLED"] = "true"
    trainer.train()

    # ---- 验证权重变化 ----
    print("\n================ 训练结束，对比权重变化 ================")
    for name, param in model.named_parameters():
        if name == target_vit_layer and initial_vit_weight is not None:
            final_weight = param.detach().cpu()
            max_diff = torch.max(torch.abs(initial_vit_weight.float() - final_weight.float())).item()
            print(f"[{target_vit_layer}] (ViT) 最大权重差异: {max_diff:.8f}")
            if max_diff > 1e-7:
                print(" 👉 结论: ViT 参数发生了变化，参与了更新！(未冻结)")
            else:
                print(" 👉 结论: ViT 参数没有变化，已被成功冻结！")
                
        elif name == target_merger_layer and initial_merger_weight is not None:
            final_weight = param.detach().cpu()
            max_diff = torch.max(torch.abs(initial_merger_weight.float() - final_weight.float())).item()
            print(f"[{target_merger_layer}] (Merger) 最大权重差异: {max_diff:.8f}")
            if max_diff > 1e-7:
                print(" 👉 结论: Merger 参数发生了变化，参与了更新！(未冻结)")
            else:
                print(" 👉 结论: Merger 参数没有变化，已被成功冻结！")
    print("========================================================\n")


if __name__ == "__main__":
    main()