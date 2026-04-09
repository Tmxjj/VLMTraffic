"""
验证 DPO 训练中 Qwen3-VL 的 ViT (Visual Encoder) 是否真正参与参数更新。

依赖版本要求（在远程服务器上执行）:
    pip install trl==0.29.1

TRL >= 0.29.1 修复了 image_grid_thw 未传递给模型的 bug（PR #3906 / Issue #4071），
无需任何 monkey-patch，直接使用官方 API 即可正确运行 Qwen3-VL 的多模态 DPO。

运行方式:
    CUDA_VISIBLE_DEVICES=0 python src/training/verify_vit_update.py
"""

import os
import torch
from transformers import AutoConfig, Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from PIL import Image


def main():
    model_path = "/root/autodl-tmp/model/qwen3_vl_8b_sft"

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

    # ---- 检查 ViT 冻结状态 ----
    print("\n检查 Visual Encoder (ViT) 参数的 requires_grad 属性:")
    vit_name = "visual"
    has_frozen_vit = False
    for name, param in model.named_parameters():
        if vit_name in name and not param.requires_grad:
            has_frozen_vit = True
            print(f"发现被冻结的视觉层: {name}")
            break
    if not has_frozen_vit:
        print("视觉编码器 (ViT) 的参数没有被冻结, requires_grad=True")

    # ---- 记录初始权重 ----
    target_layer_name = None
    initial_weight = None
    for name, param in model.named_parameters():
        if vit_name in name and "weight" in name and param.requires_grad:
            target_layer_name = name
            initial_weight = param.clone().detach().cpu()
            break
    print(f"\n选取观察的视觉层: {target_layer_name}")

    # ---- 构造数据集 ----
    # TRL 0.29.1 的 DataCollatorForVisionPreference 期望原始格式（不做预处理）:
    #   - images: 每个样本的 PIL 图像列表
    #   - prompt / chosen / rejected: 对话格式（message dict 列表，content 为纯字符串即可）
    # prepare_multimodal_messages 会自动把 images 注入 prompt 消息的 image 占位符，
    # 无需手动写 {"type": "image"} 占位符。
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
    # TRL 0.29.1 的 DPOConfig:
    #   - 移除了 max_prompt_length、max_completion_length（仅保留 max_length）
    #   - 视觉数据集自动保留所需列（无需 remove_unused_columns=False）
    #   - 视觉数据集跳过预处理流水线，由 DataCollatorForVisionPreference 负责 on-the-fly 处理
    print("\n开始 DPO 训练...")
    training_args = DPOConfig(
        output_dir="./tmp_dpo_verify",
        per_device_train_batch_size=1,
        max_length=None,  # VLM 必须禁用截断：截断会移除部分图像token，导致与pixel_values的特征数不匹配
        max_steps=2,
        learning_rate=1e-5,
        precompute_ref_log_probs=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
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
    print("\n训练结束，对比权重变化...")
    for name, param in model.named_parameters():
        if name == target_layer_name:
            final_weight = param.detach().cpu()
            max_diff = torch.max(torch.abs(initial_weight.float() - final_weight.float())).item()
            print(f"[{target_layer_name}] 最大权重差异: {max_diff:.8f}")
            if max_diff > 0:
                print("结论: ViT 参数发生了变化，在 DPO 训练中参与了反向传播和更新！")
            else:
                print("结论: ViT 参数没有变化，可能被冻结或学习率过小。")
            break


if __name__ == "__main__":
    main()
