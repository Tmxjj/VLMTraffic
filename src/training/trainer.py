'''
Author: yufei Ji
Date: 2026-01-14 16:43:47
LastEditTime: 2026-04-09 23:32:53
Description: 使用不同偏好优化方法（DPO / RPO / SFT）对 Qwen3-VL 进行多模态微调。
             适配 TRL >= 0.29.1，修复了 image_grid_thw 未传递的 bug（PR #3906）。
FilePath: /VLMTraffic/src/training/trainer.py

依赖版本:
    trl >= 0.29.1
    transformers >= 4.56.2
    accelerate >= 1.4.0

运行方式 (通过 rpo_trainer.sh 调用):
    accelerate launch --config_file configs/accelerate_config.yaml \
        src/training/trainer.py --method rpo --model_path ... ...
'''

import argparse
import os

import PIL.Image
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from trl import DPOConfig, DPOTrainer

# ===========================================================================
# 模型参数冻结函数 
# ===========================================================================

def freeze_multimodal_components(model, freeze_vit=True, freeze_merger=True):
    """
    根据 Qwen3-VL 的具体参数架构，精准冻结多模态组件以节省显存。
    """
    print("\n================ 开始配置参数冻结 ================")
    # ViT 相关的参数前缀
    vit_keywords = ["visual.patch_embed", "visual.pos_embed", "visual.blocks"]
    # Merger/Projector 相关的参数前缀 (包含深层 merger 列表)
    merger_keywords = ["visual.merger", "visual.deepstack_merger_list"]

    frozen_vit_params = 0
    frozen_merger_params = 0

    for name, param in model.named_parameters():
        # 1. 冻结 ViT
        if freeze_vit and any(kw in name for kw in vit_keywords):
            param.requires_grad = False
            frozen_vit_params += param.numel()
            
        # 2. 冻结 Merger
        elif freeze_merger and any(kw in name for kw in merger_keywords):
            param.requires_grad = False
            frozen_merger_params += param.numel()

    print(f"✅ 冻结配置完成:")
    print(f" - 冻结 ViT: {freeze_vit} (锁定了 {frozen_vit_params / 1e6:.2f} M 个参数)")
    print(f" - 冻结 Merger: {freeze_merger} (锁定了 {frozen_merger_params / 1e6:.2f} M 个参数)")
    print("==================================================\n")
    return model
# ===========================================================================
# 数据预处理函数
# ===========================================================================

def format_qwen_dpo_dataset(example):
    """
    用于 dataset.map() 的单条数据处理函数。

    将原始 JSON 数据转换为 TRL 0.29.1 的 DataCollatorForVisionPreference 所需格式：
      - prompt  : 包含 {"type": "image"} 占位符的消息列表（不含 PIL 对象）
      - chosen  : assistant 回复消息列表
      - rejected: assistant 回复消息列表
      - images  : 该样本对应的 PIL 图像列表

    设计原则：
      PIL 图像只放在 images 字段，不放在 content["image"] 里。
      TRL 的 prepare_multimodal_messages 会在 collate 阶段自动将 images 中的
      PIL 图像注入到消息内容的 {"type": "image"} 占位符中，避免序列化问题。
    """
    # === 1. 处理 Prompt：提取图片路径，构建干净的占位符消息 ===
    prompt_messages = example["prompt"]
    images_list = []  # 用于存放该样本的所有 PIL 图像

    for msg in prompt_messages:
        if msg["role"] == "user":
            for content in msg["content"]:
                if content["type"] == "image":
                    img_path = content.get("image", "")
                    if isinstance(img_path, str) and os.path.exists(img_path):
                        # 加载 PIL 图像，追加到独立列表
                        images_list.append(PIL.Image.open(img_path).convert("RGB"))
                        # 移除路径键，只保留 {"type": "image"} 占位符
                        # DataCollatorForVisionPreference 会在 collate 时注入真实 PIL 对象
                        content.pop("image", None)
                    elif isinstance(img_path, str):
                        print(f"警告: 图片路径不存在 → {img_path}")

    # === 2. 处理 Chosen：统一为 [{"role": "assistant", "content": str}] 格式 ===
    # 原始数据可能是 [{"role": "assistant", "content": "..."}]（content 为字符串）
    # 或 [{"role": "assistant", "content": [{"type": "text", "text": "..."}]}]（已结构化）
    # TRL 0.29.1 的 prepare_multimodal_messages 两种格式都能处理，此处统一为字符串。
    chosen_raw = example["chosen"]
    if isinstance(chosen_raw[0]["content"], list):
        # 已结构化 → 提取纯文本
        chosen_text = "".join(
            part["text"] for part in chosen_raw[0]["content"] if part.get("type") == "text"
        )
    else:
        chosen_text = chosen_raw[0]["content"]
    chosen_messages = [{"role": "assistant", "content": chosen_text}]

    # === 3. 处理 Rejected：与 Chosen 处理方式相同 ===
    rejected_raw = example["rejected"]
    if isinstance(rejected_raw[0]["content"], list):
        rejected_text = "".join(
            part["text"] for part in rejected_raw[0]["content"] if part.get("type") == "text"
        )
    else:
        rejected_text = rejected_raw[0]["content"]
    rejected_messages = [{"role": "assistant", "content": rejected_text}]

    return {
        "prompt": prompt_messages,       # 含 {"type": "image"} 占位符的消息列表
        "chosen": chosen_messages,        # assistant 偏好回复
        "rejected": rejected_messages,    # assistant 非偏好回复
        "images": images_list,            # 对应的 PIL 图像列表
    }


# ===========================================================================
# 主函数
# ===========================================================================

def main():
    # ------------------------------------------------------------------
    # 1. 命令行参数解析
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Qwen3-VL 多模态偏好优化训练脚本 (TRL >= 0.29.1)")
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["sft", "dpo", "rpo"],
        help="偏好优化方法: sft=监督微调, dpo=直接偏好优化, rpo=鲁棒偏好优化"
    )
    parser.add_argument("--model_path", type=str, required=True, help="预训练模型路径")
    parser.add_argument("--dataset_path", type=str, required=True, help="训练数据集路径 (.jsonl)")
    parser.add_argument("--output_dir", type=str, required=True, help="模型检查点输出目录")
    parser.add_argument("--beta", type=float, required=True, help="DPO KL 散度惩罚系数（越大越保守）")
    parser.add_argument("--num_train_epochs", type=int, required=True, help="训练轮数")
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True, help="梯度累积步数")
    parser.add_argument("--per_device_train_batch_size", type=int, required=True, help="每卡 batch size")
    parser.add_argument("--learning_rate", type=float, required=True, help="学习率")
    args, _ = parser.parse_known_args()

    parser.add_argument("--freeze_vit", type=str, default="true", choices=["true", "false"], help="是否冻结视觉主干参数")
    parser.add_argument("--freeze_merger", type=str, default="true", choices=["true", "false"], help="是否冻结投影融合层参数")
    
    args, _ = parser.parse_known_args()

    # 参数类型转换
    freeze_vit_bool = args.freeze_vit.lower() == "true"
    freeze_merger_bool = args.freeze_merger.lower() == "true"

    print(f"训练参数: {args}")

    # 梯度异常检测（调试用，生产环境可关闭以提升性能）
    torch.autograd.set_detect_anomaly(True)

    # ------------------------------------------------------------------
    # 2. 模型加载
    # ------------------------------------------------------------------

    # 加载模型配置，并处理 transformers 版本兼容问题：
    # transformers >= 5.x 使用 rope_parameters，而 4.x 使用 rope_scaling，
    # 此处做一次兼容转换，确保旧版 transformers 能正确读取新版保存的配置。
    config = AutoConfig.from_pretrained(args.model_path)
    if hasattr(config.text_config, "rope_parameters") and \
            getattr(config.text_config, "rope_scaling", None) is None:
        print("检测到 transformers 版本配置差异，正在适配 rope_parameters → rope_scaling ...")
        config.text_config.rope_scaling = config.text_config.rope_parameters

    # 加载策略模型（Policy Model）：即需要被优化的模型
    # dtype=torch.bfloat16 可节省约 50% 显存，同时保持训练稳定性
    # device_map=None：ZeRO-3 要求不使用 device_map，由 deepspeed 统一管理参数分布
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map=None,  # ZeRO-3 必须为 None，参数由 deepspeed 跨卡分片
    )
    #在模型加载后立即应用冻结策略
    model = freeze_multimodal_components(model, freeze_vit=freeze_vit_bool, freeze_merger=freeze_merger_bool)

    # 加载参考模型（Reference Model）：用于计算 KL 散度，参数在训练过程中固定不变。
    # ZeRO-3 下必须显式实例化，不能让 TRL 自动复制（自动复制与 ZeRO-3 的参数分片机制不兼容）。
    ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map=None,
    )

    # 加载多模态处理器（Processor）：同时包含 tokenizer 和图像处理器
    processor = AutoProcessor.from_pretrained(args.model_path)

    # ------------------------------------------------------------------
    # 3. 数据集加载与预处理
    # ------------------------------------------------------------------

    # 从 JSONL 文件加载原始数据集
    train_dataset = load_dataset(
        "json",
        data_files={"train": args.dataset_path},
        split="train",
    )

    # 将原始数据转换为 TRL 0.29.1 所需格式：
    #   - 提取图片路径 → 加载为 PIL 图像 → 存入 images 字段
    #   - 清理消息内容中的路径，只保留 {"type": "image"} 占位符
    # 注意：num_proc=1 避免多进程 PIL 序列化问题（PIL 对象跨进程传输不稳定）
    train_dataset = train_dataset.map(
        format_qwen_dpo_dataset,
        num_proc=1,
        desc="格式化数据集（加载图片）",
    )

    # ------------------------------------------------------------------
    # 4. 公共训练参数（所有方法共用）
    # ------------------------------------------------------------------

    # 通用 DPOConfig 参数说明：
    #
    # max_length=None：
    #   VLM 必须禁用截断。图像被转化为大量视觉 token（本项目约 2048 个），
    #   任何截断都会导致 pixel_values 的特征数与 input_ids 中的图像 token 数不一致，
    #   引发 "Image features and image tokens do not match" 错误。
    #
    # precompute_ref_log_probs=False：
    #   ZeRO-3 下参考模型参数分散在多张卡上，无法在训练前统一预计算 log_probs，
    #   必须在每个训练步骤动态计算，代价是每步多一次 ref_model 的前向传播。
    #
    # gradient_checkpointing=True：
    #   用计算换显存：前向传播时不保存所有中间激活值，反向传播时重新计算。
    #   对 8B 规模的模型在 A100 上训练是必须开启的。
    #
    # gradient_checkpointing_kwargs={"use_reentrant": False}：
    #   PyTorch 推荐的新版检查点实现，避免 use_reentrant=True 的内存泄漏问题。

    common_config = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_length=None,                # VLM 必须禁用截断，见上方说明
        lr_scheduler_type="cosine",     # 余弦退火，训练后期平滑降低学习率
        warmup_ratio=0.05,              # 前 5% 步骤线性预热，防止初期梯度爆炸
        learning_rate=args.learning_rate,
        beta=args.beta,
        save_strategy="epoch",          # 每个 epoch 保存一次检查点
        save_only_model=True,           # 只保存模型权重，不保存优化器状态（节省磁盘）
        save_total_limit=1,             # 最多保留 1 个检查点，自动删除旧的
        eval_strategy="no",             # 不做评估，节省训练时间
        logging_steps=1,                # 每步打印一次训练日志
        max_grad_norm=1.0,              # 梯度裁剪，防止梯度爆炸
        precompute_ref_log_probs=False, # ZeRO-3 必须为 False，见上方说明
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,                      # 使用 bfloat16 混合精度训练
    )

    # ------------------------------------------------------------------
    # 5. 按方法构建 Trainer
    # ------------------------------------------------------------------

    if args.method == "sft":
        # ------ SFT（监督微调）------
        # 用 DPOConfig 的 loss_type="sft" 实现：
        # 只在 chosen 回复上计算 NLL 损失，等价于标准 SFT，忽略 rejected 数据。
        # 通常用于 DPO 的热身阶段，先在 chosen 数据上做 SFT，再做 DPO。
        training_args = DPOConfig(
            loss_type="sft",
            **common_config,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,             # SFT 不需要参考模型
            args=training_args,
            train_dataset=train_dataset,
            processing_class=processor,
        )

    elif args.method == "dpo":
        # ------ DPO（直接偏好优化）------
        # 标准 sigmoid DPO 损失：
        #   L = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
        # β 控制与参考模型的偏离程度：β 越大，策略越保守（越靠近 ref_model）。
        training_args = DPOConfig(
            loss_type="sigmoid",
            **common_config,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=processor,
        )

    elif args.method == "rpo":
        # ------ RPO（鲁棒偏好优化）------
        # RPO = DPO 损失 + SFT 损失（在 chosen 回复上的 NLL），即：
        #   L_RPO = L_DPO + α * L_SFT(chosen)
        #
        # TRL 0.13.0 用 rpo_alpha 参数实现，TRL 0.29.1 已移除该参数，
        # 改用 loss_type 多损失组合方式实现等价效果：
        #   loss_type=["sigmoid", "sft"] + loss_weights=[1.0, rpo_alpha]
        #
        # RPO 的优势：在偏好学习的同时，SFT 损失防止 chosen 回复质量退化，
        # 对于复杂推理任务（如交通信号相位决策）比纯 DPO 更稳定。
        rpo_alpha = 1.0  # SFT 损失权重，与原 rpo_alpha 含义相同，默认 1.0
        training_args = DPOConfig(
            loss_type=["sigmoid", "sft"],   # DPO loss + SFT loss 组合
            loss_weights=[1.0, rpo_alpha],  # L_total = 1.0 * L_DPO + rpo_alpha * L_SFT
            **common_config,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=processor,
        )

    else:
        raise NotImplementedError(f"不支持的训练方法: {args.method}")

    # ------------------------------------------------------------------
    # 6. 启动训练
    # ------------------------------------------------------------------
    trainer.train()


if __name__ == "__main__":
    main()
