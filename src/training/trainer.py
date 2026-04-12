'''
Author: yufei Ji
Date: 2026-01-14 16:43:47
LastEditTime: 2026-04-11 10:00:00
Description: 使用不同偏好优化方法（DPO / RPO / SFT / NCDPO）对 Qwen3-VL 进行多模态微调。
             适配 TRL >= 0.29.1，修复了 image_grid_thw 未传递的 bug（PR #3906）。

             新增 NCDPO（数值距离连续惩罚 DPO）方法：
             在 RPO 框架（sigmoid + sft）基础上，引入每样本动态 β：
               β_i = β_base × (1 + α × tanh(ε_i / ε_scale))
             其中 ε_i 为第 i 条样本正负回答的车辆计数 MAE。
FilePath: /VLMTraffic/src/training/trainer.py

依赖版本:
    trl >= 0.29.1
    transformers >= 4.56.2
    accelerate >= 1.4.0

运行方式:
    # RPO 训练
    accelerate launch --config_file configs/accelerate_config.yaml \
        src/training/trainer.py --method rpo --model_path ... ...

    # NCDPO 训练（需先运行 augment_ncdpo_weights.py 生成增强数据集）
    accelerate launch --config_file configs/accelerate_config.yaml \
        src/training/trainer.py --method ncdpo --model_path ... \
        --ncdpo_alpha 1.0 --ncdpo_eps_scale 2.0 ...
'''

import argparse
import os
from dataclasses import dataclass
from typing import Any

import PIL.Image
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import DataCollatorForVisionPreference


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

    注意：HuggingFace .map() 会保留此函数未返回的列（如 id、count_error_weight），
    因此 NCDPO 所需的 count_error_weight 字段在 .map() 后仍可访问。

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
                        content.pop("image", None)
                    elif isinstance(img_path, str):
                        print(f"警告: 图片路径不存在 → {img_path}")

    # === 2. 处理 Chosen：统一为 [{"role": "assistant", "content": str}] 格式 ===
    chosen_raw = example["chosen"]
    if isinstance(chosen_raw[0]["content"], list):
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
        # 注意：其他字段（id、count_error_weight 等）由 HuggingFace .map() 自动保留
    }


# ===========================================================================
# NCDPO 专属组件：自定义 Collator + 自定义 Trainer
# ===========================================================================

@dataclass
class NCDPODataCollator(DataCollatorForVisionPreference):
    """
    扩展 VisionPreference 数据整理器，将 count_error_weight 字段透传到 batch 中。

    TRL 的 DataCollatorForVisionPreference 只处理固定的视觉/文本字段，
    无法自动传递额外的浮点字段。本 Collator 在调用父类前先将
    count_error_weight 取出，再作为张量追加到 batch 字典中。

    batch 输出额外字段：
        count_error_weight (torch.FloatTensor, shape: batch_size):
            每条样本的计数 MAE，由 augment_ncdpo_weights.py 预计算得到。
    """

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # 先弹出计数误差权重，避免父类因未知字段报错
        weights = [float(ex.pop("count_error_weight", 0.0)) for ex in examples]

        # 调用父类完成图像处理、tokenization 和 padding
        batch = super().torch_call(examples)

        # 将计数误差权重作为 FloatTensor 追加到 batch
        # 形状: (batch_size,) 与 delta_score 对齐，支持逐样本乘法
        batch["count_error_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


class NCDPOTrainer(DPOTrainer):
    """
    数值距离连续惩罚 DPO Trainer（NCDPO）。

    在 RPO（sigmoid + sft）框架基础上，对 sigmoid DPO 损失引入逐样本动态 β：

        β_i = β_base × (1 + α × tanh(ε_i / ε_scale))

    其中：
        ε_i     : 第 i 条样本正负回答之间的车辆计数 MAE（由数据集预先计算）
        ε_scale : 归一化尺度（控制 tanh 饱和点，默认 2.0）
        α       : 幅值系数（控制动态范围，默认 1.0）

    当 ε_i = 0（计数完全一致）时，β_i = β_base（退化为标准 RPO）。
    当 ε_i → ∞（计数严重错误）时，β_i → β_base × (1 + α)（最大惩罚）。

    实现机制（最小侵入设计）：
        将 self.beta 重定义为 Python property。
        - setter: 父类 __init__ 执行 self.beta = args.beta 时，存储为 _base_beta
        - getter: 若当前 batch 设置了 _ncdpo_errors，返回形状 (B,) 的 per-sample 张量；
                  否则退化为标量 float，行为与父类完全一致。
        仅需在 _compute_loss 开头注入 count_error_weight，无需复制 TRL 源码。
    """

    # ----------------------------------------------------------------
    # 类级别的 beta property（覆盖实例属性，使 per-sample 动态 β 成为可能）
    # ----------------------------------------------------------------
    @property
    def beta(self):
        """
        返回当前有效的 β 值：
          - 标量 float：没有正在处理的 batch，或 batch 中无计数误差
          - 形状 (B,) 的 FloatTensor：正在处理的 batch 包含 count_error_weight
        """
        if self._ncdpo_errors is not None:
            # w_i = tanh(ε_i / ε_scale) ∈ (0, 1)
            w = torch.tanh(self._ncdpo_errors / self._ncdpo_eps_scale)
            # β_i = β_base × (1 + α × w_i)
            return self._base_beta * (1.0 + self._ncdpo_alpha * w)
        return self._base_beta

    @beta.setter
    def beta(self, value: float):
        """父类 __init__ 执行 self.beta = args.beta 时触发，存储为标量基准值"""
        self._base_beta = float(value)

    def __init__(
        self,
        *args,
        ncdpo_alpha: float = 1.0,
        ncdpo_eps_scale: float = 2.0,
        **kwargs
    ):
        """
        Args:
            ncdpo_alpha     : 动态 β 的幅值系数 α（默认 1.0，β 最大翻倍）
            ncdpo_eps_scale : tanh 归一化尺度 ε_scale（默认 2.0，对应均值 MAE≈0.34 时放大约 17%）
            *args, **kwargs : 透传给父类 DPOTrainer
        """
        # 注意：_ncdpo_errors 必须在 super().__init__() 之前初始化，
        # 因为父类 __init__ 会触发 self.beta = args.beta（调用 setter）
        self._ncdpo_errors = None
        self._ncdpo_alpha = ncdpo_alpha
        self._ncdpo_eps_scale = ncdpo_eps_scale
        # 调用父类（内部执行 self.beta = args.beta → 触发 setter → 存为 _base_beta）
        super().__init__(*args, **kwargs)

        print(f"\n{'='*55}")
        print(f"  NCDPOTrainer 初始化完成")
        print(f"  基准 β (beta_base):    {self._base_beta}")
        print(f"  幅值系数 α (alpha):    {self._ncdpo_alpha}")
        print(f"  归一化尺度 (eps_scale):{self._ncdpo_eps_scale}")
        print(f"  β 动态范围:            [{self._base_beta:.4f}, {self._base_beta * (1 + self._ncdpo_alpha):.4f}]")
        print(f"{'='*55}\n")

    def _compute_loss(self, model, inputs, return_outputs):
        """
        覆盖 TRL 的 _compute_loss，在调用父类前注入逐样本计数误差权重。

        操作流程：
          1. 从 inputs 中弹出 count_error_weight（避免传入模型 forward）
          2. 将其转为与模型同设备的 FloatTensor，存储到 self._ncdpo_errors
          3. 调用父类 _compute_loss（此时 self.beta property 将返回 per-sample 张量）
          4. 清除 self._ncdpo_errors（恢复标量模式，避免 eval 时污染）
        """
        # 提取 NCDPO 专属字段（必须在父类前 pop，否则模型 forward 会收到未知 key）
        count_errors = inputs.pop("count_error_weight", None)

        if count_errors is not None:
            # 移至当前计算设备，确保与 delta_score 同设备做 element-wise 乘法
            device = self.accelerator.device
            self._ncdpo_errors = count_errors.float().to(device)
        else:
            self._ncdpo_errors = None

        try:
            # 调用父类：内部 self.beta 调用 property getter
            # → 若 _ncdpo_errors 非 None，返回形状 (B,) 的张量
            # → TRL 的 `-F.logsigmoid(self.beta * delta_score)` 变为逐样本计算
            result = super()._compute_loss(model, inputs, return_outputs)
        finally:
            # 无论成功或异常，都清除状态，确保下次 eval step 恢复为标量 β
            self._ncdpo_errors = None

        return result


# ===========================================================================
# 主函数
# ===========================================================================

def main():
    # ------------------------------------------------------------------
    # 1. 命令行参数解析
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Qwen3-VL 多模态偏好优化训练脚本 (TRL >= 0.29.1)"
    )
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["sft", "dpo", "rpo", "ncdpo"],
        help="偏好优化方法: sft=监督微调, dpo=直接偏好优化, rpo=鲁棒偏好优化, ncdpo=数值距离连续惩罚DPO"
    )
    parser.add_argument("--model_path",   type=str, required=True, help="预训练模型路径")
    parser.add_argument("--dataset_path", type=str, required=True, help="训练数据集路径 (.jsonl)")
    parser.add_argument("--output_dir",   type=str, required=True, help="模型检查点输出目录")
    parser.add_argument("--beta",         type=float, required=True, help="DPO KL 散度惩罚基准系数")
    parser.add_argument("--num_train_epochs",          type=int,   required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--learning_rate",               type=float, required=True)

    # 冻结策略
    parser.add_argument("--freeze_vit",    type=str, default="true",  choices=["true", "false"])
    parser.add_argument("--freeze_merger", type=str, default="true",  choices=["true", "false"])

    # NCDPO 专属超参数
    parser.add_argument(
        "--ncdpo_alpha", type=float, default=1.0,
        help="[NCDPO] 动态 β 的幅值系数 α（默认 1.0，使 β 最大翻倍）"
    )
    parser.add_argument(
        "--ncdpo_eps_scale", type=float, default=2.0,
        help="[NCDPO] 计数误差的 tanh 归一化尺度 ε_scale（默认 2.0）"
    )

    args, _ = parser.parse_known_args()

    freeze_vit_bool    = args.freeze_vit.lower()    == "true"
    freeze_merger_bool = args.freeze_merger.lower() == "true"

    print(f"训练参数: {args}")

    # 梯度异常检测（调试用，生产环境可关闭）
    torch.autograd.set_detect_anomaly(True)

    # ------------------------------------------------------------------
    # 2. 模型加载
    # ------------------------------------------------------------------
    config = AutoConfig.from_pretrained(args.model_path)
    if hasattr(config.text_config, "rope_parameters") and \
            getattr(config.text_config, "rope_scaling", None) is None:
        print("检测到 transformers 版本配置差异，正在适配 rope_parameters → rope_scaling ...")
        config.text_config.rope_scaling = config.text_config.rope_parameters

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map=None,  # ZeRO-3 必须为 None
    )
    model = freeze_multimodal_components(model, freeze_vit=freeze_vit_bool, freeze_merger=freeze_merger_bool)

    ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map=None,
    )

    processor = AutoProcessor.from_pretrained(args.model_path)

    # ------------------------------------------------------------------
    # 3. 数据集加载与预处理
    # ------------------------------------------------------------------
    train_dataset = load_dataset(
        "json",
        data_files={"train": args.dataset_path},
        split="train",
    )

    # num_proc=1：避免多进程 PIL 序列化问题
    # HuggingFace .map() 会保留函数未返回的列（如 count_error_weight）
    train_dataset = train_dataset.map(
        format_qwen_dpo_dataset,
        num_proc=1,
        desc="格式化数据集（加载图片）",
    )

    # ------------------------------------------------------------------
    # 4. 公共训练参数（所有方法共用）
    # ------------------------------------------------------------------
    common_config = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_length=None,                # VLM 必须禁用截断（避免图像 token 不对齐）
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        learning_rate=args.learning_rate,
        beta=args.beta,
        save_strategy="epoch",
        save_only_model=True,
        save_total_limit=1,
        eval_strategy="no",
        logging_steps=1,
        max_grad_norm=1.0,
        precompute_ref_log_probs=False, # ZeRO-3 必须为 False
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
    )

    # ------------------------------------------------------------------
    # 5. 按方法构建 Trainer
    # ------------------------------------------------------------------

    if args.method == "sft":
        # ------ SFT（监督微调）------
        # loss_type="sft"：只在 chosen 上计算 NLL 损失
        training_args = DPOConfig(loss_type="sft", **common_config)
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=processor,
        )

    elif args.method == "dpo":
        # ------ DPO（直接偏好优化）------
        # L = -E[log σ(β * (logπ(y_w|x)/π_ref - logπ(y_l|x)/π_ref))]
        training_args = DPOConfig(loss_type="sigmoid", **common_config)
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=processor,
        )

    elif args.method == "rpo":
        # ------ RPO（鲁棒偏好优化）------
        # L_RPO = L_DPO(sigmoid) + rpo_alpha × L_SFT(chosen)
        # TRL 0.29.1 用 loss_type 列表实现多损失组合
        rpo_alpha = 1.0
        training_args = DPOConfig(
            loss_type=["sigmoid", "sft"],
            loss_weights=[1.0, rpo_alpha],
            **common_config,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=processor,
        )

    elif args.method == "ncdpo":
        # ------ NCDPO（数值距离连续惩罚 DPO）------
        #
        # 损失公式：
        #   L_NCDPO = -E[log σ(β_i × Δ_i)] + λ × L_SFT(chosen)
        #   β_i = β_base × (1 + α × tanh(ε_i / ε_scale))
        #   Δ_i = (logπ(y_w|x) - logπ_ref(y_w|x)) - (logπ(y_l|x) - logπ_ref(y_l|x))
        #
        # 数据集要求：
        #   使用 src/dataset/DPO_data_construct/augment_ncdpo_weights.py
        #   生成的 dpo_dataset_ncdpo.jsonl（含 count_error_weight 字段）
        #
        # 实现机制（TRL 0.29.1 兼容）：
        #   NCDPODataCollator: 将 count_error_weight 从 examples 透传到 batch
        #   NCDPOTrainer:      将 self.beta 重定义为 property，在 _compute_loss 时
        #                      注入 per-sample 张量，TRL 的损失计算对此透明
        rpo_alpha = 1.0
        training_args = DPOConfig(
            loss_type=["sigmoid", "sft"],   # 同 RPO：sigmoid DPO + SFT 正则
            loss_weights=[1.0, rpo_alpha],
            **common_config,
        )
        # 使用自定义 Collator 和 Trainer
        ncdpo_collator = NCDPODataCollator(processor=processor)
        trainer = NCDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=processor,
            data_collator=ncdpo_collator,
            ncdpo_alpha=args.ncdpo_alpha,
            ncdpo_eps_scale=args.ncdpo_eps_scale,
        )

    else:
        raise NotImplementedError(f"不支持的训练方法: {args.method}")

    # ------------------------------------------------------------------
    # 6. 启动训练
    # ------------------------------------------------------------------
    trainer.train()


if __name__ == "__main__":
    main()
