'''
Author: yufei Ji
Date: 2026-04-11
Description: Simulation-Grounded Dual-Verifiable RLVR 训练主脚本。
             使用 TRL GRPOTrainer 对 Qwen3-VL 进行在线强化学习微调。

    核心流程：
      1. 加载 SFT/DPO checkpoint 作为初始策略模型
      2. 对同一 BEV 图像采样 G=8 个完整 CoT 响应
      3. 用双路可验证奖励（r_perc + r_env）对每个响应打分
      4. 计算组内相对优势 A_i = (r_i - mean(r)) / std(r)
      5. GRPO 梯度更新策略模型，无需 Critic 网络和 Value Model

    奖励信号（见 rlvr_reward.py）：
      r_perc : -clamp(|pred_count - gt_count| / gt_count, 0, 1)  ∈ [-1, 0]
      r_env  : phase_pressure[selected_phase] / max_pressure      ∈ [0, 1]
      r      : α × r_perc + β × r_env                           ∈ [-0.7, 1.0]

    模型参数冻结策略：
      - 默认冻结 ViT（visual.patch_embed / pos_embed / blocks）
      - 默认冻结 Merger（visual.merger / deepstack_merger_list）
      - LoRA 适配 LLM 主干的 q_proj / k_proj / v_proj / o_proj

    数据集格式（由 rlvr_data_collector.py 生成）：
      每行 JSONL：
        sample_id, scenario_key, route_file, junction_id, step,
        current_phase_id, image_path, prompt,
        gt_per_movement, gt_total_all, phase_pressure, optimal_phase

FilePath: /VLMTraffic/src/training/rlvr/rlvr_grpo_trainer.py

依赖版本（在 VLMTraffic 虚拟环境基础上额外需要）：
    trl >= 0.29.1         (已要求，需确认 GRPOTrainer 可用)
    peft >= 0.14.0        (LoRA 支持，pip install peft>=0.14.0)
    transformers >= 4.56.2
    accelerate >= 1.4.0

运行方式：
    accelerate launch --config_file configs/accelerate_config.yaml \\
        src/training/rlvr/rlvr_grpo_trainer.py \\
        --model_path /root/autodl-tmp/model/qwen3_vl_8b_sft \\
        --dataset_path /root/autodl-tmp/rlvr_dataset/rlvr_train.jsonl \\
        --output_dir /root/autodl-tmp/results/checkpoints_rlvr \\
        --num_generations 8 --alpha 0.7 --beta_reward 0.3
'''

import argparse
import json
import os
import sys

import PIL.Image
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

# 将项目根目录加入 sys.path，以便导入 rlvr_reward
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.training.rlvr.rlvr_reward import batch_rlvr_reward_fn  # noqa: E402


# ===========================================================================
# 参数冻结（与 trainer.py 保持一致）
# ===========================================================================

def freeze_multimodal_components(
    model,
    freeze_vit: bool = True,
    freeze_merger: bool = True,
) -> torch.nn.Module:
    """
    冻结 Qwen3-VL 多模态组件（ViT 编码器 + Merger 投影层）。

    RLVR 阶段仅更新 LLM 主干（+ LoRA 适配器），视觉组件保持 SFT/DPO
    训练后的权重不变，既节省显存又防止视觉特征退化。

    Args:
        model      : Qwen3VLForConditionalGeneration 实例
        freeze_vit : 是否冻结 ViT patch_embed / pos_embed / blocks
        freeze_merger : 是否冻结 Merger / deepstack_merger_list

    Returns:
        model (原地修改参数的 requires_grad，直接返回同一对象)
    """
    print("\n================ 开始配置参数冻结 ================")

    # ViT 相关参数前缀
    vit_keywords = ["visual.patch_embed", "visual.pos_embed", "visual.blocks"]
    # Merger / Projector 相关参数前缀
    merger_keywords = ["visual.merger", "visual.deepstack_merger_list"]

    frozen_vit_params = 0
    frozen_merger_params = 0

    for name, param in model.named_parameters():
        if freeze_vit and any(kw in name for kw in vit_keywords):
            param.requires_grad = False
            frozen_vit_params += param.numel()
        elif freeze_merger and any(kw in name for kw in merger_keywords):
            param.requires_grad = False
            frozen_merger_params += param.numel()

    print(f"✅ 冻结配置完成:")
    print(f"   - 冻结 ViT:    {freeze_vit} ({frozen_vit_params / 1e6:.2f} M 参数)")
    print(f"   - 冻结 Merger: {freeze_merger} ({frozen_merger_params / 1e6:.2f} M 参数)")
    print("==================================================\n")
    return model


# ===========================================================================
# LoRA 配置
# ===========================================================================

def apply_lora(
    model,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> torch.nn.Module:
    """
    对 LLM 主干的注意力投影层施加 LoRA 适配器。

    LoRA 仅更新少量参数（约 0.1% 参数量），适合 GRPO 阶段在线策略更新，
    防止灾难性遗忘，同时保持 VRAM 占用可控。

    Args:
        model       : 已冻结多模态组件的 Qwen3VL 模型
        lora_r      : LoRA 秩（默认 16，平衡表达力与参数量）
        lora_alpha  : LoRA 缩放系数（默认 32）
        lora_dropout: LoRA dropout（默认 0.05，防止过拟合）

    Returns:
        包含 LoRA 适配器的 PeftModel
    """
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # 仅对 LLM 主干的注意力投影层施加 LoRA，跳过视觉组件
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ===========================================================================
# 数据集加载与预处理
# ===========================================================================

def load_rlvr_dataset(dataset_path: str) -> Dataset:
    """
    加载 rlvr_data_collector.py 生成的 JSONL 数据集，转换为 HuggingFace Dataset。

    JSONL 每行字段：
        sample_id       (str)  : 样本 ID
        scenario_key    (str)  : 场景名称，如 "JiNan"
        image_path      (str)  : BEV 图像绝对路径
        prompt          (list) : 消息列表（含 {"type": "image"} 内容项）
        gt_total_all    (float): GT 车辆总数（来自 jam_length_vehicle）
        phase_pressure  (dict) : {phase_idx_str: pressure_float}
        optimal_phase   (int)  : MaxPressure 推荐的最优相位

    处理步骤：
        1. 读取 JSONL 每行，提取所需字段
        2. 将 prompt 中的图像路径替换为 {"type": "image"} 占位符
        3. 加载 PIL 图像存入 images 列表
        4. 将 phase_pressure 序列化为 JSON 字符串（Dataset 不支持 dict 列）

    Args:
        dataset_path (str): JSONL 文件路径

    Returns:
        HuggingFace Dataset，含以下列：
            prompt, images, gt_total_all, phase_pressure, sample_id
    """
    records = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  第 {line_idx + 1} 行 JSON 解析失败，跳过: {e}")
                continue

            # --- 提取图像路径并加载 PIL 图像 ---
            image_path = sample.get("image_path", "")
            if not image_path or not os.path.exists(image_path):
                print(f"⚠️  样本 {sample.get('sample_id', '?')} 图像不存在: {image_path}，跳过")
                continue

            try:
                pil_image = PIL.Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"⚠️  样本 {sample.get('sample_id', '?')} 图像加载失败: {e}，跳过")
                continue

            # --- 清理 prompt 中的图像路径（替换为纯占位符）---
            prompt_messages = sample.get("prompt", [])
            cleaned_prompt = _clean_prompt_image_paths(prompt_messages)

            # --- phase_pressure 序列化为 JSON 字符串（Dataset 要求列类型统一）---
            phase_pressure_raw = sample.get("phase_pressure", {})
            if isinstance(phase_pressure_raw, dict):
                phase_pressure_str = json.dumps(phase_pressure_raw)
            else:
                phase_pressure_str = str(phase_pressure_raw)

            records.append({
                "sample_id":     sample.get("sample_id", f"sample_{line_idx}"),
                "prompt":        cleaned_prompt,
                "images":        [pil_image],          # 每条样本恰好一张 BEV 图
                "gt_total_all":  float(sample.get("gt_total_all", 0.0)),
                "phase_pressure": phase_pressure_str,  # JSON 字符串，奖励函数内反序列化
            })

    if not records:
        raise ValueError(f"数据集为空或全部样本解析失败: {dataset_path}")

    print(f"✅ 数据集加载完成: {len(records)} 条样本（来自 {dataset_path}）")
    return Dataset.from_list(records)


def _clean_prompt_image_paths(prompt_messages: list) -> list:
    """
    将 prompt 消息列表中的图像路径键（"image": "/path/..."）清理掉，
    只保留 {"type": "image"} 占位符。

    TRL 的 GRPOTrainer 会在 collate 阶段将 images 列中的 PIL 对象
    注入到消息内容的 {"type": "image"} 占位符，因此 prompt 里不应
    再携带路径字符串，避免与 PIL 注入冲突。

    Args:
        prompt_messages (list): 原始消息列表

    Returns:
        清理后的消息列表（原地修改副本）
    """
    import copy
    cleaned = copy.deepcopy(prompt_messages)
    for msg in cleaned:
        if not isinstance(msg.get("content"), list):
            continue
        for content_item in msg["content"]:
            if content_item.get("type") == "image":
                # 移除路径键，只保留 {"type": "image"} 占位符
                content_item.pop("image", None)
    return cleaned


# ===========================================================================
# 奖励函数包装器
# ===========================================================================

def make_reward_fn(alpha: float, beta_reward: float):
    """
    生成绑定了 alpha/beta 超参数的奖励函数闭包，满足 TRL GRPOTrainer 接口。

    TRL 调用约定：
        reward_fn(completions, **kwargs) -> list[float]
    其中 kwargs 由数据集中的同名列自动注入。本函数利用 gt_total_all
    和 phase_pressure 两列作为验证信号。

    Args:
        alpha       (float): r_perc 权重（感知奖励）
        beta_reward (float): r_env 权重（效率奖励）

    Returns:
        符合 TRL 接口的奖励函数
    """
    def _reward_fn(
        completions: list[str],
        gt_total_all: list,
        phase_pressure: list,
        **kwargs,
    ) -> list[float]:
        """
        TRL GRPOTrainer 调用的批量奖励函数。

        Args:
            completions    : G 个模型生成响应（已解码字符串列表）
            gt_total_all   : 对应 batch 的 GT 车辆总数列表
            phase_pressure : 对应 batch 的相位压力字典列表（JSON 字符串）

        Returns:
            list[float]: 长度 len(completions) 的奖励分值
        """
        return batch_rlvr_reward_fn(
            completions=completions,
            gt_total_all=gt_total_all,
            phase_pressure=phase_pressure,
            alpha=alpha,
            beta=beta_reward,
        )

    # 为 TRL 识别 kwargs 列名提供函数名（便于日志追踪）
    _reward_fn.__name__ = f"rlvr_reward_a{alpha}_b{beta_reward}"
    return _reward_fn


# ===========================================================================
# 命令行参数解析
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数，覆盖所有可配置的训练超参数。"""
    parser = argparse.ArgumentParser(
        description="Simulation-Grounded Dual-Verifiable RLVR 训练（GRPO 范式）"
    )

    # --- 路径参数 ---
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="初始策略模型路径（建议使用 SFT/DPO checkpoint）",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="RLVR 训练数据集路径（rlvr_data_collector.py 生成的 JSONL）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="训练结果和 checkpoint 保存目录",
    )

    # --- GRPO 核心超参数 ---
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="每个 prompt 采样的响应数量 G（默认 8，GRPO 组内对比）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="模型生成响应的最大 token 数（默认 512，CoT 推理链）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="采样温度（默认 0.9，鼓励探索多样化 CoT）",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="nucleus sampling top-p（默认 0.95）",
    )
    parser.add_argument(
        "--kl_coef",
        type=float,
        default=0.04,
        help="KL 散度惩罚系数（防止策略过度偏离参考模型，默认 0.04）",
    )

    # --- 奖励权重 ---
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="r_perc 感知奖励权重（默认 0.7）",
    )
    parser.add_argument(
        "--beta_reward",
        type=float,
        default=0.3,
        help="r_env 效率奖励权重（默认 0.3）",
    )

    # --- 训练超参数 ---
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="训练轮数（默认 3）",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="每卡训练 batch size（GRPO 显存占用高，建议设为 1）",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="梯度累积步数（默认 8，有效 batch = 8 × num_gpus）",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="学习率（GRPO 训练建议使用较小值，默认 1e-6）",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="学习率 warmup 比例（默认 5%）",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="每 N 步保存一次 checkpoint（默认 50）",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=5,
        help="每 N 步打印一次训练日志（默认 5）",
    )

    # --- LoRA 超参数 ---
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA 秩（默认 16）",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA 缩放系数（默认 32）",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout（默认 0.05）",
    )

    # --- 参数冻结策略 ---
    parser.add_argument(
        "--freeze_vit",
        type=str,
        default="true",
        choices=["true", "false"],
        help="是否冻结 ViT 编码器（默认 true）",
    )
    parser.add_argument(
        "--freeze_merger",
        type=str,
        default="true",
        choices=["true", "false"],
        help="是否冻结 Merger 投影层（默认 true）",
    )

    # --- 其他 ---
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="使用 bfloat16 混合精度训练（默认开启）",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=2048,
        help="prompt 最大 token 长度（默认 2048）",
    )

    return parser.parse_args()


# ===========================================================================
# 主训练入口
# ===========================================================================

def main():
    """GRPO RLVR 训练主函数。"""
    args = parse_args()

    # -------------------------------------------------------------------------
    # Step 1：加载基础模型（使用 SFT/DPO checkpoint 初始化策略）
    # -------------------------------------------------------------------------
    print(f"\n[Step 1] 加载基础模型: {args.model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # -------------------------------------------------------------------------
    # Step 2：冻结视觉组件（ViT + Merger），仅训练 LLM 主干
    # -------------------------------------------------------------------------
    print("\n[Step 2] 配置参数冻结策略")
    freeze_vit = args.freeze_vit.lower() == "true"
    freeze_merger = args.freeze_merger.lower() == "true"
    model = freeze_multimodal_components(model, freeze_vit, freeze_merger)

    # -------------------------------------------------------------------------
    # Step 3：施加 LoRA 适配器
    # -------------------------------------------------------------------------
    print("\n[Step 3] 施加 LoRA 适配器")
    model = apply_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # -------------------------------------------------------------------------
    # Step 4：加载 RLVR 训练数据集
    # -------------------------------------------------------------------------
    print(f"\n[Step 4] 加载 RLVR 数据集: {args.dataset_path}")
    dataset = load_rlvr_dataset(args.dataset_path)

    # -------------------------------------------------------------------------
    # Step 5：配置 GRPOConfig 训练参数
    # -------------------------------------------------------------------------
    print("\n[Step 5] 配置 GRPOConfig")
    grpo_config = GRPOConfig(
        # --- 输出路径 ---
        output_dir=args.output_dir,

        # --- GRPO 核心参数 ---
        num_generations=args.num_generations,     # G=8：每 prompt 采样 8 个响应
        max_new_tokens=args.max_new_tokens,       # CoT 推理链最大长度
        temperature=args.temperature,             # 采样温度（探索性）
        top_p=args.top_p,
        kl_coef=args.kl_coef,                    # KL 惩罚（防策略偏离过大）

        # --- 训练超参数 ---
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,

        # --- 精度与内存 ---
        bf16=args.bf16,
        fp16=False,

        # --- 日志与保存 ---
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,              # 只保留最近 3 个 checkpoint 节省磁盘
        report_to="none",                # 不使用 wandb/tensorboard（可按需修改）

        # --- 数据集列透传配置 ---
        # GRPOTrainer 会将这些列名的数据注入到奖励函数的 **kwargs 中
        reward_fn_kwargs_dataset_columns=["gt_total_all", "phase_pressure"],

        # --- Prompt 长度限制 ---
        max_prompt_length=args.max_prompt_length,

        # --- 其他 ---
        dataloader_num_workers=2,
        remove_unused_columns=False,     # 保留 gt_total_all / phase_pressure 列
    )

    # -------------------------------------------------------------------------
    # Step 6：构建奖励函数
    # -------------------------------------------------------------------------
    print(f"\n[Step 6] 构建双路可验证奖励函数 (α={args.alpha}, β={args.beta_reward})")
    reward_fn = make_reward_fn(alpha=args.alpha, beta_reward=args.beta_reward)

    # -------------------------------------------------------------------------
    # Step 7：实例化 GRPOTrainer 并启动训练
    # -------------------------------------------------------------------------
    print("\n[Step 7] 初始化 GRPOTrainer")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=processor,
        reward_funcs=[reward_fn],        # 双路联合奖励（单个函数返回 r = α·r_perc + β·r_env）
        train_dataset=dataset,
    )

    print("\n" + "=" * 60)
    print("  开始 RLVR GRPO 训练")
    print(f"  模型路径:        {args.model_path}")
    print(f"  数据集路径:      {args.dataset_path}")
    print(f"  输出目录:        {args.output_dir}")
    print(f"  样本数量:        {len(dataset)}")
    print(f"  每 prompt 采样:  G = {args.num_generations}")
    print(f"  奖励权重:        α(r_perc)={args.alpha}, β(r_env)={args.beta_reward}")
    print(f"  LoRA 秩:         r = {args.lora_r}")
    print(f"  学习率:          {args.learning_rate}")
    print("=" * 60 + "\n")

    trainer.train()

    # -------------------------------------------------------------------------
    # Step 8：保存最终模型
    # -------------------------------------------------------------------------
    print(f"\n[Step 8] 保存最终模型到: {args.output_dir}/final_model")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_model"))
    print("✅ RLVR GRPO 训练完成！")


if __name__ == "__main__":
    main()
