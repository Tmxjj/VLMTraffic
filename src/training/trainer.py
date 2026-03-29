'''
Author: yufei Ji
Date: 2026-01-14 16:43:47
LastEditTime: 2026-01-30 10:55:21
Description: this script is used to train the VLM model with different methods
FilePath: /VLMTraffic/src/training/trainer.py
'''


import argparse
import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer, KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer,Qwen3VLForConditionalGeneration, AutoProcessor, AutoConfig
# from ddpo_config import DDPOConfig
from accelerate import Accelerator
import PIL.Image
import os
import debugpy
import sys

# # 同时兼容多种启动器的 Rank 环境变量
# local_rank = os.environ.get("LOCAL_RANK", os.environ.get("ACCELERATE_LOCAL_RANK", "0"))

# if not (hasattr(sys, 'gettrace') and sys.gettrace() is not None):
#     # 严格限定只有 local_rank 为 "0" 的主进程才能启动调试器
#     if str(local_rank) == "0" and os.environ.get("ENABLE_DEBUGPY") == "true":
#         import debugpy
#         print(f"!!! [Rank {local_rank}] DEBUGPY IS WAITING FOR ATTACHMENT ON PORT 5679 !!!")
#         debugpy.listen(("0.0.0.0", 5679))
#         debugpy.wait_for_client()
#         print(f"!!! [Rank {local_rank}] DEBUGPY CLIENT ATTACHED !!!")


def format_qwen_dpo_dataset(example):
    """
    用于 dataset.map() 的单条数据处理函数，
    将不完全规范的 chosen/rejected 转换为标准多模态对话格式。
    """
    # === 1. 处理 Prompt (读取图片并继承格式) ===
    prompt_messages = example["prompt"]
    images_list = []  # 初始化图片列表
    for msg in prompt_messages:
        if msg["role"] == "user":
            for content in msg["content"]:
                if content["type"] == "image" and isinstance(content["image"], str):
                    img_path = content["image"]
                    if os.path.exists(img_path):
                        # 读取为 PIL 对象，Qwen-VL 的 processor 需要这个
                        img = PIL.Image.open(img_path).convert("RGB")
                        content["image"] = img # 留在消息体内，兼容 Qwen
                        images_list.append(img) # 🟢 追加到独立列表中，兼容 TRL

                    else:
                        print(f"⚠️ 警告: 找不到图片 {img_path}")

    # === 2. 处理 Chosen (将其转化为 List of Dict 格式) ===
    # 取出原来 dict 里的字符串
    chosen_raw_text = example["chosen"][0]["content"] 
    chosen_messages = [
        {
            "role": "assistant", 
            "content": [
                {"type": "text", "text": chosen_raw_text}
            ]
        }
    ]

    # === 3. 处理 Rejected (将其转化为 List of Dict 格式) ===
    rejected_raw_text = example["rejected"][0]["content"]
    rejected_messages = [
        {
            "role": "assistant", 
            "content": [
                {"type": "text", "text": rejected_raw_text}
            ]
        }
    ]

    return {
        "prompt": prompt_messages,
        "chosen": chosen_messages,
        "rejected": rejected_messages,
        "images": images_list, 
    }
def main():
    # 1. 创建解析器并定义命令行参数
    parser = argparse.ArgumentParser(description="Run fine-tuning with different methods.")
    parser.add_argument("--method", type=str, required=True, choices=["sft", "dpo", "rpo", "nca", "kto", "ddpo","mdpo","mrpo"], help="The fine-tuning method to use.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training dataset file (e.g., .parquet).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output checkpoints.")
    parser.add_argument("--beta", type=float, required=True, help="Beta value.")
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)

    # 2. 解析参数
    args, _ = parser.parse_known_args()

    torch.autograd.set_detect_anomaly(True)

    print(args)

    # 3. 使用从命令行传入的参数
    method = args.method
    model_path = args.model_path
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    beta = args.beta
    num_train_epochs = args.num_train_epochs
    gradient_accumulation_steps= args.gradient_accumulation_steps
    per_device_train_batch_size= args.per_device_train_batch_size
    learning_rate= args.learning_rate

    # 对于Qwen3-vl 加载模型和tokenizer需要特殊处理
    config = AutoConfig.from_pretrained(model_path)
    
    # 补丁：处理 transformers 5.x 到 4.x 的版本兼容问题
    if hasattr(config.text_config, "rope_parameters") and getattr(config.text_config, "rope_scaling", None) is None:
        print("⚠️ 检测到 transformers 版本配置冲突，正在将 rope_parameters 转换为 rope_scaling...")
        # 把新版的 rope_parameters 原封不动地赋给旧版的 rope_scaling
        config.text_config.rope_scaling = config.text_config.rope_parameters

    # 🟢 加载训练模型 (Policy Model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16
    )

    # 🟢 加载参考模型 (Reference Model)
    # ZeRO-3 要求必须手动实例化，不能让 Trainer 自动创建
    ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_path)
    if hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    train_dataset = load_dataset(
        "json",                           # 指定文件格式为 "json"
        data_files={"train": dataset_path}, # 提供一个字典，将文件名映射到你想要创建的split名称
        split="train"                        # 从上一步创建的splits中选择 "train" split
    )
    # num_proc=8 开启多进程加速处理，remove_columns 丢掉多余的 id 等字段
    train_dataset = train_dataset.map(
        format_qwen_dpo_dataset,
        num_proc=8,
        desc="Formatting DPO dataset for Qwen3-VL"
    )

    if method == "sft":
        training_args = DPOConfig(
            loss_type=["sft"],
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=8,
            max_prompt_length=4096,
            # max_completion_length=512,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            learning_rate=learning_rate,
            beta=beta,
            save_strategy="epoch",
            save_only_model=True,
            save_total_limit=None,
            eval_strategy="steps",
            eval_steps=40,
            logging_steps=1,
        )

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=test_dataset
        )
    elif method == "dpo":
        training_args = DPOConfig(
            loss_type=["sigmoid"],
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=1,
            max_prompt_length=4096,
            #For VLMs, truncating may remove image tokens, leading to errors during training. To avoid this, set max_length=None in the DPOConfig. This allows the model to process the full sequence length without truncating image tokens. 但是在评测中 tokne为2376，避免显存溢出，暂时设置为4086，后续可以根据实际情况调整。
            max_length=4096, 
            max_completion_length=4096,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            learning_rate=learning_rate,
            beta=beta,
            save_strategy="epoch",
            save_only_model=True,
            save_total_limit=None,
            eval_strategy="no",
            # eval_steps=40,
            logging_steps=1,
            max_grad_norm=1.0,
            remove_unused_columns=True, # 防止移除我们需要的手动构建的 labels
            # precompute_ref_log_probs 不支持zero-3，因为 ref_model 的参数不在 Trainer 管理的范围内，设置为 True 会导致训练时找不到预计算的 log_probs，从而报错。对于 Zero-3，必须保持 precompute_ref_log_probs=False，让 Trainer 在每个训练步骤动态计算参考模型的 log_probs。
            precompute_ref_log_probs=False, 
            bf16=True,
        )

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            ref_model=ref_model,
            train_dataset=train_dataset,
            # eval_dataset=test_dataset,
            processing_class = processor,
        )
    elif method == 'mdpo':
        training_args = DPOConfig(
            loss_type=["sigmoid"],
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=1,
            max_prompt_length=4096,
            max_length=4096,
            max_completion_length=4096,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            learning_rate=learning_rate,
            beta=beta,
            save_strategy="epoch",
            save_only_model=True,
            save_total_limit=None,
            eval_strategy="no",
            # eval_steps=40,
            logging_steps=1,
            max_grad_norm=1.0,
            use_liger_loss=False
        )

        trainer = MDPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=test_dataset,
        )
    elif method == 'mrpo':
        training_args = DPOConfig(
            loss_type=["sigmoid"],
            rpo_alpha=1.0,
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=1,
            max_prompt_length=4096,
            max_length=4096,
            max_completion_length=4096,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            learning_rate=learning_rate,
            beta=beta,
            save_strategy="epoch",
            save_only_model=True,
            save_total_limit=None,
            eval_strategy="no",
            # eval_steps=40,
            logging_steps=1,
            max_grad_norm=1.0,
            use_liger_loss=False
        )

        trainer = MDPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=test_dataset,
        )
    elif method == "rpo":
        training_args = DPOConfig(
            loss_type=["sigmoid"],
            rpo_alpha=1.0,
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=1,
            max_prompt_length=7680,
            max_length = 8192,
            max_completion_length=512,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            learning_rate=learning_rate,
            beta=beta,
            save_strategy="epoch",
            save_only_model=True,
            save_total_limit= None,
            eval_strategy="no",
            eval_steps=40,
            logging_steps=1,
        )

        trainer = DPOTrainer(
            model=model,
            # ref_model=ref_model, #for zero-3
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=test_dataset,
        )
    elif method == "nca":
        training_args = DPOConfig(
            loss_type=["nca_pair"],
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=8,
            max_prompt_length=512,
            # max_completion_length=512,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            learning_rate=learning_rate,
            beta=beta,
            save_strategy="epoch",
            save_only_model=True,
            save_total_limit=1,
            eval_strategy="steps",
            eval_steps=40,
            logging_steps=1,
        )

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=test_dataset,
            tokenizer = tokenizer
        )
    elif method == "kto":
        training_args = KTOConfig(
            loss_type="kto",
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=8,
            max_prompt_length=512,
            max_completion_length=512,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            learning_rate=learning_rate,
            beta=beta,
            save_strategy="epoch",
            save_only_model=True,
            save_total_limit=1,
            eval_strategy="no",
            eval_steps=40,
            logging_steps=1,
        )

        trainer = KTOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
    elif method == "ddpo":
        ddpo_training_args = DDPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=8,
            num_generations=4,
            temperature=1.0,
            max_prompt_length=512,
            max_completion_length=512,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            learning_rate=learning_rate,
            beta=beta,
            save_strategy="epoch",
            save_only_model=True,
            save_total_limit=1,
            eval_strategy="steps",
            eval_steps=40,
            logging_steps=1,
        )

        trainer = DDPOTrainer(
            model=model,
            args=ddpo_training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
    else:
        raise NotImplementedError

    trainer.train()
    
if __name__ == "__main__":
    main()