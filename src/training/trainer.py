'''
Author: yufei Ji
Date: 2026-01-14 16:43:47
LastEditTime: 2026-01-14 16:43:58
Description: this script is used to train the VLM model with different methods
FilePath: /VLMTraffic/src/training/trainer.py
'''


import argparse
import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer, KTOConfig, KTOTrainer, MDPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from ddpo_config import DDPOConfig
from accelerate import Accelerator
import os
import debugpy

# 从环境变量中读取是否开启调试
# 只有在主进程 (local_rank 0) 且环境变量设置为 "true" 时才启动调试器
if os.environ.get("ACCELERATE_LOCAL_RANK", "0") == "0" and os.environ.get("ENABLE_DEBUGPY") == "true":
    print("!!! DEBUGPY IS WAITING FOR ATTACHMENT !!!")
    # 允许远程连接
    debugpy.listen(("0.0.0.0", 5679))
    # 等待 VS Code 等客户端连接
    debugpy.wait_for_client()
    print("!!! DEBUGPY CLIENT ATTACHED !!!")


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

    # TODO:对于Qwen3-vl 加载模型和tokenizer需要特殊处理
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)    

    train_dataset = load_dataset(
        "parquet",                           # 指定文件格式为 "parquet"
        data_files={"train": dataset_path}, # 提供一个字典，将文件名映射到你想要创建的split名称
        split="train"                        # 从上一步创建的splits中选择 "train" split
    )

    if method == "sft":
        training_args = DPOConfig(
            loss_type=["sft"],
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=8,
            max_prompt_length=512,
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
            remove_unused_columns=False, # 防止移除我们需要的手动构建的 labels
        )

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=test_dataset,
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