'''
Author: yufei Ji
Date: 2026-04-14 22:11:46
LastEditTime: 2026-04-14 22:12:59
Description: this script is used to 
FilePath: /VLMTraffic/scripts/test_qwen_vl_tokens.py
--------------------------------------------------
分辨率类型        | 尺寸 (W x H)      | 总 Token 数  | 图像 Token 数 (估算)
--------------------------------------------------
1080p       | 1920x1080       | 2056        | 2042      
720p        | 1280x720        | 896         | 882       
1024*1024   | 1024x1024       | 1040        | 1026      
480p        | 640x480         | 316         | 302       
320p        | 480x320         | 166         | 152       
--------------------------------------------------
'''

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

def test_token_length():
    """
    测试不同分辨率图像在 Qwen3-VL-4B-Instruct 模型中的 token 占用情况。
    将分别测试: 1080p, 720p, 1024x1024, 720x720, 480p, 320p
    """
    model_path = "models/base_models/Qwen3-VL-4B-Instruct"
    print(f"正在加载处理器和模型 (这可能需要一些时间)...\n模型路径: {model_path}")
    
    try:
        # 加载 processor 和 tokenizer，通常计算 token 只需 processor
        processor = AutoProcessor.from_pretrained(model_path)
        
        # 如果只需要测试 token，可以选择不加载完整的模型权重以节省显存。
        # 这里按照需求加载模型，可以自动映射到合适的设备
        # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        print("加载成功！\n")
    except Exception as e:
        print(f"加载失败，请检查模型路径或当前 transformers 版本是否支持 Qwen3-VL。\n错误信息: {e}")
        return

    # 定义要测试的不同分辨率
    # (宽, 高)
    resolutions = {
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "1024*1024": (1024, 1024),
        "720*720": (720, 720),
        "480p": (640, 480), # 标准 480p 比例
        "320p": (480, 320)  # 标准 320p 比例
    }

    # 测试文本 prompt
    text_prompt = "请详细描述这张图片。"
    
    print("-" * 50)
    print(f"{'分辨率类型':<12} | {'尺寸 (W x H)':<15} | {'总 Token 数':<10} | {'图像 Token 数 (估算)':<10}")
    print("-" * 50)

    for res_name, (width, height) in resolutions.items():
        # 生成对应分辨率的纯色随机测试图像
        dummy_image = Image.new('RGB', (width, height), color=(73, 109, 137))
        
        # 构造对话格式 (Qwen-VL 通常的输入格式)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        try:
            # 准备模型的输入
            # 使用 processor 处理图像和文本
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[dummy_image],
                padding=True,
                return_tensors="pt"
            )
            
            # 获取总的 token 数量
            # input_ids 包含了文本和图像被处理后的所有 token
            total_tokens = inputs.input_ids.shape[1]
            
            # 为了估算图像占用的 token 数，可以计算仅输入文本时的 token 数
            # 文本 token 计算
            text_only_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]
            text_only_str = processor.apply_chat_template(text_only_messages, tokenize=False, add_generation_prompt=True)
            text_only_inputs = processor.tokenizer(text_only_str, return_tensors="pt")
            text_tokens = text_only_inputs.input_ids.shape[1]
            
            # 图像 token 数量 = 总 token 数量 - 纯文本 token 数量
            image_tokens = total_tokens - text_tokens
            
            print(f"{res_name:<11} | {str(width)+'x'+str(height):<15} | {total_tokens:<11} | {image_tokens:<10}")

        except Exception as e:
             print(f"{res_name:<11} | {str(width)+'x'+str(height):<15} | 处理失败: {str(e)}")

    print("-" * 50)

if __name__ == "__main__":
    test_token_length()
