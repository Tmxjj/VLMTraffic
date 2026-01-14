import torch
from PIL import Image
import requests
import json
import base64
import time
from io import BytesIO
from configs.prompt_builder import PromptBuilder
from configs.model_config import MODEL_CONFIG
import re

# 需要安装的库：
# pip install transformers google openai requests

class VLMAgent:
    """
    支持多种后端的 VLM Agent：
    1. local_model: 加载本地 HuggingFace 模型
    2. sdk: 使用官方 SDK (OpenAI / Gemini)
    3. requests: 使用原生 HTTP 请求 (本地API端口 或 远程通用API)
    """
    def __init__(self, api_type=None, **kwargs):
        """
        Args:
            api_type: "local_model", "openai_sdk", "gemini_sdk", "requests". If None, use config default.
            kwargs: Override options from MODEL_CONFIG
        """
        # Determine API type: Arg > Config > Default
        self.api_type = api_type or MODEL_CONFIG.get("api_type", "local_model")
        
        # Load configuration: Global Defaults -> Backend Specific -> User Overrides
        backend_config = MODEL_CONFIG.get(self.api_type, {})
        self.config = {**MODEL_CONFIG, **backend_config, **kwargs}
        
        # Common generation parameters
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_new_tokens", 100) # Compatible with max_new_tokens logic if needed

        self.client = None
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        self._initialize_backend()

    def _initialize_backend(self):
        """根据配置初始化对应的后端"""
        print(f"Initializing VLMAgent with backend: {self.api_type}...")
        
        #  加载本地 模型 (支持 Qwen3-VL 和 通用 HF) ===
        if self.api_type == "local_model":
            model_path = self.config.get("model_path")
            device = self.config.get("device", "cuda")
            
            # 尝试作为 Qwen3-VL (ModelScope) 加载
            if model_path and ("qwen3" in model_path.lower()):
                try:
                    from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
                    self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_path, 
                        dtype="auto",
                        device_map=device, 
                        trust_remote_code=True
                    ).eval()
                    print("Local Qwen3-VL Model loaded successfully via ModelScope.")
                    return
                except Exception as e:
                    print(f"Failed to load Qwen3-VL via ModelScope: {e}")
                    print("Falling back to transformers...")

            # Fallback / Default Transformers Logic
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    device_map=device, 
                    trust_remote_code=True
                ).eval()
                print("Local HF Model loaded successfully.")
            except Exception as e:
                print(f"Failed to load local model: {e}")

        # 官方 SDK (OpenAI) 
        elif self.api_type == "openai_sdk":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.config.get("api_key"),
                base_url=self.config.get("base_url") # 可选，兼容本地 vLLM/Ollama
            )

        # 官方 SDK (Gemini)
        elif self.api_type == "gemini_sdk":
            from google import genai
            self.model = genai.Client(api_key=self.config.get("api_key"))
           

        # Python Requests (通用 HTTP / 本地端口)
        elif self.api_type == "requests":
            self.url = self.config.get("url")
            self.headers = self.config.get("headers", {"Content-Type": "application/json"})
            
        else:
            raise ValueError(f"Unsupported api_type: {self.api_type}")

    def _encode_image(self, image_path):
        """辅助函数：将图片转为 base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def _parse_action(self, response: str) -> int:
        """From VLM text response to action index."""
        try:
            # Look for patterns like "Action: [0]", "Action: 0", "**Action:** [0]"
            match = re.search(r"Action:?\s*\[?(\d+)\]?", response, re.IGNORECASE)
            if match:
                return int(match.group(1))
            else:
                print(f"Warning: Could not parse action from response: {response}. Defaulting to 0.")
                return 0 # Default fallback
        except Exception as e:
            print(f"Error parsing action: {e}. Defaulting to 0.")
            return 0

    def get_decision(self, image_path: str, prompt: str):
        """
        统一推理入口
        Output: 0, 1, 2, 3 (String)
        """
        response = "ERROR"
        start_time = time.perf_counter()
        try:
            # === 本地模型推理 ===
            if self.api_type == "local_model":
                # Check if using Processor (e.g. Qwen3-VL)
                if hasattr(self, 'processor') and self.processor is not None:
                    # Construct messages
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    
                    # Prepare inputs
                    inputs = self.processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True, # 是否在输入末尾添加生成提示（如<|im_start|>assistant\n）
                        return_dict=True,
                        return_tensors="pt"
                    )
                    inputs= inputs.to(self.model.device)
                    # Generate
                    gen_kwargs = {
                        "max_new_tokens": self.max_tokens,
                        "temperature": self.temperature
                    }
                    if self.temperature > 0:
                        gen_kwargs["do_sample"] = True
                    generated_ids = self.model.generate(**inputs, **gen_kwargs)
                    
                    # Trim input tokens
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0].strip()
                    
                else: 
                    # Default/Legacy logic (e.g. Qwen-VL v1, InternVL using AutoTokenizer)
                    query = self.tokenizer.from_list_format([
                        {'image': image_path},
                        {'text': prompt},
                    ])
                    inputs = self.tokenizer(query, return_tensors='pt')
                    inputs = inputs.to(self.model.device)
                    
                    gen_kwargs = {
                        "max_new_tokens": self.max_tokens,
                        "temperature": self.temperature
                    }
                    if self.temperature > 0:
                        gen_kwargs["do_sample"] = True

                    pred = self.model.generate(**inputs, **gen_kwargs)
                    raw_response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
                    response = raw_response.strip()
    
            # === OpenAI SDK ===
            elif self.api_type == "openai_sdk":
                base64_image = self._encode_image(image_path)
                api_resp = self.client.chat.completions.create(
                    model=self.config.get("model_name", "gpt-4-vision-preview"),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                response = api_resp.choices[0].message.content.strip()

            # === Gemini SDK === 
            elif self.api_type == "gemini_sdk":
                from google.genai import types
                model_name = self.config.get("model_name", "gemini-2.0-flash")
                img = Image.open(image_path)
                
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )

                api_resp = self.model.models.generate_content(
                    model=model_name, 
                    contents=[prompt, img],
                    config=config
                )            
                response = api_resp.text.strip()

            # === Requests ===
            elif self.api_type == "requests":
                base64_image = self._encode_image(image_path)
                payload = {
                    "model": self.config.get("model_name", "default"),
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{prompt} [IMAGE]" # 简化示例
                        }
                    ],
                    "image": base64_image,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
                
                req_resp = requests.post(self.url, headers=self.headers, json=payload)
                if req_resp.status_code == 200:
                    result = req_resp.json()
                    response = result['choices'][0]['message']['content']
                else:
                    response = f"Error: {req_resp.status_code}"

        except Exception as e:
            print(f"Inference Error: {e}")
            response = "ERROR"

        end_time = time.perf_counter()
        latency = end_time - start_time
        print(f"Inference Time: {end_time - start_time:.2f} seconds")

        action = self._parse_action(response)

        return response, latency, action


if __name__ == "__main__":

    prompt_text = PromptBuilder.build_decision_prompt(current_phase_id=1)
    test_img = "data/test/Hongkong_YMT/5/bev_aircraft_offline.jpg"

    # 使用 configs/model_config.py 中的默认配置初始化
    agent = VLMAgent()
    
    print(agent.get_decision(test_img, prompt_text))

    # 也可以动态覆盖参数
    # agent_custom = VLMAgent(api_type="requests", temperature=0.1)
