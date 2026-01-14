import torch
from PIL import Image
import requests
import json
import base64
import time
from io import BytesIO
from prompt_builder import PromptBuilder
# 需要安装的库：
# pip install transformers google openai requests

class VLMAgent:
    """
    支持多种后端的 VLM Agent：
    1. local_hf: 加载本地 HuggingFace 模型
    2. sdk: 使用官方 SDK (OpenAI / Gemini)
    3. requests: 使用原生 HTTP 请求 (本地API端口 或 远程通用API)
    """
    def __init__(self, api_type="local_hf", **kwargs):
        """
        Args:
            api_type: "local_hf", "openai_sdk", "gemini_sdk", "requests"
            kwargs: 根据 api_type 传入不同的配置参数
                - local_hf: model_path, device
                - openai_sdk: api_key, base_url, model_name
                - gemini_sdk: api_key, model_name
                - requests: url, headers, model_name
        """
        self.api_type = api_type
        self.config = kwargs
        self.client = None
        self.model = None
        self.tokenizer = None
        
        self._initialize_backend()

    def _initialize_backend(self):
        """根据配置初始化对应的后端"""
        print(f"Initializing VLMAgent with backend: {self.api_type}...")
        
        #  加载本地 HuggingFace 模型 ===
        if self.api_type == "local_model":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_path = self.config.get("model_path")
            device = self.config.get("device", "cuda")
            try:
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
                query = self.tokenizer.from_list_format([
                    {'image': image_path},
                    {'text': prompt},
                ])
                inputs = self.tokenizer(query, return_tensors='pt')
                inputs = inputs.to(self.model.device)
                pred = self.model.generate(**inputs, max_new_tokens=10)
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
                    max_tokens=10
                )
                response = api_resp.choices[0].message.content.strip()

            # === Gemini SDK ===
            elif self.api_type == "gemini_sdk":
                model_name = self.config.get("model_name", "gemini-2.0-flash")
                img = Image.open(image_path)
                api_resp = self.model.models.generate_content(
                    model=model_name, contents=[prompt, img]
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
                    "prompt": prompt
                }
                
                req_resp = requests.post(self.url, headers=self.headers, json=payload)
                if req_resp.status_code == 200:
                    response = req_resp.json().get("result", "").strip()
                else:
                    response = f"Error: {req_resp.status_code}"

        except Exception as e:
            print(f"Inference Error: {e}")
            response = "ERROR"

        end_time = time.perf_counter()
        latency = end_time - start_time
        print(f"Inference Time: {end_time - start_time:.2f} seconds")

        return response, latency


if __name__ == "__main__":

    prompt_text = PromptBuilder.build_decision_prompt(current_phase_id=1, phase_explanation=
        "- Phase 0: NS Straight\n"
        "- Phase 1: NS Left\n"
        "- Phase 2: EW Straight\n"      
        "- Phase 3: EW Left"
    )
    test_img = "data/test/Hongkong_YMT/1/bev_aircraft_offline.jpg"

    # 官方 SDK (gemini) 
    agent_sdk = VLMAgent(
        api_type="gemini_sdk",
        api_key="***",
        model_name="gemini-2.5-flash",
        
    )
    print(agent_sdk.get_decision(test_img, prompt_text))

    # Python Requests (调用本地封装的 FastAPI 端口)
    # agent_req = VLMAgent(
    #     api_type="requests",
    #     url="http://127.0.0.1:5000/predict",
    #     model_name="custom-vlm"
    # )
    # print(agent_req.get_decision(test_img, prompt_text))

    # 加载本地模型
    # agent_local = VLMAgent(
    #     api_type="local_model",
    #     model_path="/path/to/local/model"
    # )