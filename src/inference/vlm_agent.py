import torch
from PIL import Image
import requests
import json
import base64
import time
import re
import random
from io import BytesIO
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import concurrent.futures

from configs.prompt_builder import PromptBuilder
from configs.model_config import MODEL_CONFIG

def is_api_retryable_error(exception):
    """定义哪些错误触发重试：503, 429 以及明确的 Overloaded 提示"""
    err_msg = str(exception).lower()
    retryable_conditions = [
        "503", 
        "429", 
        "unavailable", 
        "overloaded", 
        "rate limit",
        "deadline exceeded"
    ]
    return any(condition in err_msg for condition in retryable_conditions)

class VLMAgent:
    def __init__(self, api_type=None, batch_size=1, **kwargs):
        self.api_type = api_type or MODEL_CONFIG.get("api_type", "local_model")
        self.batch_size = batch_size  # 新增：并发推理数量
        backend_config = MODEL_CONFIG.get(self.api_type, {})
        self.config = {**MODEL_CONFIG, **backend_config, **kwargs}
        
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_new_tokens", 100)

        self.client = None
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        self._initialize_backend()

    def _initialize_backend(self):
        logger.info(f"[EVAL] Initializing VLMAgent with backend: {self.api_type}...")
        
        if self.api_type == "local_model":
            model_path = self.config.get("model_path")
            device = self.config.get("device", "cuda")
            
            if model_path and ("qwen3" in model_path.lower()):
                try:
                    from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
                    self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_path, dtype="auto", device_map=device, trust_remote_code=True
                    ).eval()
                    logger.info("[EVAL] Local Qwen3-VL Model loaded.")
                    return
                except Exception as e:
                    logger.warning(f"[EVAL] ModelScope load failed: {e}")

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map=device, trust_remote_code=True
                ).eval()
            except Exception as e:
                logger.error(f"[EVAL] Failed to load local model: {e}")

        elif self.api_type == "openai_sdk":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.get("api_key"), base_url=self.config.get("base_url"))

        elif self.api_type == "gemini_sdk":
            from google import genai
            self.model = genai.Client(api_key=self.config.get("api_key"))

        elif self.api_type == "requests":
            self.url = self.config.get("url")
            self.headers = self.config.get("headers", {"Content-Type": "application/json"})

    def get_batch_decision(self, image_paths: list, prompts: list):
        """支持多图多Prompt的Batch并发推理"""
        if not image_paths or not prompts:
            return []

        # ==========================================
        # 1. 针对 vLLM / API 请求：使用多线程触发服务端 Continuous Batching
        # ==========================================
        if self.api_type == "requests":
            start_time = time.perf_counter()
            results = [None] * len(image_paths) # 预占位，保证返回顺序与输入一致
            
            # 使用 batch_size 限制最大并发连接数（保护服务端不被压垮）
            max_workers = min(self.batch_size, len(image_paths)) if self.batch_size > 0 else len(image_paths)
            logger.info(f"[EVAL] Starting ThreadPoolExecutor with {max_workers} workers for vLLM API batching.")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_idx = {
                    executor.submit(self.get_decision, img, p): idx 
                    for idx, (img, p) in enumerate(zip(image_paths, prompts))
                }
                
                # 收集结果
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"[EVAL] Batch API item {idx} failed: {e}")
                        results[idx] = ("ERROR", 0.0, 0, None)

            total_latency = time.perf_counter() - start_time
            logger.info(f"[EVAL] API Batch Inference (Size: {len(image_paths)}) completed in {total_latency:.2f}s total.")
            return results

        # ==========================================
        # 2. 针对 HuggingFace 本地模型：使用原生的张量 Batch (Padding)
        # ==========================================
        elif self.api_type == "local_model" and hasattr(self, 'processor') and self.processor is not None:
            start_time = time.perf_counter()
            try:
                messages_batch = []
                for img_path, p in zip(image_paths, prompts):
                    messages_batch.append([
                        {"role": "user", "content": [
                            {"type": "image", "image": img_path}, 
                            {"type": "text", "text": p}
                        ]}
                    ])

                if self.processor.tokenizer.pad_token_id is None:
                    self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
                    
                inputs = self.processor.apply_chat_template(
                    messages_batch, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    return_dict=True, 
                    return_tensors="pt",
                    padding=True
                ).to(self.model.device)

                gen_kwargs = {
                    "max_new_tokens": self.max_tokens, 
                    "temperature": self.temperature, 
                    "do_sample": self.temperature > 0
                }
                generated_ids = self.model.generate(**inputs, **gen_kwargs)

                trimmed_ids = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                responses = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)

                latency = (time.perf_counter() - start_time) / len(image_paths)
                batch_results = []
                for resp in responses:
                    resp = resp.strip()
                    action = self._parse_action(resp)
                    batch_results.append((resp, latency, action, None))
                
                logger.info(f"[EVAL] Tensor Batch Inference (Size: {len(image_paths)}) completed in {latency * len(image_paths):.2f}s total.")
                return batch_results

            except Exception as e:
                logger.error(f"[EVAL] Tensor Batch Inference Failed: {e}")
                return [("ERROR", 0.0, 0, None)] * len(image_paths)

        # ==========================================
        # 3. 其他情况 (OpenAI/Gemini 且未要求并发时)，降级为串行循环
        # ==========================================
        else:
            logger.info("[EVAL] Falling back to sequential processing.")
            results = []
            for img, p in zip(image_paths, prompts):
                results.append(self.get_decision(img, p))
            return results
        

    # --- 核心重试逻辑包装 (保留用于单路口或 API 调用) ---
    @retry(
        retry=retry_if_exception(is_api_retryable_error),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=lambda retry_state: logger.warning(
            f"[EVAL] API Error (Attempt {retry_state.attempt_number}). Retrying in {retry_state.next_action.sleep}s..."
        ),
        reraise=True
    )
    def _execute_inference(self, image_path, prompt):
        """仅包含网络/模型请求的核心逻辑"""
        if self.api_type == "local_model":
            if hasattr(self, 'processor') and self.processor is not None:
                messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
                inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
                inputs = inputs.to(self.model.device)
                gen_kwargs = {"max_new_tokens": self.max_tokens, "temperature": self.temperature, "do_sample": self.temperature > 0}
                return self.model.generate(**inputs, **gen_kwargs), inputs
            else:
                query = self.tokenizer.from_list_format([{'image': image_path}, {'text': prompt}])
                inputs = self.tokenizer(query, return_tensors='pt').to(self.model.device)
                return self.model.generate(**inputs, max_new_tokens=self.max_tokens, temperature=self.temperature), inputs

        elif self.api_type == "openai_sdk":
            base64_image = self._encode_image(image_path)
            return self.client.chat.completions.create(
                model=self.config.get("model_name", "gpt-4-vision-preview"),
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
                max_tokens=self.max_tokens, temperature=self.temperature
            )

        elif self.api_type == "gemini_sdk":
            from google.genai import types
            img = Image.open(image_path)
            config = types.GenerateContentConfig(
                temperature=self.temperature, max_output_tokens=self.max_tokens,
                thinking_config=types.ThinkingConfig(include_thoughts=True)
            )
            return self.model.models.generate_content(model=self.config.get("model_name", "gemini-3-pro-preview"), contents=[prompt, img], config=config)

        elif self.api_type == "requests":
            base64_image = self._encode_image(image_path)
            payload = {
                "model": self.config.get("model_name", "qwen3-vl-4b"),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            resp = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()

    def get_decision(self, image_path: str, prompt: str):
        """单图推理主入口"""
        response, thought = "ERROR", None
        input_tokens, output_tokens = 0, 0
        start_time = time.perf_counter()

        try:
            raw_result = self._execute_inference(image_path, prompt)

            if self.api_type == "local_model":
                gen_ids, inputs = raw_result
                input_tokens = inputs.input_ids.shape[1]
                if hasattr(self, 'processor') and self.processor:
                    trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
                    output_tokens = trimmed_ids[0].shape[0]
                    response = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()
                else:
                    output_tokens = gen_ids.shape[1] - input_tokens
                    response = self.tokenizer.decode(gen_ids.cpu()[0], skip_special_tokens=True).strip()

            elif self.api_type == "openai_sdk":
                response = raw_result.choices[0].message.content.strip()
                if raw_result.usage:
                    input_tokens, output_tokens = raw_result.usage.prompt_tokens, raw_result.usage.completion_tokens

            elif self.api_type == "gemini_sdk":
                res_parts, thought_parts = [], []
                if raw_result.candidates:
                    for part in raw_result.candidates[0].content.parts:
                        if hasattr(part, 'thought') and part.thought:
                            thought_parts.append(part.text)
                        else:
                            res_parts.append(part.text)
                response = "".join(res_parts).strip()
                thought = "\n".join(thought_parts).strip()
                if raw_result.usage_metadata:
                    input_tokens = raw_result.usage_metadata.prompt_token_count
                    output_tokens = raw_result.usage_metadata.candidates_token_count

            elif self.api_type == "requests":
                response = raw_result['choices'][0]['message']['content']
                input_tokens = raw_result.get('usage', {}).get('prompt_tokens', 0)
                output_tokens = raw_result.get('usage', {}).get('completion_tokens', 0)

        except Exception as e:
            logger.error(f"[EVAL] Inference Failed after retries: {e}")
            response = "ERROR"

        latency = time.perf_counter() - start_time
        logger.info(f"[EVAL] Latency: {latency:.2f}s | Tokens: {input_tokens} -> {output_tokens}")
        
        return response, latency, self._parse_action(response), thought

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _parse_action(self, response: str) -> int:
        match = re.search(r"Action:?\s*\[?(\d+)\]?", response, re.IGNORECASE)
        return int(match.group(1)) if match else 0