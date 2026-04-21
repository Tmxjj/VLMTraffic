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

    def get_batch_decision(self, image_paths_list: list, prompts: list):
        """支持多图多Prompt的Batch并发推理。

        Args:
            image_paths_list: List[List[str]] — 每个元素是一个路口的多张图像路径列表（8张：4进口道+4上游）
            prompts: List[str] — 每个路口对应的 prompt
        """
        if not image_paths_list or not prompts:
            return []

        # ==========================================
        # 1. 针对 vLLM / API 请求：使用多线程触发服务端 Continuous Batching
        # ==========================================
        if self.api_type == "requests":
            start_time = time.perf_counter()
            results = [None] * len(image_paths_list)

            max_workers = min(self.batch_size, len(image_paths_list)) if self.batch_size > 0 else len(image_paths_list)
            logger.info(f"[EVAL] Starting ThreadPoolExecutor with {max_workers} workers for vLLM API batching.")

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(self.get_decision, imgs, p): idx
                    for idx, (imgs, p) in enumerate(zip(image_paths_list, prompts))
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"[EVAL] Batch API item {idx} failed: {e}")
                        results[idx] = ("ERROR", 0.0, (0, 25), None)

            total_latency = time.perf_counter() - start_time
            logger.info(f"[EVAL] API Batch Inference (Size: {len(image_paths_list)}) completed in {total_latency:.2f}s total.")
            return results

        # ==========================================
        # 2. 针对 HuggingFace 本地模型：使用原生的张量 Batch (Padding)
        # ==========================================
        elif self.api_type == "local_model" and hasattr(self, 'processor') and self.processor is not None:
            start_time = time.perf_counter()
            try:
                messages_batch = []
                for img_paths, p in zip(image_paths_list, prompts):
                    # 每个路口有多张图像
                    if isinstance(img_paths, (list, tuple)):
                        content = [{"type": "image", "image": ip} for ip in img_paths]
                    else:
                        content = [{"type": "image", "image": img_paths}]
                    content.append({"type": "text", "text": p})
                    messages_batch.append([{"role": "user", "content": content}])

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

                latency = (time.perf_counter() - start_time) / len(image_paths_list)
                batch_results = []
                for resp in responses:
                    resp = resp.strip()
                    action = self._parse_action(resp)
                    batch_results.append((resp, latency, action, None))

                logger.info(f"[EVAL] Tensor Batch Inference (Size: {len(image_paths_list)}) completed in {latency * len(image_paths_list):.2f}s total.")
                return batch_results

            except Exception as e:
                logger.error(f"[EVAL] Tensor Batch Inference Failed: {e}")
                return [("ERROR", 0.0, (0, 25), None)] * len(image_paths_list)

        # ==========================================
        # 3. 其他情况 (OpenAI/Gemini 且未要求并发时)，降级为串行循环
        # ==========================================
        else:
            logger.info("[EVAL] Falling back to sequential processing.")
            results = []
            for imgs, p in zip(image_paths_list, prompts):
                results.append(self.get_decision(imgs, p))
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
    def _execute_inference(self, image_paths, prompt):
        """仅包含网络/模型请求的核心逻辑。

        Args:
            image_paths: str 或 List[str]，支持单图或多图（8张）输入
        """
        # 统一为列表
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if self.api_type == "local_model":
            if hasattr(self, 'processor') and self.processor is not None:
                content = [{"type": "image", "image": ip} for ip in image_paths]
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
                inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
                inputs = inputs.to(self.model.device)
                gen_kwargs = {"max_new_tokens": self.max_tokens, "temperature": self.temperature, "do_sample": self.temperature > 0}
                return self.model.generate(**inputs, **gen_kwargs), inputs
            else:
                fmt = [{'image': ip} for ip in image_paths] + [{'text': prompt}]
                query = self.tokenizer.from_list_format(fmt)
                inputs = self.tokenizer(query, return_tensors='pt').to(self.model.device)
                return self.model.generate(**inputs, max_new_tokens=self.max_tokens, temperature=self.temperature), inputs

        elif self.api_type == "openai_sdk":
            content = [{"type": "text", "text": prompt}]
            for ip in image_paths:
                b64 = self._encode_image(ip)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            return self.client.chat.completions.create(
                model=self.config.get("model_name", "gpt-4-vision-preview"),
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_tokens, temperature=self.temperature
            )

        elif self.api_type == "gemini_sdk":
            from google.genai import types
            imgs = [Image.open(ip) for ip in image_paths]
            config = types.GenerateContentConfig(
                temperature=self.temperature, max_output_tokens=self.max_tokens,
                thinking_config=types.ThinkingConfig(include_thoughts=True)
            )
            contents = [prompt] + imgs
            return self.model.models.generate_content(model=self.config.get("model_name", "gemini-3-pro-preview"), contents=contents, config=config)

        elif self.api_type == "requests":
            content = [{"type": "text", "text": prompt}]
            for ip in image_paths:
                b64 = self._encode_image(ip)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            payload = {
                "model": self.config.get("model_name", "qwen3-vl-4b"),
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            resp = requests.post(self.url, headers=self.headers, json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json()

    def get_decision(self, image_paths, prompt: str):
        """多图推理主入口。

        Args:
            image_paths: str 或 List[str]，支持单图或8张多视角图像
        """
        response, thought = "ERROR", None
        input_tokens, output_tokens = 0, 0
        start_time = time.perf_counter()

        try:
            raw_result = self._execute_inference(image_paths, prompt)

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

    def _parse_action(self, response: str) -> tuple:
        """解析 VLM 输出的动作，返回 (phase_id, green_duration) 元组。

        支持新格式: Action: phase=X, duration=Y
        兼容旧格式: Action: X
        duration 必须在候选集 [10, 15, 20, 25, 30, 35] 内，否则取最近合法值。
        """
        from src.utils.tsc_env.tsc_wrapper import GREEN_DURATION_CANDIDATES

        # 新格式：phase=X, duration=Y
        new_fmt = re.search(r"Action:.*?phase\s*=\s*(\d+).*?duration\s*=\s*(\d+)", response, re.IGNORECASE | re.DOTALL)
        if new_fmt:
            phase_id = int(new_fmt.group(1))
            duration = int(new_fmt.group(2))
            # 取最近候选值
            duration = min(GREEN_DURATION_CANDIDATES, key=lambda x: abs(x - duration))
            return phase_id, duration

        # 兼容旧格式：Action: X
        old_fmt = re.search(r"Action:?\s*\[?(\d+)\]?", response, re.IGNORECASE)
        if old_fmt:
            return int(old_fmt.group(1)), GREEN_DURATION_CANDIDATES[len(GREEN_DURATION_CANDIDATES) // 2]

        return 0, GREEN_DURATION_CANDIDATES[len(GREEN_DURATION_CANDIDATES) // 2]