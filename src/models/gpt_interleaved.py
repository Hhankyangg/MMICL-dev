import os
import io
import json
import base64
import time
import requests
from PIL import Image
from typing import List, Dict, Any, Union

# 引入 OpenAI SDK (基于 Script 2)
from openai import AzureOpenAI, OpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError, BadRequestError
from azure.identity import AzureCliCredential, ManagedIdentityCredential, ChainedTokenCredential, get_bearer_token_provider

# 假设基类定义 (保持与 GeminiGenerator 一致)
from src.models.base_model import BaseImageGenerator

class GPTImageInterleaved(BaseImageGenerator):
    def __init__(self, 
                 azure_endpoint: str,
                 api_key: str = None,
                 api_version: str = "2024-02-15-preview", # Script 1 config
                 deployment_name: str = "gpt-4.1",        # Script 2 model name
                 timeout: int = 120,
                 max_retries: int = 3,
                 max_image_size: int = 1500):             # Script 1 config
        
        super().__init__()
        
        self.deployment_name = deployment_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_image_size = max_image_size
        
        if api_key:
            # 方式 A: 使用 API Key (传统方式)
            print(f"[GPT] Initializing with API Key...")
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,  # <--- 直接传 Key
                api_version=api_version,
                timeout=timeout
            )
        else:
            # 方式 B: 使用 Azure AD Token (免密/企业方式)
            print(f"[GPT] Initializing with Azure AD Token (CLI/Managed Identity)...")
            scope = "api://trapi/.default"
            credential = get_bearer_token_provider(ChainedTokenCredential(
                AzureCliCredential(),
                ManagedIdentityCredential(),
            ), scope)
            
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                azure_ad_token_provider=credential, # <--- 传 Token Provider
                api_version=api_version,
                timeout=timeout
            )

    @property
    def is_api_model(self) -> bool:
        return True

    @property
    def can_generate_text(self) -> bool:
        return True

    def _resize_and_encode_image(self, image_path: str) -> str:
        """
        [Helper] 读取本地图片，调整大小(Script 1逻辑)，并转为Base64
        """
        if not os.path.exists(image_path):
            print(f"[GPT Warning] Image path not found: {image_path}")
            return None
            
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                
                # Resize logic from Script 1
                w, h = img.size
                if max(w, h) > self.max_image_size:
                    scale = self.max_image_size / max(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                buffered = io.BytesIO()
                img.save(buffered, format="PNG") # Keep PNG for quality
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"[GPT Error] Failed to process image {image_path}: {e}")
            return None

    def _construct_payload(self, history_contents: List[Dict]) -> List[Dict]:
        """
        [Helper] 将 pipeline 标准 history 转换为 Script 2 要求的 Interleaved Input 格式
        Key format: 'input_text' / 'input_image' (Preview API specific)
        """
        # Script 2 的 responses.create 接口看起来是一次性调用，
        # 但如果是多轮对话，通常将历史 flatten 或者放在 content 列表里。
        # 这里我们将 history 转换为 OpenAI 的 messages 列表结构。
        
        messages = []
        
        for turn in history_contents:
            role = turn["role"]
            if role == "model": role = "assistant"
            
            content_parts = []
            
            for item in turn["content"]:
                # 处理文本
                if "text" in item and item["text"]:
                    content_parts.append({
                        "type": "input_text",  # Script 2 specific key
                        "text": item["text"]
                    })
                
                # 处理图片
                elif "image" in item and item["image"]:
                    base64_img = self._resize_and_encode_image(item["image"])
                    if base64_img:
                        content_parts.append({
                            "type": "input_image", # Script 2 specific key
                            "image_url": f"data:image/png;base64,{base64_img}"
                        })
            
            if content_parts:
                messages.append({
                    "role": role,
                    "content": content_parts
                })
        
        return messages

    def _call_with_retry(self, func, **kwargs):
        """
        [Helper] 带有重试机制的 SDK 调用封装
        """
        for attempt in range(self.max_retries):
            try:
                return func(**kwargs)
            except (RateLimitError, APIConnectionError, APIStatusError) as e:
                # 遇到 5xx, 网络错误 或 429 RateLimit 进行重试
                # 400 Bad Request 等错误不在此列
                if isinstance(e, APIStatusError) and 400 <= e.status_code < 500 and e.status_code != 429:
                    raise e # 客户端错误直接抛出

                if attempt == self.max_retries - 1:
                    print(f"[GPT] Request failed after {self.max_retries} attempts. Error: {e}")
                    raise e
                
                sleep_time = 2 * (attempt + 1)
                print(f"[GPT] Connection issue/Timeout ({e}), retrying in {sleep_time}s... (Attempt {attempt + 1})")
                time.sleep(sleep_time)
            except Exception as e:
                raise e

    def generate_text_core(self, 
                           history_contents: List[Dict], 
                           **kwargs) -> List[Dict]:
        """
        核心生文逻辑：不使用 Tool，仅进行多模态理解
        """
        # Script 2 的 input 参数结构是一个 list[dict]
        messages = self._construct_payload(history_contents)
        
        try:
            # 使用 responses.create 接口 (Script 2)
            response = self._call_with_retry(
                self.client.responses.create,
                model=self.deployment_name,
                input=messages,
                # 这里不传 tools，即为纯文本生成/理解模式
            )
            
            # 解析输出 (Script 2 风格)
            # 假设输出在 response.output 中，类型为 message
            text_content = ""
            for output in response.output:
                if output.type == "message":
                    # 假设 content 是字符串或 list
                    content_val = output.content
                    if isinstance(content_val, list):
                        for c in content_val:
                            if c.get("type") == "text":
                                text_content += c.get("text", "")
                    else:
                        text_content += str(content_val)

            history_contents.append({
                "role": "assistant",
                "content": [{"text": text_content}]
            })
            return history_contents

        except Exception as e:
            print(f"[GPT Text Gen Error]: {e}")
            return history_contents

    def generate_image_core(self, 
                            history_contents: List[Dict], 
                            output_image_file: str,
                            **kwargs) -> List[Dict]:
        """
        核心生图逻辑：Interleaved Input + Image Generation Tool
        """
        messages = self._construct_payload(history_contents)
        
        try:
            print(f"[GPT] Requesting Image Generation (Timeout: {self.timeout}s)...")
            
            # 调用 API (Script 2 核心逻辑)
            response = self._call_with_retry(
                self.client.responses.create,
                model=self.deployment_name,
                input=messages,
                tools=[{"type": "image_generation"}] # [Script 2] 显式开启生图工具
            )
            
            image_base64 = None
            text_comment = ""
            
            # 解析 Response Output (Script 2 逻辑)
            # 遍历 output 寻找 image_generation_call
            if hasattr(response, 'output'):
                for output in response.output:
                    # 获取生图结果
                    if output.type == "image_generation_call":
                        image_base64 = output.result # Script 2: output.result 包含 base64
                    
                    # 同时获取可能存在的文本解释
                    elif output.type == "message":
                         # 简化处理，提取文本
                         if isinstance(output.content, str):
                             text_comment += output.content
            
            if not image_base64:
                raise ValueError(f"No 'image_generation_call' found in response. Text output: {text_comment}")

            # 解码并保存图片 (Script 1 逻辑)
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            if output_image_file:
                os.makedirs(os.path.dirname(output_image_file), exist_ok=True)
                image.save(output_image_file)
                print(f"[GPT] Image saved to {output_image_file}")
            
            # 更新历史
            new_content = []
            if text_comment:
                new_content.append({"text": text_comment})
            if output_image_file:
                new_content.append({"image": output_image_file}) 
            
            history_contents.append({
                "role": "assistant",
                "content": new_content
            })
            
            return history_contents

        except Exception as e:
            print(f"[GPT Image Gen Error]: {e}")
            # 调试信息
            if 'response' in locals():
                try:
                    print(f"Response debug: {response.output}")
                except: pass
            return history_contents