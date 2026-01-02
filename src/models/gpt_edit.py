import os
import io
import base64
import time
import requests
from typing import List, Dict, Any, Tuple
from PIL import Image

# 引入 OpenAI 和 Azure 库 (基于 Script 1)
from openai import AzureOpenAI, OpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError, BadRequestError
from azure.identity import AzureCliCredential, ManagedIdentityCredential, ChainedTokenCredential, get_bearer_token_provider

# 假设基类定义
from src.models.base_model import BaseImageGenerator

class GPTEdit(BaseImageGenerator):
    def __init__(self, 
                 azure_endpoint: str,
                 api_key: str = None,
                 api_version: str = "2024-02-15-preview", # Script 1 默认版本
                 deployment_name: str = "gpt-4o",         # Script 1 默认模型
                 timeout: int = 120,
                 max_retries: int = 3,
                 max_image_size: int = 1500):             # Script 1 配置
        
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
                azure_ad_token_provider=credential,
                api_version=api_version,
                timeout=timeout
            )

    @property
    def is_api_model(self) -> bool:
        return True

    @property
    def can_generate_text(self) -> bool:
        # 明确标记不支持纯文本对话生成
        return False

    # ================= 工具方法 (复用 Script 1) =================

    def _resize_image_if_needed(self, image: Image.Image) -> Image.Image:
        """Script 1: 单独调整图片尺寸"""
        w, h = image.size
        if max(w, h) > self.max_image_size:
            scale = self.max_image_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return image

    def _pil_to_bytes_stream(self, image: Image.Image, name="image.png") -> io.BytesIO:
        """Script 1: 将PIL Image转换为BytesIO流"""
        img_byte_arr = io.BytesIO()
        # 强制转换为 RGB 防止 PNG RGBA 兼容性问题 (Script 1 process_and_merge 中也有类似逻辑)
        if image.mode in ('RGBA', 'LA'): 
            image = image.convert('RGB')
        image.save(img_byte_arr, format='PNG') 
        img_byte_arr.seek(0)
        img_byte_arr.name = name
        return img_byte_arr

    # ================= 核心逻辑：Interleaved 转 Edit 格式 =================

    def _flatten_interleaved_content(self, content_list: List[Dict]) -> Tuple[str, List[io.BytesIO]]:
        """
        [关键处理逻辑]
        将 [{"text": "A"}, {"image": "path1"}, {"text": "B"}] 
        转换为:
        1. Prompt: "A <image 1> B"
        2. Image Streams: [stream(path1)]
        """
        full_prompt = ""
        image_streams = []
        img_counter = 1

        for item in content_list:
            # 处理文本
            if "text" in item and item["text"]:
                full_prompt += item["text"]
            
            # 处理图片
            elif "image" in item and item["image"]:
                image_path = item["image"]
                
                # 1. 拼接文本占位符
                # Script 1 的逻辑是把图片列表传进去，这里我们在 Prompt 里显式指代它们
                placeholder = f" <image {img_counter}> "
                full_prompt += placeholder
                
                # 2. 加载并处理图片
                try:
                    if os.path.exists(image_path):
                        with Image.open(image_path) as img:
                            # 必须 Copy 一份，否则退出 with 块后文件关闭会导致 stream 出错
                            img_copy = img.copy() 
                            img_resized = self._resize_image_if_needed(img_copy)
                            stream = self._pil_to_bytes_stream(img_resized, name=f"image_{img_counter}.png")
                            image_streams.append(stream)
                            img_counter += 1
                    else:
                        print(f"[GPTEdit Warning] Image path not found: {image_path}")
                except Exception as e:
                    print(f"[GPTEdit Error] Failed to load image {image_path}: {e}")

        return full_prompt, image_streams

    def _call_with_retry(self, func, **kwargs):
        """通用重试逻辑"""
        for attempt in range(self.max_retries):
            try:
                return func(**kwargs)
            except (RateLimitError, APIConnectionError, APIStatusError) as e:
                # 400 错误通常是参数问题，重试无效
                if isinstance(e, APIStatusError) and 400 <= e.status_code < 500 and e.status_code != 429:
                    raise e 

                if attempt == self.max_retries - 1:
                    print(f"[GPTEdit] Request failed after {self.max_retries} attempts. Error: {e}")
                    raise e
                
                sleep_time = 2 * (attempt + 1)
                print(f"[GPTEdit] Network/RateLimit issue ({e}), retrying in {sleep_time}s... (Attempt {attempt + 1})")
                time.sleep(sleep_time)
            except Exception as e:
                raise e

    # ================= 接口实现 =================

    def generate_text_core(self, history_contents: List[Dict], **kwargs) -> List[Dict]:
        """
        [Restriction] GPTEdit 模式不支持纯文本对话。
        """
        print("[GPTEdit] Warning: generate_text_core called but this model only supports image editing.")
        return history_contents

    def generate_image_core(self, 
                            history_contents: List[Dict], 
                            output_image_file: str,
                            **kwargs) -> List[Dict]:
        """
        实现核心生图逻辑：使用 images.edit 接口
        """
        # 1. 提取最后一轮的用户输入 (我们假设最后一轮包含了所有的 Prompt 和 Reference Images)
        last_turn = history_contents[-1]
        if last_turn["role"] != "user":
            print("[GPTEdit] Warning: Last turn is not user input.")
            return history_contents
        
        # 2. 将 Interleaved 格式转换为 Edit API 需要的 (Prompt String, Image List)
        prompt_text, image_streams = self._flatten_interleaved_content(last_turn["content"])
        
        # 如果没有图片，images.edit 可能会报错或者退化为 DALL-E，
        # 但 Script 1 强依赖于输入图片，所以这里最好做个检查
        if not image_streams:
            print("[GPTEdit] Warning: No source images found in input. Edit API might fail.")

        try:
            print(f"[GPTEdit] Requesting Image Edit (Prompt len: {len(prompt_text)}, Imgs: {len(image_streams)})...")
            
            # 3. 调用 API (Script 1 方式)
            # 注意：Script 1 中 extra_query 是为了指定 api-version，但我们在 Client 初始化时已经指定了
            # 这里如果不放心可以保留
            response = self._call_with_retry(
                self.client.images.edit,
                model=self.deployment_name,
                image=image_streams, # 传入流列表
                prompt=prompt_text,
                n=1,
                # extra_query={"api-version": "2025-04-01-preview"} # 如有必要可加
            )
            
            # 4. 解析结果 (Script 1 方式)
            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            
            # 5. 保存图片
            if output_image_file:
                os.makedirs(os.path.dirname(output_image_file), exist_ok=True)
                with open(output_image_file, "wb") as img_file:
                    img_file.write(image_bytes)
                print(f"[GPTEdit] Image saved to {output_image_file}")
            
            # 6. 更新历史
            # 由于不支持多轮对话，这里的更新主要是为了 pipeline 不报错
            history_contents.append({
                "role": "assistant",
                "content": [
                    {"text": f"Image edited based on: {prompt_text[:50]}..."},
                    {"image": output_image_file}
                ]
            })
            
            return history_contents

        except Exception as e:
            print(f"[GPTEdit Image Gen Error]: {e}")
            return history_contents