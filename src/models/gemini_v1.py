import os
import io
import json
import base64
import time  # [Added] 用于重试等待
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError # [Added] 具体异常
from PIL import Image
from typing import List, Dict, Any

from src.models.base_model import BaseImageGenerator
# from src.utils.config_loader import load_config # 假设在外部
# from src.utils.utils import encode_image # 假设在外部

class GeminiGenerator(BaseImageGenerator):
    def __init__(self, 
                 url: str,
                 api_key: str,
                 generation_model_name: str = "gemini-2.5-flash-image-preview",
                 understanding_model_name: str = "gemini-2.5-flash",
                 timeout: int = 120,       # [Added] 超时时间 (秒)
                 max_retries: int = 3):   # [Added] 最大重试次数
        
        super().__init__()
        
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 构造 API URL
        self.generation_model_url = f"{url}/v1beta/models/{generation_model_name}:generateContent/"
        self.understanding_model_url = f"{url}/v1beta/models/{understanding_model_name}:generateContent/"


    @property
    def is_api_model(self) -> bool:
        return True


    @property
    def can_generate_text(self) -> bool:
        return True


    def _construct_payload(self, history_contents: List[Dict]) -> List[Dict]:
        """
        [Helper] 将标准格式的 history_contents 转换为 Gemini API 需要的 contents payload。
        (代码保持不变)
        """
        gemini_contents = []
        
        for turn in history_contents:
            role = "user" if turn["role"] == "user" else "model"
            parts = []
            
            for item in turn["content"]:
                if "text" in item:
                    parts.append({"text": item["text"]})
                elif "image" in item:
                    # 假设 encode_image 是外部引入的
                    from src.utils.utils import encode_image
                    image_path = item["image"]
                    encoded_data = encode_image(image_path)
                    if encoded_data:
                        parts.append({
                            "inlineData": {
                                "data": encoded_data,
                                "mimeType": "image/png"
                            }
                        })
            
            gemini_contents.append({
                "role": role,
                "parts": parts
            })
            
        return gemini_contents

    def _post_with_retry(self, url: str, payload: Dict, headers: Dict) -> requests.Response:
        """
        [Added Helper] 带有重试和超时机制的 POST 请求封装
        """
        for attempt in range(self.max_retries):
            try:
                # 发起请求
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=payload, 
                    timeout=self.timeout
                )
                
                # 检查 HTTP 状态码
                # 注意：如果是 4xx 客户端错误 (如 400 参数错误)，通常重试无用，应该直接抛出
                # 如果是 5xx 服务端错误 或 429 Too Many Requests，则应该重试
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    response.raise_for_status()
                    return response
                
                response.raise_for_status()
                return response

            except (Timeout, ConnectionError, HTTPError) as e:
                # 判断是否是最后一次尝试
                if attempt == self.max_retries - 1:
                    print(f"[Gemini] Request failed after {self.max_retries} attempts. Error: {e}")
                    raise e # 抛出异常供外层捕获
                
                # 遇到 5xx, 429, Timeout, ConnectionError 进行等待后重试
                # 指数退避：第一次等 2秒，第二次等 4秒...
                sleep_time = 2 * (attempt + 1)
                print(f"[Gemini] Connection issue/Timeout ({e}), retrying in {sleep_time}s... (Attempt {attempt + 1}/{self.max_retries})")
                time.sleep(sleep_time)
            
            except Exception as e:
                # 其他未知错误直接抛出，不重试
                raise e
                
        return None # Should not allow reach here

    def generate_text_core(self, 
                           history_contents: List[Dict], 
                           **kwargs) -> List[Dict]:
        """
        实现核心生文逻辑
        """
        payload = {
            "contents": self._construct_payload(history_contents),
            "generationConfig": {
                "responseModalities": ["TEXT"]
            }
        }
        
        headers = {
           'Authorization': f'Bearer {self.api_key}',
           'Content-Type': 'application/json'
        }
        
        try:
            # [Modified] 使用带重试的方法
            response = self._post_with_retry(self.understanding_model_url, payload, headers)
            response_json = response.json()
            
            # 解析文本回复
            try:
                text_content = response_json["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                print(f"[Gemini Error] Unexpected response structure: {response_json}")
                text_content = ""

            # 更新历史并返回
            history_contents.append({
                "role": "assistant",
                "content": [{"text": text_content}]
            })
            return history_contents

        except Exception as e:
            print(f"[Gemini Text Gen Error]: {e}")
            # 即使出错也返回原 history，避免 pipeline 崩溃
            return history_contents


    def generate_image_core(self, 
                            history_contents: List[Dict], 
                            output_image_file: str,
                            **kwargs) -> List[Dict]:
        """
        实现核心生图逻辑
        """
        # 获取可选参数
        aspect_ratio = kwargs.get("image_ratio", "1:1")
        image_size = kwargs.get("image_size", "1K")
        
        payload = {
            "contents": self._construct_payload(history_contents),
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size
                }
            }
        }
        
        headers = {
           'Authorization': f'Bearer {self.api_key}',
           'Content-Type': 'application/json'
        }
        
        try:
            print(f"[Gemini] Requesting Image Generation (Timeout: {self.timeout}s)...")
            # [Modified] 使用带重试的方法
            response = self._post_with_retry(self.generation_model_url, payload, headers)
            response_json = response.json()
            
            # 解析图片 Base64
            image_base64 = None
            text_comment = ""
            
            candidates = response_json.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates returned")
                
            parts = candidates[0].get("content", {}).get("parts", [])
            
            for part in parts:
                if "inlineData" in part:
                    image_base64 = part["inlineData"]["data"]
                if "text" in part:
                    text_comment += part["text"]
            
            if not image_base64:
                raise ValueError(f"No image data found in response. Text: {text_comment}")

            # 解码并保存
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # 确保目录存在
            if output_image_file:
                os.makedirs(os.path.dirname(output_image_file), exist_ok=True)
                image.save(output_image_file)
                print(f"[Gemini] Image saved to {output_image_file}")
            
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
            print(f"[Gemini Image Gen Error]: {e}")
            if "response" in locals() and response:
                 # 注意：如果是 Timeout 异常，可能没有 response 对象，这里要做个安全判断
                try:
                    print(f"Response dump: {response.text}")
                except:
                    pass
            return history_contents