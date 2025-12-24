import os
import io
import json
import base64
import requests
from PIL import Image
from typing import List, Dict, Any

from src.models.base_model import BaseImageGenerator
from src.utils.utils import encode_image


class GeminiGenerator(BaseImageGenerator):
    def __init__(self, 
                 url: str,
                 api_key: str,
                 generation_model_name: str = "gemini-2.5-flash-image-preview",
                 understanding_model_name: str = "gemini-2.5-flash"):
        
        super().__init__()
        
        self.api_key = api_key
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
        
        Input Standard Format:
        [
            {"role": "user", "content": [{"text": "hi"}, {"image": "/path.png"}]},
            {"role": "assistant", "content": [{"text": "ok"}]}
        ]
        
        Output Gemini Format:
        [
            {"role": "user", "parts": [{"text": "hi"}, {"inlineData": {...}}]},
            {"role": "model", "parts": [{"text": "ok"}]}
        ]
        """
        gemini_contents = []
        
        for turn in history_contents:
            role = "user" if turn["role"] == "user" else "model"
            parts = []
            
            # content 是一个 list，包含 text 段和 image 段
            for item in turn["content"]:
                if "text" in item:
                    parts.append({"text": item["text"]})
                elif "image" in item:
                    # 处理图片路径 -> Base64
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


    def generate_text_core(self, 
                           history_contents: List[Dict], 
                           **kwargs) -> List[Dict]:
        """
        实现核心生文逻辑 (用于多轮对话的中间轮次)
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
            response = requests.post(self.understanding_model_url, headers=headers, json=payload)
            response.raise_for_status() # 检查 HTTP 错误
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
        Kwargs:
            image_ratio (str): e.g. "1:1"
            image_size (str): e.g. "1K" 
        """
        # 获取可选参数
        aspect_ratio = kwargs.get("image_ratio", "1:1")
        image_size = kwargs.get("image_size", "1K")
        
        payload = {
            "contents": self._construct_payload(history_contents),
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"], # 请求同时返回图文
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
            print(f"[Gemini] Requesting Image Generation...")
            response = requests.post(self.generation_model_url, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            
            # 解析图片 Base64
            # Gemini 返回结构可能在 parts[0] 也可能在 parts[1] (如果附带了文字解释)
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
            
            # 更新历史 (Assistant 回复了图片 + 可能的一段话)
            new_content = []
            if text_comment:
                new_content.append({"text": text_comment})
            # 这里我们在历史里记录图片保存的路径，而不是 Base64，保持历史轻量
            if output_image_file:
                new_content.append({"image": output_image_file}) 
            
            history_contents.append({
                "role": "assistant",
                "content": new_content
            })
            
            return history_contents

        except Exception as e:
            print(f"[Gemini Image Gen Error]: {e}")
            if "response" in locals():
                print(f"Response dump: {response.text}")
            return history_contents
