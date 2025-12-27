from PIL import Image
import requests
import io
import json
import base64
import re
import os


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def dataset_input_content_process(content: str, image_dir: str = "./images"):
    # 查找 user_content 中 <image:xxx>
    image_pattern = re.compile(r"<image:\s*(.*?)\s*>")  # 匹配 <image:xxx> 中的 xxx
    image_matches = image_pattern.findall(content)    # ["xxx", "yyy" ...]
    image_paths = {f"<image:{image_match}>": os.path.join(image_dir, f"{image_match}.png") for image_match in image_matches}
    image_placeholders = [f"<image:{image_match}>" for image_match in image_matches]
    # 将 user_content 按照 image_placeholders 分割成多个 str
    content_parts = re.split(f"({'|'.join(image_placeholders)})", content)
    
    user_content = []
    for content in content_parts:
        if content in image_placeholders:
            user_content.append({"image": image_paths[content]})
        else:
            if content != "":
                user_content.append({"text": content})
    
    return user_content

