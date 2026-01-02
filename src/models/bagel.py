import os
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import sys
import requests
from io import BytesIO

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from safetensors.torch import load_file

from src.models.bagel_data.transforms import ImageTransform
from src.models.bagel_data.data_utils import pil_img2rgb, add_special_tokens
from src.models.bagel_modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from src.models.bagel_modeling.qwen2 import Qwen2Tokenizer
from src.models.bagel_modeling.bagel.qwen2_navit import NaiveCache
from src.models.bagel_modeling.autoencoder import load_ae
from src.models.bagel_modeling.inferencer import InterleaveInferencer

from src.models.base_model import BaseImageGenerator



class BagelGenerator(BaseImageGenerator):

    # INSTALL_REQ = True
    # INTERLEAVE = False

    def __init__(self, 
                 model_path='ByteDance-Seed/BAGEL-7B-MoT', 
                 max_mem_per_gpu = '80GiB',
                 think_understand=False,
                 think_generation=False,
                 **kwargs):

        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        self.vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )

        with init_empty_weights():
            self.language_model = Qwen2ForCausalLM(llm_config)
            self.vit_model = SiglipVisionModel(vit_config)
            self.model = Bagel(self.language_model, self.vit_model, config)
            self.model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)

        self.vae_transform = ImageTransform(1024, 512, 16)
        self.vit_transform = ImageTransform(980, 224, 14)

        self.device_map = infer_auto_device_map(
            self.model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        if torch.cuda.device_count() == 1:
            first_device = self.device_map.get(same_device_modules[0], "cuda0")
            for k in same_device_modules:
                if k in self.device_map:
                    self.device_map[k] = first_device
                else:
                    self.device_map[k] = "cuda:0"
        else:
            first_device = self.device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in self.device_map:
                    self.device_map[k] = first_device

        self.model = load_checkpoint_and_dispatch(
            self.model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=self.device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload"
        )

        self.model = self.model.eval()

        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids
        )

        kwargs_default = {
            "max_think_token_n": 1000,
            "do_sample": False,
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.0,
            "cfg_interval": [0.4, 1.0],
            "timestep_shift": 3.0,
            "num_timesteps": 50,
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "global"
        }

        self.kwargs = kwargs_default
        kwargs_default.update(kwargs)

        self.think_understand = think_understand
        self.think_generation = think_generation


    @property
    def is_api_model(self) -> bool:
        return False


    @property
    def can_generate_text(self) -> bool:
        return True

    
    def _prepare_inputs(self, history_contents: List[Dict]) -> List[Union[str, Image.Image]]:
            """
            辅助函数：将 standard history format 转换为 Bagel interleave_inference 需要的 input_list。
            """
            input_list = []
            for turn in history_contents:
                # 遍历每一轮对话 (user 或 assistant)
                for item in turn.get('content', []):
                    if 'text' in item:
                        # 提取文本
                        input_list.append(item['text'])
                    elif 'image' in item:
                        # 提取图片路径并加载为 PIL Image
                        image_path = item['image']
                        if isinstance(image_path, str):
                            try:
                                image = Image.open(image_path).convert("RGB")
                                input_list.append(image)
                            except Exception as e:
                                print(f"Error loading image {image_path}: {e}")
                        elif isinstance(image_path, Image.Image):
                            input_list.append(image_path.convert("RGB"))
            return input_list


    def generate_text_core(self, 
                            history_contents: List[Dict], 
                            **kwargs) -> List[Dict]:
        """
        实现文本生成核心逻辑 (Understanding mode)
        Args:
            think (bool): 是否输出思考过程 (<think>...</think>)
        """
        # 1. 准备输入数据
        input_list = self._prepare_inputs(history_contents)
        
        # 2. 合并推理参数
        gen_kwargs = self.kwargs.copy()
        gen_kwargs.update(kwargs)
        
        # 3. 调用 Bagel 推理器
        # understanding_output=True 模式用于文本生成
        # 显式传入 think 参数
        with torch.no_grad():
            outputs = self.inferencer.interleave_inference(
                input_list,
                understanding_output=True,
                think=self.think_understand, 
                **gen_kwargs
            )
        
        # 4. 处理输出并更新历史
        # 对于文本生成，outputs 通常是 [text_string]
        # 如果 think=True，返回的 text_string 内部会包含 <think>...</think> 标签
        if outputs and isinstance(outputs[-1], str):
            generated_text = outputs[-1]
            
            # 构建 assistant 回复
            new_turn = {
                "role": "assistant",
                "content": [{"text": generated_text}]
            }
            history_contents.append(new_turn)
            
        return history_contents


    def generate_image_core(self, 
                            history_contents: List[Dict], 
                            output_image_file: str,
                            **kwargs) -> List[Dict]:
        """
        实现图像生成核心逻辑 (Generation mode)
        Args:
            think (bool): 是否在生图前进行思考/规划 (Plan -> Image)
        """
        # 1. 准备输入数据
        input_list = self._prepare_inputs(history_contents)
        
        # 2. 合并推理参数
        gen_kwargs = self.kwargs.copy()
        gen_kwargs.update(kwargs)
        
        # 3. 调用 Bagel 推理器
        # understanding_output=False 模式用于图像生成
        with torch.no_grad():
            outputs = self.inferencer.interleave_inference(
                input_list,
                understanding_output=False,
                think=self.think_generation,
                **gen_kwargs
            )
        
        # 4. 处理输出
        # outputs 可能是 [Image] 或者 [Plan_Text, Image]
        generated_content = []
        
        for out in outputs:
            if isinstance(out, str):
                # 这是 think=True 时生成的 Planning 文本
                generated_content.append({"text": out})
                # 也可以选择打印出来看看
                print(f"[Bagel Plan]: {out[:100]}...") 
                
            elif isinstance(out, Image.Image):
                # 保存生成的图像
                try:
                    os.makedirs(os.path.dirname(output_image_file), exist_ok=True)
                    out.save(output_image_file)
                    # 在历史记录中记录保存的文件路径
                    generated_content.append({"image": output_image_file})
                except Exception as e:
                    print(f"Error saving generated image: {e}")

        # 5. 更新历史记录
        # 如果有 think，历史记录会变成 [{"text": "<think>..."}, {"image": "/path/to/img"}]
        if generated_content:
            history_contents.append({
                "role": "assistant",
                "content": generated_content
            })
            
        return history_contents
