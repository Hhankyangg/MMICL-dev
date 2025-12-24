from abc import ABC, abstractmethod
from PIL import Image

from src.utils.utils import dataset_input_content_process

class BaseImageGenerator(ABC):
    
    def __init__(self):
      pass
    
    
    @property
    def is_api_model(self) -> bool:
        """
        开关：是否为 API 模型。
        决定是否开多线程
        例如：Gemini/GPT-4o 设为 True，SD/Flux 设为 False。
        """
        return False
    
    @property
    def can_generate_text(self) -> bool:
        """
        开关：控制多轮对话中是否尝试调用 generate_text_core。
        对于纯生图模型 (SD/Flux) 设为 False。
        对于多模态模型 (Gemini/GPT-4o) 设为 True。
        """
        return False
    
    
    @abstractmethod
    def generate_image_core(self, 
                            history_contents: list, 
                            output_image_file: str,
                            **kwargs) -> list:
        """
        Args:
            history_contents (list): Example: [
                                                {
                                                  "role": "user", 
                                                  "content": [{"text": "repaint this image: "}, {"image": "/path/to/image0.png"}, {"text": " into a Van Gogh style painting"}]
                                                },
                                                {
                                                  "role": "assistant", 
                                                  "content": [{"text": "OK"}, {"image": "/path/to/image1.png"}]
                                                },
                                                {
                                                  "role": "user", 
                                                  "content": [{"text": "repaint this image: "}, {"image": "/path/to/image2.png"}, {"text": " into a Van Gogh style painting"}]
                                                }
                                              ]. 
                                     Defaults to [].

        Returns:
            assistant_response (list): Example: [..., {"role": "assistant", "content": [...]}]
        """
        pass
    
    
    def generate_text_core(self, 
                            history_contents: list, 
                            **kwargs) -> list:
        pass
    
    
    def generate_image_single_turn(self,
                            context: str,
                            instruction: str,
                            output_image_file: str,
                            source_image_dir: str = "./images",
                            **kwargs) -> list:
        
        user_content = f"{context}\n{instruction}"
        print(f"==================== USER_CONTENT: ====================\n{user_content}")
        user_content = dataset_input_content_process(user_content, source_image_dir)
        history_contents = [{"role": "user", "content": user_content}]
        
        return self.generate_image_core(history_contents, output_image_file, **kwargs)


    def generate_image_multi_turn_core(self,
                            user_turns: list[str],
                            output_image_file: str,
                            source_image_dir: str = "./images",
                            **kwargs) -> list:
        
        understanding_turns = user_turns[:-1]
        generate_turn = user_turns[-1]
        history_contents = []
        
        for user_turn in understanding_turns:
            print(f"==================== USER_TURN: ====================\n{user_turn}")
            user_turn = dataset_input_content_process(user_turn, source_image_dir)
            history_contents.append({"role": "user", "content": user_turn})
            
            history_contents = self.generate_text_core(history_contents, **kwargs)  # add assistant response
            print(f"==================== ASSISTANT_RESPONSE: ====================\n{history_contents[-1]['content'][0]['text']}")
            
        print(f"==================== USER_TURN: ====================\n{generate_turn}")
        user_turn = dataset_input_content_process(generate_turn, source_image_dir)
        history_contents.append({"role": "user", "content": user_turn})
        
        history_contents = self.generate_image_core(history_contents, output_image_file, **kwargs)  # add assistant response
        return history_contents
    
    
    def generate_image_multi_turn(self,
                            context: str,
                            instruction: str,
                            output_image_file: str,
                            source_image_dir: str = "./images",
                            **kwargs) -> list:
        
        user_turns = [context, instruction]
        return self.generate_image_multi_turn_core(user_turns, output_image_file, source_image_dir, **kwargs)
