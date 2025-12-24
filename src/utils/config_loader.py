import os
import yaml
from typing import Dict, Any

def get_project_root() -> str:
    """
    获取项目根目录的绝对路径。
    假设 src/utils/config_loader.py 位于项目根目录的 src/utils/ 下，
    往上推两级即可找到根目录。
    """
    current_file_path = os.path.abspath(__file__)
    utils_dir = os.path.dirname(current_file_path) # src/utils
    src_dir = os.path.dirname(utils_dir)           # src
    project_root = os.path.dirname(src_dir)        # Project_Root
    return project_root

def load_config(config_name: str = "config.yaml") -> Dict[str, Any]:
    """
    加载配置文件。
    
    逻辑：
    1. 首先尝试直接读取 config_name (假设是在当前工作目录)。
    2. 如果找不到，尝试去项目根目录查找。
    
    Args:
        config_name (str): 配置文件名或路径。
        
    Returns:
        dict: 解析后的配置字典。
        
    Raises:
        FileNotFoundError: 如果找不到文件。
        ValueError: 如果 YAML 格式错误。
    """
    
    # 路径尝试列表
    possible_paths = [
        config_name,  # 当前目录
        os.path.join(get_project_root(), config_name) # 项目根目录
    ]
    
    final_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            final_path = path
            break
            
    if final_path is None:
        raise FileNotFoundError(
            f"Config file '{config_name}' not found. "
            f"Searched in: {[os.path.abspath(p) for p in possible_paths]}"
        )
        
    # 读取并解析
    with open(final_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            if not config:
                return {}
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file at {final_path}: {e}")

# 简单的测试代码
if __name__ == "__main__":
    try:
        cfg = load_config()
        print("Config loaded successfully:")
        print(cfg)
    except Exception as e:
        print(e)