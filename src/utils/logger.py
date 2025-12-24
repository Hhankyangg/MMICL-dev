import logging
import os
import sys
from datetime import datetime

def setup_logger(name: str, log_dir: str = "logs"):
    """
    配置 Logger：同时输出到控制台和日志文件
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = [] # 清除已有的 handlers，防止重复打印

    # 1. 定义格式
    # 格式：[时间] [级别] [Logger名] 消息
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 2. 文件 Handler (按时间命名)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{timestamp}.log"), 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 3. 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger