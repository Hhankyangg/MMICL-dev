import os
import json
import argparse
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

from src.models.base_model import BaseImageGenerator
from src.models.gemini_v1 import GeminiGenerator


def get_image_generator(model_type: str, config: dict, logger) -> BaseImageGenerator:
    """初始化模型工厂函数"""
    generator = None
    
    if model_type == "nanobanana1":
        logger.info("Initializing NaNoBanana1 Model...")
        cfg = config.get("nanobanana1_model")
        if not cfg:
            logger.error("Config missing 'nanobanana1_model' section!")
            return None
        try:
            generator = GeminiGenerator(
                url=cfg['base_url'],
                api_key=cfg['api_key'],
                generation_model_name=cfg['generation_model_name'],
                understanding_model_name=cfg['understanding_model_name']
            )
        except KeyError as e:
            logger.error(f"Missing key in nanobanana1_model config: {e}")
            return None

    elif model_type == "nanobanana2":
        logger.info("Initializing NaNoBanana2 Model...")
        cfg = config.get("nanobanana2_model")
        if not cfg:
            logger.error("Config missing 'nanobanana2_model' section!")
            return None
        try:
            generator = GeminiGenerator(
                url=cfg['base_url'],
                api_key=cfg['api_key'],
                generation_model_name=cfg['generation_model_name'],
                understanding_model_name=cfg['understanding_model_name']
            )
        except KeyError as e:
            logger.error(f"Missing key in nanobanana2_model config: {e}")
            return None
            
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None
        
    logger.info(f"{model_type} initialized successfully.")
    return generator


def process_one_case(item: dict, generator: BaseImageGenerator, output_dir: str, source_images_dir: str):
    """
    处理单条数据的原子函数，用于多线程调用。
    返回: (status, case_id, message/result)
    status: 'success', 'skip', 'fail'
    """
    case_id = item['id']
    context = item['context']
    instruction = item['instruction']
    
    save_path = os.path.join(output_dir, f"{case_id}.png")

    # 1. 断点续传检查
    if os.path.exists(save_path):
        return "skip", case_id, "File exists"

    # 2. 生成逻辑
    try:
        # 调用模型的单轮生成方法
        history_contents = generator.generate_image_single_turn(
            context=context,
            instruction=instruction,
            output_image_file=save_path,
            source_image_dir=source_images_dir
        )
        return "success", case_id, history_contents
    except Exception as e:
        return "fail", case_id, str(e)


def run_generation(args):
    # 1. 初始化日志
    logger = setup_logger(f"gen_{args.model}_{args.dim}")
    logger.info(f"Starting Generation Pipeline...")
    
    # 2. 加载配置
    try:
        config = load_config(args.config_file)
        logger.info("Config loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # 3. 准备路径
    exp_name = args.exp_name
    dimension_name = args.dim
    
    dataset_dir = os.path.join("dataset", dimension_name)
    json_path = os.path.join(dataset_dir, "interleaved_data.json")
    source_images_dir = os.path.join(dataset_dir, "images")
    
    output_root = config['paths'].get('output_root', 'outputs')
    output_dir = os.path.join(output_root, exp_name, args.model, dimension_name)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(json_path):
        logger.error(f"Data file not found: {json_path}")
        return

    logger.info(f"Experiment: {exp_name} | Model: {args.model} | Dim: {dimension_name}")
    logger.info(f"Output Directory: {output_dir}")

    # 4. 初始化模型
    generator = get_image_generator(args.model, config, logger)
    if not generator:
        logger.error("Failed to initialize generator. Exiting.")
        return

    # 5. 加载数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data_items = json.load(f)
    logger.info(f"Loaded {len(data_items)} items.")

    # 6. 核心生成逻辑 (串行 vs 并行)
    success_count = 0
    skip_count = 0
    fail_count = 0

    # 判断是否为 API 模型 (支持多线程)
    is_api = generator.is_api_model

    if is_api:
        logger.info(f"Model identified as API Model. Using ThreadPoolExecutor with {args.num_workers} workers.")
        
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # 提交所有任务
            future_to_case = {
                executor.submit(process_one_case, item, generator, output_dir, source_images_dir): item['id']
                for item in data_items
            }

            # 使用 tqdm 监控进度
            for future in tqdm(as_completed(future_to_case), total=len(data_items), desc="Generating (Parallel)"):
                case_id = future_to_case[future]
                try:
                    status, cid, msg = future.result()
                    
                    if status == "success":
                        success_count += 1
                        logger.info(f"[Success] Case {cid}: {msg}")
                    elif status == "skip":
                        skip_count += 1
                        # logger.info(f"[Skipped] Case {cid}") 
                    elif status == "fail":
                        fail_count += 1
                        logger.error(f"[Failed] Case {cid}: {msg}")
                        
                except Exception as exc:
                    fail_count += 1
                    logger.error(f"[Exception] Case {case_id} crashed thread: {exc}")

    else:
        logger.info("Model identified as Local/Sequential Model. Running in main thread.")
        
        for item in tqdm(data_items, desc="Generating (Sequential)"):
            status, cid, msg = process_one_case(item, generator, output_dir, source_images_dir)
            
            if status == "success":
                success_count += 1
                logger.info(f"[Success] Case {cid}")
            elif status == "skip":
                skip_count += 1
            elif status == "fail":
                fail_count += 1
                logger.error(f"[Failed] Case {cid}: {msg}")

    # 7. 总结
    logger.info("========== Generation Summary ==========")
    logger.info(f"Total: {len(data_items)}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Skipped: {skip_count}")
    logger.info(f"Failed:  {fail_count}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image generation benchmark.")
    
    parser.add_argument("--config_file", type=str, default="config.yaml",
                        help="Path to the config file.")    
    parser.add_argument("--model", type=str, required=True, 
                        help="The model type to run.")
    parser.add_argument("--dim", type=str, default="dimension_visual_link", 
                        help="The dimension folder name under dataset/.")
    parser.add_argument("--exp_name", type=str, default="benchmark_v1", 
                        help="Experiment name for output folder.")
    parser.add_argument("--num_workers", type=int, default=20, 
                        help="Number of threads for API models. Ignored for local models.")

    args = parser.parse_args()

    run_generation(args)
