#!/usr/bin/env python3
"""
增强版批量图像mask生成脚本
基于SAM2自动mask生成器，具有更好的错误处理、配置管理和性能优化
"""

import os
import sys
import argparse
import random
import logging
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import psutil

# 添加sam2路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    print(f"错误: 无法导入SAM2模块: {e}")
    print("请确保SAM2已正确安装")
    sys.exit(1)


@dataclass
class ProcessingConfig:
    """处理配置类"""
    input_dir: str
    output_dir: str = "output"
    limit: Optional[int] = None
    checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    config: str = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    seed: int = 42
    output_format: str = "png"
    quality: int = 95
    save_masks: bool = False
    max_workers: int = 1
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 0
    use_m2m: bool = False


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def check_memory(self, threshold: float = 90.0) -> bool:
        """检查内存使用率"""
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > threshold:
            self.logger.warning(f"内存使用率过高: {memory_percent:.1f}%")
            return False
        return True
    
    def check_gpu_memory(self) -> Optional[float]:
        """检查GPU内存使用"""
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
            self.logger.info(f"GPU内存使用: {gpu_memory_gb:.2f}GB")
            return gpu_memory_gb
        return None
    
    @contextmanager
    def memory_cleanup(self):
        """内存清理上下文管理器"""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_paths(config: ProcessingConfig) -> List[str]:
        """验证路径配置"""
        errors = []
        
        if not os.path.exists(config.input_dir):
            errors.append(f"输入目录不存在: {config.input_dir}")
        
        # 检查模型文件
        if not os.path.exists(config.checkpoint):
            errors.append(f"模型checkpoint不存在: {config.checkpoint}")
            # 提供可用的模型文件建议
            checkpoint_dir = os.path.dirname(config.checkpoint)
            if os.path.exists(checkpoint_dir):
                available_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
                if available_files:
                    errors.append(f"可用的模型文件: {', '.join(available_files)}")
        
        # 检查配置文件
        # 对于configs路径，Hydra会自动处理，不需要检查物理文件存在性
        if not config.config.startswith('configs/') and not os.path.exists(config.config):
            errors.append(f"配置文件不存在: {config.config}")
            # 提供可用的配置文件建议
            config_dir = os.path.dirname(config.config)
            if os.path.exists(config_dir):
                available_configs = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
                if available_configs:
                    errors.append(f"可用的配置文件: {', '.join(available_configs)}")
        
        return errors
    
    @staticmethod
    def validate_parameters(config: ProcessingConfig) -> List[str]:
        """验证参数配置"""
        errors = []
        
        if config.limit is not None and config.limit <= 0:
            errors.append("limit参数必须大于0")
        
        if not 1 <= config.quality <= 100:
            errors.append("quality参数必须在1-100之间")
        
        if config.output_format not in ['png', 'jpg', 'both']:
            errors.append("output_format必须是png、jpg或both")
        
        if config.max_workers <= 0:
            errors.append("max_workers必须大于0")
        
        return errors


class EnhancedMaskGenerator:
    """增强版mask生成器"""
    
    def __init__(self, config: ProcessingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.monitor = ResourceMonitor(logger)
        self.mask_generator = None
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def setup_device(self) -> torch.device:
        """设置计算设备"""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info(f"使用设备: {device}")
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            self.logger.info(f"使用设备: {device}")
            self.logger.warning("MPS设备支持是初步的，可能会有数值差异")
        else:
            device = torch.device("cpu")
            self.logger.info(f"使用设备: {device}")
        
        return device
    
    def load_model(self) -> bool:
        """加载SAM2模型"""
        try:
            device = self.setup_device()
            self.logger.info(f"加载SAM2模型: {self.config.checkpoint}")
            
            sam2 = build_sam2(
                config_file=self.config.config, 
                ckpt_path=self.config.checkpoint, 
                device=device, 
                apply_postprocessing=False
            )
            
            # 配置mask生成器参数
            self.mask_generator = SAM2AutomaticMaskGenerator(
                sam2,
                points_per_side=self.config.points_per_side,
                pred_iou_thresh=self.config.pred_iou_thresh,
                stability_score_thresh=self.config.stability_score_thresh,
                crop_n_layers=self.config.crop_n_layers,
                crop_n_points_downscale_factor=self.config.crop_n_points_downscale_factor,
                min_mask_region_area=self.config.min_mask_region_area,
                use_m2m=self.config.use_m2m
            )
            
            self.logger.info("模型加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            return False
    
    def get_image_files(self) -> List[str]:
        """获取图像文件列表"""
        image_files = []
        for file_path in Path(self.config.input_dir).rglob('*'):
            if file_path.suffix.lower() in self.supported_formats:
                image_files.append(str(file_path))
        
        self.logger.info(f"找到 {len(image_files)} 张图片")
        
        if self.config.limit and self.config.limit < len(image_files):
            image_files = random.sample(image_files, self.config.limit)
            self.logger.info(f"随机选择 {self.config.limit} 张图片进行处理")
        
        return image_files
    
    def show_anns(self, anns: List[Dict[str, Any]], borders: bool = True) -> Optional[np.ndarray]:
        """显示annotations并返回可视化图像"""
        if len(anns) == 0:
            return None
        
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], 
                       sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
            
            if borders:
                contours, _ = cv2.findContours(m.astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_NONE)
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) 
                           for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
        
        return img
    
    def save_single_mask(self, image_rgb: np.ndarray, mask: Dict[str, Any], 
                        output_path: str, base_name: str, mask_index: int) -> bool:
        """保存单个mask"""
        try:
            # 创建单个mask的可视化
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            
            # 显示单个mask
            m = mask['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 4))
            img[:, :, 3] = 0
            
            # 为mask分配颜色
            color_mask = np.concatenate([np.random.random(3), [0.7]])
            img[m] = color_mask
            
            # 添加边框
            contours, _ = cv2.findContours(m.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) 
                       for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.8), thickness=2)
            
            plt.imshow(img)
            plt.axis('off')
            
            # 根据mask属性确定文件名
            area = mask.get('area', 0)
            stability_score = mask.get('stability_score', 0)
            predicted_iou = mask.get('predicted_iou', 0)
            
            # 根据面积大小分类mask类型
            if area > 50000:
                mask_type = "large"
            elif area > 10000:
                mask_type = "medium"
            elif area > 1000:
                mask_type = "small"
            else:
                mask_type = "tiny"
            
            # 构建文件名：基础名_mask类型_索引_面积_稳定性分数_预测IoU
            filename = f"{base_name}_mask_{mask_type}_{mask_index:03d}_area{area}_stab{stability_score:.2f}_iou{predicted_iou:.2f}"
            
            # 保存单个mask图片
            if self.config.output_format in ['png', 'both']:
                png_path = os.path.join(output_path, f"{filename}.png")
                plt.savefig(png_path, bbox_inches='tight', dpi=150, pad_inches=0)
                self.logger.debug(f"保存单个mask PNG: {png_path}")
            
            if self.config.output_format in ['jpg', 'both']:
                jpg_path = os.path.join(output_path, f"{filename}.jpg")
                plt.savefig(jpg_path, bbox_inches='tight', dpi=150, pad_inches=0, 
                           quality=self.config.quality)
                self.logger.debug(f"保存单个mask JPG: {jpg_path}")
            
            plt.close()
            
            # 保存单个mask数据
            if self.config.save_masks:
                mask_data_path = os.path.join(output_path, f"{filename}.npy")
                np.save(mask_data_path, mask)
                self.logger.debug(f"保存单个mask数据: {mask_data_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存单个mask失败: {str(e)}")
            return False
    
    def save_results(self, image_rgb: np.ndarray, masks: List[Dict], 
                    output_path: str, base_name: str) -> bool:
        """保存处理结果"""
        try:
            # 保存每个mask为单独的图片
            success_count = 0
            for i, mask in enumerate(masks):
                if self.save_single_mask(image_rgb, mask, output_path, base_name, i):
                    success_count += 1
            
            self.logger.info(f"成功保存 {success_count}/{len(masks)} 个单独的mask图片")
            
            # 可选：同时保存所有mask的合成图
            plt.figure(figsize=(20, 20))
            plt.imshow(image_rgb)
            
            mask_img = self.show_anns(masks)
            if mask_img is not None:
                plt.imshow(mask_img)
            
            plt.axis('off')
            
            # 保存合成可视化结果
            if self.config.output_format in ['png', 'both']:
                png_path = os.path.join(output_path, f"{base_name}_all_masks.png")
                plt.savefig(png_path, bbox_inches='tight', dpi=150, pad_inches=0)
                self.logger.debug(f"保存合成PNG: {png_path}")
            
            if self.config.output_format in ['jpg', 'both']:
                jpg_path = os.path.join(output_path, f"{base_name}_all_masks.jpg")
                plt.savefig(jpg_path, bbox_inches='tight', dpi=150, pad_inches=0, 
                           quality=self.config.quality)
                self.logger.debug(f"保存合成JPG: {jpg_path}")
            
            plt.close()
            
            # 保存原始mask数据
            if self.config.save_masks:
                mask_data_path = os.path.join(output_path, f"{base_name}_all_masks.npy")
                np.save(mask_data_path, masks)
                self.logger.debug(f"保存所有mask数据: {mask_data_path}")
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {str(e)}")
            return False
    
    def process_single_image(self, image_path: str) -> Tuple[bool, str]:
        """处理单张图片"""
        try:
            with self.monitor.memory_cleanup():
                # 检查内存
                if not self.monitor.check_memory():
                    return False, "内存不足"
                
                # 加载图片
                image = Image.open(image_path)
                image_rgb = np.array(image.convert("RGB"))
                
                self.logger.debug(f"处理图片: {os.path.basename(image_path)} ({image_rgb.shape})")
                
                # 生成masks
                masks = self.mask_generator.generate(image_rgb)
                self.logger.debug(f"生成了 {len(masks)} 个mask")
                
                # 保存结果
                base_name = Path(image_path).stem
                success = self.save_results(image_rgb, masks, self.config.output_dir, base_name)
                
                if success:
                    return True, f"成功处理 {len(masks)} 个mask"
                else:
                    return False, "保存失败"
                
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            self.logger.error(f"处理图片 {image_path} 时出错: {error_msg}")
            return False, error_msg
    
    def process_images_parallel(self, image_files: List[str]) -> Dict[str, int]:
        """并行处理图片"""
        results = {"success": 0, "failed": 0}
        
        if self.config.max_workers == 1:
            # 单线程处理
            for image_path in tqdm(image_files, desc="处理图片", unit="张"):
                success, message = self.process_single_image(image_path)
                if success:
                    results["success"] += 1
                    self.logger.info(f"✓ {os.path.basename(image_path)}: {message}")
                else:
                    results["failed"] += 1
                    self.logger.error(f"✗ {os.path.basename(image_path)}: {message}")
        else:
            # 多线程处理
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_image = {
                    executor.submit(self.process_single_image, img): img 
                    for img in image_files
                }
                
                for future in tqdm(as_completed(future_to_image), 
                                 total=len(image_files), desc="处理图片", unit="张"):
                    image_path = future_to_image[future]
                    try:
                        success, message = future.result()
                        if success:
                            results["success"] += 1
                            self.logger.info(f"✓ {os.path.basename(image_path)}: {message}")
                        else:
                            results["failed"] += 1
                            self.logger.error(f"✗ {os.path.basename(image_path)}: {message}")
                    except Exception as e:
                        results["failed"] += 1
                        self.logger.error(f"✗ {os.path.basename(image_path)}: 处理异常: {e}")
        
        return results
    
    def run(self) -> bool:
        """运行批量处理"""
        # 验证配置
        validator = ConfigValidator()
        path_errors = validator.validate_paths(self.config)
        param_errors = validator.validate_parameters(self.config)
        
        if path_errors or param_errors:
            for error in path_errors + param_errors:
                self.logger.error(error)
            return False
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.logger.info(f"输出目录: {self.config.output_dir}")
        
        # 加载模型
        if not self.load_model():
            return False
        
        # 获取图片文件
        image_files = self.get_image_files()
        if not image_files:
            self.logger.error("没有找到支持的图片文件")
            return False
        
        # 处理图片
        self.logger.info(f"开始处理 {len(image_files)} 张图片...")
        results = self.process_images_parallel(image_files)
        
        # 输出结果
        total = results["success"] + results["failed"]
        self.logger.info(f"处理完成! 成功: {results['success']}/{total}")
        
        if results["failed"] > 0:
            self.logger.warning(f"失败: {results['failed']} 张图片")
        
        return results["success"] > 0


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """设置日志"""
    logger = logging.getLogger("batch_mask_generator")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config_from_file(config_file: str) -> Dict:
    """从文件加载配置"""
    with open(config_file, 'r', encoding='utf-8') as f:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_file.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError("不支持的配置文件格式")


def auto_detect_config() -> Tuple[str, str]:
    """自动检测可用的模型和配置文件"""
    # 检测checkpoint
    checkpoint_dirs = ['checkpoints', '../checkpoints']
    checkpoint = None
    model_type = None
    
    for dir_path in checkpoint_dirs:
        if os.path.exists(dir_path):
            pt_files = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
            if pt_files:
                # 按优先级选择模型：large > base_plus > small > tiny
                priority_order = ['large', 'base_plus', 'b+', 'small', 'tiny']
                for priority in priority_order:
                    for pt_file in pt_files:
                        if priority in pt_file:
                            checkpoint = os.path.abspath(os.path.join(dir_path, pt_file))
                            # 确定模型类型
                            if 'large' in pt_file or '_l.' in pt_file:
                                model_type = 'l'
                            elif 'base_plus' in pt_file or 'b+' in pt_file:
                                model_type = 'b+'
                            elif 'small' in pt_file or '_s.' in pt_file:
                                model_type = 's'
                            elif 'tiny' in pt_file or '_t.' in pt_file:
                                model_type = 't'
                            break
                    if checkpoint:
                        break
                
                # 如果没有匹配到优先级，选择第一个
                if not checkpoint:
                    checkpoint = os.path.abspath(os.path.join(dir_path, pt_files[0]))
                    # 尝试从文件名推断类型
                    filename = pt_files[0].lower()
                    if 'large' in filename:
                        model_type = 'l'
                    elif 'base' in filename or 'b+' in filename:
                        model_type = 'b+'
                    elif 'small' in filename:
                        model_type = 's'
                    elif 'tiny' in filename:
                        model_type = 't'
                    else:
                        model_type = 'l'  # 默认使用large配置
                break
    
    # 检测配置文件，根据模型类型选择匹配的配置
    config_dir = 'sam2/configs/sam2.1'
    config = None
    
    if os.path.exists(config_dir):
        yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
        if yaml_files:
            # 根据模型类型选择对应的配置文件
            target_config = f'sam2.1_hiera_{model_type}.yaml'
            for yaml_file in yaml_files:
                if yaml_file == target_config:
                    # 返回Hydra期望的configs路径格式
                    config = f'configs/sam2.1/{yaml_file}'
                    break
            
            # 如果没有找到匹配的配置，选择第一个可用的
            if not config:
                config = f'configs/sam2.1/{yaml_files[0]}'
    
    return checkpoint, config


def main():
    parser = argparse.ArgumentParser(
        description='增强版批量图像mask生成脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s ./images --limit 10
  %(prog)s ./images --output ./results --max-workers 4
  %(prog)s ./images --config-file config.yaml
        """
    )
    
    # 基本参数
    parser.add_argument('input_dir', help='输入图片文件夹路径')
    parser.add_argument('--output', '-o', default='output', help='输出文件夹路径')
    parser.add_argument('--limit', '-l', type=int, help='随机选择处理的图片数量')
    
    # 模型参数
    parser.add_argument('--checkpoint', '-c', help='SAM2模型checkpoint路径')
    parser.add_argument('--config', help='SAM2模型配置文件路径')
    parser.add_argument('--auto-detect', action='store_true', help='自动检测模型和配置文件')
    
    # 输出参数
    parser.add_argument('--output-format', choices=['png', 'jpg', 'both'], 
                       default='png', help='输出图片格式')
    parser.add_argument('--quality', type=int, default=95, help='JPEG输出质量 (1-100)')
    parser.add_argument('--save-masks', required=False, default=False, help='保存原始mask数据')
    
    # 性能参数
    parser.add_argument('--max-workers', type=int, default=1, help='最大并行工作线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # SAM2参数
    parser.add_argument('--points-per-side', type=int, default=32, help='每边采样点数')
    parser.add_argument('--pred-iou-thresh', type=float, default=0.88, help='预测IoU阈值')
    parser.add_argument('--stability-score-thresh', type=float, default=0.95, help='稳定性分数阈值')
    
    # 日志参数
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别')
    parser.add_argument('--log-file', help='日志文件路径')
    
    # 配置文件
    parser.add_argument('--config-file', help='YAML/JSON配置文件路径')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level, args.log_file)
    
    try:
        # 从配置文件加载参数
        config_dict = {}
        if args.config_file:
            config_dict = load_config_from_file(args.config_file)
            logger.info(f"从配置文件加载参数: {args.config_file}")
        
        # 自动检测模型文件
        if args.auto_detect or (not args.checkpoint and not args.config):
            auto_checkpoint, auto_config = auto_detect_config()
            if not args.checkpoint:
                args.checkpoint = auto_checkpoint
            if not args.config:
                args.config = auto_config
            logger.info(f"自动检测到模型: {args.checkpoint}")
            logger.info(f"自动检测到配置: {args.config}")
        
        # 创建配置对象
        config = ProcessingConfig(
            input_dir=args.input_dir,
            output_dir=args.output,
            limit=args.limit,
            checkpoint=args.checkpoint or config_dict.get('checkpoint', 'checkpoints/sam2.1_hiera_large.pt'),
            config=args.config or config_dict.get('config', 'sam2/configs/sam2.1/sam2.1_hiera_l.yaml'),
            seed=args.seed,
            output_format=args.output_format,
            quality=args.quality,
            save_masks=args.save_masks,
            max_workers=args.max_workers,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh
        )
        
        # 设置随机种子
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # 创建并运行生成器
        generator = EnhancedMaskGenerator(config, logger)
        success = generator.run()
        
        if success:
            logger.info("批量处理完成!")
            sys.exit(0)
        else:
            logger.error("批量处理失败!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("用户中断处理")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序异常: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()