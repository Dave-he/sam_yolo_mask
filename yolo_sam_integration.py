#!/usr/bin/env python3
"""
YOLO11 + SAM2 集成工具
使用YOLO11检测图片中的物体，使用SAM2模型分割物体的轮廓，生成带物体名称的彩色mask图
"""

import os
import sys
import argparse
import random
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import colorsys

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm

# YOLO导入
from ultralytics import YOLO

# SAM2导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"错误: 无法导入SAM2模块: {e}")
    print("请确保SAM2已正确安装")
    sys.exit(1)


class YOLOSAMIntegrator:
    """YOLO + SAM2 集成处理器"""
    
    def __init__(self, 
                 yolo_model_path: str = "yolo11x.pt",
                 sam_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt",
                 sam_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 confidence_threshold: float = 0.5,
                 logger: Optional[logging.Logger] = None):
        """
        初始化YOLO + SAM2集成器
        
        Args:
            yolo_model_path: YOLO模型路径
            sam_checkpoint: SAM2模型checkpoint路径
            sam_config: SAM2配置文件路径
            confidence_threshold: YOLO检测置信度阈值
            logger: 日志记录器
        """
        self.yolo_model_path = yolo_model_path
        self.sam_checkpoint = sam_checkpoint
        self.sam_config = sam_config
        self.confidence_threshold = confidence_threshold
        self.logger = logger or self._setup_logger()
        
        # 模型实例
        self.yolo_model = None
        self.sam_predictor = None
        self.device = None
        
        # 颜色映射
        self.class_colors = {}
        
    def _setup_logger(self) -> logging.Logger:
        """设置默认日志记录器"""
        logger = logging.getLogger("yolo_sam_integrator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
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
        else:
            device = torch.device("cpu")
            self.logger.info(f"使用设备: {device}")
        
        self.device = device
        return device
    
    def load_models(self) -> bool:
        """加载YOLO和SAM2模型"""
        try:
            # 设置设备
            self.setup_device()
            
            # 加载YOLO模型
            self.logger.info(f"加载YOLO模型: {self.yolo_model_path}")
            self.yolo_model = YOLO(self.yolo_model_path)
            
            # 加载SAM2模型
            self.logger.info(f"加载SAM2模型: {self.sam_checkpoint}")
            sam2_model = build_sam2(
                config_file=self.sam_config,
                ckpt_path=self.sam_checkpoint,
                device=self.device,
                apply_postprocessing=False
            )
            
            self.sam_predictor = SAM2ImagePredictor(sam2_model)
            
            self.logger.info("所有模型加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            return False
    
    def generate_class_colors(self, class_names: List[str]) -> Dict[str, Tuple[int, int, int]]:
        """为每个类别生成唯一的颜色"""
        colors = {}
        num_classes = len(class_names)
        
        for i, class_name in enumerate(class_names):
            # 使用HSV色彩空间生成均匀分布的颜色
            hue = i / num_classes
            saturation = 0.8 + (i % 3) * 0.1  # 0.8, 0.9, 1.0
            value = 0.8 + (i % 2) * 0.2       # 0.8, 1.0
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors[class_name] = tuple(int(c * 255) for c in rgb)
        
        return colors
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """使用YOLO检测图像中的物体"""
        try:
            # YOLO推理
            results = self.yolo_model(image, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # 获取边界框坐标
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        confidence = boxes.conf[i].cpu().numpy()
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        detections.append(detection)
            
            self.logger.info(f"检测到 {len(detections)} 个物体")
            return detections
            
        except Exception as e:
            self.logger.error(f"YOLO检测失败: {str(e)}")
            return []
    
    def segment_with_sam(self, image: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """使用SAM2基于边界框进行分割"""
        try:
            # 设置图像
            self.sam_predictor.set_image(image)
            
            # 将边界框转换为SAM2格式 [x1, y1, x2, y2]
            input_box = np.array(bbox)
            
            # 进行分割预测
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            # 返回最佳mask
            if len(masks) > 0:
                return masks[0]  # 返回第一个mask
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"SAM2分割失败: {str(e)}")
            return None
    
    def create_colored_mask(self, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.6) -> np.ndarray:
        """创建彩色mask"""
        colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
        colored_mask[mask] = [*color, int(255 * alpha)]
        return colored_mask
    
    def draw_text_with_background(self, draw: ImageDraw.Draw, position: Tuple[int, int], 
                                text: str, font: ImageFont.ImageFont, 
                                text_color: Tuple[int, int, int], 
                                bg_color: Tuple[int, int, int]) -> None:
        """绘制带背景的文本"""
        # 获取文本边界框
        bbox = draw.textbbox(position, text, font=font)
        
        # 绘制背景矩形
        draw.rectangle(bbox, fill=bg_color)
        
        # 绘制文本
        draw.text(position, text, fill=text_color, font=font)
    
    def process_image(self, image_path: str, output_dir: str, 
                     save_individual_masks: bool = True) -> bool:
        """处理单张图像"""
        try:
            # 加载图像
            image = Image.open(image_path)
            image_rgb = np.array(image.convert("RGB"))
            
            self.logger.info(f"处理图像: {os.path.basename(image_path)} ({image_rgb.shape})")
            
            # YOLO检测
            detections = self.detect_objects(image_rgb)
            if not detections:
                self.logger.warning("未检测到任何物体")
                return False
            
            # 生成类别颜色
            class_names = list(set(det['class_name'] for det in detections))
            if not self.class_colors:
                self.class_colors = self.generate_class_colors(class_names)
            
            # 创建输出图像
            result_image = image.copy()
            result_array = np.array(result_image)
            
            # 创建合成mask图像
            composite_mask = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
            
            # 处理每个检测结果
            processed_objects = []
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # 使用SAM2进行分割
                mask = self.segment_with_sam(image_rgb, bbox)
                if mask is None:
                    self.logger.warning(f"无法为 {class_name} 生成mask")
                    continue
                
                # 获取类别颜色
                color = self.class_colors.get(class_name, (255, 255, 255))
                
                # 创建彩色mask
                colored_mask = self.create_colored_mask(mask, color)
                
                # 添加到合成mask
                composite_mask[mask] = colored_mask[mask]
                
                # 保存单个mask（如果需要）
                if save_individual_masks:
                    base_name = Path(image_path).stem
                    individual_mask_path = os.path.join(
                        output_dir, 
                        f"{base_name}_{class_name}_{i:02d}_conf{confidence:.2f}.png"
                    )
                    
                    # 创建单个物体的可视化
                    single_result = image.copy()
                    single_array = np.array(single_result)
                    single_colored_mask = self.create_colored_mask(mask, color, alpha=0.7)
                    
                    # 叠加mask
                    mask_indices = single_colored_mask[:, :, 3] > 0
                    for c in range(3):
                        single_array[mask_indices, c] = (
                            single_array[mask_indices, c] * (1 - single_colored_mask[mask_indices, 3] / 255) +
                            single_colored_mask[mask_indices, c] * (single_colored_mask[mask_indices, 3] / 255)
                        ).astype(np.uint8)
                    
                    # 绘制边界框和标签
                    single_pil = Image.fromarray(single_array)
                    draw = ImageDraw.Draw(single_pil)
                    
                    # 绘制边界框
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # 绘制标签
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
                    except:
                        font = ImageFont.load_default()
                    
                    label = f"{class_name} {confidence:.2f}"
                    self.draw_text_with_background(
                        draw, (x1, y1 - 25), label, font, 
                        (255, 255, 255), color
                    )
                    
                    single_pil.save(individual_mask_path)
                    self.logger.debug(f"保存单个mask: {individual_mask_path}")
                
                processed_objects.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'color': color
                })
            
            if not processed_objects:
                self.logger.warning("没有成功处理任何物体")
                return False
            
            # 创建最终的合成图像
            final_array = result_array.copy()
            
            # 叠加所有masks
            mask_indices = composite_mask[:, :, 3] > 0
            for c in range(3):
                final_array[mask_indices, c] = (
                    final_array[mask_indices, c] * (1 - composite_mask[mask_indices, 3] / 255) +
                    composite_mask[mask_indices, c] * (composite_mask[mask_indices, 3] / 255)
                ).astype(np.uint8)
            
            # 绘制所有边界框和标签
            final_pil = Image.fromarray(final_array)
            draw = ImageDraw.Draw(final_pil)
            
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            for obj in processed_objects:
                x1, y1, x2, y2 = obj['bbox']
                color = obj['color']
                
                # 绘制边界框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                
                # 绘制标签
                label = f"{obj['class_name']} {obj['confidence']:.2f}"
                self.draw_text_with_background(
                    draw, (x1, y1 - 30), label, font, 
                    (255, 255, 255), color
                )
            
            # 保存最终结果
            base_name = Path(image_path).stem
            final_output_path = os.path.join(output_dir, f"{base_name}_yolo_sam_result.png")
            final_pil.save(final_output_path)
            
            # 保存处理信息
            info_path = os.path.join(output_dir, f"{base_name}_detection_info.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'image_path': image_path,
                    'detections': processed_objects,
                    'total_objects': len(processed_objects)
                }, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✓ 成功处理图像，检测到 {len(processed_objects)} 个物体")
            self.logger.info(f"结果保存至: {final_output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理图像失败: {str(e)}")
            return False
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         save_individual_masks: bool = True) -> Dict[str, int]:
        """批量处理目录中的图像"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取图像文件
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        for file_path in Path(input_dir).rglob('*'):
            if file_path.suffix.lower() in supported_formats:
                image_files.append(str(file_path))
        
        if not image_files:
            self.logger.error(f"在目录 {input_dir} 中未找到支持的图像文件")
            return {'success': 0, 'failed': 0}
        
        self.logger.info(f"找到 {len(image_files)} 张图像")
        
        # 处理图像
        results = {'success': 0, 'failed': 0}
        for image_path in tqdm(image_files, desc="处理图像", unit="张"):
            if self.process_image(image_path, output_dir, save_individual_masks):
                results['success'] += 1
            else:
                results['failed'] += 1
        
        self.logger.info(f"处理完成! 成功: {results['success']}/{len(image_files)}")
        if results['failed'] > 0:
            self.logger.warning(f"失败: {results['failed']} 张图像")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLO11 + SAM2 物体检测与分割工具")
    parser.add_argument("--input", "-i", required=True, 
                       help="输入图像文件或目录路径")
    parser.add_argument("--output", "-o", default="output", 
                       help="输出目录路径 (默认: output)")
    parser.add_argument("--yolo-model", default="yolo11x.pt", 
                       help="YOLO模型路径 (默认: yolo11x.pt)")
    parser.add_argument("--sam-checkpoint", default="checkpoints/sam2.1_hiera_large.pt", 
                       help="SAM2模型checkpoint路径")
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml", 
                       help="SAM2配置文件路径")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, 
                       help="YOLO检测置信度阈值 (默认: 0.5)")
    parser.add_argument("--no-individual-masks", action="store_true", 
                       help="不保存单个物体的mask图像")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("main")
    
    # 创建集成器
    integrator = YOLOSAMIntegrator(
        yolo_model_path=args.yolo_model,
        sam_checkpoint=args.sam_checkpoint,
        sam_config=args.sam_config,
        confidence_threshold=args.confidence,
        logger=logger
    )
    
    # 加载模型
    if not integrator.load_models():
        logger.error("模型加载失败，退出程序")
        return 1
    
    # 处理输入
    input_path = Path(args.input)
    if input_path.is_file():
        # 处理单个文件
        logger.info(f"处理单个图像: {input_path}")
        success = integrator.process_image(
            str(input_path), 
            args.output, 
            not args.no_individual_masks
        )
        return 0 if success else 1
    elif input_path.is_dir():
        # 处理目录
        logger.info(f"批量处理目录: {input_path}")
        results = integrator.process_directory(
            str(input_path), 
            args.output, 
            not args.no_individual_masks
        )
        return 0 if results['success'] > 0 else 1
    else:
        logger.error(f"输入路径不存在: {input_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())