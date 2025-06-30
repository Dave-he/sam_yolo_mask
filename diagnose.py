#!/usr/bin/env python3
"""
诊断脚本：检查YOLO模型在测试图像上的检测效果
"""

import os
import sys
import argparse
from pathlib import Path
import random

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import numpy as np
from PIL import Image

def test_yolo_detection(image_path, model_path="models/yolo11x.pt", confidence_thresholds=[0.1, 0.25, 0.5, 0.75]):
    """
    测试YOLO模型在不同置信度阈值下的检测效果
    """
    print(f"\n🔍 分析图像: {os.path.basename(image_path)}")
    
    # 加载模型
    try:
        model = YOLO(model_path)
        print(f"✅ 成功加载YOLO模型: {model_path}")
    except Exception as e:
        print(f"❌ 加载YOLO模型失败: {e}")
        return
    
    # 加载图像
    try:
        image = Image.open(image_path)
        image_rgb = np.array(image.convert("RGB"))
        print(f"📷 图像尺寸: {image_rgb.shape}")
    except Exception as e:
        print(f"❌ 加载图像失败: {e}")
        return
    
    # 测试不同置信度阈值
    for conf_threshold in confidence_thresholds:
        print(f"\n📊 置信度阈值: {conf_threshold}")
        
        try:
            # YOLO推理
            results = model(image_rgb, conf=conf_threshold, verbose=False)
            
            total_detections = 0
            class_counts = {}
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        confidence = boxes.conf[i].cpu().numpy()
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = model.names[class_id]
                        
                        total_detections += 1
                        if class_name not in class_counts:
                            class_counts[class_name] = []
                        class_counts[class_name].append(confidence)
            
            print(f"  🎯 检测到 {total_detections} 个物体")
            
            if class_counts:
                print("  📋 检测到的类别:")
                for class_name, confidences in class_counts.items():
                    avg_conf = np.mean(confidences)
                    max_conf = np.max(confidences)
                    min_conf = np.min(confidences)
                    print(f"    • {class_name}: {len(confidences)}个 (置信度: {min_conf:.3f}-{max_conf:.3f}, 平均: {avg_conf:.3f})")
            else:
                print("  ❌ 未检测到任何物体")
                
        except Exception as e:
            print(f"  ❌ 检测失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="诊断YOLO检测问题")
    parser.add_argument("--input", type=str, default="/Users/hyx/unet-watermark/data/test", help="输入图像目录")
    parser.add_argument("--model", type=str, default="models/yolo11x.pt", help="YOLO模型路径")
    parser.add_argument("--samples", type=int, default=3, help="测试样本数量")
    parser.add_argument("--confidence_thresholds", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75], help="测试的置信度阈值")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLO检测诊断工具")
    print("=" * 60)
    
    # 检查输入目录
    image_dir = Path(args.input)
    if not image_dir.exists():
        print(f"❌ 输入目录不存在: {args.input}")
        return
    
    # 查找图像文件
    supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
    image_files = []
    for ext in supported_extensions:
        image_files.extend(list(image_dir.glob(ext)))
        image_files.extend(list(image_dir.glob(ext.upper())))
    
    if not image_files:
        print(f"❌ 在 {args.input} 中没有找到图像文件")
        return
    
    print(f"📁 找到 {len(image_files)} 张图像")
    
    # 随机选择样本
    if len(image_files) > args.samples:
        sample_files = random.sample(image_files, args.samples)
        print(f"🎲 随机选择 {args.samples} 张图像进行测试")
    else:
        sample_files = image_files
        print(f"📋 测试所有 {len(sample_files)} 张图像")
    
    # 显示YOLO模型支持的类别
    try:
        model = YOLO(args.model)
        print(f"\n📚 YOLO模型支持的类别数量: {len(model.names)}")
        print("🏷️  部分支持的类别:")
        class_names = list(model.names.values())
        for i in range(0, min(20, len(class_names)), 4):
            row = class_names[i:i+4]
            print(f"    {', '.join(row)}")
        if len(class_names) > 20:
            print(f"    ... 还有 {len(class_names) - 20} 个类别")
    except Exception as e:
        print(f"❌ 无法加载模型获取类别信息: {e}")
        return
    
    # 测试每张图像
    for image_file in sample_files:
        test_yolo_detection(str(image_file), args.model, args.confidence_thresholds)
    
    print("\n" + "=" * 60)
    print("🔧 建议:")
    print("1. 如果所有置信度阈值下都检测不到物体，可能是:")
    print("   • 图像中没有YOLO模型训练的常见物体类别")
    print("   • 图像质量问题（模糊、光线不足等）")
    print("   • 需要使用专门训练的模型")
    print("2. 如果低置信度下能检测到物体，建议降低置信度阈值")
    print("3. 可以尝试使用 --confidence_threshold 0.1 或更低的值")
    print("=" * 60)

if __name__ == "__main__":
    main()