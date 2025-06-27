#!/usr/bin/env python3
"""
简单的演示脚本
展示如何使用YOLO11 + SAM2进行物体检测和分割
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yolo_sam_integration import YOLOSAMIntegrator


def demo():
    """演示函数"""
    print("=" * 60)
    print("YOLO11 + SAM2 物体检测与分割演示")
    print("=" * 60)
    
    # 检查必要的目录和文件
    required_dirs = ["images", "checkpoints"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"✓ 创建目录: {dir_name}")
    
    # 检查是否有测试图像
    image_dir = Path("images")
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not image_files:
        print("\n⚠️  警告: images目录中没有找到图像文件")
        print("请将要处理的图像文件放入images目录中")
        print("支持的格式: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
        return
    
    print(f"\n✓ 找到 {len(image_files)} 张图像")
    
    # 检查模型文件
    yolo_model = "yolo11x.pt"  # YOLO会自动下载
    sam_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    sam_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    if not os.path.exists(sam_checkpoint):
        print(f"\n⚠️  警告: SAM2模型文件不存在: {sam_checkpoint}")
        print("请下载SAM2模型文件并放置在checkpoints目录中")
        print("下载地址: https://github.com/facebookresearch/segment-anything-2")
        return
    
    if not os.path.exists(sam_config):
        print(f"\n⚠️  警告: SAM2配置文件不存在: {sam_config}")
        print("请确保configs目录中有正确的配置文件")
        return
    
    print("\n✓ 模型文件检查通过")
    
    # 创建集成器
    print("\n🚀 初始化YOLO + SAM2集成器...")
    integrator = YOLOSAMIntegrator(
        yolo_model_path=yolo_model,
        sam_checkpoint=sam_checkpoint,
        sam_config=sam_config,
        confidence_threshold=0.5
    )
    
    # 加载模型
    print("📦 加载模型...")
    if not integrator.load_models():
        print("❌ 模型加载失败")
        return
    
    print("✅ 模型加载成功")
    
    # 处理图像
    output_dir = "demo_output"
    print(f"\n🎯 开始处理图像，输出目录: {output_dir}")
    
    results = integrator.process_directory(
        input_dir="images",
        output_dir=output_dir,
        save_individual_masks=True
    )
    
    # 显示结果
    print("\n" + "=" * 60)
    print("处理结果:")
    print(f"✅ 成功处理: {results['success']} 张图像")
    print(f"❌ 处理失败: {results['failed']} 张图像")
    
    if results['success'] > 0:
        print(f"\n📁 结果文件保存在: {output_dir}")
        print("\n生成的文件类型:")
        print("  • *_yolo_sam_result.png - 完整的检测和分割结果")
        print("  • *_<类别名>_<序号>_conf<置信度>.png - 单个物体的分割结果")
        print("  • *_detection_info.json - 检测信息的JSON文件")
    
    print("\n🎉 演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()