#!/usr/bin/env python3
"""
主脚本，演示如何使用YOLO11 + SAM2进行物体检测和分割
支持命令行参数，例如 --limit 来限制处理的图像数量
"""

import os
import sys
import argparse
import random
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.yolo_sam_integration import YOLOSAMIntegrator


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="YOLO11 + SAM2 物体检测与分割演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py                           # 处理images目录中的所有图像
  python main.py --limit 5                 # 随机选择5张图像进行处理
  python main.py --input_dir my_images     # 指定输入目录
  python main.py --output_dir results      # 指定输出目录
  python main.py --confidence_threshold 0.7 # 设置检测置信度阈值
        """
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="限制处理的图像数量 (0表示处理所有图像，>0表示随机选择指定数量的图像)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="images",
        help="输入图像目录 (默认: images)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="输出结果目录 (默认: output)"
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="models/yolo11x.pt",
        help="YOLO模型路径 (默认: models/yolo11x.pt)"
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="models/sam2.1_hiera_base_plus.pt",
        help="SAM2模型检查点路径 (默认: models/sam2.1_hiera_base_plus.pt)"
    )
    parser.add_argument(
        "--sam_config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM2模型配置文件路径 (默认: configs/sam2.1/sam2.1_hiera_b+.yaml)"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="检测置信度阈值 (默认: 0.5)"
    )
    parser.add_argument(
        "--save_individual_masks",
        action="store_true",
        default=True,
        help="是否保存单个物体的分割结果 (默认: True)"
    )
    parser.add_argument(
        "--no_individual_masks",
        action="store_true",
        help="不保存单个物体的分割结果"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，用于可重复的随机选择 (默认: None)"
    )

    args = parser.parse_args()
    
    # 处理互斥的参数
    if args.no_individual_masks:
        args.save_individual_masks = False

    print("=" * 60)
    print("YOLO11 + SAM2 物体检测与分割演示")
    print("=" * 60)
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        print(f"✓ 设置随机种子: {args.seed}")
    
    # 检查必要的目录和文件
    required_dirs = [args.input_dir, "models"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"✓ 创建目录: {dir_name}")
    
    # 检查是否有测试图像
    image_dir = Path(args.input_dir)
    supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
    image_files = []
    for ext in supported_extensions:
        image_files.extend(list(image_dir.glob(ext)))
        image_files.extend(list(image_dir.glob(ext.upper())))
    
    if not image_files:
        print(f"\n⚠️  警告: {args.input_dir}目录中没有找到图像文件")
        print("请将要处理的图像文件放入指定目录中")
        print("支持的格式: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
        return
    
    print(f"\n✓ 找到 {len(image_files)} 张图像")

    # 根据 --limit 参数选择图像
    if args.limit > 0 and len(image_files) > args.limit:
        print(f"\n✂️  根据 --limit 参数，随机选择 {args.limit} 张图像进行处理...")
        image_files = random.sample(image_files, args.limit)
        print(f"已选择 {len(image_files)} 张图像")
        for i, img_file in enumerate(image_files, 1):
            print(f"  {i}. {img_file.name}")
    elif args.limit > 0:
        print(f"\n📋 图像数量({len(image_files)})不超过限制({args.limit})，将处理所有图像")
    
    # 检查模型文件
    if not os.path.exists(args.sam_checkpoint):
        print(f"\n⚠️  警告: SAM2模型文件不存在: {args.sam_checkpoint}")
        print("请下载SAM2模型文件并放置在models目录中")
        print("下载地址: https://github.com/facebookresearch/segment-anything-2")
        return
    
    if not os.path.exists(args.sam_config):
        print(f"\n⚠️  警告: SAM2配置文件不存在: {args.sam_config}")
        print("请确保configs目录中有正确的配置文件")
        return
    
    print("\n✓ 模型文件检查通过")
    
    # 创建集成器
    print("\n🚀 初始化YOLO + SAM2集成器...")
    integrator = YOLOSAMIntegrator(
        yolo_model_path=args.yolo_model,
        sam_checkpoint=args.sam_checkpoint,
        sam_config=args.sam_config,
        confidence_threshold=args.confidence_threshold
    )
    
    # 加载模型
    print("📦 加载模型...")
    if not integrator.load_models():
        print("❌ 模型加载失败")
        return
    
    print("✅ 模型加载成功")
    
    # 处理图像
    print(f"\n🎯 开始处理图像，输出目录: {args.output_dir}")
    print(f"📊 参数设置:")
    print(f"  • 输入目录: {args.input_dir}")
    print(f"  • 输出目录: {args.output_dir}")
    print(f"  • 处理图像数: {len(image_files)}")
    print(f"  • 置信度阈值: {args.confidence_threshold}")
    print(f"  • 保存单个掩码: {args.save_individual_masks}")
    
    # 处理图像
    results = integrator.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        save_individual_masks=args.save_individual_masks,
        image_paths=[str(p) for p in image_files]  # 传递筛选后的图像路径
    )
    
    # 显示结果
    print("\n" + "=" * 60)
    print("处理结果:")
    print(f"✅ 成功处理: {results['success']} 张图像")
    print(f"❌ 处理失败: {results['failed']} 张图像")
    
    if results['success'] > 0:
        print(f"\n📁 结果文件保存在: {args.output_dir}")
        print("\n生成的文件类型:")
        print("  • *_yolo_sam_result.png - 完整的检测和分割结果")
        if args.save_individual_masks:
            print("  • *_<类别名>_<序号>_conf<置信度>.png - 单个物体的分割结果")
        print("  • *_detection_info.json - 检测信息的JSON文件")
    
    print("\n🎉 演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()