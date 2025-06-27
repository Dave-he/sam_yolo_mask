#!/usr/bin/env python3
"""
项目安装和设置脚本
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path


def run_command(command, description):
    """运行命令并处理错误"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               capture_output=True, text=True)
        print(f"✅ {description}完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}失败: {e}")
        if e.stdout:
            print(f"输出: {e.stdout}")
        if e.stderr:
            print(f"错误: {e.stderr}")
        return False


def download_file(url, filepath, description):
    """下载文件"""
    print(f"\n📥 下载{description}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ {description}下载完成: {filepath}")
        return True
    except Exception as e:
        print(f"❌ {description}下载失败: {e}")
        return False


def setup_directories():
    """创建必要的目录"""
    print("\n📁 创建项目目录...")
    directories = ['checkpoints', 'images', 'output', 'demo_output']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ 创建目录: {directory}")
    
    return True


def install_dependencies():
    """安装Python依赖"""
    print("\n📦 安装Python依赖...")
    
    # 检查requirements.txt是否存在
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt文件不存在")
        return False
    
    # 安装基础依赖
    if not run_command("pip install -r requirements.txt", "安装基础依赖"):
        return False
    
    # 安装SAM2
    if not run_command(
        "pip install git+https://github.com/facebookresearch/segment-anything-2.git",
        "安装SAM2"
    ):
        print("⚠️  SAM2安装失败，请手动安装")
        return False
    
    return True


def download_sam2_models():
    """下载SAM2模型"""
    print("\n🤖 下载SAM2模型...")
    
    models = {
        'sam2.1_hiera_large.pt': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
        'sam2.1_hiera_base_plus.pt': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
        'sam2.1_hiera_small.pt': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt'
    }
    
    print("可用的SAM2模型:")
    print("1. Large (推荐，精度最高，约2.3GB)")
    print("2. Base+ (平衡性能，约900MB)")
    print("3. Small (速度最快，约180MB)")
    print("4. 全部下载")
    print("5. 跳过下载")
    
    choice = input("\n请选择要下载的模型 (1-5): ").strip()
    
    if choice == '5':
        print("跳过模型下载")
        return True
    elif choice == '1':
        selected_models = {'sam2.1_hiera_large.pt': models['sam2.1_hiera_large.pt']}
    elif choice == '2':
        selected_models = {'sam2.1_hiera_base_plus.pt': models['sam2.1_hiera_base_plus.pt']}
    elif choice == '3':
        selected_models = {'sam2.1_hiera_small.pt': models['sam2.1_hiera_small.pt']}
    elif choice == '4':
        selected_models = models
    else:
        print("无效选择，跳过模型下载")
        return True
    
    success_count = 0
    for filename, url in selected_models.items():
        filepath = os.path.join('checkpoints', filename)
        
        # 检查文件是否已存在
        if os.path.exists(filepath):
            print(f"✓ 模型已存在: {filename}")
            success_count += 1
            continue
        
        if download_file(url, filepath, f"SAM2模型 {filename}"):
            success_count += 1
    
    print(f"\n✅ 成功下载 {success_count}/{len(selected_models)} 个模型")
    return success_count > 0


def verify_installation():
    """验证安装"""
    print("\n🔍 验证安装...")
    
    try:
        # 检查基础依赖
        import torch
        import ultralytics
        import cv2
        import numpy as np
        from PIL import Image
        print("✓ 基础依赖检查通过")
        
        # 检查SAM2
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            print("✓ SAM2导入成功")
        except ImportError:
            print("❌ SAM2导入失败")
            return False
        
        # 检查YOLO
        from ultralytics import YOLO
        print("✓ YOLO导入成功")
        
        # 检查设备
        if torch.cuda.is_available():
            print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            print("✓ MPS可用 (Apple Silicon)")
        else:
            print("⚠️  仅CPU可用，性能可能较慢")
        
        # 检查模型文件
        checkpoint_dir = Path('checkpoints')
        sam_models = list(checkpoint_dir.glob('sam2*.pt'))
        if sam_models:
            print(f"✓ 找到 {len(sam_models)} 个SAM2模型")
            for model in sam_models:
                print(f"  - {model.name}")
        else:
            print("⚠️  未找到SAM2模型文件")
        
        print("\n🎉 安装验证完成！")
        return True
        
    except ImportError as e:
        print(f"❌ 依赖检查失败: {e}")
        return False


def main():
    """主安装流程"""
    print("=" * 60)
    print("YOLO11 + SAM2 项目安装脚本")
    print("=" * 60)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return 1
    
    print(f"✓ Python版本: {sys.version}")
    
    # 安装步骤
    steps = [
        ("创建目录", setup_directories),
        ("安装依赖", install_dependencies),
        ("下载模型", download_sam2_models),
        ("验证安装", verify_installation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"\n❌ {step_name}失败，安装中止")
            return 1
    
    print("\n" + "=" * 60)
    print("🎉 安装完成！")
    print("\n下一步:")
    print("1. 将测试图像放入 'images' 目录")
    print("2. 运行演示: python demo.py")
    print("3. 或使用命令行: python yolo_sam_integration.py --help")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  安装被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 安装过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)