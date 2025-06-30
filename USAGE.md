# YOLO11 + SAM2 主脚本使用指南

## 概述

`main.py` 是升级版的演示脚本，支持命令行参数，可以灵活地配置各种处理选项。

## 主要功能

- 🎯 **YOLO11物体检测** + **SAM2精确分割**
- 📊 **命令行参数支持**，灵活配置
- 🎲 **随机图像选择**，支持 `--limit` 参数
- 🎨 **彩色分割掩码**，每个类别使用不同颜色
- 💾 **多种输出格式**，包括合成结果和单个物体掩码
- 🔧 **可配置的模型路径和参数**

## 安装要求

确保已安装所有依赖：

```bash
pip install -r requirements.txt
```

## 基本用法

### 1. 处理所有图像

```bash
python3 main.py
```

这将处理 `images/` 目录中的所有图像，结果保存到 `output/` 目录。

### 2. 随机选择图像处理

```bash
# 随机选择5张图像进行处理
python3 main.py --limit 5

# 随机选择10张图像，使用固定种子确保可重复
python3 main.py --limit 10 --seed 42
```

### 3. 指定输入和输出目录

```bash
python3 main.py --input_dir my_images --output_dir results
```

### 4. 调整检测参数

```bash
# 提高置信度阈值，只检测高置信度的物体
python3 main.py --confidence_threshold 0.7

# 不保存单个物体的分割结果
python3 main.py --no_individual_masks
```

### 5. 使用自定义模型

```bash
python3 main.py \
  --yolo_model models/yolo11s.pt \
  --sam_checkpoint models/sam2.1_hiera_small.pt \
  --sam_config configs/sam2.1/sam2.1_hiera_s.yaml
```

## 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--limit` | int | 0 | 限制处理的图像数量，0表示处理所有图像 |
| `--input_dir` | str | "images" | 输入图像目录 |
| `--output_dir` | str | "output" | 输出结果目录 |
| `--yolo_model` | str | "models/yolo11x.pt" | YOLO模型路径 |
| `--sam_checkpoint` | str | "models/sam2.1_hiera_base_plus.pt" | SAM2模型检查点路径 |
| `--sam_config` | str | "configs/sam2.1/sam2.1_hiera_b+.yaml" | SAM2模型配置文件路径 |
| `--confidence_threshold` | float | 0.5 | 检测置信度阈值 |
| `--save_individual_masks` | flag | True | 是否保存单个物体的分割结果 |
| `--no_individual_masks` | flag | False | 不保存单个物体的分割结果 |
| `--seed` | int | None | 随机种子，用于可重复的随机选择 |

## 输出文件说明

处理完成后，输出目录将包含以下文件：

### 1. 合成结果图像
- **文件名格式**: `{原文件名}_yolo_sam_result.png`
- **内容**: 包含所有检测物体的边界框、标签和分割掩码的完整结果图

### 2. 单个物体分割结果（可选）
- **文件名格式**: `{原文件名}_{类别名}_{序号}_conf{置信度}.png`
- **内容**: 单个物体的分割结果，包含原图和该物体的彩色掩码
- **示例**: `image1_person_01_conf0.85.png`

### 3. 检测信息JSON文件
- **文件名格式**: `{原文件名}_detection_info.json`
- **内容**: 包含所有检测结果的详细信息，包括边界框坐标、置信度、类别等

## 实用示例

### 快速测试（处理少量图像）

```bash
# 随机选择3张图像进行快速测试
python3 main.py --limit 3 --confidence_threshold 0.6
```

### 高质量处理

```bash
# 使用高置信度阈值，保存所有结果
python3 main.py --confidence_threshold 0.8 --save_individual_masks
```

### 批量处理大量图像

```bash
# 处理大量图像，只保存合成结果以节省空间
python3 main.py --input_dir large_dataset --no_individual_masks
```

### 可重复的随机采样

```bash
# 使用固定种子进行可重复的随机选择
python3 main.py --limit 20 --seed 123 --output_dir sample_results
```

## 注意事项

1. **模型文件**: 确保SAM2模型文件已下载并放置在正确位置
2. **GPU内存**: 处理大图像时可能需要较多GPU内存
3. **输出目录**: 输出目录会自动创建，如果已存在同名文件会被覆盖
4. **支持格式**: 支持常见图像格式：jpg, jpeg, png, bmp, tiff, tif
5. **随机选择**: 使用 `--limit` 参数时，每次运行会随机选择不同的图像，除非指定 `--seed`

## 故障排除

### 常见问题

1. **检测不到物体 (最常见问题)**
   - **症状**: 显示 "检测到 0 个物体"，所有图像处理失败
   - **原因**: 置信度阈值过高 (默认0.25)
   - **解决方案**: 
     ```bash
     # 使用更低的置信度阈值
     python3 main.py --confidence_threshold 0.1
     
     # 运行诊断工具
     python3 diagnose.py --input your_image_dir --samples 5
     ```
   - **详细指南**: 参见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

2. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确保有足够的内存和存储空间

3. **没有找到图像文件**
   - 检查输入目录路径是否正确
   - 确保目录中包含支持的图像格式

4. **处理速度慢**
   - 考虑使用较小的YOLO模型（如yolo11s.pt）
   - 提高置信度阈值以减少检测数量
   - 使用 `--limit` 参数处理部分图像

### 诊断工具

使用内置的诊断脚本来分析检测问题：

```bash
# 基本诊断
python3 diagnose.py

# 指定输入目录和样本数
python3 diagnose.py --input /path/to/images --samples 5

# 测试不同置信度阈值
python3 diagnose.py --confidence_thresholds 0.1 0.2 0.3 0.5
```

### 获取详细日志

如果遇到问题，可以查看详细的处理日志来诊断问题。

## 从demo.py迁移

如果你之前使用的是 `demo.py`，迁移到 `main.py` 非常简单：

```bash
# 原来的demo.py
python3 demo.py

# 现在的main.py（等效功能）
python3 main.py

# 新增功能：随机选择图像
python3 main.py --limit 5
```

所有原有功能都保持不变，同时增加了更多灵活的配置选项。