# YOLO11 + SAM2 物体检测与分割工具

基于Meta的SAM2 (Segment Anything Model 2) 和 YOLO11 的智能物体检测与精确分割工具。

## 🌟 功能特点

- **🎯 智能检测**: 使用YOLO11检测图像中的各种物体
- **✂️ 精确分割**: 使用SAM2对检测到的物体进行高精度轮廓分割
- **🎨 彩色标注**: 为不同类别的物体生成不同颜色的mask
- **📊 详细信息**: 显示物体类别名称和置信度
- **💾 多格式输出**: 支持单个物体和合成结果的多种保存格式
- **⚡ 批量处理**: 支持单张图片和批量目录处理

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd sam_yolo_mask

# 安装依赖
pip install -r requirements.txt

# 安装SAM2（从GitHub）
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 2. 模型准备

#### YOLO11模型
YOLO11模型会在首次运行时自动下载，无需手动准备。

#### SAM2模型
下载SAM2预训练模型并放置在`checkpoints`目录中：

```bash
# 创建checkpoints目录
mkdir -p checkpoints

# 下载SAM2模型（选择其中一个）
# Large模型（推荐，精度最高）
wget -O checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Base模型（平衡性能和速度）
wget -O checkpoints/sam2.1_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# Small模型（速度最快）
wget -O checkpoints/sam2.1_hiera_small.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

### 3. 准备测试图像

```bash
# 创建images目录
mkdir -p images

# 将要处理的图像放入images目录
# 支持格式：.jpg, .jpeg, .png, .bmp, .tiff, .tif
```

### 4. 运行演示

```bash
# 运行简单演示
python demo.py
```

## 📖 使用方法

### 命令行使用

#### 处理单张图片
```bash
python yolo_sam_integration.py --input path/to/image.jpg --output results
```

#### 批量处理目录
```bash
python yolo_sam_integration.py --input images/ --output results
```

#### 高级参数
```bash
python yolo_sam_integration.py \
    --input images/ \
    --output results \
    --yolo-model yolo11x.pt \
    --sam-checkpoint checkpoints/sam2.1_hiera_large.pt \
    --sam-config configs/sam2.1/sam2.1_hiera_l.yaml \
    --confidence 0.5 \
    --log-level INFO
```

### Python API使用

```python
from yolo_sam_integration import YOLOSAMIntegrator

# 创建集成器
integrator = YOLOSAMIntegrator(
    yolo_model_path="yolo11x.pt",
    sam_checkpoint="checkpoints/sam2.1_hiera_large.pt",
    sam_config="configs/sam2.1/sam2.1_hiera_l.yaml",
    confidence_threshold=0.5
)

# 加载模型
integrator.load_models()

# 处理单张图片
integrator.process_image("path/to/image.jpg", "output_dir")

# 批量处理
results = integrator.process_directory("images/", "output_dir")
print(f"成功处理: {results['success']} 张图片")
```

## 📁 输出文件说明

处理完成后，会在输出目录中生成以下文件：

- `*_yolo_sam_result.png` - 完整的检测和分割结果图
- `*_<类别名>_<序号>_conf<置信度>.png` - 单个物体的分割结果
- `*_detection_info.json` - 检测信息的JSON文件

### JSON文件格式
```json
{
  "image_path": "path/to/original/image.jpg",
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.95,
      "bbox": [100, 50, 300, 400],
      "color": [255, 0, 0]
    }
  ],
  "total_objects": 1
}
```

## ⚙️ 配置选项

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | - | 输入图像文件或目录路径（必需） |
| `--output` | `output` | 输出目录路径 |
| `--yolo-model` | `yolo11x.pt` | YOLO模型路径 |
| `--sam-checkpoint` | `checkpoints/sam2.1_hiera_large.pt` | SAM2模型checkpoint路径 |
| `--sam-config` | `configs/sam2.1/sam2.1_hiera_l.yaml` | SAM2配置文件路径 |
| `--confidence` | `0.5` | YOLO检测置信度阈值 |
| `--no-individual-masks` | `False` | 不保存单个物体的mask图像 |
| `--log-level` | `INFO` | 日志级别 (DEBUG/INFO/WARNING/ERROR) |

### 支持的模型

#### YOLO11模型
- `yolo11n.pt` - Nano（最快）
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra Large（最准确，默认）

#### SAM2模型
- `sam2.1_hiera_tiny.pt` - Tiny
- `sam2.1_hiera_small.pt` - Small
- `sam2.1_hiera_base_plus.pt` - Base+
- `sam2.1_hiera_large.pt` - Large（默认）

## 🔧 故障排除

### 常见问题

1. **模型下载失败**
   - 检查网络连接
   - 手动下载模型文件到对应目录

2. **CUDA内存不足**
   - 使用较小的模型（如`yolo11s.pt`和`sam2.1_hiera_small.pt`）
   - 减少批处理大小

3. **SAM2导入错误**
   - 确保正确安装SAM2：`pip install git+https://github.com/facebookresearch/segment-anything-2.git`

4. **配置文件不存在**
   - 检查`configs`目录是否完整
   - 重新克隆项目或下载配置文件

### 性能优化

- **GPU加速**: 确保安装了CUDA版本的PyTorch
- **模型选择**: 根据需求平衡精度和速度
- **图像尺寸**: 较大图像会消耗更多内存和时间

## 📊 性能基准

| 模型组合 | GPU内存 | 处理速度 | 精度 |
|----------|---------|----------|------|
| YOLO11n + SAM2-tiny | ~2GB | 最快 | 良好 |
| YOLO11s + SAM2-small | ~4GB | 快 | 较好 |
| YOLO11m + SAM2-base+ | ~6GB | 中等 | 好 |
| YOLO11x + SAM2-large | ~8GB | 慢 | 最佳 |

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目基于MIT许可证开源。

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO11实现
- [Meta SAM2](https://github.com/facebookresearch/segment-anything-2) - SAM2分割模型
- 所有为开源社区做出贡献的开发者们

## 📞 联系

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至：[your-email@example.com]

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！