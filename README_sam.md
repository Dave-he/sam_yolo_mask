# SAM2 增强版批量图像Mask生成器

基于Meta的SAM2 (Segment Anything Model 2) 的增强版批量图像分割工具，具有更好的错误处理、配置管理和性能优化。

## 🚀 快速开始


### 手动安装和运行
```bash
# 1. 安装依赖
pip install torch torchvision numpy matplotlib pillow opencv-python tqdm psutil pyyaml

# 2. 下载模型文件
cd checkpoints
bash download_ckpts.sh

# 3. 运行批量处理
python mask_generator.py ./images
```

## 📁 文件说明

- `mask_generator.py` - 增强版主程序

## ✨ 主要功能

### 🔧 增强功能
- **自动配置检测**: 自动查找可用的模型和配置文件
- **配置文件支持**: 支持YAML/JSON配置文件管理参数
- **错误处理**: 完善的错误处理和用户友好的错误信息
- **资源监控**: 内存和GPU使用监控
- **并行处理**: 支持多线程并行处理（适用于CPU密集型任务）
- **进度显示**: 实时进度条和处理状态
- **日志系统**: 可配置的日志级别和文件输出
- **参数验证**: 自动验证输入参数和文件路径

### 🎯 核心功能
- **批量处理**: 处理整个文件夹中的所有图片
- **随机采样**: 支持随机选择N张图片进行处理
- **多格式支持**: 支持JPG、PNG、BMP、TIFF等格式
- **多种输出**: PNG、JPG或两种格式同时输出
- **设备自适应**: 自动检测并使用CUDA、MPS或CPU
- **可视化结果**: 生成带有mask叠加的可视化图像
- **原始数据保存**: 可选保存原始mask数据（.npy格式）

## 📖 使用方法

### 基本用法
```bash
# 处理所有图片
python mask_generator.py ./images

# 随机处理10张图片
python mask_generator.py ./images --limit 10

# 指定输出目录
python mask_generator.py ./images --output ./results

# 自动检测模型文件
python mask_generator.py ./images --auto-detect
```

### 使用配置文件
```bash
# 使用配置文件
python mask_generator.py ./images --config-file config_example.yaml

# 配置文件 + 命令行参数（命令行参数优先）
python mask_generator.py ./images --config-file config.yaml --limit 5
```

### 高级用法
```bash
# 高质量模式
python mask_generator.py ./images \
  --points-per-side 64 \
  --pred-iou-thresh 0.92 \
  --stability-score-thresh 0.97

# 快速模式
python mask_generator.py ./images \
  --points-per-side 16 \
  --pred-iou-thresh 0.85 \
  --max-workers 4

# 保存详细日志
python mask_generator.py ./images \
  --log-level DEBUG \
  --log-file processing.log
```

## ⚙️ 参数说明

### 基本参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_dir` | 输入图片目录 | 必需 |
| `--output, -o` | 输出目录 | `output` |
| `--limit, -l` | 处理图片数量限制 | 无限制 |
| `--seed` | 随机种子 | `42` |

### 模型参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint, -c` | 模型文件路径 | 自动检测 |
| `--config` | 配置文件路径 | 自动检测 |
| `--auto-detect` | 自动检测模型文件 | `False` |

### 输出参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output-format` | 输出格式 (png/jpg/both) | `png` |
| `--quality` | JPEG质量 (1-100) | `95` |
| `--save-masks` | 保存原始mask数据 | `False` |

### 性能参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max-workers` | 并行线程数 | `1` |

### SAM2模型参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--points-per-side` | 每边采样点数 | `32` |
| `--pred-iou-thresh` | 预测IoU阈值 | `0.88` |
| `--stability-score-thresh` | 稳定性分数阈值 | `0.95` |

### 日志参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--log-level` | 日志级别 | `INFO` |
| `--log-file` | 日志文件路径 | 无 |

## 📋 配置文件示例

```yaml
# config_example.yaml
input_dir: "./images"
output_dir: "./output"
limit: 10
seed: 42

# 模型设置
checkpoint: "checkpoints/sam2.1_hiera_large.pt"
config: "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

# 输出设置
output_format: "png"
quality: 95
save_masks: false

# 性能设置
max_workers: 1

# SAM2参数
points_per_side: 32
pred_iou_thresh: 0.88
stability_score_thresh: 0.95

# 日志设置
log_level: "INFO"
log_file: null
```

## 🎛️ 不同场景的推荐配置

### 高质量模式（速度慢，质量高）
```yaml
points_per_side: 64
pred_iou_thresh: 0.92
stability_score_thresh: 0.97
max_workers: 1
```

### 快速模式（速度快，质量中等）
```yaml
points_per_side: 16
pred_iou_thresh: 0.85
stability_score_thresh: 0.90
max_workers: 4  # 仅CPU时有效
```

### 平衡模式（默认设置）
```yaml
points_per_side: 32
pred_iou_thresh: 0.88
stability_score_thresh: 0.95
max_workers: 1
```

## 🔧 环境要求

### Python版本
- Python 3.8+

### 必需依赖
```bash
pip install torch torchvision numpy matplotlib pillow opencv-python tqdm psutil pyyaml
```

### 可选依赖
- CUDA (用于GPU加速)
- MPS (用于Apple Silicon Mac GPU加速)

### 模型文件
下载SAM2模型文件到 `checkpoints/` 目录：
- `sam2.1_hiera_large.pt` (推荐)
- `sam2.1_hiera_base_plus.pt`
- `sam2.1_hiera_small.pt`
- `sam2.1_hiera_tiny.pt`

## 📊 输出结果

处理完成后，输出目录将包含：

```
output/
├── image1_masks.png          # 可视化结果
├── image2_masks.png
├── image1_masks.npy          # 原始mask数据（可选）
└── image2_masks.npy
```

每个可视化图像包含：
- 原始图像
- 彩色mask叠加
- mask边界轮廓

## 🚀 性能优化建议

### GPU使用
- 使用CUDA或MPS设备可显著提升速度
- GPU模式下建议 `max_workers=1`
- 监控GPU内存使用，避免OOM

### CPU使用
- CPU模式下可适当增加 `max_workers`
- 建议根据CPU核心数设置（通常为核心数的1-2倍）

### 内存管理
- 大图片或大批量处理时注意内存使用
- 程序会自动进行内存清理
- 可通过日志监控内存使用情况

### 参数调优
- `points_per_side`: 影响精度和速度，16-64之间选择
- `pred_iou_thresh`: 影响mask质量，0.8-0.95之间选择
- `stability_score_thresh`: 影响mask稳定性，0.9-0.98之间选择

## 🐛 故障排除

### 常见问题

1. **找不到模型文件**
   ```
   错误: 模型checkpoint不存在
   解决: 运行 --auto-detect 或手动下载模型文件
   ```

2. **CUDA内存不足**
   ```
   错误: CUDA out of memory
   解决: 减少batch size或使用CPU模式
   ```

3. **配置文件格式错误**
   ```
   错误: 不支持的配置文件格式
   解决: 使用YAML或JSON格式
   ```

4. **图片格式不支持**
   ```
   错误: 无法打开图片文件
   解决: 确保图片格式为JPG、PNG、BMP或TIFF
   ```

### 调试模式
```bash
# 启用详细日志
python mask_generator.py ./images --log-level DEBUG

# 保存日志到文件
python mask_generator.py ./images --log-file debug.log
```

## 🧪 测试

```bash
# 运行测试脚本
python test_batch.py

# 快速测试
python setup_and_run.py
```

## 📝 更新日志

### v2.0 (增强版)
- ✅ 添加配置文件支持
- ✅ 改进错误处理和用户体验
- ✅ 添加资源监控和内存管理
- ✅ 支持并行处理
- ✅ 添加进度条和详细日志
- ✅ 自动模型检测
- ✅ 参数验证和配置管理
- ✅ 快速安装脚本

### v1.0 (基础版)
- ✅ 基本批量处理功能
- ✅ 随机采样支持
- ✅ 多设备支持
- ✅ 可视化输出

## 📄 许可证

本项目基于SAM2项目，遵循相应的开源许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具！

## 📞 支持

如果遇到问题，请：
1. 查看故障排除部分
2. 运行调试模式获取详细信息
3. 提交Issue并附上错误日志