# 故障排除指南

## 问题：处理图像时检测不到物体

### 症状
- 运行 `main.py` 时显示 "检测到 0 个物体"
- 所有图像处理失败
- 输出目录中没有生成结果文件

### 根本原因
默认的置信度阈值 (0.5) 过高，导致许多有效的检测结果被过滤掉。

### 解决方案

#### 1. 快速修复：降低置信度阈值
```bash
# 使用更低的置信度阈值
python3 main.py --confidence_threshold 0.25

# 或者使用更激进的阈值
python3 main.py --confidence_threshold 0.1
```

#### 2. 诊断工具
使用提供的诊断脚本来分析问题：
```bash
# 分析图像检测效果
python3 diagnose.py --input /path/to/images --samples 5

# 测试不同的置信度阈值
python3 diagnose.py --confidence_thresholds 0.1 0.2 0.3 0.4 0.5
```

#### 3. 推荐的置信度阈值设置
- **通用场景**: 0.25 (新的默认值)
- **宽松检测**: 0.1-0.2 (可能包含误检，但检测更全面)
- **严格检测**: 0.5-0.8 (减少误检，但可能遗漏物体)

### 验证修复

运行以下命令验证问题已解决：
```bash
# 测试处理少量图像
python3 main.py --limit 3 --confidence_threshold 0.25

# 检查输出目录
ls -la output/
```

成功的输出应该包含：
- `*_yolo_sam_result.png` - 完整的检测和分割结果
- `*_<类别名>_<序号>_conf<置信度>.png` - 单个物体的分割结果
- `*_detection_info.json` - 检测信息的JSON文件

### 其他可能的问题

#### 1. 图像中没有YOLO支持的物体类别
YOLO11模型支持80个常见物体类别，包括：
- 人物：person
- 车辆：car, truck, bus, motorcycle, bicycle
- 动物：cat, dog, horse, cow, etc.
- 日常物品：chair, table, laptop, phone, etc.
- 食物：apple, banana, cake, pizza, etc.

#### 2. 图像质量问题
- 图像过于模糊
- 光线不足
- 物体被严重遮挡
- 图像分辨率过低

#### 3. 模型文件问题
确保模型文件存在且完整：
```bash
# 检查YOLO模型
ls -la models/yolo11x.pt

# 检查SAM2模型
ls -la models/sam2.1_hiera_base_plus.pt

# 检查配置文件
ls -la configs/sam2.1/sam2.1_hiera_b+.yaml
```

### 性能优化建议

1. **批量处理优化**
   ```bash
   # 处理大量图像时使用限制
   python3 main.py --limit 50 --confidence_threshold 0.25
   ```

2. **内存优化**
   ```bash
   # 不保存单个掩码以节省空间
   python3 main.py --no_individual_masks
   ```

3. **可重复性**
   ```bash
   # 使用固定随机种子
   python3 main.py --seed 42 --limit 10
   ```

### 常见错误信息

| 错误信息 | 可能原因 | 解决方案 |
|---------|---------|----------|
| "检测到 0 个物体" | 置信度阈值过高 | 降低 `--confidence_threshold` |
| "模型加载失败" | 模型文件缺失或损坏 | 重新下载模型文件 |
| "未检测到任何物体" | 图像中没有支持的物体类别 | 尝试其他图像或降低阈值 |
| "SAM2分割失败" | SAM2模型问题 | 检查SAM2模型和配置文件 |

### 联系支持

如果问题仍然存在，请提供以下信息：
1. 运行 `diagnose.py` 的完整输出
2. 使用的命令行参数
3. 错误信息的完整日志
4. 示例图像（如果可能）