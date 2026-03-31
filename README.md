# 一、YOLO目标检测数据采集，清理，划分，检查工具包
- data_collection: ROS2机器人数据采集（主要是RGB和RGBD数据，以及过滤掉损坏图片的工具）
- format_conversion: 数据转换工具，数据标注结合labelme（内含SAM3分割大模型可以辅助标注数据，标注完后通过mask转txt可以快速生成标签）
- train_tricks:后续提升模型的泛化性和提升单个模型的检测数量的数据处理工具
- Dataset_tools:数据集处理工具，包括标签图像匹配，批量修改类别id，划分数据集，数据统计，模糊图像去除，去重等

# 二、模型训练（ultralytics文档说明，模型训练以及机器人目标检测训练trick，相关参数快速入门，以及模型结构修改教程）
## ultralytics文档说明
### 1.安装
```bash
# bash
pip install ultralytics
```
