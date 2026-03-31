# 一、YOLO目标检测数据采集，清理，划分，检查工具包
- data_collection: ROS2机器人数据采集（主要是RGB和RGBD数据，以及过滤掉损坏图片的工具）
- format_conversion: 数据转换工具，数据标注结合labelme（内含SAM3分割大模型可以辅助标注数据，标注完后通过mask转txt可以快速生成标签）
- train_tricks:后续提升模型的泛化性和提升单个模型的检测数量的数据处理工具
- Dataset_tools:数据集处理工具，包括标签图像匹配，批量修改类别id，划分数据集，数据统计，模糊图像去除，去重等
- label_tools: 数据标注工具

# 二、模型训练（ultralytics文档说明，模型训练以及机器人目标检测训练技巧，相关参数）
## ultralytics文档说明
### 1.安装及 ultralytics说明
```bash
# bash（安装的包一般是在：miniconda3/lib/python3.8/site-packages/ultralytics）
pip install ultralytics
```

<div align="center">
  <img src="ultralytics.png" alt="ultralytics" width="1000">
  <br>
  <em>图1: ultralytics目录</em>
</div>

### 2.模型的配置文件
```txt
模型的配置文件在ultralytics/cfg/models目录下（yolo3，5，6，8，9，10，11，12，26）

```

<div align="center">
  <img src="cfg.png" alt="ultralytics" width="1000">
  <br>
  <em>图2: yolo11的结构图</em>
</div>
对比可以很清楚的看到各个模型之间结构的差异，其中yolo11_Adown.yaml为修改结构的目标检测文件，best.pt为其训练的最佳模型权重文件


### 3.模型的训练参考
```txt
官方的模型的配置文件在ultralytics/cfg/models目录下（yolo3，5，6，8，9，10，11，12，26）
模型训练微调量化和转出推理预测参考模型训练微调量化和转出推理预测.ipynb
其中类别的文件为data.yaml,训练配置文件为arg.yaml(各个参数的详细说明在arg.yaml做了说明)
```
### 4.最实用的模型的训练技巧

- 超参数配置：warmup_epochs: 0    # 预热轮数（学习率从低到高线性增长及训练开始由0-初始学习率线性增加），好处是模型前期快速拟合,降低训练周期，一般设置为总训练轮数的10%左右比较合。
- cos_lr: true                   # 是否使用余弦退火学习率调度器
- close_mosaic: 0                # 马赛克增强将多张图片拼接为一张后输入模型训练，好处是增强模型的泛化能力，坏处是训练的数据很多可能不符合现实规律，易造成误检，建议模型最后20%轮关闭它
- erasing: 0.4                   # 随机擦除的概率（随机遮挡矩形区域），模拟随机遮挡。

### 5.模型优化之数据分布分析和优化方法参考

- 数据集分析，以yolo11_Adown.yaml为例：
  模型训练开始时会生成相关数据集label统计信息,如实例的数量（检测框的数量），检测目标的长宽，数量越多颜色越深，以及检测目标的尺度分布
  <div align="center">
  <img src="labels.jpg" alt="ultralytics" width="1000">
  <br>
  <em>图3: labels示例</em>
</div>
图3示例可知尺度跨度大，大量小尺度的目标，即可以在原yolo结构的基础上，从小尺度目标上改模型结构，所以借鉴了yolov9的Adown结构，同时修改检测头以及特征融合方式以提升对于小尺度目标的检测效果

在数据量足够大的情况下可以只微调检测头，来适应不同区域下的检测任务，提升区域检测性能，微调代码参考模型训练微调量化和转出推理预测.ipynb文件

