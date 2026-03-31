# LabelMe JSON 批量转 Mask 工具

## 功能简介:json2mask.py
批量将 [LabelMe](https://github.com/wkentaro/labelme) 标注的 JSON 文件转换为二值掩码图像（`.png`）。  
支持 **多边形（polygon）** 和 **矩形（rectangle）** 两种标注形状。  
生成的掩码中，标注区域为白色（像素值 1），背景为黑色（像素值 0），保存为单通道 8 位 PNG 图像。

## 函数原型
```python
batch_json_to_mask(input_dir: str, output_dir: str, target_class: Union[str, int, None] = None) -> None
```

# VOC 格式 XML 转 YOLO 格式 TXT 转换工具

## 功能简介:xml2yolo.py
将 **Pascal VOC 格式**的目标检测标注文件（`.xml`）批量转换为 **YOLO 格式**的文本文件（`.txt`）。  
- 自动提取 XML 中的所有类别，生成类别映射文件 `classes.txt`。
- 支持从已有的类别文件加载类别映射（避免重复提取）。
- 自动验证边界框坐标的有效性，并裁剪到图像范围内。
- 生成转换统计信息（转换文件数、总对象数、类别映射等），保存为 JSON 文件。

## 支持的 VOC XML 结构
```xml
<annotation>
    <size>
        <width>1920</width>
        <height>1080</height>
    </size>
    <object>
        <name>person</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>150</ymin>
            <xmax>300</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
    ...
</annotation>
