import json
import cv2
import numpy as np
import os
from pathlib import Path


def batch_json_to_mask(input_dir, output_dir, target_class=None):
    """
    批量转换JSON文件为Mask
    :param input_dir: JSON文件目录
    :param output_dir: 输出目录
    :param target_class: 指定要转换的类别（如只转换特定类别）
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob('*.json'))

    for json_file in json_files:
        # 读取JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 创建mask
        height = data['imageHeight']
        width = data['imageWidth']
        mask = np.zeros((height, width), dtype=np.uint8)

        # 处理每个标注
        for shape in data['shapes']:
            label = shape['label']

            # 如果指定了目标类别，只转换该类别
            if target_class and label != target_class:
                continue

            points = np.array(shape['points'], dtype=np.int32)

            if shape['shape_type'] == 'polygon':
                cv2.fillPoly(mask, [points], 1)
            elif shape['shape_type'] == 'rectangle':
                cv2.rectangle(mask,
                              (int(points[0][0]), int(points[0][1])),
                              (int(points[1][0]), int(points[1][1])),
                              1, -1)
            # 可以添加其他形状类型的处理

        # 保存mask（使用与原图相同的名字）
        mask_name = json_file.stem + '.png'
        mask_path = output_dir / mask_name
        cv2.imwrite(str(mask_path), mask)

        print(f'已转换: {json_file.name} -> {mask_name}')


# 使用示例
batch_json_to_mask(r'C:\Users\TJDX\Desktop\video\lane\labels', r"C:\Users\TJDX\Desktop\video\lane\mask", target_class=0)