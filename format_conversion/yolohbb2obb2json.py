import os
import json
import cv2
import glob
import numpy as np

def obb_txt_to_labelme(txt_dir, img_dir, output_dir, class_map, img_ext='.jpg'):
    """
    将 YOLO OBB 格式（四点坐标）的 txt 文件转换为 LabelMe JSON（多边形）。
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))

    for txt_file in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        image_file = os.path.join(img_dir, base_name + img_ext)

        if not os.path.exists(image_file):
            print(f"警告: 找不到图片 {image_file}，跳过 {txt_file}")
            continue

        img = cv2.imread(image_file)
        if img is None:
            print(f"警告: 无法读取图片 {image_file}，跳过")
            continue
        h, w = img.shape[:2]

        # 构造 JSON 基础结构
        labelme_json = {
            "version": "5.8.1",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_file),
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w
        }

        with open(txt_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:  # class_id + 8 coordinates
                continue

            class_id = int(parts[0])
            # 归一化坐标转像素
            coords_norm = list(map(float, parts[1:9]))
            coords_pixel = [coord * (w if i % 2 == 0 else h) for i, coord in enumerate(coords_norm)]
            # 重组为点列表：[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            points = [[coords_pixel[i], coords_pixel[i+1]] for i in range(0, 8, 2)]

            label_name = class_map.get(class_id, str(class_id))
            labelme_json["shapes"].append({
                "label": label_name,
                "points": points,
                "group_id": None,
                "shape_type": "oriented_rectangle",   # 多边形，可表示任意旋转矩形
                "flags": {}
            })

        output_file = os.path.join(output_dir, base_name + '.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(labelme_json, f, indent=2, ensure_ascii=False)
        print(f"已转换: {txt_file} -> {output_file}")


CLASS_MAP = {0: 'wire', 1: 'water_pipe'}   # 按实际情况

obb_txt_to_labelme(
    txt_dir=r"C:/Users/TJDX/Desktop/优化",   # 刚才生成的OBB txt目录
    img_dir=r"C:\Users\TJDX\Desktop\new\alltotalwire_yolo\alltotalwire_yolo\images\val",  # 原始图片目录
    output_dir=r"C:\Users\TJDX\Desktop\new\change\val",      # 输出JSON目录
    class_map=CLASS_MAP,
    img_ext='.png'
)