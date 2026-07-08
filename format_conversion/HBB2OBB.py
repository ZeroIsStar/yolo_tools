import os
import numpy as np
from pathlib import Path

def hbb_to_obb_plain(image_path, hbb_ann_path, output_dir, img_ext='.jpg'):
    """
    仅根据 HBB 标注（中心+宽高）生成 OBB 四点坐标（角度恒为0），
    不读取图片，不做任何图像处理。
    """
    # 读取标注文件
    with open(hbb_ann_path, 'r') as f:
        lines = f.readlines()

    obb_annotations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        x_c, y_c, box_w, box_h = map(float, parts[1:5])

        # 直接使用归一化坐标，无需图片尺寸
        # 四个角点（未旋转，即角度=0）
        # 顺序：左上、右上、右下、左下（顺时针）
        x1 = x_c - box_w / 2
        y1 = y_c - box_h / 2
        x2 = x_c + box_w / 2
        y2 = y_c - box_h / 2
        x3 = x_c + box_w / 2
        y3 = y_c + box_h / 2
        x4 = x_c - box_w / 2
        y4 = y_c + box_h / 2

        # 裁剪到 [0,1] 防止数值溢出（但通常不会）
        coords = np.clip([x1, y1, x2, y2, x3, y3, x4, y4], 0, 1)

        obb_line = f"{class_id} " + " ".join([f"{c:.6f}" for c in coords])
        obb_annotations.append(obb_line)

    # 写入输出文件
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, Path(hbb_ann_path).name)
    with open(output_path, 'w') as f:
        f.write("\n".join(obb_annotations))

    print(f"转换完成: {hbb_ann_path} -> {output_path}")


def batch_convert_hbb_to_obb_plain(image_dir, hbb_dir, output_dir, img_ext='.jpg'):
    """批量转换（无需真实图片，只需标注文件）"""
    hbb_files = list(Path(hbb_dir).glob("*.txt"))
    for hbb_path in hbb_files:
        # 这里不再需要检查图片是否存在，因为完全不依赖图片
        hbb_to_obb_plain(None, hbb_path, output_dir, img_ext)


if __name__ == "__main__":
    image_dir = r"..."  # 实际上用不到，但保留占位
    hbb_dir = r"C:\Users\TJDX\Desktop\new\alltotalwire_yolo\alltotalwire_yolo\labels\train"
    output_dir = r"C:\Users\TJDX\Desktop\new\hbb2obb\images\train"
    batch_convert_hbb_to_obb_plain(image_dir, hbb_dir, output_dir, img_ext='.png')