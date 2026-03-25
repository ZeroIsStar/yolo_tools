import os
import random
import shutil
from pathlib import Path


def split_dataset_simple(images_path="image", labels_path="labels", train_ratio=0.8):
    """
    简化版本的数据集分割
    """
    # 创建目录
    Path("images/train").mkdir(parents=True, exist_ok=True)
    Path("images/val").mkdir(parents=True, exist_ok=True)
    Path("labels/train").mkdir(parents=True, exist_ok=True)
    Path("labels/val").mkdir(parents=True, exist_ok=True)

    # 获取所有有效的图像-标签对
    valid_pairs = []
    images_path = Path(images_path)
    labels_path = Path(labels_path)

    for img_file in images_path.glob("*.*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            label_file = labels_path / (img_file.stem + '.txt')
            if label_file.exists():
                valid_pairs.append((img_file, label_file))
            else:
                print(f"警告: {img_file.name} 没有对应的标签文件")

    # 随机打乱并分割
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    print(f"总样本数: {len(valid_pairs)}")
    print(f"训练集: {len(train_pairs)}")
    print(f"验证集: {len(val_pairs)}")

    # 复制训练集
    for img_path, label_path in train_pairs:
        shutil.copy2(img_path, f"images/train/{img_path.name}")
        shutil.copy2(label_path, f"labels/train/{label_path.name}")

    # 复制验证集
    for img_path, label_path in val_pairs:
        shutil.copy2(img_path, f"images/val/{img_path.name}")
        shutil.copy2(label_path, f"labels/val/{label_path.name}")

    print("数据集分割完成!")


# 使用
if __name__ == "__main__":
    split_dataset_simple(r'C:\Users\TJDX\Desktop\FireAndSmoke\images',r'C:\Users\TJDX\Desktop\FireAndSmoke\labels')