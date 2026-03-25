#!/usr/bin/env python3
import os
import glob

image_dir = r"C:\Users\TJDX\Desktop\clean_roboot\image"
text_dir = r"C:\Users\TJDX\Desktop\clean_roboot\labels"

# 支持的图片扩展名
image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

# 获取所有图片文件名（不含扩展名）
image_files = set()
for ext in image_extensions:
    for img in glob.glob(os.path.join(image_dir, f"*{ext}")):
        base = os.path.splitext(os.path.basename(img))[0]
        image_files.add(base)

# 获取所有文本文件名（不含扩展名）
text_files = set()
for txt in glob.glob(os.path.join(text_dir, "*.txt")):
    base = os.path.splitext(os.path.basename(txt))[0]
    text_files.add(base)

# 删除没有对应txt的图片
for ext in image_extensions:
    for img in glob.glob(os.path.join(image_dir, f"*{ext}")):
        base = os.path.splitext(os.path.basename(img))[0]
        if base not in text_files:
            print(f"删除图片: {img}")
            os.remove(img)

# 删除没有对应图片的文本
for txt in glob.glob(os.path.join(text_dir, "*.txt")):
    base = os.path.splitext(os.path.basename(txt))[0]
    if base not in image_files:
        print(f"删除文本: {txt}")
        os.remove(txt)

print("清理完成！")

