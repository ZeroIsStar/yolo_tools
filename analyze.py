import os
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
# ========== 配置五个文件夹路径 ==========
folders = [
    r"C:\Users\TJDX\Downloads\runs\runs\detect\predict_car_S\labels",
    r"C:\Users\TJDX\Downloads\runs\runs\detect\predict_car_F\labels",
    r"C:\Users\TJDX\Downloads\runs\runs\detect\predict_car_eq\labels",
    r"C:\Users\TJDX\Downloads\runs\runs\detect\predict_car_BGR\labels",
    r"C:\Users\TJDX\Downloads\runs\runs\detect\predict_car\labels"
]
# =====================================

def get_txt_files(folder):
    """返回文件夹中所有 .txt 文件的文件名（不含路径）"""
    files = set()
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.lower().endswith('.txt'):
                files.add(f)
    return files

def count_classes_in_txt(filepath):
    """读取 YOLO 格式 txt，返回不重复的类别 ID 数量"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        classes = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                classes.add(parts[0])   # 第一个字段是类别ID
        return len(classes)
    except:
        return None   # 文件损坏或无法读取

# 1. 收集所有文件夹中的 txt 文件名
all_files = set()
folder_files = []      # 每个文件夹的文件名集合
for folder in folders:
    files = get_txt_files(folder)
    folder_files.append(files)
    all_files.update(files)

print(f"在五个文件夹中共发现 {len(all_files)} 个不同的 txt 文件名。\n")

# 2. 检查文件名一致性
print("=" * 60)
print("【文件名不一致的文件】（至少一个文件夹缺失）")
print("=" * 60)
name_inconsistent = []
for fname in sorted(all_files):
    present = [i+1 for i, files in enumerate(folder_files) if fname in files]
    if len(present) != len(folders):   # 不是所有文件夹都有
        name_inconsistent.append(fname)
        missing = [i+1 for i in range(len(folders)) if i+1 not in present]
        print(f"📄 {fname}")
        print(f"   ✅ 存在于文件夹: {present}")
        print(f"   ❌ 缺失于文件夹: {missing}")
        print()

if not name_inconsistent:
    print("✅ 所有文件在五个文件夹中均存在，文件名完全一致。\n")

# 3. 检查内容类别数一致性（仅针对所有文件夹都存在的文件）
print("\n" + "=" * 60)
print("【类别数不一致的文件】（五个文件夹均存在，但类别ID数量不同）")
print("=" * 60)
content_inconsistent = []
common_files = [f for f in all_files if all(f in files for files in folder_files)]
for fname in sorted(common_files):
    class_counts = []
    filepaths = []
    for i, folder in enumerate(folders):
        path = os.path.join(folder, fname)
        cnt = count_classes_in_txt(path)
        class_counts.append(cnt)
        filepaths.append(path)
    # 检查是否所有计数相等（忽略 None 表示读取失败）
    if len(set(class_counts)) != 1:
        content_inconsistent.append(fname)
        print(f"📄 {fname}")
        for i, cnt in enumerate(class_counts, 1):
            status = f"{cnt} 个类别" if cnt is not None else "读取失败"
            print(f"   文件夹{i}: {status}")
        print()

if not content_inconsistent:
    print("✅ 所有共同存在的文件类别数完全一致。\n")

# 4. 输出总结
print("\n" + "=" * 60)
print("统计汇总")
print("=" * 60)
print(f"总不同文件名数量: {len(all_files)}")
print(f"文件名不一致数量: {len(name_inconsistent)}")
print(f"共同存在的文件数量: {len(common_files)}")
print(f"类别数不一致数量: {len(content_inconsistent)}")


