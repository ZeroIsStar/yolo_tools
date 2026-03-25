import os
import glob
from collections import Counter


"""
检查标签类别
"""
def count_yolo_labels(txt_folder_path):
    """
    统计YOLO格式标签文件夹中的类别和数量

    参数:
        txt_folder_path: 存放YOLO标签txt文件的文件夹路径
    """
    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(txt_folder_path, "*.txt"))

    if not txt_files:
        print(f"在文件夹 '{txt_folder_path}' 中没有找到txt文件")
        return

    print(f"找到 {len(txt_files)} 个标签文件")

    # 统计所有类别的数量
    class_counter = Counter()
    total_objects = 0

    # 遍历所有txt文件
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if line:  # 非空行
                        parts = line.split()
                        if parts:  # 确保有内容
                            class_id = int(parts[0])
                            class_counter[class_id] += 1
                            total_objects += 1
        except Exception as e:
            print(f"读取文件 {txt_file} 时出错: {e}")

    # 打印统计结果
    print("\n" + "=" * 50)
    print("YOLO标签统计结果")
    print("=" * 50)
    print(f"总标签文件数: {len(txt_files)}")
    print(f"总目标数量: {total_objects}")
    print(f"类别数量: {len(class_counter)}")
    print("\n类别分布:")
    print("-" * 30)

    # 按类别ID排序后显示
    for class_id in sorted(class_counter.keys()):
        count = class_counter[class_id]
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"类别 {class_id}: {count} 个 ({percentage:.2f}%)")

    return class_counter


# 使用方法
if __name__ == "__main__":
    # 指定你的YOLO标签文件夹路径
    txt_folder = r"C:\Users\TJDX\PyCharmMiscProject\yolo_tool\labels\train"  # 修改为你的路径

    # 执行统计
    stats = count_yolo_labels(txt_folder)