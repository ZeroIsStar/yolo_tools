import os
import numpy as np
from PIL import Image
from collections import defaultdict

def count_classes_pil(folder_path):
    """
    使用PIL和Numpy统计文件夹内所有PNG标签文件的类别。

    Args:
        folder_path (str): 包含PNG标签文件的文件夹路径。

    Returns:
        dict: 包含总类别数和各类别出现文件数的字典。
    """
    all_unique_classes = set()
    class_file_count = defaultdict(int) # 记录每个类别出现在多少个文件中
    processed_files = 0

    # 遍历文件夹内所有.png文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            try:
                # 1. 使用PIL打开图像，并转换为numpy数组
                # 注意：PNG标签通常是单通道灰度图，模式为'L'（8位）或 'I'（32位）
                with Image.open(file_path) as img:
                    label_array = np.array(img) # 此时数组形状为 (H, W)

                # 2. 找出当前文件中的唯一类别ID
                unique_in_file = np.unique(label_array)
                # 3. 更新总类别集合和文件计数
                all_unique_classes.update(unique_in_file)
                for cls in unique_in_file:
                    class_file_count[cls] += 1

                processed_files += 1
                if processed_files % 100 == 0:
                    print(f"已处理 {processed_files} 个文件...")

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    print(f"\n处理完成！共扫描 {processed_files} 个文件。")
    print(f"所有文件中出现的不重复类别总数为: {len(all_unique_classes)}")
    print(f"类别ID列表 (已排序): {sorted(all_unique_classes)}")

    # 返回详细统计结果
    return {
        'total_class_count': len(all_unique_classes),
        'class_ids': sorted(all_unique_classes),
        'class_distribution': dict(sorted(class_file_count.items())), # 每个类别出现的文件数
        'files_processed': processed_files
    }


# ====== 使用示例 ======
if __name__ == "__main__":
    # 请将此处路径替换为你的实际文件夹路径
    label_folder = r"C:\Users\TJDX\Desktop\CropsOrWeed9"
    results = count_classes_pil(label_folder)

    # 打印更详细的分布（可选）
    print("\n===== 类别详情 =====")
    for cls_id, file_count in results['class_distribution'].items():
        print(f"  类别 {cls_id:3d} -> 出现在 {file_count:4d} 个文件中")