import os
from pathlib import Path


def clean_yolo_txt_files_advanced(txt_folder, remove_class_ids=None, remove_duplicate_classes=False):
    """
    清理YOLO格式的txt文件，支持更灵活的配置

    参数:
    - txt_folder: txt文件所在文件夹
    - remove_class_ids: 要移除的类别ID列表，默认[3]
    - remove_duplicate_classes: 是否删除同一类别的重复标注（即使坐标不同）
    """
    if remove_class_ids is None:
        remove_class_ids = ['3']

    txt_folder = Path(txt_folder)

    # 支持的文本格式
    txt_extensions = ['.txt']

    # 统计信息
    stats = {
        'total_files': 0,
        'total_original_annotations': 0,
        'total_final_annotations': 0,
        'removed_by_class': {cid: 0 for cid in remove_class_ids},
        'removed_duplicates': 0,
        'removed_same_class_duplicates': 0,
        'files_modified': 0
    }

    # 遍历所有txt文件
    for txt_file in txt_folder.iterdir():
        if txt_file.suffix.lower() in txt_extensions:
            stats['total_files'] += 1

            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            original_count = len(lines)
            stats['total_original_annotations'] += original_count

            # 处理每一行
            processed_lines = []
            processed_annotations = set()  # 用于检测完全相同的标注
            class_counts = {}  # 用于统计每个类别的出现次数

            for line in lines:
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                parts = line.split()
                if len(parts) != 5:  # 确保是有效的YOLO格式
                    processed_lines.append(line)  # 保持非标准格式不变
                    continue

                class_id = parts[0]
                coordinates = tuple(parts[1:])

                # 检查是否是要移除的类别
                if class_id in remove_class_ids:
                    stats['removed_by_class'][class_id] += 1
                    continue

                # 检查是否是完全相同的标注
                annotation_key = (class_id,) + coordinates
                if annotation_key in processed_annotations:
                    stats['removed_duplicates'] += 1
                    continue

                # 检查是否要删除同一类别的重复标注
                if remove_duplicate_classes:
                    # 更新类别计数
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1

                    # 如果这是该类别第一次出现，保留；否则删除
                    if class_counts[class_id] > 1:
                        stats['removed_same_class_duplicates'] += 1
                        continue

                # 记录并保留标注
                processed_annotations.add(annotation_key)
                processed_lines.append(line)

            # 更新统计信息
            final_count = len(processed_lines)
            stats['total_final_annotations'] += final_count

            # 如果有修改，写回文件
            if original_count != final_count:
                stats['files_modified'] += 1
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(processed_lines))

                # 打印每个文件的详细修改信息
                print(f"已处理: {txt_file.name}")
                print(f"  原始: {original_count}个标注 → 最终: {final_count}个标注")
                removed_count = original_count - final_count
                if removed_count > 0:
                    print(f"  移除: {removed_count}个标注")

    # 打印总体统计信息
    print("\n" + "=" * 60)
    print("处理完成！详细统计信息：")
    print("=" * 60)
    print(f"总处理文件数: {stats['total_files']}")
    print(f"修改的文件数: {stats['files_modified']}")
    print(f"总原始标注数: {stats['total_original_annotations']}")
    print(f"总最终标注数: {stats['total_final_annotations']}")
    print(f"净移除标注数: {stats['total_original_annotations'] - stats['total_final_annotations']}")
    print()

    print("按类别移除统计：")
    for class_id, count in stats['removed_by_class'].items():
        if count > 0:
            print(f"  类别 {class_id}: {count}个")

    print(f"移除完全相同标注: {stats['removed_duplicates']}个")

    if remove_duplicate_classes:
        print(f"移除同类重复标注: {stats['removed_same_class_duplicates']}个")

    print("=" * 60)


# 使用示例
if __name__ == "__main__":
    # 设置你的YOLO txt文件夹路径
    txt_folder_path = r"C:\Users\TJDX\PyCharmMiscProject\yolo_tool\labels\val"  # 修改为你的路径

    # 方案1: 基本清理（只移除类别3和完全相同的标注）
    print("执行基本清理...")
    clean_yolo_txt_files_advanced(
        txt_folder=txt_folder_path,
        remove_class_ids=['1'],  # 要移除的类别
        remove_duplicate_classes=False  # 不删除同一类别的重复标注
    )

    # 方案2: 严格清理（移除类别3和所有重复标注）
    # print("\n执行严格清理...")
    # clean_yolo_txt_files_advanced(
    #     txt_folder=txt_folder_path,
    #     remove_class_ids=['3'],  # 要移除的类别
    #     remove_duplicate_classes=True  # 删除同一类别的重复标注
    # )

    # 方案3: 移除多个类别（如3和5）
    # print("\n执行多类别清理...")
    # clean_yolo_txt_files_advanced(
    #     txt_folder=txt_folder_path,
    #     remove_class_ids=['3', '5'],  # 要移除的类别
    #     remove_duplicate_classes=False
    # )