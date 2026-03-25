import os
import shutil


def move_matching_files(folder_a, folder_b, folder_c):
    """
    将A文件夹中与B文件夹同名的文件移动到C文件夹

    Args:
        folder_a: 源文件所在的A文件夹路径
        folder_b: 用于匹配名称的B文件夹路径
        folder_c: 目标C文件夹路径
    """
    # 确保C文件夹存在
    os.makedirs(folder_c, exist_ok=True)

    # 获取B文件夹中所有文件的名称（不带扩展名或带扩展名）
    b_files = set()

    # 方法1：只匹配文件名（不带扩展名）
    for file in os.listdir(folder_b):
        if os.path.isfile(os.path.join(folder_b, file)):
            # 去掉扩展名，只取文件名部分
            filename_without_ext = os.path.splitext(file)[0]
            b_files.add(filename_without_ext)

            # 如果需要完全匹配（包括扩展名），使用：
            # b_files.add(file)

    # 遍历A文件夹，移动匹配的文件
    moved_count = 0
    for file in os.listdir(folder_a):
        source_path = os.path.join(folder_a, file)

        if os.path.isfile(source_path):
            # 获取文件名（不带扩展名）
            filename_without_ext = os.path.splitext(file)[0]

            # 检查是否在B文件夹中存在同名文件
            if filename_without_ext in b_files:
                # 构建目标路径
                destination_path = os.path.join(folder_c, file)

                # 移动文件
                shutil.move(source_path, destination_path)
                print(f"已移动: {file} -> {folder_c}")
                moved_count += 1

    print(f"操作完成！共移动了 {moved_count} 个文件。")


# 使用示例
folder_a = r"C:\Users\TJDX\Desktop\tissue\images"  # A文件夹路径
folder_b = r"C:\Users\TJDX\Desktop\tissue\labels"  # B文件夹路径
folder_c = r"C:\Users\TJDX\Desktop\tissue\train"  # C文件夹路径

move_matching_files(folder_a, folder_b, folder_c)

# import os
#
#
# def find_different_files(folder1, folder2, ignore_ext=False, ignore_case=False):
#     """
#     找出两个文件夹中不同的文件
#
#     Args:
#         folder1: 第一个文件夹路径
#         folder2: 第二个文件夹路径
#         ignore_ext: 是否忽略文件扩展名
#         ignore_case: 是否忽略大小写
#     """
#     # 获取两个文件夹的文件列表
#     files1 = set()
#     files2 = set()
#
#     # 读取第一个文件夹
#     for root, dirs, files in os.walk(folder1):
#         for file in files:
#             file_path = os.path.join(root, file)
#             relative_path = os.path.relpath(file_path, folder1)
#
#             if ignore_ext:
#                 # 去掉扩展名
#                 name, ext = os.path.splitext(relative_path)
#                 key = name
#             else:
#                 key = relative_path
#
#             if ignore_case:
#                 key = key.lower()
#
#             files1.add(key)
#
#     # 读取第二个文件夹
#     for root, dirs, files in os.walk(folder2):
#         for file in files:
#             file_path = os.path.join(root, file)
#             relative_path = os.path.relpath(file_path, folder2)
#
#             if ignore_ext:
#                 name, ext = os.path.splitext(relative_path)
#                 key = name
#             else:
#                 key = relative_path
#
#             if ignore_case:
#                 key = key.lower()
#
#             files2.add(key)
#
#     # 找出差异
#     only_in_folder1 = files1 - files2
#     only_in_folder2 = files2 - files1
#
#     return only_in_folder1, only_in_folder2
#
#
# # 使用示例
# folder1 = r"C:\Users\TJDX\Desktop\video\lane\all"
# folder2 = r"C:\Users\TJDX\Desktop\video\lane\mask"
#
# only_in_1, only_in_2 = find_different_files(folder1, folder2)
#
# print("只在文件夹1中的文件:")
# for file in sorted(only_in_1):
#     print(f"  - {file}")
#
# print("\n只在文件夹2中的文件:")
# for file in sorted(only_in_2):
#     print(f"  - {file}")
#
# print(f"\n统计:")
# print(f"文件夹1独有的文件数: {len(only_in_1)}")
# print(f"文件夹2独有的文件数: {len(only_in_2)}")