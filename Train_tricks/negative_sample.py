import os
from pathlib import Path


def create_txt_for_images(folder_path):
    """
    为文件夹中的所有图片创建同名的txt空文件

    参数:
    folder_path: 图片文件夹路径
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ico'}

    # 转换为Path对象
    folder = Path(folder_path)

    # 检查文件夹是否存在
    if not folder.exists():
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return

    if not folder.is_dir():
        print(f"错误：'{folder_path}' 不是文件夹")
        return

    # 遍历文件夹中的文件
    created_count = 0
    skipped_count = 0

    for file_path in folder.iterdir():
        if file_path.is_file():
            # 检查文件扩展名是否为图片格式
            if file_path.suffix.lower() in image_extensions:
                # 构建对应的txt文件名
                txt_path = file_path.with_suffix('.txt')

                # 检查是否已存在同名txt文件
                if not txt_path.exists():
                    try:
                        # 创建空txt文件
                        txt_path.touch()
                        print(f"✓ 已创建: {txt_path.name}")
                        created_count += 1
                    except Exception as e:
                        print(f"✗ 创建失败 {file_path.name}: {e}")
                else:
                    print(f"○ 已存在: {txt_path.name}")
                    skipped_count += 1

    print(f"\n完成！共创建 {created_count} 个文件，跳过 {skipped_count} 个已存在的文件")


# 使用示例
if __name__ == "__main__":
    # 方法1: 使用当前脚本所在目录
    # create_txt_for_images(".")

    # 方法2: 指定特定文件夹
    folder_path = input("请输入图片文件夹路径（直接按Enter使用当前目录）: ").strip()

    if not folder_path:
        folder_path = "."  # 使用当前目录

    create_txt_for_images(folder_path)
