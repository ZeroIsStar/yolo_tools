from PIL import Image
import os

def resize_images(input_folder, output_folder, size=(640, 640)):
    """
    批量调整图片大小
    :param input_folder: 包含图片的输入文件夹路径
    :param output_folder: 输出调整大小后的图片的文件夹路径
    :param size: 目标尺寸，默认为 (640, 640)
    """
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否是图片文件（通过扩展名）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 构造完整的文件路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开图片并调整大小
            try:
                with Image.open(input_path) as img:
                    img_resized = img.resize(size, Image.LANCZOS)  # 使用高质量的缩放算法
                    img_resized.save(output_path)  # 保存调整大小后的图片
                    print(f"Resized and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# 使用示例
input_folder = r"C:\Users\TJDX\Desktop\add"  # 替换为你的输入文件夹路径
output_folder = r"C:\Users\TJDX\Desktop\add"  # 替换为你的输出文件夹路径
resize_images(input_folder, output_folder)