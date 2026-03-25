import os
import glob
import shutil


def generate_missing_yolo_labels(image_dir, label_dir, output_dir=None):
    """
    生成缺失的YOLO标注文件

    参数:
        image_dir: 图片文件夹路径
        label_dir: 现有标注文件夹路径
        output_dir: 输出文件夹路径（如果为None，则使用label_dir）
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = label_dir

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG']

    # 获取所有图片文件（不带扩展名）
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))

    image_names = {os.path.splitext(os.path.basename(img))[0] for img in image_files}

    # 获取现有标注文件（不带扩展名）
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    label_names = {os.path.splitext(os.path.basename(label))[0] for label in label_files}

    print(f"A文件夹（图片）: {len(image_names)} 个文件")
    print(f"B文件夹（现有标注）: {len(label_names)} 个文件")

    # 找出缺失的标注文件
    missing_names = image_names - label_names

    print(f"\n需要生成 {len(missing_names)} 个缺失的标注文件:")

    # 生成缺失的标注文件（空文件）
    generated_count = 0
    for name in missing_names:
        # 生成对应的TXT文件路径
        txt_path = os.path.join(output_dir, f"{name}.txt")

        # 检查是否已经有文件存在（避免覆盖）
        if not os.path.exists(txt_path):
            # 创建空文件
            with open(txt_path, 'w', encoding='utf-8') as f:
                # YOLO格式：如果没有对象，文件可以是空的
                # 或者可以写注释行，如：＃ No objects in this image
                pass

            generated_count += 1
            print(f"✓ 已生成: {name}.txt")
        else:
            print(f"⚠ 文件已存在: {name}.txt")

    print(f"\n完成！共生成 {generated_count} 个标注文件")

    return missing_names, generated_count


def generate_yolo_dataset_structure(image_dir, label_dir, output_base_dir):
    """
    生成完整的YOLO数据集结构（包括train/val/test划分）
    """
    # 创建标准YOLO数据集目录结构
    directories = [
        'images/train',
        'images/val',
        'images/test',
        'labels/train',
        'labels/val',
        'labels/test'
    ]

    for dir_path in directories:
        os.makedirs(os.path.join(output_base_dir, dir_path), exist_ok=True)

    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    all_images = []

    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        all_images.extend(glob.glob(os.path.join(image_dir, f"*{ext.upper()}")))

    # 获取现有标注文件
    existing_labels = {os.path.splitext(os.path.basename(f))[0]
                       for f in glob.glob(os.path.join(label_dir, "*.txt"))}

    print(f"总图片数: {len(all_images)}")
    print(f"现有标注数: {len(existing_labels)}")

    # 按需生成缺失的标注文件
    for img_path in all_images:
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # 对应的标注文件路径
        label_src_path = os.path.join(label_dir, f"{img_name}.txt")

        # 如果标注文件不存在，创建空文件
        if not os.path.exists(label_src_path):
            label_src_path = os.path.join(label_dir, f"{img_name}.txt")
            with open(label_src_path, 'w', encoding='utf-8') as f:
                # 创建空文件
                pass

    print("数据集结构生成完成！")
    return True


def validate_dataset(image_dir, label_dir):
    """
    验证数据集完整性
    """
    print("=" * 50)
    print("数据集完整性验证")
    print("=" * 50)

    # 获取图片和标注文件
    image_files = glob.glob(os.path.join(image_dir, "*.*"))
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in
                   ['.jpg', '.jpeg', '.png', '.bmp', '.tif']]

    label_files = glob.glob(os.path.join(label_dir, "*.txt"))

    # 提取文件名（不带扩展名）
    image_names = {os.path.splitext(os.path.basename(f))[0] for f in image_files}
    label_names = {os.path.splitext(os.path.basename(f))[0] for f in label_files}

    # 分析
    common_files = image_names & label_names
    images_only = image_names - label_names
    labels_only = label_names - image_names

    print(f"总图片数: {len(image_names)}")
    print(f"总标注数: {len(label_names)}")
    print(f"匹配的文件: {len(common_files)}")
    print(f"只有图片没有标注: {len(images_only)}")
    print(f"只有标注没有图片: {len(labels_only)}")

    if images_only:
        print(f"\n缺少标注的图片: {list(images_only)[:10]}")  # 只显示前10个
        if len(images_only) > 10:
            print(f"... 还有 {len(images_only) - 10} 个")

    if labels_only:
        print(f"\n缺少图片的标注: {list(labels_only)[:10]}")
        if len(labels_only) > 10:
            print(f"... 还有 {len(labels_only) - 10} 个")

    # 生成统计报告
    report = {
        'total_images': len(image_names),
        'total_labels': len(label_names),
        'matched_pairs': len(common_files),
        'images_without_labels': len(images_only),
        'labels_without_images': len(labels_only),
        'completeness_rate': f"{len(common_files) / len(image_names) * 100:.2f}%" if image_names else "0%"
    }

    print(f"\n数据集完整率: {report['completeness_rate']}")

    return report


# 使用示例
if __name__ == "__main__":
    # 设置路径
    A_folder = r"C:\Users\TJDX\Desktop\test\collect\images\images"  # 图片文件夹
    B_folder = r"C:\Users\TJDX\Desktop\test\collect\labels(2)\labels"  # 现有标注文件夹

    # 1. 生成缺失的标注文件
    print("生成缺失的标注文件...")
    missing, generated = generate_missing_yolo_labels(A_folder, B_folder)

    # 2. 验证数据集完整性
    print("\n" + "=" * 50)
    print("验证数据集完整性")
    print("=" * 50)
    report = validate_dataset(A_folder, B_folder)

    # 3. 可选：创建完整的数据集结构
    create_full_structure = input("\n是否创建完整的YOLO数据集结构？(y/n): ")
    if create_full_structure.lower() == 'y':
        output_dir = "./yolo_dataset"
        generate_yolo_dataset_structure(A_folder, B_folder, output_dir)
        print(f"完整数据集已创建到: {output_dir}")