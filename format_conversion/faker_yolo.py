from pathlib import Path


def create_empty_txt_for_unlabeled_images(image_folder, label_folder):
    """
    为图像文件夹中的每张图片，在标签文件夹中检查是否存在对应的txt文件，
    若不存在则创建空txt文件（即视为无标注）。

    参数:
    image_folder: 图片所在文件夹路径
    label_folder: 标签文件存放文件夹路径
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ico'}

    image_dir = Path(image_folder)
    label_dir = Path(label_folder)

    # 检查文件夹是否存在
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"错误：图片文件夹 '{image_folder}' 不存在或不是文件夹")
        return
    if not label_dir.exists():
        print(f"标签文件夹 '{label_folder}' 不存在，正在创建...")
        try:
            label_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"创建标签文件夹失败: {e}")
            return

    created_count = 0
    skipped_count = 0

    for file_path in image_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            # 对应的txt文件路径（在标签文件夹中，文件名相同但扩展名为.txt）
            txt_path = label_dir / (file_path.stem + '.txt')

            if not txt_path.exists():
                try:
                    txt_path.touch()
                    print(f"✓ 创建空标注: {txt_path.name} (对应图片: {file_path.name})")
                    created_count += 1
                except Exception as e:
                    print(f"✗ 创建失败 {txt_path.name}: {e}")
            else:
                skipped_count += 1

    print(f"\n完成！共为 {created_count} 张无标注图像创建了空txt文件，跳过 {skipped_count} 张已有标注的图像。")


if __name__ == "__main__":
    # 交互式输入两个路径
    img_folder = input("请输入图片文件夹路径: ").strip()
    label_folder = input("请输入标签文件夹路径（若不存在将自动创建）: ").strip()
    create_empty_txt_for_unlabeled_images(img_folder, label_folder)