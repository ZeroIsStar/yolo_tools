import os
import argparse
from pathlib import Path
import numpy as np


def obb_to_aabb(obb_points):
    """
    将OBB的4个角点转换为轴对齐框（AABB）

    Args:
        obb_points: list of 8个坐标值 [x1,y1,x2,y2,x3,y3,x4,y4] (归一化坐标)

    Returns:
        (x_center, y_center, width, height) 归一化的中心点和宽高
    """
    # 提取x和y坐标
    x_coords = obb_points[0::2]  # 索引0,2,4,6
    y_coords = obb_points[1::2]  # 索引1,3,5,7

    # 计算最小最大值
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # 计算中心点和宽高
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # 确保坐标在[0,1]范围内（防止浮点误差）
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def convert_obb_file(txt_path, backup=True):
    """
    转换单个OBB标签文件为AABB标签文件
    """
    # 读取原始标签
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 检查是否有OBB标签（9个数值：class_id + 8个坐标）
    has_obb = False
    new_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) == 9:  # OBB格式: class_id + 8个坐标
            has_obb = True
            class_id = parts[0]
            obb_points = [float(x) for x in parts[1:]]

            # 转换为AABB
            x_center, y_center, width, height = obb_to_aabb(obb_points)

            # 生成新的标签行
            new_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            new_lines.append(new_line)
        elif len(parts) == 5:  # 已经是AABB格式
            new_lines.append(line)
        else:
            # 未知格式，保留原样
            print(f"⚠️  警告: {txt_path.name} 包含未知格式: {line}")
            new_lines.append(line)

    # 如果没有OBB标签，跳过
    if not has_obb:
        print(f"⏭️  跳过: {txt_path.name} (没有OBB标签)")
        return False

    # 如果需要备份
    if backup:
        backup_path = txt_path.parent / f"{txt_path.stem}_backup{txt_path.suffix}"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"💾 已备份: {backup_path.name}")

    # 写入新标签
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
        if new_lines:
            f.write('\n')  # 末尾加换行

    print(f"✅ 已转换: {txt_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='批量将YOLO-OBB标签（9个值）转换为YOLO标准标签（5个值）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理labels目录下所有txt文件
  python obb_to_aabb.py ./labels/

  # 处理并递归子目录
  python obb_to_aabb.py ./dataset/ --recursive

  # 不创建备份（直接覆盖）
  python obb_to_aabb.py ./labels/ --no-backup
        """
    )
    parser.add_argument('input_dir', type=str, help='包含标签txt文件的目录路径')
    parser.add_argument('--no-backup', action='store_true', help='不创建备份文件（默认会备份）')
    parser.add_argument('--recursive', action='store_true', help='递归处理子目录')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出到指定目录（不覆盖原文件）。如果不指定，则直接修改原文件')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ 错误: 目录 '{input_dir}' 不存在")
        return

    # 如果指定了输出目录，创建它
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 输出目录: {output_dir}")

    # 收集所有txt文件
    if args.recursive:
        txt_files = list(input_dir.rglob('*.txt'))
    else:
        txt_files = list(input_dir.glob('*.txt'))

    # 过滤掉备份文件
    txt_files = [f for f in txt_files if not f.name.endswith('_backup.txt')]

    if not txt_files:
        print(f"❌ 在 '{input_dir}' 中未找到txt标签文件")
        return

    print(f"📁 找到 {len(txt_files)} 个标签文件")
    print("=" * 60)

    # 处理每个文件
    processed_count = 0
    for txt_file in txt_files:
        if output_dir:
            # 如果指定了输出目录，复制文件到输出目录再处理
            output_file = output_dir / txt_file.name
            # 读取原文件内容
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            # 转换输出目录中的文件（不创建备份）
            if convert_obb_file(output_file, backup=False):
                processed_count += 1
        else:
            # 直接处理原文件（会创建备份）
            if convert_obb_file(txt_file, backup=not args.no_backup):
                processed_count += 1

    print("=" * 60)
    print(f"🎉 处理完成! 转换了 {processed_count} 个文件")
    if not args.no_backup and not output_dir:
        print("💡 备份文件已保存，后缀为 _backup.txt")
        print("   如果确认无误，可以删除这些备份文件")
    if output_dir:
        print(f"💡 转换后的文件保存在: {output_dir}")


if __name__ == '__main__':
    main()