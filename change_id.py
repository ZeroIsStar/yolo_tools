#!/usr/bin/env python3
"""
批量修改YOLO标签文件中的类别ID
用法: python change_id.py --label-dir /path/to/labels --old-id 2 --new-id 1 --no-backup
"""

import os
import glob
import argparse


def change_class_id(label_dir, old_id, new_id, backup=False):
    """
    修改标签文件中的类别ID

    Args:
        label_dir: 存放.txt文件的目录
        old_id: 要修改的旧ID，如果为-1则修改所有ID
        new_id: 新的ID
        backup: 是否创建备份
    """
    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    if not txt_files:
        print(f"在 {label_dir} 中没有找到.txt文件")
        return

    print(f"找到 {len(txt_files)} 个标签文件")

    modified_count = 0
    total_lines = 0

    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        file_modified = False

        for line in lines:
            line = line.strip()
            if not line:  # 跳过空行
                continue

            parts = line.split()
            if not parts:
                continue

            # 检查是否是有效的YOLO格式
            try:
                current_id = int(parts[0])
                total_lines += 1

                # 判断是否需要修改
                if old_id == -1 or current_id == old_id:
                    parts[0] = str(new_id)
                    file_modified = True

                new_lines.append(" ".join(parts))
            except (ValueError, IndexError):
                # 不是有效的YOLO格式，保持原样
                new_lines.append(line)

        # 如果文件有修改，写回
        if file_modified:
            if backup:
                # 创建备份
                backup_file = txt_file + ".bak"
                import shutil
                shutil.copy2(txt_file, backup_file)

            with open(txt_file, 'w') as f:
                f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

            modified_count += 1

    print(f"处理完成！")
    print(f"扫描了 {total_lines} 行标注")
    print(f"修改了 {modified_count} 个文件")
    if backup:
        print(f"原始文件已备份为 .bak 文件")


def main():
    parser = argparse.ArgumentParser(description="批量修改YOLO标签文件类别ID")
    parser.add_argument("--label-dir", type=str, required=True,
                        help="存放标签文件的目录路径")
    parser.add_argument("--old-id", type=int, default=-1,
                        help="要修改的旧ID，-1表示修改所有ID（默认）")
    parser.add_argument("--new-id", type=int, required=True,
                        help="新的类别ID")
    parser.add_argument("--no-backup", action="store_true",
                        help="不创建备份文件")
    args = parser.parse_args()

    change_class_id(
        label_dir=args.label_dir,
        old_id=args.old_id,
        new_id=args.new_id,
        backup=not args.no_backup
    )

if __name__ == "__main__":
    main()