#!/usr/bin/env python3
"""
YOLO数据集检查工具 - 手动验证和标记问题图像
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
from pathlib import Path
import json
import shutil


class YOLODatasetInspector:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.labels_dir = self.data_dir / 'labels'

        # 获取所有图像文件
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) +
                                  list(self.images_dir.glob('*.png')) +
                                  list(self.images_dir.glob('*.jpeg')))

        self.current_idx = 0
        self.problem_images = set()
        self.load_progress()

        # 创建GUI
        self.root = tk.Tk()
        self.root.title(f"YOLO数据集检查工具 - {data_dir}")
        self.root.geometry("1200x800")

        self.setup_ui()

    def setup_ui(self):
        # 控制面板
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # 图像显示区域
        self.image_frame = ttk.Frame(self.root, padding="10")
        self.image_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        # 信息显示区域
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))

        # 控制按钮
        ttk.Button(control_frame, text="上一张", command=self.prev_image).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="下一张", command=self.next_image).grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="跳转到:").grid(row=0, column=2, padx=5)
        self.goto_entry = ttk.Entry(control_frame, width=10)
        self.goto_entry.grid(row=0, column=3, padx=5)
        ttk.Button(control_frame, text="跳转", command=self.goto_image).grid(row=0, column=4, padx=5)

        # 问题标记按钮
        ttk.Button(control_frame, text="标记为损坏", command=lambda: self.mark_problem('corrupted')).grid(row=0,
                                                                                                          column=5,
                                                                                                          padx=5)
        ttk.Button(control_frame, text="标记为模糊", command=lambda: self.mark_problem('blurry')).grid(row=0, column=6,
                                                                                                       padx=5)
        ttk.Button(control_frame, text="标记为森林火灾", command=lambda: self.mark_problem('forest_fire')).grid(row=0,
                                                                                                                column=7,
                                                                                                                padx=5)
        ttk.Button(control_frame, text="标记为重复", command=lambda: self.mark_problem('duplicate')).grid(row=0,
                                                                                                          column=8,
                                                                                                          padx=5)
        ttk.Button(control_frame, text="取消标记", command=self.unmark_problem).grid(row=0, column=9, padx=5)

        # 统计和保存
        ttk.Button(control_frame, text="保存标记", command=self.save_progress).grid(row=0, column=10, padx=5)
        ttk.Button(control_frame, text="删除标记项", command=self.delete_marked).grid(row=0, column=11, padx=5)

        # 图像显示
        self.canvas = tk.Canvas(self.image_frame, width=800, height=600, bg='black')
        self.canvas.pack()

        # 信息显示
        self.info_label = ttk.Label(info_frame, text="")
        self.info_label.pack()

        # 进度显示
        self.progress_label = ttk.Label(info_frame, text="")
        self.progress_label.pack()

        # 绑定键盘事件
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('c', lambda e: self.mark_problem('corrupted'))
        self.root.bind('b', lambda e: self.mark_problem('blurry'))
        self.root.bind('f', lambda e: self.mark_problem('forest_fire'))
        self.root.bind('d', lambda e: self.mark_problem('duplicate'))
        self.root.bind('u', lambda e: self.unmark_problem())

        # 显示第一张图像
        self.load_current_image()

    def load_current_image(self):
        if self.current_idx >= len(self.image_files):
            return

        img_path = self.image_files[self.current_idx]

        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            self.canvas.delete("all")
            self.canvas.create_text(400, 300, text="无法加载图像", fill="white", font=("Arial", 20))
            return

        # 调整大小以适应画布
        height, width = img.shape[:2]
        scale = min(800 / width, 600 / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = cv2.resize(img, (new_width, new_height))

        # 转换为PIL格式
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.img_tk = ImageTk.PhotoImage(img_pil)

        # 显示图像
        self.canvas.delete("all")
        self.canvas.create_image(400, 300, image=self.img_tk, anchor=tk.CENTER)

        # 显示边界框（如果存在标签）
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            self.draw_bounding_boxes(img_path, label_path, scale)

        # 更新信息
        filename = img_path.name
        is_problem = str(img_path) in self.problem_images
        problem_type = self.get_problem_type(str(img_path))

        info_text = f"文件名: {filename} | 尺寸: {width}x{height} | "
        info_text += f"当前: {self.current_idx + 1}/{len(self.image_files)} | "

        if is_problem:
            info_text += f"问题: {problem_type} | 标记: ✅"
        else:
            info_text += "标记: ❌"

        self.info_label.config(text=info_text)

        # 更新进度
        problem_count = len(self.problem_images)
        progress_text = f"已标记问题图像: {problem_count}/{len(self.image_files)} ({problem_count / len(self.image_files) * 100:.1f}%)"
        self.progress_label.config(text=progress_text)

    def draw_bounding_boxes(self, img_path, label_path, scale):
        """绘制YOLO格式的边界框"""
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # YOLO格式: class x_center y_center width height (归一化坐标)
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # 转换为像素坐标
                img = cv2.imread(str(img_path))
                img_h, img_w = img.shape[:2]

                x1 = int((x_center - width / 2) * img_w * scale)
                y1 = int((y_center - height / 2) * img_h * scale)
                x2 = int((x_center + width / 2) * img_w * scale)
                y2 = int((y_center + height / 2) * img_h * scale)

                # 绘制矩形（考虑画布偏移）
                canvas_x1 = 400 - (img_w * scale) / 2 + x1
                canvas_y1 = 300 - (img_h * scale) / 2 + y1
                canvas_x2 = 400 - (img_w * scale) / 2 + x2
                canvas_y2 = 300 - (img_h * scale) / 2 + y2

                # 根据类别选择颜色
                colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
                color = colors[class_id % len(colors)]

                self.canvas.create_rectangle(canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                                             outline=color, width=2)

                # 显示类别标签
                class_names = ['fire', 'smoke', 'person', 'vehicle']  # 根据你的数据集修改
                class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
                self.canvas.create_text(canvas_x1, canvas_y1 - 10, text=class_name,
                                        fill=color, anchor=tk.W, font=("Arial", 10))

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_image()

    def next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.load_current_image()

    def goto_image(self):
        try:
            idx = int(self.goto_entry.get()) - 1
            if 0 <= idx < len(self.image_files):
                self.current_idx = idx
                self.load_current_image()
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")

    def mark_problem(self, problem_type):
        img_path = self.image_files[self.current_idx]
        self.problem_images.add(str(img_path))

        # 保存问题类型
        progress_file = self.data_dir / "inspection_progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        else:
            progress = {'problem_images': {}}

        progress['problem_images'][str(img_path)] = problem_type

        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

        self.load_current_image()

    def unmark_problem(self):
        img_path = self.image_files[self.current_idx]
        if str(img_path) in self.problem_images:
            self.problem_images.remove(str(img_path))

            # 更新进度文件
            progress_file = self.data_dir / "inspection_progress.json"
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    progress = json.load(f)

                if 'problem_images' in progress and str(img_path) in progress['problem_images']:
                    del progress['problem_images'][str(img_path)]

                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=2)

        self.load_current_image()

    def get_problem_type(self, img_path_str):
        progress_file = self.data_dir / "inspection_progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)

            if 'problem_images' in progress and img_path_str in progress['problem_images']:
                return progress['problem_images'][img_path_str]

        return "unknown"

    def load_progress(self):
        progress_file = self.data_dir / "inspection_progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)

            if 'problem_images' in progress:
                self.problem_images = set(progress['problem_images'].keys())

    def save_progress(self):
        progress_file = self.data_dir / "inspection_progress.json"

        progress = {
            'total_images': len(self.image_files),
            'problem_images': {},
            'timestamp': str(datetime.now())
        }

        for img_path_str in self.problem_images:
            progress['problem_images'][img_path_str] = self.get_problem_type(img_path_str)

        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

        messagebox.showinfo("保存成功", f"已保存标记进度到: {progress_file}")

    def delete_marked(self):
        if not messagebox.askyesno("确认", f"确定要删除 {len(self.problem_images)} 个标记的问题图像吗？"):
            return

        backup_dir = self.data_dir / "deleted_images_backup"
        backup_dir.mkdir(exist_ok=True)

        deleted_count = 0
        for img_path_str in self.problem_images:
            img_path = Path(img_path_str)
            label_path = self.labels_dir / f"{img_path.stem}.txt"

            # 备份
            shutil.move(str(img_path), backup_dir / img_path.name)
            if label_path.exists():
                shutil.move(str(label_path), backup_dir / label_path.name)

            deleted_count += 1

        # 重新加载图像列表
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) +
                                  list(self.images_dir.glob('*.png')) +
                                  list(self.images_dir.glob('*.jpeg')))
        self.current_idx = min(self.current_idx, len(self.image_files) - 1)
        self.problem_images.clear()

        self.load_current_image()
        messagebox.showinfo("删除完成", f"已删除 {deleted_count} 个问题图像，备份在: {backup_dir}")

    def run(self):
        self.root.mainloop()


def inspect_dataset():
    import argparse

    parser = argparse.ArgumentParser(description='YOLO数据集检查工具（GUI）')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集目录')

    args = parser.parse_args()

    inspector = YOLODatasetInspector(args.data_dir)
    inspector.run()


if __name__ == "__main__":
    # # 启动GUI检查工具
    # python hand_check.py --data_dir /path/to/your/dataset
    # GUI工具快捷键：
    #
    # 左箭头 / 右箭头：上一张 / 下一张图像
    #
    # C：标记为损坏
    #
    # B：标记为模糊
    #
    # F：标记为森林火灾
    #
    # D：标记为重复
    #
    # U：取消标记
    from datetime import datetime

    inspect_dataset()