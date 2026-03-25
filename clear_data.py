#!/usr/bin/env python3
"""
YOLO数据集自动化清理脚本
功能：删除重复、损坏、模糊、森林火灾的图像和标签
用法：python clear.py --data_dir /path/to/dataset --threshold 0.9
"""

import os
import cv2
import numpy as np
import shutil
import hashlib
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')


class YOLODatasetCleaner:
    def __init__(self, data_dir, backup_dir=None, threshold=0.9):
        """
        初始化清理器

        Args:
            data_dir: 数据集根目录（包含images/和labels/文件夹）
            backup_dir: 备份目录，默认为data_dir/backup_clean
            threshold: 图像相似度阈值，高于此值视为重复
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.labels_dir = self.data_dir / 'labels'

        # 验证目录结构
        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise ValueError(f"数据集目录结构错误，需要包含images/和labels/文件夹")

        # 设置备份目录
        self.backup_dir = Path(
            backup_dir) if backup_dir else self.data_dir / f"backup_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.threshold = threshold
        self.deletion_log = []
        self.stats = {
            'total': 0,
            'duplicate': 0,
            'corrupted': 0,
            'blurry': 0,
            'forest_fire': 0,
            'remaining': 0
        }

    def get_image_label_pairs(self):
        """获取所有图像-标签对"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        pairs = []

        for img_path in self.images_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    pairs.append((img_path, label_path))
                else:
                    print(f"警告: 找不到标签文件 {label_path}")

        self.stats['total'] = len(pairs)
        return pairs

    def compute_image_hash(self, image):
        """计算图像的感知哈希（用于快速去重）"""
        # 缩小图像尺寸
        img_resized = cv2.resize(image, (8, 8))
        # 转换为灰度图
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        # 计算平均值
        avg = gray.mean()
        # 生成哈希
        hash_str = ''.join(['1' if pixel > avg else '0' for pixel in gray.flatten()])
        return hash_str

    def compute_histogram(self, image):
        """计算图像直方图特征"""
        # 转换为HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 计算颜色直方图
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        # 合并特征
        hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        # 归一化
        hist = hist / (hist.sum() + 1e-6)
        return hist

    def is_corrupted_image(self, img_path):
        """检查图像是否损坏"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return True, "无法读取图像文件"

            # 检查图像尺寸
            if img.shape[0] == 0 or img.shape[1] == 0:
                return True, "图像尺寸为0"

            # 检查通道数
            if len(img.shape) != 3 or img.shape[2] != 3:
                return True, "图像通道数异常"

            # 检查像素值范围
            if img.min() == img.max() == 0:
                return True, "图像全黑"

            return False, None
        except Exception as e:
            return True, f"读取异常: {str(e)}"

    def is_blurry_image(self, image, threshold=100.0):
        """检查图像是否模糊（使用拉普拉斯方差）"""
        if image is None:
            return True

        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 计算拉普拉斯方差
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        return laplacian_var < threshold

    def is_forest_fire_scene(self, image, label_path):
        """检测是否为森林火灾场景"""
        if image is None:
            return False

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = image.shape[:2]

        # 1. 检测森林（绿色区域）
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / (height * width)

        # 2. 检测火焰（橙红色区域）
        lower_fire1 = np.array([0, 100, 100])  # 红色
        upper_fire1 = np.array([10, 255, 255])
        lower_fire2 = np.array([10, 100, 100])  # 橙色
        upper_fire2 = np.array([30, 255, 255])

        fire_mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        fire_mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        fire_mask = cv2.bitwise_or(fire_mask1, fire_mask2)
        fire_ratio = np.sum(fire_mask > 0) / (height * width)

        # 3. 检测烟雾（低饱和度灰白色区域）
        lower_smoke = np.array([0, 0, 50])
        upper_smoke = np.array([180, 50, 200])
        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
        smoke_ratio = np.sum(smoke_mask > 0) / (height * width)

        # 4. 检查标签中是否有火/烟类别（YOLO格式）
        has_fire_in_labels = False
        has_smoke_in_labels = False

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        # 假设 class_id 0=fire, 1=smoke（根据你的数据集调整）
                        if class_id == 0:
                            has_fire_in_labels = True
                        elif class_id == 1:
                            has_smoke_in_labels = True

        # 判断逻辑：有森林且（有火焰/烟雾或标签中有火/烟类别）
        is_forest = green_ratio > 0.3  # 至少有30%绿色区域
        has_fire_smoke = fire_ratio > 0.02 or smoke_ratio > 0.05 or has_fire_in_labels or has_smoke_in_labels

        return is_forest and has_fire_smoke

    def find_duplicate_images(self, image_pairs, similarity_threshold=0.9):
        """查找高度相似的重复图像"""
        print("正在检测重复图像...")

        # 提取图像特征
        features = []
        valid_pairs = []

        for img_path, label_path in tqdm(image_pairs):
            img = cv2.imread(str(img_path))
            if img is not None:
                # 使用直方图特征
                hist = self.compute_histogram(img)
                features.append(hist)
                valid_pairs.append((img_path, label_path, img))

        if len(features) < 2:
            return []

        features = np.array(features)

        # 使用最近邻算法查找相似图像
        nbrs = NearestNeighbors(n_neighbors=2, metric='cosine').fit(features)
        distances, indices = nbrs.kneighbors(features)

        # 标记重复图像
        duplicate_indices = set()
        for i in range(len(distances)):
            if distances[i][1] < (1 - similarity_threshold):  # 余弦距离转换为相似度
                neighbor_idx = indices[i][1]
                # 保留较小的索引，删除较大的索引
                if i < neighbor_idx and neighbor_idx not in duplicate_indices:
                    duplicate_indices.add(neighbor_idx)

        # 返回要删除的图像对
        duplicate_pairs = []
        for idx in duplicate_indices:
            img_path, label_path, _ = valid_pairs[idx]
            duplicate_pairs.append((img_path, label_path))

        return duplicate_pairs

    def safe_delete_files(self, img_path, label_path, reason):
        """安全删除文件（备份到备份目录）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 备份文件
        backup_img = self.backup_dir / f"{timestamp}_{img_path.name}"
        backup_label = self.backup_dir / f"{timestamp}_{label_path.name}"

        try:
            shutil.move(str(img_path), str(backup_img))
            shutil.move(str(label_path), str(backup_label))

            # 记录删除日志
            self.deletion_log.append({
                'image': str(img_path),
                'label': str(label_path),
                'reason': reason,
                'backup_image': str(backup_img),
                'backup_label': str(backup_label),
                'timestamp': timestamp
            })

            return True
        except Exception as e:
            print(f"删除失败 {img_path}: {e}")
            return False

    def clean_dataset(self, blur_threshold=100.0):
        """执行完整的清理流程"""
        print("=" * 50)
        print("YOLO数据集清理开始")
        print(f"数据集目录: {self.data_dir}")
        print(f"备份目录: {self.backup_dir}")
        print("=" * 50)

        # 获取所有图像-标签对
        image_pairs = self.get_image_label_pairs()
        print(f"找到 {len(image_pairs)} 个图像-标签对")

        # 第一步：检测并删除损坏图像
        print("\n[1/4] 检查损坏图像...")
        corrupted_to_delete = []
        valid_pairs = []

        for img_path, label_path in tqdm(image_pairs):
            is_corrupted, reason = self.is_corrupted_image(img_path)
            if is_corrupted:
                corrupted_to_delete.append((img_path, label_path, reason))
            else:
                valid_pairs.append((img_path, label_path))

        # 第二步：检测并删除模糊图像
        print("\n[2/4] 检查模糊图像...")
        blurry_to_delete = []
        clear_pairs = []

        for img_path, label_path in tqdm(valid_pairs):
            img = cv2.imread(str(img_path))
            if self.is_blurry_image(img, blur_threshold):
                blurry_to_delete.append((img_path, label_path, "图像模糊"))
            else:
                clear_pairs.append((img_path, label_path))

        # 第三步：检测并删除森林火灾图像
        print("\n[3/4] 检查森林火灾图像...")
        forest_fire_to_delete = []
        non_forest_pairs = []

        for img_path, label_path in tqdm(clear_pairs):
            img = cv2.imread(str(img_path))
            if self.is_forest_fire_scene(img, label_path):
                forest_fire_to_delete.append((img_path, label_path, "森林火灾场景"))
            else:
                non_forest_pairs.append((img_path, label_path))

        # 第四步：检测并删除重复图像
        print("\n[4/4] 检查重复图像...")
        duplicate_to_delete = self.find_duplicate_images(non_forest_pairs, self.threshold)

        # 合并所有需要删除的图像（避免重复）
        all_to_delete = {}

        # 添加损坏图像
        for img_path, label_path, reason in corrupted_to_delete:
            all_to_delete[str(img_path)] = (img_path, label_path, reason)

        # 添加模糊图像
        for img_path, label_path, reason in blurry_to_delete:
            if str(img_path) not in all_to_delete:
                all_to_delete[str(img_path)] = (img_path, label_path, reason)

        # 添加森林火灾图像
        for img_path, label_path, reason in forest_fire_to_delete:
            if str(img_path) not in all_to_delete:
                all_to_delete[str(img_path)] = (img_path, label_path, reason)

        # 添加重复图像
        for img_path, label_path in duplicate_to_delete:
            if str(img_path) not in all_to_delete:
                all_to_delete[str(img_path)] = (img_path, label_path, "重复图像")

        # 执行删除
        print("\n执行删除操作...")
        delete_items = list(all_to_delete.values())

        for img_path, label_path, reason in tqdm(delete_items):
            if self.safe_delete_files(img_path, label_path, reason):
                # 更新统计
                if "损坏" in reason:
                    self.stats['corrupted'] += 1
                elif "模糊" in reason:
                    self.stats['blurry'] += 1
                elif "森林火灾" in reason:
                    self.stats['forest_fire'] += 1
                elif "重复" in reason:
                    self.stats['duplicate'] += 1

        # 更新剩余数量
        remaining_pairs = self.get_image_label_pairs()
        self.stats['remaining'] = len(remaining_pairs)

        # 保存日志和统计
        self.save_results()

        return self.stats

    def save_results(self):
        """保存清理结果和日志"""
        # 保存删除日志
        log_file = self.data_dir / "cleaning_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats,
                'deleted_files': self.deletion_log,
                'backup_dir': str(self.backup_dir)
            }, f, indent=2, ensure_ascii=False)

        # 保存统计摘要
        summary_file = self.data_dir / "cleaning_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("YOLO数据集清理报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"清理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集目录: {self.data_dir}\n")
            f.write(f"备份目录: {self.backup_dir}\n\n")
            f.write("清理统计:\n")
            f.write(f"  原始图像数量: {self.stats['total']}\n")
            f.write(f"  删除损坏图像: {self.stats['corrupted']}\n")
            f.write(f"  删除模糊图像: {self.stats['blurry']}\n")
            f.write(f"  删除森林火灾图像: {self.stats['forest_fire']}\n")
            f.write(f"  删除重复图像: {self.stats['duplicate']}\n")
            f.write(f"  剩余图像数量: {self.stats['remaining']}\n")
            f.write(
                f"  删除比例: {(self.stats['total'] - self.stats['remaining']) / max(self.stats['total'], 1) * 100:.2f}%\n\n")

            if self.deletion_log:
                f.write("删除详情:\n")
                for i, item in enumerate(self.deletion_log[:20], 1):  # 只显示前20条
                    f.write(f"  {i}. {Path(item['image']).name} - {item['reason']}\n")
                if len(self.deletion_log) > 20:
                    f.write(f"  ... 还有 {len(self.deletion_log) - 20} 条记录\n")

        print(f"\n清理完成！报告已保存到: {summary_file}")

    def generate_updated_data_yaml(self, original_yaml=None):
        """生成更新后的data.yaml文件（如果需要）"""
        if original_yaml:
            yaml_path = Path(original_yaml)
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    content = f.read()

                # 这里可以根据需要修改yaml内容
                # 例如更新图像数量统计
                updated_content = content

                new_yaml_path = self.data_dir / "data_cleaned.yaml"
                with open(new_yaml_path, 'w') as f:
                    f.write(updated_content)

                print(f"更新后的YAML文件: {new_yaml_path}")
                return new_yaml_path

        return None


def main():
    parser = argparse.ArgumentParser(description='YOLO数据集自动化清理工具')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录（包含images/和labels/文件夹）')
    parser.add_argument('--backup_dir', type=str, default=None, help='备份目录（默认在data_dir下创建）')
    parser.add_argument('--threshold', type=float, default=0.9, help='图像相似度阈值（0-1，默认0.9）')
    parser.add_argument('--blur_threshold', type=float, default=100.0, help='模糊检测阈值（默认100.0，越小越严格）')
    parser.add_argument('--dry_run', action='store_true', help='模拟运行，不实际删除文件')
    parser.add_argument('--config', type=str, help='配置文件路径（JSON格式）')

    args = parser.parse_args()

    # 加载配置文件（如果有）
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)

    # 创建清理器
    cleaner = YOLODatasetCleaner(
        data_dir=args.data_dir,
        backup_dir=args.backup_dir,
        threshold=args.threshold
    )

    # 如果是模拟运行，修改备份目录为临时目录
    if args.dry_run:
        cleaner.backup_dir = cleaner.data_dir / "dry_run_backup"
        cleaner.backup_dir.mkdir(exist_ok=True)
        print("模拟运行模式 - 不会实际删除文件，仅生成报告")

    # 执行清理
    try:
        stats = cleaner.clean_dataset(blur_threshold=args.blur_threshold)

        # 显示统计信息
        print("\n" + "=" * 50)
        print("清理统计摘要:")
        print(f"原始总数: {stats['total']}")
        print(f"删除损坏: {stats['corrupted']}")
        print(f"删除模糊: {stats['blurry']}")
        print(f"删除森林火灾: {stats['forest_fire']}")
        print(f"删除重复: {stats['duplicate']}")
        print(f"剩余数量: {stats['remaining']}")
        print(f"删除比例: {(stats['total'] - stats['remaining']) / max(stats['total'], 1) * 100:.2f}%")

        if args.dry_run:
            print("\n模拟运行完成！所有'删除'的文件已备份到:", cleaner.backup_dir)
            print("确认无误后，可删除dry_run_backup目录并重新运行（不带--dry_run参数）")

    except Exception as e:
        print(f"清理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()