import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from collections import defaultdict
import argparse
from pathlib import Path


# ---------- 工具函数 ----------
def parse_obb_line(line):
    """解析一行YOLO OBB标签，返回类别ID和四个角点（归一化坐标）。"""
    parts = line.strip().split()
    if len(parts) != 9:
        raise ValueError(f"无效的OBB行，期望9个数值，实际得到 {len(parts)}: {line}")
    class_id = int(parts[0])
    coords = list(map(float, parts[1:]))
    points = np.array([[coords[i], coords[i + 1]] for i in range(0, 8, 2)], dtype=np.float32)
    return class_id, points


def format_obb_line(class_id, points):
    """将类别ID和四个角点格式化为YOLO OBB字符串。"""
    flat = points.flatten().tolist()
    return f"{class_id} " + " ".join([f"{v:.6f}" for v in flat])


def get_center(points):
    """计算旋转框的中心点（使用minAreaRect获得）。"""
    rect = cv2.minAreaRect(points)
    return np.array(rect[0], dtype=np.float32)


def boundary_distance(points1, points2):
    """计算两个旋转框边界之间的最小距离（如果相交则为0）。"""
    poly1 = Polygon(points1)
    poly2 = Polygon(points2)
    if not poly1.is_valid:
        poly1 = poly1.buffer(0)
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)
    return poly1.distance(poly2)  # shapely的distance方法


def center_distance(points1, points2):
    """计算两个旋转框中心点的欧氏距离。"""
    c1 = get_center(points1)
    c2 = get_center(points2)
    return np.linalg.norm(c1 - c2)


def merge_boxes_by_min_rect(boxes):
    """
    合并一组属于同一类别的旋转框（取所有顶点并计算最小外接旋转矩形）。
    boxes: list of (class_id, points)
    返回合并后的点集 (4x2 numpy array)。
    """
    all_points = []
    for _, pts in boxes:
        all_points.extend(pts.tolist())
    all_points = np.array(all_points, dtype=np.float32)
    if len(all_points) < 3:
        return boxes[0][1].copy()
    rect = cv2.minAreaRect(all_points)
    box_pts = cv2.boxPoints(rect)
    return np.array(box_pts, dtype=np.float32)


def merge_boxes_by_average(boxes):
    """合并一组框（取中心、宽高和角度的平均值）。"""
    centers, whs, angles = [], [], []
    for _, pts in boxes:
        rect = cv2.minAreaRect(pts)
        centers.append(rect[0])
        whs.append(rect[1])
        angles.append(rect[2])
    avg_cx = np.mean([c[0] for c in centers])
    avg_cy = np.mean([c[1] for c in centers])
    avg_w = np.mean([w for w, _ in whs])
    avg_h = np.mean([h for _, h in whs])
    avg_angle = np.mean(angles)
    rect = ((avg_cx, avg_cy), (avg_w, avg_h), avg_angle)
    return np.array(cv2.boxPoints(rect), dtype=np.float32)


def cluster_boxes_by_distance(boxes, distance_threshold, distance_metric='center'):
    """
    根据距离对同类框进行聚类（使用并查集）。
    boxes: list of (class_id, points)
    返回分组列表，每个组是一个list of boxes。
    """
    n = len(boxes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # 计算所有对之间的距离
    for i in range(n):
        for j in range(i + 1, n):
            if distance_metric == 'center':
                dist = center_distance(boxes[i][1], boxes[j][1])
            elif distance_metric == 'boundary':
                dist = boundary_distance(boxes[i][1], boxes[j][1])
            else:
                raise ValueError(f"不支持的距离度量: {distance_metric}")
            if dist <= distance_threshold:
                union(i, j)

    groups = defaultdict(list)
    for idx in range(n):
        root = find(idx)
        groups[root].append(boxes[idx])
    return list(groups.values())


def process_file(input_path, output_path, distance_threshold, distance_metric, merge_method):
    """处理单个标签文件：读取、合并、写入。"""
    # 按类别分组读取所有标注
    class_boxes = defaultdict(list)
    with open(input_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            cls_id, points = parse_obb_line(line)
            class_boxes[cls_id].append((cls_id, points))
        except ValueError as e:
            print(f"警告: 跳过无效行 '{line}' in {input_path}: {e}")

    merged_lines = []
    for cls_id, boxes in class_boxes.items():
        if len(boxes) <= 1:
            for _, pts in boxes:
                merged_lines.append(format_obb_line(cls_id, pts))
            continue

        # 聚类
        groups = cluster_boxes_by_distance(boxes, distance_threshold, distance_metric)
        for group in groups:
            if len(group) == 1:
                _, pts = group[0]
                merged_lines.append(format_obb_line(cls_id, pts))
            else:
                if merge_method == 'min_rect':
                    merged_pts = merge_boxes_by_min_rect(group)
                elif merge_method == 'average':
                    merged_pts = merge_boxes_by_average(group)
                else:
                    raise ValueError(f"不支持的合并方法: {merge_method}")
                merged_pts = np.clip(merged_pts, 0.0, 1.0)
                merged_lines.append(format_obb_line(cls_id, merged_pts))

    with open(output_path, 'w') as f:
        f.write('\n'.join(merged_lines))
        if merged_lines:
            f.write('\n')


def main():
    parser = argparse.ArgumentParser(description='根据距离合并YOLO OBB标签中相近的同类别旋转框')
    parser.add_argument('--input_dir', required=True, help='包含YOLO OBB标签txt文件的文件夹')
    parser.add_argument('--output_dir', required=True, help='输出文件夹（将被创建）')
    parser.add_argument('--distance_threshold', type=float, default=0.05,
                        help='距离阈值（归一化坐标），默认0.05')
    parser.add_argument('--distance_metric', choices=['center', 'boundary'], default='center',
                        help='距离度量方式：center（中心距离）或 boundary（边界最小距离），默认center')
    parser.add_argument('--merge_method', choices=['min_rect', 'average'], default='min_rect',
                        help='合并方法：min_rect（最小外接矩形）或 average（平均），默认min_rect')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(input_dir.glob('*.txt'))
    if not txt_files:
        print(f"警告: 在 {input_dir} 中未找到任何 .txt 文件")
        return

    for txt_path in txt_files:
        out_path = output_dir / txt_path.name
        process_file(txt_path, out_path, args.distance_threshold, args.distance_metric, args.merge_method)
        print(f"处理完成: {txt_path.name} -> {out_path}")

    print("所有文件处理完成！")


if __name__ == '__main__':
    main()
  # python merge_obb_by_distance.py --input_dir /path/to/labels --output_dir /path/to/merged_labels --distance_threshold 0.2 --distance_metric boundary --merge_method min_rect