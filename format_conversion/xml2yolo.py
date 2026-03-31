import xml.etree.ElementTree as ET
import os
import glob
import json


class VOC2YOLOConverter:
    def __init__(self):
        self.classes = []
        self.class_to_id = {}

    def extract_classes_from_xmls(self, xml_dir):
        """从所有XML文件中提取所有类别"""
        classes_set = set()
        xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))

        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    classes_set.add(class_name)
            except Exception as e:
                print(f"读取 {xml_file} 时出错: {e}")

        self.classes = sorted(list(classes_set))
        self.class_to_id = {name: idx for idx, name in enumerate(self.classes)}

        print(f"发现 {len(self.classes)} 个类别:")
        for i, cls in enumerate(self.classes):
            print(f"  {i}: {cls}")

        return self.classes

    def convert_single_file(self, xml_path, output_dir=None):
        """转换单个文件"""
        # 解析XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取基本信息
        filename = os.path.splitext(os.path.basename(xml_path))[0]
        size = root.find('size')

        if size is None:
            raise ValueError(f"{xml_path} 中没有找到size标签")

        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        # 获取所有对象
        objects = []
        for obj in root.findall('object'):
            # 类别
            class_name = obj.find('name').text

            if class_name not in self.class_to_id:
                print(f"警告: 未知类别 '{class_name}'，跳过")
                continue

            class_id = self.class_to_id[class_name]

            # 边界框
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue

            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # 验证坐标
            if xmin >= xmax or ymin >= ymax:
                print(f"警告: {filename} 中的边界框坐标无效，跳过")
                continue

            if xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
                print(f"警告: {filename} 中的边界框超出图像范围")

            # 转换为YOLO格式
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # 确保在[0,1]范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            objects.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 确定输出路径
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            txt_path = os.path.join(output_dir, f"{filename}.txt")
        else:
            txt_path = xml_path.replace('.xml', '.txt')

        # 写入TXT文件
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(objects))

        return txt_path, len(objects)

    def batch_convert(self, xml_dir, output_dir, classes_file=None):
        """批量转换"""
        # 如果提供了类别文件，则从文件加载
        if classes_file and os.path.exists(classes_file):
            with open(classes_file, 'r', encoding='utf-8') as f:
                self.classes = [line.strip() for line in f if line.strip()]
            self.class_to_id = {name: idx for idx, name in enumerate(self.classes)}
        else:
            # 否则从XML中提取
            self.extract_classes_from_xmls(xml_dir)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 保存类别文件
        classes_output = os.path.join(output_dir, "classes.txt")
        with open(classes_output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.classes))

        # 获取所有XML文件
        xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))

        print(f"开始转换 {len(xml_files)} 个文件...")
        print(f"类别映射: {self.class_to_id}")

        success_count = 0
        total_objects = 0

        for xml_file in xml_files:
            try:
                output_path, obj_count = self.convert_single_file(xml_file, output_dir)
                success_count += 1
                total_objects += obj_count
                print(f"✓ {os.path.basename(xml_file)} -> {os.path.basename(output_path)} "
                      f"({obj_count} 个对象)")
            except Exception as e:
                print(f"✗ 转换失败 {os.path.basename(xml_file)}: {e}")

        # 生成统计信息
        stats = {
            "total_xml_files": len(xml_files),
            "successfully_converted": success_count,
            "total_objects": total_objects,
            "classes": self.classes,
            "class_mapping": self.class_to_id
        }

        stats_file = os.path.join(output_dir, "conversion_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"\n转换完成！")
        print(f"成功转换: {success_count}/{len(xml_files)}")
        print(f"总对象数: {total_objects}")
        print(f"类别文件已保存: {classes_output}")
        print(f"统计信息: {stats_file}")


# 使用示例
if __name__ == "__main__":
    converter = VOC2YOLOConverter()

    # 设置路径
    xml_folder = "C:/Users/TJDX/Desktop/windows_v1.8.1/labels"  # XML文件夹
    output_folder = "C:/Users/TJDX/Desktop/windows_v1.8.1/labels"  # 输出文件夹

    # 如果有单独的类别文件（每行一个类别名）
    classes_file = "classes.txt"  # 可选

    # 执行批量转换
    converter.batch_convert(xml_folder, output_folder, classes_file)
