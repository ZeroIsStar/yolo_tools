import json
from pathlib import Path

# ===== 请修改以下配置 =====
json_folder = r"C:\Users\TJDX\Desktop\new\change\val"  # 存放JSON的文件夹
output_folder = r"C:\Users\TJDX\Desktop\new_obb\val"  # 输出TXT的文件夹
class_names = ["wire", "water_pipe"]  # 按顺序列出所有类别，索引即为class_id
# =========================

# 构建类别映射
class_mapping = {name: idx for idx, name in enumerate(class_names)}

# 创建输出目录
Path(output_folder).mkdir(parents=True, exist_ok=True)

# 遍历所有JSON文件
for json_path in Path(json_folder).glob("*.json"):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w = data['imageWidth']
    img_h = data['imageHeight']

    txt_path = Path(output_folder) / (json_path.stem + '.txt')

    with open(txt_path, 'w', encoding='utf-8') as out_f:
        for shape in data['shapes']:
            # 只处理有向矩形（如果混有普通矩形也一并处理）
            if shape['shape_type'] not in ['oriented_rectangle', 'rectangle']:
                continue

            # 获取类别ID
            label = shape['label']
            if label not in class_mapping:
                print(f"警告: 未找到类别 '{label}'，已跳过")
                continue
            class_id = class_mapping[label]

            # 直接读取四个角点（已经是 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]）
            points = shape['points']

            # 归一化坐标
            normalized = []
            for x, y in points:
                normalized.append(x / img_w)
                normalized.append(y / img_h)

            # 写入TXT: class_id x1 y1 x2 y2 x3 y3 x4 y4
            line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized])
            out_f.write(line + '\n')

    print(f"✅ 转换完成: {txt_path}")

print("🎉 所有文件处理完毕！")