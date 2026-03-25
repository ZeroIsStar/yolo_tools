import os
import shutil

# 配置路径（请根据实际情况修改）
A_folder = r'C:\Users\TJDX\Desktop\windows_v1.8.1\calibrate\images'   # 图片文件夹
B_folder = r'C:\Users\TJDX\Desktop\windows_v1.8.1\calibrate\train'   # txt文件夹
C_folder = r'C:\Users\TJDX\Desktop\windows_v1.8.1\calibrate\labels'   # 目标文件夹

# 确保目标文件夹存在
os.makedirs(C_folder, exist_ok=True)

# 1. 获取A文件夹中所有图片的文件名（不含扩展名）
# 常见的图片扩展名，可根据需要增删
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'}

# 存储A中图片文件名（无扩展名）的集合
image_names = set()

for filename in os.listdir(A_folder):
    file_path = os.path.join(A_folder, filename)
    if os.path.isfile(file_path):
        name, ext = os.path.splitext(filename)
        if ext.lower() in image_extensions:
            image_names.add(name)

# 2. 遍历B文件夹中的txt文件，如果名称在集合中，则移动到C
for filename in os.listdir(B_folder):
    if not filename.lower().endswith('.txt'):
        continue
    name, ext = os.path.splitext(filename)  # ext 将是 .txt
    if name in image_names:
        src = os.path.join(B_folder, filename)
        dst = os.path.join(C_folder, filename)
        shutil.copy(src, dst)
        print(f'已移动: {filename}')

print('操作完成！')