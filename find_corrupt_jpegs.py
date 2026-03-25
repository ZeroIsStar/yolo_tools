import os
import sys
import subprocess
from tqdm import tqdm
import cv2

# ================= 配置区域 =================
# 图片文件夹路径
dataset_path = r"C:\Users\TJDX\Desktop\test\2_12"


# ===========================================

def check_image_via_subprocess(filepath):
    """
    启动一个子进程单独读取这张图片。
    如果读取过程中产生标准错误输出（stderr），则说明图片有问题。
    """
    # 构造一段简短的 Python 代码，只做 imread
    # 我们把路径作为参数传递，避免反斜杠转义问题
    cmd = [
        sys.executable,
        "-c",
        "import cv2, sys; cv2.imread(sys.argv[1])",
        filepath
    ]

    # 运行子进程，捕获 stderr (标准错误输出)
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 检查 stderr 中是否有 YOLO/OpenCV 常见的 JPEG 警告
    # "Corrupt JPEG" 是最关键的关键词
    if "Corrupt JPEG" in result.stderr or "extraneous bytes" in result.stderr:
        return True, result.stderr.strip()
    return False, None


print(f"正在扫描文件夹: {dataset_path}")
print("这可能比平时慢一点，因为每张图都在独立进程中检查...")
print("-" * 60)

corrupted_files = []

# 获取所有图片列表
image_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(os.path.join(root, file))

# 使用 tqdm 显示进度条，不用担心刷屏，因为我们只打印坏文件
for file_path in tqdm(image_files, desc="精确扫描中"):
    try:
        # ---------- 第1步：子进程快速预检 ----------
        is_corrupted, error_msg = check_image_via_subprocess(file_path)
        if is_corrupted:
            raise ValueError(f"子进程预检失败: {error_msg}")

        # ---------- 第2步：OpenCV 正式读取验证 ----------
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("OpenCV 无法读取图像（返回 None）")

    except Exception as e:
        # 任何异常都视为文件损坏/不可用，统一记录
        corrupted_files.append(file_path)
        tqdm.write(f"\n[!!! 找到坏文件]: {file_path}")
        tqdm.write(f"    └─ 错误信息: {e}")

print("-" * 60)
print(f"扫描结束。共发现 {len(corrupted_files)} 个损坏文件：")

# 最后统一打印一遍，方便复制
for f in corrupted_files:
    print(f)

# ================= 新增删除功能 =================
if corrupted_files:
    print("\n" + "=" * 60)
    user_input = input("是否要删除这些损坏文件？(y/n): ").strip().lower()

    if user_input == 'y':
        deleted_count = 0
        print("开始删除损坏文件...")
        for file_path in corrupted_files:
            try:
                os.remove(file_path)
                print(f"已删除: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"删除失败 {file_path}: {e}")

        print(f"\n成功删除 {deleted_count} 个损坏文件。")
    else:
        print("已取消删除操作。")
else:
    print("没有发现损坏文件，无需删除。")

# import cv2
# import os
# from pathlib import Path
#
# input_dir =  r"C:\Users\TJDX\Downloads\data\RGB"
# output_dir =  r"C:\Users\TJDX\Downloads\data\BGR"
# Path(output_dir).mkdir(parents=True, exist_ok=True)
#
# exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
#
# for fname in os.listdir(input_dir):
#     if fname.lower().endswith(exts):
#         path_in = os.path.join(input_dir, fname)
#         img = cv2.imread(path_in)               # 此时OpenCV按BGR理解，颜色错误（如果原图是RGB）
#         if img is None:
#             continue
#         img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 关键：将BGR重新映射为RGB，实际是交换R/B
#         # 或者手动交换： img_bgr = img[:, :, ::-1]
#         path_out = os.path.join(output_dir, fname)
#         cv2.imwrite(path_out, img_bgr)          # 保存为BGR顺序
#         print(f"转换并保存: {fname}")
#
# import cv2
# import os
# import numpy as np
# from pathlib import Path
#
# # ===== 配置路径 =====
# src_folder = r"C:\Users\TJDX\Downloads\data\RGB"
# dst_folder = r"C:\Users\TJDX\Downloads\data\RGB_Equalized"
# # ===================
#
# Path(dst_folder).mkdir(parents=True, exist_ok=True)
#
# # 支持的图像扩展名
# exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
#
# for filename in os.listdir(src_folder):
#     if filename.lower().endswith(exts):
#         path_in = os.path.join(src_folder, filename)
#         img = cv2.imread(path_in)
#
#         if img is None:
#             print(f"❌ 无法读取: {filename}")
#             continue
#
#         # ----- 彩色图处理 -----
#         if len(img.shape) == 3:
#             # 1. BGR -> HSV
#             hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#             h, s, v = cv2.split(hsv)
#             # 2. 仅对亮度(V)通道做直方图均衡化
#             v_eq = cv2.equalizeHist(v)
#             # 3. 合并 H, S, V_eq
#             hsv_eq = cv2.merge([h, s, v_eq])
#             # 4. HSV -> BGR
#             img_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
#
#         # ----- 灰度图处理 -----
#         else:
#             img_eq = cv2.equalizeHist(img)
#
#         # 保存结果
#         path_out = os.path.join(dst_folder, filename)
#         cv2.imwrite(path_out, img_eq)
#         print(f"✅ 已处理: {filename}")
# print("🎉 全部处理完成！")