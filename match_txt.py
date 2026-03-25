import os
import sys

def merge_matched_txts(folder_a, folder_b, output_folder):
    """
    匹配并合并两个文件夹中的同名txt文件。
    folder_a: 第一个文件夹路径
    folder_b: 第二个文件夹路径（内容将追加到第一个文件末尾）
    output_folder: 合并后文件的输出文件夹
    """
    # 1. 扫描两个文件夹，建立文件名到完整路径的映射
    files_in_a = {f: os.path.join(folder_a, f) for f in os.listdir(folder_a) if f.endswith('.txt')}
    files_in_b = {f: os.path.join(folder_b, f) for f in os.listdir(folder_b) if f.endswith('.txt')}

    # 2. 找出两个文件夹中都存在的共同文件名
    common_filenames = set(files_in_a.keys()) & set(files_in_b.keys())

    if not common_filenames:
        print("警告：未在两个文件夹中找到同名的txt文件。")
        return

    # 3. 为每个共同文件执行合并操作
    for filename in common_filenames:
        path_a = files_in_a[filename]
        path_b = files_in_b[filename]
        output_path = os.path.join(output_folder, filename)

        try:
            with open(path_a, 'r', encoding='utf-8') as fa, \
                 open(path_b, 'r', encoding='utf-8') as fb:
                content_a = fa.read()
                content_b = fb.read()

            # 将folder_b的内容追加到folder_a内容的末尾
            merged_content = content_a + content_b  # 中间加一个空行分隔

            with open(output_path, 'w', encoding='utf-8') as f_out:
                f_out.write(merged_content)

            print(f"已合并：{filename}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错：{e}")

    print(f"\n合并完成！文件已保存至：{output_folder}")

# === 使用方法：修改下面的路径，然后运行脚本 ===
if __name__ == "__main__":
    # 请将以下路径替换为你实际的文件夹路径
    folder_1 = r"C:\Users\TJDX\Desktop\garbage\garbage\labels\val"   # 第一个文件夹
    folder_2 = r"C:\Users\TJDX\Desktop\labels1086\labels1086"  # 第二个文件夹
    output_dir = r"C:\Users\TJDX\Desktop\mix_label"            # 输出文件夹（需要提前创建好）

    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    merge_matched_txts(folder_1, folder_2, output_dir)