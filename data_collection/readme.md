# <span style="text-transform: uppercase;">数据采集脚本</span>
## RGB.py\
### 功能说明\
'''txt
订阅四个相机话题（前、后、左、右），话题名与原始代码保持一致。

每个相机图像保存到独立的子目录（例如 RGB_data/front/），避免文件覆盖。

保存间隔为 3 秒（可通过修改 self.save_interval 调整）。

图像文件名格式为 YYYYMMDD_HHMMSS.jpg。

使用多线程执行器，保证四个相机回调互不阻塞。
'''

RGBD.py
功能说明
订阅三个彩色图像话题：/camera1/color/image_raw、/camera2/color/image_raw、/camera4/color/image_raw。

每个相机独立保存图像，互不干扰。

保存间隔 3 秒（可通过修改 self.save_interval 调整）。

图像保存路径为 /capella/lib/python3.10/site-packages/garbage_detection_node/data，文件名包含相机ID和时间戳，例如 camera1_20250331_123456.jpg。

使用多线程执行器，确保多相机回调并行处理，避免阻塞。

数据集采集标准
尽可能的覆盖可能涉及的场景（晴天，阴天，户外室内）

运行
python3 RGB.py/RGBD.py

