import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from functools import partial
import cv2
import numpy as np
import os
import time


class GarbageDetectionDemonstration(Node):
    def __init__(self):
        super().__init__('garbage_detection_demo')
        self.last_save_time = 0.0          # 全局保存间隔控制
        self.camera_ids = [1, 2, 4]        # 需要采集的相机编号

        # QoS 配置
        qos_ = QoSProfile(depth=1)
        qos_.reliability = ReliabilityPolicy.BEST_EFFORT

        # 为每个相机创建独立的回调组并订阅彩色图像
        self.subscription_list = []
        for camera_id in self.camera_ids:
            group_color = MutuallyExclusiveCallbackGroup()
            color_topic = f'/camera{camera_id}/color/image_raw'
            self.subscription_list.append(
                self.create_subscription(
                    Image,
                    color_topic,
                    partial(self.color_callback, camera_id=camera_id),
                    qos_,
                    callback_group=group_color,
                )
            )
        self.get_logger().info("Image collection node started.")

    def save_image_every_3s(self, color_data, save_dir, camera_id):
        """每隔3秒保存一张图片（文件名包含相机ID）"""
        now = time.time()
        if now - self.last_save_time < 3.0:
            return
        os.makedirs(save_dir, exist_ok=True)
        # 时间戳 + 相机ID 避免覆盖
        filename = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + f"_cam{camera_id}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), color_data)
        self.last_save_time = now

    def color_callback(self, msg, camera_id: int):
        """彩色图像回调：解码并保存"""
        if len(msg.data) == 0:
            return
        # 解析图像数据（假设为BGR格式，这里保持与原逻辑一致）
        color_height = msg.height
        color_width = msg.width
        color_data = np.array(msg.data).reshape((color_height, color_width, 3))
        # 原代码中转换了颜色空间，保留此转换（若需保存正确颜色，可去掉或调整）
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        # 保存路径（可按需修改）
        save_dir = '/capella/lib/python3.10/site-packages/garbage_detection_node/data'
        self.save_image_every_3s(color_data, save_dir, camera_id)


def main(args=None):
    rclpy.init(args=args)
    node = GarbageDetectionDemonstration()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        executor.shutdown()


if __name__ == '__main__':
    main()
