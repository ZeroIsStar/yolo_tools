#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile

from sensor_msgs.msg import CompressedImage

import cv2
import time
import numpy as np
import os
from functools import partial

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')

        # 相机配置
        self.cameras = {
            'front': {
                'topic': '/rgb_camera_front/compressed',
                'save_dir': '/capella/lib/python3.10/site-packages/garbage_detection_node/RGB_data/front',
            },
            'back': {
                'topic': '/rgb_camera_back/compressed',
                'save_dir': '/capella/lib/python3.10/site-packages/garbage_detection_node/RGB_data/back',
            },
            'left': {
                'topic': '/rgb_camera_left/compressed',
                'save_dir': '/capella/lib/python3.10/site-packages/garbage_detection_node/RGB_data/left',
            },
            'right': {
                'topic': '/rgb_camera_right/compressed',
                'save_dir': '/capella/lib/python3.10/site-packages/garbage_detection_node/RGB_data/right',
            },
        }

        # QoS 配置（与原始代码一致）
        qos_ = QoSProfile(depth=1)
        qos_.reliability = ReliabilityPolicy.BEST_EFFORT

        # 为每个相机创建订阅
        # 修改：将 subscriptions 改为 camera_subs，避免与 Node 基类属性冲突
        self.camera_subs = {}
        for camera_name, cam_info in self.cameras.items():
            # 创建互斥回调组，避免多线程同时保存（可选）
            callback_group = MutuallyExclusiveCallbackGroup()
            self.camera_subs[camera_name] = self.create_subscription(
                CompressedImage,
                cam_info['topic'],
                partial(self.image_callback, camera_name=camera_name),
                qos_,
                callback_group=callback_group,
            )
            self.get_logger().info(f"Subscribed to {cam_info['topic']}")

        # 记录每个相机上次保存时间
        self.last_save_time = {name: 0.0 for name in self.cameras.keys()}
        self.save_interval = 3.0  # 保存间隔（秒）

        self.get_logger().info("Image saver node started")

    def image_callback(self, msg, camera_name: str):
        """
        压缩图像回调：解码并定期保存
        """
        # 解码图像
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().warn(f"[{camera_name}] Failed to decode image", once=True)
                return
        except Exception as e:
            self.get_logger().warn(f"[{camera_name}] Decode error: {e}", once=True)
            return

        # 检查保存间隔
        now = time.time()
        if now - self.last_save_time[camera_name] < self.save_interval:
            return

        # 保存图像
        save_dir = self.cameras[camera_name]['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        filename = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + ".jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, img)
        self.last_save_time[camera_name] = now
        self.get_logger().info(f"[{camera_name}] Saved image to {filepath}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    # 使用多线程执行器，保证四个相机回调并行处理
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
