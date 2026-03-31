import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, DurabilityPolicy, ReliabilityPolicy, QoSProfile, HistoryPolicy
# garbage_cord_markers
import tf2_geometry_msgs
from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from capella_ros_msg.msg import GarbageDetect
from visualization_msgs.msg import Marker, MarkerArray

import torch

import cv2
import time
import math
import numpy as np
import os
from collections import deque
from functools import partial

from ultralytics import YOLO

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class GarbageDetectionDemonstration(Node):
    def __init__(self):
        super().__init__('garbage_detection_demo')
        # 保存图像的起始时间
        self.last_save_time = 0.0

        # 相机编号列表
        self.camera_ids = [1, 2, 4]

        self.callback_group_list = []
        self.subscription_list = []

        # 每个相机独立的数据存储
        self.camera_states = {}

        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'

        self.model = YOLO(os.path.join(os.path.dirname(__file__),
                                       '/capella/lib/python3.10/site-packages/garbage_detection_node/garbage_detection_yolo11_324.pt'))

        self.get_logger().info("Model initialization successful !")

        qos_ = QoSProfile(depth=1)
        qos_.reliability = ReliabilityPolicy.BEST_EFFORT

        # 为每个摄像头初始化数据存储和订阅
        for camera_id in self.camera_ids:
            # 初始化相机状态
            self.camera_states[camera_id] = {
                'K': None,
                'depth_data': None,
                'depth_x': None,
                'depth_y': None,
                'depth_width': None,
                'depth_height': None,
                'depth_frame_id': None,
            }

            # 为每个摄像头创建独立的回调组
            group_color = MutuallyExclusiveCallbackGroup()
            group_depth = MutuallyExclusiveCallbackGroup()
            group_info = MutuallyExclusiveCallbackGroup()
            self.callback_group_list.extend([group_color, group_depth, group_info])

            # topic 名称
            color_topic = f'/camera{camera_id}/color/image_raw'
            depth_topic = f'/camera{camera_id}/depth/image_raw'
            camera_info_topic = f'/camera{camera_id}/depth/camera_info'

            # 订阅color图像、深度图像和CameraInfo
            self.subscription_list.append(
                self.create_subscription(
                    Image,
                    color_topic,
                    partial(self.color_callback, camera_id=camera_id),
                    qos_,
                    callback_group=group_color,
                )
            )

            self.subscription_list.append(
                self.create_subscription(
                    Image,
                    depth_topic,
                    partial(self.depth_callback, camera_id=camera_id),
                    qos_,
                    callback_group=group_depth,
                )
            )

            self.subscription_list.append(
                self.create_subscription(
                    CameraInfo,
                    camera_info_topic,
                    partial(self.camera_info_callback, camera_id=camera_id),
                    qos_,
                    callback_group=group_info,
                )
            )

        self.pose_publisher = self.create_publisher(GarbageDetect, '/garbage_cord', 1)

        # 垃圾历史点可视化发布器
        self.garbage_markers_pub = self.create_publisher(MarkerArray, '/garbage_cord_markers', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.horizontal_fov = 86

        # 计数器
        self.unique_garbage_count = 0
        self._published_garbage_xy_map: list[tuple[float, float]] = []

        # 可视化的垃圾点，用一个列表来装
        self._visualization_garbage_list = deque(maxlen=6)
        # 垃圾可视化
        self.garbage_visualization_timer = self.create_timer(0.5, self._update_garbage_visualization)
        self.declare_parameter('unique_distance_threshold_m', 0.15)
        self._unique_distance_threshold_m = float(self.get_parameter('unique_distance_threshold_m').value)

    def _is_new_garbage_xy_map(self, x_m: float, y_m: float) -> bool:
        for px, py in self._published_garbage_xy_map:
            if (x_m - px) ** 2 + (y_m - py) ** 2 <= self._unique_distance_threshold_m ** 2:
                return False
        return True

    def camera_info_callback(self, msg, camera_id: int):
        self.camera_states[camera_id]['K'] = msg.k

    def depth_callback(self, msg, camera_id: int):
        state = self.camera_states[camera_id]
        if len(msg.data) > 0:
            depth_width = msg.width
            depth_height = msg.height
            state['depth_width'] = depth_width
            state['depth_height'] = depth_height
            state['depth_frame_id'] = msg.header.frame_id

            state['depth_data'] = (
                torch.from_numpy(np.array(msg.data))
                .to(self.device)
                .to(torch.float32)
                .reshape((depth_height, depth_width, -1))
            )

            # 深度相机中每个像素的x和y
            if (
                    state['depth_x'] is None
                    or state['depth_y'] is None
                    or state['depth_x'].shape[0] != depth_height
                    or state['depth_x'].shape[1] != depth_width
            ):
                x = torch.arange(depth_width, device=self.device).repeat(depth_height, 1)
                y = torch.arange(depth_height, device=self.device).unsqueeze(1).repeat(1, depth_width)
                state['depth_x'] = x
                state['depth_y'] = y
        else:
            state['depth_data'] = None
            self.get_logger().info(f"[camera{camera_id}] Depth cameras do not have depth data !")

    def save_image_every_1s(self, color_data, save_dir):

        now = time.time()
        if now - self.last_save_time < 3.0:
            return

        os.makedirs(save_dir, exist_ok=True)

        filename = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + ".jpg"
        cv2.imwrite(os.path.join(save_dir, filename), color_data)

        self.last_save_time = now

    def color_callback(self, msg, camera_id: int):
        if len(msg.data) > 0:
            # state = self.camera_states[camera_id]
            # color_time = msg.header.stamp
            color_width = msg.width
            color_height = msg.height

            color_data = np.array(msg.data).reshape((color_height, color_width, 3))
            color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
            self.save_image_every_1s(color_data,
                                     save_dir='/capella/lib/python3.10/site-packages/garbage_detection_node/data')


def main(args=None):
    rclpy.init(args=args)
    garbage_detection_demonstration = GarbageDetectionDemonstration()

    multi_executor = MultiThreadedExecutor(num_threads=4)
    multi_executor.add_node(garbage_detection_demonstration)
    multi_executor.spin()
    garbage_detection_demonstration.destroy_node()
    multi_executor.shutdown()