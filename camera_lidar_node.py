# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:02:21 2024

@author: gao
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
from sensor_msgs_py.point_cloud2 import read_points_list

K = np.array([[633.0128, 0., 425.0031],
                [0., 635.3088, 228.2753],
                [0.,0.,1.]
                ])
DIST = np.array([0.1020, -0.1315, 0, 0, 0])
# dist_coeffs = np.array([.0, .0, .0, .0, .0])

class Transformation:
    def __init__(self, x, y, z, roll, pitch, yaw):
        self.translation = np.array([x, y, z])
        self.rotation_matrix = self.euler_to_rot_matrix(roll, pitch, yaw)

    @staticmethod
    def euler_to_rot_matrix(roll, pitch, yaw):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        return R_z @ R_y @ R_x

    def transform_points(self, points):
        return np.dot(points, self.rotation_matrix.T) + self.translation

class Camera:
    def __init__(self, K, dist_coeffs):
        self.K = K
        self.dist_coeffs = dist_coeffs

    def project_points(self, points):
        points = points.reshape(-1, 1, 3)
        image_points, _ = cv2.projectPoints(points, np.eye(3), np.zeros((3,1)), self.K, self.dist_coeffs)
        return image_points.reshape(-1, 2)

class LidarCameraFusion(Node):
    def __init__(self):
        super().__init__('lidar_camera_fusion')
        self.subscription_lidar = self.create_subscription(
            PointCloud2,
            'scala_decoder_sdk_points',
            self.listener_callback_lidar,
            10)
        self.subscription_lidar  # prevent unused variable warning

        self.subscription_camera = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback_camera,
            10)
        self.subscription_camera  # prevent unused variable warning

        self.bridge = CvBridge()
        self.lidar_points_transformed = None
        self.camera_pose = Transformation(0, 0, 0, 0, 0, 0)  
        self.lidar_pose = Transformation(-1, 0, 0, 0, 0, 0) 
        self.camera = Camera(K, DIST)

    def listener_callback_lidar(self, msg):
        # Convert PointCloud2 to array of points
        points = read_points_list(msg)
        points_array = np.array(points, dtype=np.float32)
        
        # Transform points
        self.lidar_points_transformed = self.transform_points(points_array[:, :3])  # Assuming XYZ are the first three fields

    def listener_callback_camera(self, msg):
        if self.lidar_points_transformed is not None:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                self.get_logger().error(f"Error converting image: {str(e)}")
                return
            
            # Project points onto the camera image
            points_camera = self.camera_pose.transform_points(self.lidar_points_transformed - self.lidar_pose.translation)
            image_points = self.camera.project_points(points_camera)
            
            # image_points = self.project_points_to_image(self.lidar_points_transformed)
            
            for point in image_points:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < cv_image.shape[1] and 0 <= y < cv_image.shape[0]:  
                    cv2.circle(cv_image, (x, y), 3, (0, 0, 255), -1)  
            
            cv2.imshow("Camera View with Lidar Points", cv_image)
            cv2.waitKey(1)

    def transform_points(self, points):
        # Similar transformation as before
        R = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        return points @ R.T

    def project_points_to_image(self, points):
        # Dummy camera matrix and projection logic
        K = np.array([
            [633.0128, 0., 425.0031],
            [0., 635.3088, 228.2753],
            [0., 0., 1.]
        ])
        points = points.reshape(-1, 1, 3)
        image_points, _ = cv2.projectPoints(points, np.eye(3), np.zeros(3), K, np.zeros(4))
        return image_points.reshape(-1, 2)

def main(args=None):
    rclpy.init(args=args)
    fusion_node = LidarCameraFusion()
    rclpy.spin(fusion_node)
    fusion_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
