# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:38:29 2024

@author: gao
"""

import numpy as np

class LidarTransform:
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
        return np.dot(R_z, np.dot(R_y, R_x))

    def transform_point_cloud(self, points):
        return np.dot(points, self.rotation_matrix.T) + self.translation

x_center, y_center, z_center, roll_center, pitch_center, yaw_center = 0, 0, 0, 0, 0, 0
x_left, y_left, z_left, roll_left, pitch_left, yaw_left = -1, 0, 0, 0, 0, 0
x_right, y_right, z_right, roll_right, pitch_right, yaw_right = 1, 0, 0, 0, 0, 0

center_lidar = LidarTransform(x_center, y_center, z_center, roll_center, pitch_center, yaw_center)
left_lidar = LidarTransform(x_left, y_left, z_left, roll_left, pitch_left, yaw_left)
right_lidar = LidarTransform(x_right, y_right, z_right, roll_right, pitch_right, yaw_right)

left_to_center_transform = LidarTransform(left_lidar.translation[0] - center_lidar.translation[0],
                                          left_lidar.translation[1] - center_lidar.translation[1],
                                          left_lidar.translation[2] - center_lidar.translation[2],
                                          0, 0, 0) 

right_to_center_transform = LidarTransform(right_lidar.translation[0] - center_lidar.translation[0],
                                           right_lidar.translation[1] - center_lidar.translation[1],
                                           right_lidar.translation[2] - center_lidar.translation[2],
                                           0, 0, 0)  


points_left = np.array([[1, 2, 3], [4, 5, 6]])  # 左边激光雷达点云
points_right = np.array([[7, 8, 9], [10, 11, 12]])  # 右边激光雷达点云

points_left_transformed = left_to_center_transform.transform_point_cloud(points_left)
points_right_transformed = right_to_center_transform.transform_point_cloud(points_right)

print("Left transformed:", points_left_transformed)
print("Right transformed:", points_right_transformed)


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
    return np.dot(R_z, np.dot(R_y, R_x))


def transform_point_cloud(points, translation, rotation_matrix):
    return np.dot(points, rotation_matrix.T) + translation


x_center, y_center, z_center, roll_center, pitch_center, yaw_center = 0, 0, 0, 0, 0, 0

x_left, y_left, z_left, roll_left, pitch_left, yaw_left = -1, 0, 0, 0, 0, 0

x_right, y_right, z_right, roll_right, pitch_right, yaw_right = 1, 0, 0, 0, 0, 0


rot_matrix_left = euler_to_rot_matrix(roll_left-roll_center, pitch_left-pitch_center, yaw_left-yaw_center)
trans_left = np.array([x_left-x_center, y_left-y_center, z_left-z_center])

rot_matrix_right = euler_to_rot_matrix(roll_right-roll_center, pitch_right-pitch_center, yaw_right-yaw_center)
trans_right = np.array([x_right-x_center, y_right-y_center, z_right-z_center])


points_left = np.array([[1, 2, 3], [4, 5, 6]])  # 左边激光雷达点云
points_right = np.array([[7, 8, 9], [10, 11, 12]])  # 右边激光雷达点云

points_left_transformed = transform_point_cloud(points_left, trans_left, rot_matrix_left)
points_right_transformed = transform_point_cloud(points_right, trans_right, rot_matrix_right)

print("Left transformed:", points_left_transformed)
print("Right transformed:", points_right_transformed)



###########
## Camera and Lidar
###########

import numpy as np
import cv2

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

K = np.array([[633.0128, 0., 425.0031],
                [0., 635.3088, 228.2753],
                [0.,0.,1.]
                ])
dist_coeffs = np.array([0.1020, -0.1315, 0, 0, 0])
# dist_coeffs = np.array([.0, .0, .0, .0, .0])

camera_pose = Transformation(0, 0, 0, 0, 0, 0)  
lidar_pose = Transformation(-1, 0, 0, 0, 0, 0)  

camera = Camera(K, dist_coeffs)

points_lidar = np.array([[20, 0, 0],
                         [20, 1, 0],
                         [20, 2, 0],
                         [20, 0, 1]])

# The LIDAR's X-axis (forward) corresponds to the camera's Z-axis
# The LIDAR's Y-axis (left side) corresponds to the negative direction of the camera's X-axis 
# The LIDAR's Z-axis (upwards) corresponds to the negative direction of the camera's Y-axis  
    

R = np.array([
    [0, -1, 0],
    [0, 0, -1],
    [1, 0, 0]
    ])

points_lidar = points_lidar @ R.T
    
points_camera = camera_pose.transform_points(points_lidar - lidar_pose.translation)

image_points = camera.project_points(points_camera)

print("Projected points on the image:", image_points)

image = cv2.imread('test.png')  

for point in image_points:
    x, y = int(point[0]), int(point[1])
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  

cv2.imshow('Image with Lidar Points', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



#
#lidar_to_camera_translation = np.array([1, 0, 0])  
#lidar_to_camera_rotation = euler_to_rot_matrix(np.radians(0), np.radians(0), np.radians(0))  
#
#K = np.array([[633.0128, 0., 425.0031],
#                [0., 635.3088, 228.2753],
#                [0.,0.,1.]
#                ])
#dist = np.array([0.1020, -0.1315, 0, 0, 0])
#
#points_lidar_frame = np.random.rand(100, 3) * 10  
#
#points_camera_frame = transform_point_cloud_to_camera_frame(points_lidar_frame, 
#                                                            lidar_to_camera_translation, 
#                                                            lidar_to_camera_rotation)
#
#points_camera_frame = points_camera_frame[points_camera_frame[:, 2] > 0]
#
#points_image_plane = project_to_image_plane(points_camera_frame, K)
#
#image = cv2.imread('your_camera_image.jpg')  
#
#for point in points_image_plane:
#    x, y = int(point[0]), int(point[1])
#    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  
#        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  
#
#cv2.imshow('Image with Lidar Points', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

