# ROS2 imports 
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

# python imports
import numpy as np
import cv2
import os, sys, time
import math
from math import sin, cos

# pycuda imports
import pycuda.autoinit  # This is needed for initializing CUDA driver
import pycuda.driver as cuda

# CV Bridge and message imports
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from vision_msgs.msg import ObjectHypothesisWithPose, BoundingBox2D, Detection2D, Detection2DArray, BoundingBox3D, Detection3D, Detection3DArray
from cv_bridge import CvBridge, CvBridgeError
from message_filters import Subscriber, TimeSynchronizer
from bboxes_msg.msg import BoundingBox, BoundingBoxes
from visualization_msgs.msg import Marker, MarkerArray

# Yolo3D Detection import
from .torch_lib.Dataset import *
from .library.Math import *
from .library.Plotting import *
from .torch_lib import Model, ClassAverages

# Torch import
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

os.environ['TORCH_HOME'] = '/media/fmon005/Data/Data/Model'

def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location, orient

def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion (w in last place)
    quat = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = [0] * 4
    q[0] = cy * cp * sr - sy * sp * cr
    q[1] = sy * cp * sr + cy * sp * cr
    q[2] = sy * cp * cr - cy * sp * sr
    q[3] = cy * cp * cr + sy * sp * sr

    return q

def line_points_from_3d_bbox(x, y, z, w, h, l, theta):
    corner_matrix = np.array(
        [[-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [ 1,  1,  1],
        [ 1, -1,  1],
        [-1, -1,  1],
        [-1,  1,  1],
        [-1,  1, -1]], dtype=np.float32
    )
    relative_eight_corners = 0.5 * corner_matrix * np.array([w, h, l]) #[8, 3]

    _cos = cos(theta)
    _sin = sin(theta)

    rotated_corners_x, rotated_corners_z = (
            relative_eight_corners[:, 2] * _cos +
                relative_eight_corners[:, 0] * _sin,
        -relative_eight_corners[:, 2] * _sin +
            relative_eight_corners[:, 0] * _cos
        ) #[8]
    rotated_corners = np.stack([rotated_corners_x, relative_eight_corners[:,1], rotated_corners_z], axis=-1) #[8, 3]
    abs_corners = rotated_corners + np.array([x, y, z])  # [8, 3]

    points = []
    for i in range(1, 5):
        points += [
            Point(x=abs_corners[i, 0], y=abs_corners[i, 1], z=abs_corners[i, 2]),
            Point(x=abs_corners[i%4+1, 0], y=abs_corners[i%4+1, 1], z=abs_corners[i%4+1, 2])
        ]
        points += [
            Point(x=abs_corners[(i + 4)%8, 0], y=abs_corners[(i + 4)%8, 1], z=abs_corners[(i + 4)%8, 2]),
            Point(x=abs_corners[((i)%4 + 5)%8, 0], y=abs_corners[((i)%4 + 5)%8, 1], z=abs_corners[((i)%4 + 5)%8, 2])
        ]
    points += [
        Point(x=abs_corners[2, 0], y=abs_corners[2, 1], z=abs_corners[2, 2]),
        Point(x=abs_corners[7, 0], y=abs_corners[7, 1], z=abs_corners[7, 2]),
        Point(x=abs_corners[3, 0], y=abs_corners[3, 1], z=abs_corners[3, 2]),
        Point(x=abs_corners[6, 0], y=abs_corners[6, 1], z=abs_corners[6, 2]),

        Point(x=abs_corners[4, 0], y=abs_corners[4, 1], z=abs_corners[4, 2]),
        Point(x=abs_corners[5, 0], y=abs_corners[5, 1], z=abs_corners[5, 2]),
        Point(x=abs_corners[0, 0], y=abs_corners[0, 1], z=abs_corners[0, 2]),
        Point(x=abs_corners[1, 0], y=abs_corners[1, 1], z=abs_corners[1, 2])
    ]

    return points

class Yolo3dDetector(Node):

    def __init__(self):
        super().__init__('yolo3ddetection_node')
        
        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        self.declare_parameter("yolo_2d_topic", "/yolo2dbboxes")
        
        # Create a subscriber to the Image topic
        camera_topic = self.get_parameter("camera_topic").get_parameter_value().string_value
        yolo_2d_topic = self.get_parameter("yolo_2d_topic").get_parameter_value().string_value
        
        self.get_logger().info("Subscription from %s, %s" % (camera_topic, yolo_2d_topic))
        
        # Create a subscriber to the Image topic and 2DBox topic
        image_msg = Subscriber(self, Image, camera_topic)
        bbox2d_msg = Subscriber(self, BoundingBoxes, yolo_2d_topic)
        ts = TimeSynchronizer([image_msg, bbox2d_msg], 10)
        ts.registerCallback(self.callback)

        self.bridge = CvBridge()
        # Create a Detection 2D array topic to publish results on
        self.detection_publisher = self.create_publisher(Detection3DArray, 'yolo3ddetection', 10)

        # Create an Image publisher for the results
        self.result_publisher = self.create_publisher(Image,'yolo3ddetection_image',10)
        self.bbox_publish = self.create_publisher(MarkerArray, "/det3dbboxes", 10)

        weights_path = os.path.expanduser('~/ros2_ws/src/yolo3d_ros2/yolo3d_ros2/weights')
        model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
        if len(model_lst) == 0:
            self.get_logger().info('No previous model found, please train first!')
            exit()
        else:
            my_vgg = vgg.vgg19_bn(pretrained=True)
            self.model3D = Model.Model(features=my_vgg.features, bins=2).cuda()
            checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
            self.model3D.load_state_dict(checkpoint['model_state_dict'])
            self.model3D.eval()        
  
        self.get_logger().info("YOLO 3D Detection Model loaded /%s" % model_lst[-1])
        self.averages = ClassAverages.ClassAverages()
        self.angle_bins = generate_bins(2)
        camera_info = np.array([[607.7498168945312, 0.0, 419.7865905761719],
                        [0.0, 606.7474365234375, 230.44131469726562],
                        [0.0, 0.0, 1.0]])
        self.proj_matrix = np.zeros((3, 4))
        self.proj_matrix[0:3, 0:3] = camera_info[0:3,0:3] 
        self.frame_id = None

    def object_to_marker(self, obj, frame_id="base", marker_id=None, duration=10, color=None):
        """ Transform an object to a marker.

        Args:
            obj: Dict
            frame_id: string; parent frame name
            marker_id: visualization_msgs.msg.Marker.id
            duration: the existence time in rviz
        
        Return:
            marker: visualization_msgs.msg.Marker

        object dictionary expectation:
            object['whl'] = [w, h, l]
            object['xyz'] = [x, y, z] # center point location in center camera coordinate
            object['theta']: float
            object['score']: float
            object['type_name']: string # should have name in constant.KITTI_NAMES

        """
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = frame_id
        if marker_id is not None:
            marker.id = marker_id
        marker.type = 5
        marker.scale.x = 0.05

        # object_cls_index = KITTI_NAMES.index(obj["type_name"])
        # if color is None:
        #     obj_color = KITTI_COLORS[object_cls_index] #[r, g, b]
        # else:
        #     obj_color = color
        obj_color = [0, 182, 182]
        marker.color.r = obj_color[0] / 255.0
        marker.color.g = obj_color[1] / 255.0
        marker.color.b = obj_color[2] / 255.0
        marker.color.a = 1.0
        marker.points = line_points_from_3d_bbox(obj["xyz"][0], obj["xyz"][1], obj["xyz"][2], obj["whl"][0], obj["whl"][1], obj["whl"][2], obj["theta"])

        marker.lifetime = Duration(seconds=duration).to_msg()
        
        return marker

    def callback(self, image_msg, bbox2ds_msg):

        try:
          cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
          cv_image = np.ones((480, 848)).astype(np.uint8)
          print(e)

        start_time = time.time()
        self.frame_id = bbox2ds_msg.header.frame_id
        img = np.copy(cv_image)
        # yolo_img = np.copy(cv_image)
        detection3d_array = Detection3DArray()
        
        boxes_confs_clss = bbox2ds_msg.bounding_boxes
        objects = []
        for box_conf_cls in boxes_confs_clss:

            xmin = box_conf_cls.xmin
            xmax = box_conf_cls.xmax
            ymin = box_conf_cls.ymin
            ymax = box_conf_cls.ymax
            cls_name = box_conf_cls.cls
            cf = box_conf_cls.probability
            if not self.averages.recognized_class(cls_name):
                continue
            top_left = (xmin, ymin)
            bottom_right = (xmax, ymax)
            box_2d = [top_left, bottom_right]
            try:
                detectedObject = DetectedObject(img, cls_name, box_2d, self.proj_matrix)
            except:
                print('DetectedObject error')
                continue
            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            detected_class = cls_name
            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = self.model3D(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += self.averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += self.angle_bins[argmax]
            alpha -= np.pi
            location, orient = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)
            location[1] = -location[1]
            self.get_logger().info(str(time.time()-start_time))
            obj = {}

            obj['whl'] = [dim[2], dim[0], dim[1]]
            obj['theta'] = orient
            obj['type_name'] = cls_name
            obj['xyz'] = [location[2], location[0], location[1]]
            objects.append(obj)
            # self.get_logger().info(" ".join(str(x) for x in obj['xyz']))
            
            # print('Estimated class %s, pose: %s' % (cls_name, location))
            quat = quaternion_from_euler(0, 0, orient)

            object_hypothesis_with_pose = ObjectHypothesisWithPose()
            object_hypothesis_with_pose.hypothesis.class_id = str(cls_name)
            object_hypothesis_with_pose.hypothesis.score = float(cf)

            bounding_box3d = BoundingBox3D()
            bounding_box3d.center.position.x = float(location[2])
            bounding_box3d.center.position.y = float(location[0])
            bounding_box3d.center.position.z = float(location[1])
            bounding_box3d.center.orientation.x = float(quat[0])
            bounding_box3d.center.orientation.y = float(quat[1])
            bounding_box3d.center.orientation.z = float(quat[2])
            bounding_box3d.center.orientation.w = float(quat[2])

            detection = Detection3D()
            detection.header = bbox2ds_msg.header
            detection.results.append(object_hypothesis_with_pose)
            detection.bbox = bounding_box3d

            detection3d_array.header = bbox2ds_msg.header
            detection3d_array.detections.append(detection) 

        markerarray_msg = MarkerArray()
        for i, obj in enumerate(objects):
            marker_msg = self.object_to_marker(obj, frame_id='map', marker_id=i, duration=1)
            markerarray_msg.markers.append(marker_msg)
            
        cv2.imshow('3D detections', img)
        cv2.waitKey(1)
        # duration = Duration(seconds=10)
        
        #self.bbox_publish.publish([self.object_to_marker(obj, frame_id=self.frame_id, marker_id=i, duration=10) for i, obj in enumerate(objects)])
        self.bbox_publish.publish(markerarray_msg)
        # Publishing the results onto the the Detection2DArray vision_msgs format
        self.detection_publisher.publish(detection3d_array)
        # Publishing the results
        ros_image = self.bridge.cv2_to_imgmsg(img)
        ros_image.header.frame_id = 'yolo3d_frame'
        self.result_publisher.publish(ros_image)


