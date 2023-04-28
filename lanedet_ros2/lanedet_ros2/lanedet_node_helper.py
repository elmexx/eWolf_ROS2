# ROS2 imports 
import rclpy
from rclpy.node import Node

import numpy as np
import os, sys, cv2, time
import os.path as osp

# CV Bridge and message imports
from sensor_msgs.msg import Image as SensorImage
from std_msgs.msg import String, Header
from lane_parameter_msg.msg import LaneParams
#from builtin_interfaces.msg import Time
import cv2
from cv_bridge import CvBridge, CvBridgeError

# Torch import 
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.linear_model import RANSACRegressor

# Lanedet imports
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network
from pathlib import Path
from .tools.lane_parameter import DictObjHolder, bev_perspective, PolynomialRegression, get_fit_param, insertLaneBoundary

# garmin 1920*1080
"""
mtx = np.array([[1.121200e+03, 0.000000e+00, 9.634416e+02],
       [0.000000e+00, 1.131900e+03, 5.268709e+02],
       [0.000000e+00, 0.000000e+00, 1.000000e+00]])

dist = np.array([-0.3051, 0.0771, 0, 0, 0])
"""
# Intel realsense 848*480
mtx = np.array([[633.0128, 0., 425.0031],
                [0., 635.3088, 228.2753],
                [0.,0.,1.]
                ])
dist = np.array([0.1020, -0.1315, 0, 0, 0])

pitch = 2
yaw = 0
roll = 0
height = 1.6

distAheadOfSensor = 30
spaceToLeftSide = 5    
spaceToRightSide = 5
bottomOffset = 1

imgw = 848
imgh = 480

COLOR_MAP = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (125, 125, 0)]


class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.x_scale = self.cfg.sensor_img_w / self.cfg.ori_img_w
        self.y_scale = self.cfg.sensor_img_h / self.cfg.ori_img_h

        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, ori_img):
        # ori_img = cv2.imread(img_path)
        ori_img = cv2.resize(ori_img, (self.cfg.ori_img_w,self.cfg.ori_img_h))
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'ori_img':ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.get_lanes(data)
        return data

    def show(self, data):
        out_file = self.cfg.savedir 
        if out_file:
            out_file = osp.join(out_file, 'img_path')
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        x_scale = self.cfg.sensor_img_w / self.cfg.ori_img_w
        y_scale = self.cfg.sensor_img_h / self.cfg.ori_img_h

        imshow_lanes(data['ori_img'], lanes, x_scale, y_scale, show=True, out_file=out_file)

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]

        if self.cfg.show or self.cfg.savedir:
            self.show(data)
        return data
    
class Lanedet(Node):

    def __init__(self):
        super().__init__('lanedet_node')

        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        # Create a subscriber to the Image topic
        camera_topic = self.get_parameter("camera_topic").get_parameter_value().string_value
        self.get_logger().info("Subscription from %s" % camera_topic)
        
        self.subscription = self.create_subscription(SensorImage, camera_topic, self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

        # Create a Lane detection topic to publish results on
        self.leftdetection_publisher = self.create_publisher(LaneParams, 'leftlanedetection', 10)
        self.rightdetection_publisher = self.create_publisher(LaneParams, 'rightlanedetection', 10)

        # Create an Image publisher for the results
        self.result_publisher = self.create_publisher(SensorImage,'lanedetection_image',10)

        # load lanedet Net
        cfg_path = os.path.join(os.getenv("HOME"), 'ros2_ws/src/lanedet_ros2/lanedet_ros2/configs/condlane/resnet101_culane.py')
        model_path = os.path.join(os.getenv("HOME"), 'ros2_ws/src/lanedet_ros2/lanedet_ros2/model/condlane_r101_culane.pth')
        cfg = Config.fromfile(cfg_path)
        cfg.show = False
        cfg.savedir = None
        cfg.load_from = model_path
        self.detect = Detect(cfg)

        self.get_logger().info("Lane Detection neural network loaded! ")
        
        # Camera Pose
        self.CameraPose = DictObjHolder({
                "Pitch": pitch,
                "Yaw": yaw,
                "Roll": roll,
                "Height": height,
                })

        # IntrinsicMatrix
        self.IntrinsicMatrix = np.transpose(mtx)

        # Out Image View
        self.OutImageView = DictObjHolder({
                "distAheadOfSensor": distAheadOfSensor,
                "spaceToLeftSide": spaceToLeftSide,
                "spaceToRightSide": spaceToRightSide,
                "bottomOffset": bottomOffset,
                })

        self.OutImageSize = np.array([np.nan, np.int_(imgw/2)])  # image H, image W
        self.left_fit_model = RANSACRegressor(PolynomialRegression(degree=2), residual_threshold=10)
        self.right_fit_model = RANSACRegressor(PolynomialRegression(degree=2), residual_threshold=10)


    def listener_callback(self, data):
        # self.get_logger().info("Received an image! ")
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          cv_image = np.zeros((480,848)).astype(np.uint8)
          print(e)
        
        outdata = self.detect.run(cv_image)
        lanes = [lane.to_array(self.detect.cfg) for lane in outdata['lanes']]
        
        
        
        
        img = outdata['ori_img']
        img = cv2.resize(img, (int(self.detect.x_scale * img.shape[1]), int(self.detect.y_scale * img.shape[0])))
        binaryimg_original = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
        color_idx = 0
        for lane in lanes:
            lane_color = COLOR_MAP[color_idx]
            for x, y in lane:
                if x <= 0 or y <= 0:
                    continue
                x = np.round(x * self.detect.x_scale)
                y = np.round(y * self.detect.y_scale)
                x, y = int(x), int(y)

                cv2.circle(binaryimg_original, (x, y), 3, 255, -1)
                cv2.circle(img, (x, y), 3, lane_color, -1)
            color_idx = color_idx + 1
        # Bird's Eye View Image
   
        warpimage, unwarp_matrix, birdseyeview = bev_perspective(binaryimg_original, mtx, self.CameraPose, self.OutImageView, self.OutImageSize)
        
        imageX, imageY = np.where(warpimage)
        xyBoundaryPoints = birdseyeview.bevimagetovehicle(np.column_stack((imageY,imageX)))

        leftlane = xyBoundaryPoints[xyBoundaryPoints[:,1]>0]
        rightlane = xyBoundaryPoints[xyBoundaryPoints[:,1]<=0]

        left_init_param = np.array([])
        right_init_param = np.array([])
        leftparam = get_fit_param(leftlane, left_init_param, self.left_fit_model)
        rightparam = get_fit_param(rightlane, right_init_param, self.right_fit_model)

        # left_lane_img = insertLaneBoundary(img, warpimage, leftparam, self.OutImageView, birdseyeview)
        # lane_img = insertLaneBoundary(left_lane_img, warpimage, rightparam, self.OutImageView, birdseyeview)
        
        cv2.imshow('view', img)
        cv2.imshow('bev', warpimage)
        cv2.waitKey(1)
        
        left_detection = LaneParams()
        right_detection = LaneParams()
        
        msg_stamp = self.get_clock().now().to_msg()
        
        left_detection.header.stamp = msg_stamp
        left_detection.header.frame_id =='left_lane'
        if leftparam.size == 0:
            left_detection.a = 0.0
            left_detection.b = 0.0
            left_detection.c = 0.0
        else:
            left_detection.a = leftparam[0]
            left_detection.b = leftparam[1]
            left_detection.c = leftparam[2]
        
        right_detection.header.stamp = msg_stamp
        right_detection.header.frame_id =='right_lane'
        if rightparam.size == 0:
            right_detection.a = 0.0
            right_detection.b = 0.0
            right_detection.c = 0.0
        else:
            right_detection.a = rightparam[0]
            right_detection.b = rightparam[1]
            right_detection.c = rightparam[2]
        
        # Publishing the results onto the the lane detection vision_msgs format
        self.leftdetection_publisher.publish(left_detection)
        self.rightdetection_publisher.publish(right_detection)
        
        ros_image = self.bridge.cv2_to_imgmsg(img)
        ros_image.header.frame_id = 'lane_detectimg'
        self.result_publisher.publish(ros_image)
        
        
        

