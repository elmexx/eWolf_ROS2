# ROS2 imports 
import rclpy
from rclpy.node import Node

import numpy as np
import os, sys, cv2, time
import os.path as osp

# CV Bridge and message imports
from sensor_msgs.msg import Image as SensorImage
from sensor_msgs.msg import Imu
from std_msgs.msg import String, Header
from lane_parameter_msg.msg import LaneParams, LaneMarkingProjected, LaneMarkingProjectedArray, LaneMarkingProjectedArrayBoth
from message_filters import Subscriber, TimeSynchronizer
from geometry_msgs.msg import PoseWithCovarianceStamped
#from builtin_interfaces.msg import Time
import cv2
from cv_bridge import CvBridge, CvBridgeError
from tf_transformations import euler_from_quaternion

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

pitch = 3
yaw = 0
roll = 0
height = 1.6

distAheadOfSensor = 30
spaceToLeftSide = 5    
spaceToRightSide = 5
bottomOffset = 1

imgw = 848
imgh = 480
window_size = 10

COLOR_MAP = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (125, 125, 0), (0, 125, 125), (125,0,125)]

def moving_average(array, window_data):
    window_data[:-1] = window_data[1:]
    window_data[-1] = array
    moving_avg = np.mean(window_data, axis=0)
    return moving_avg

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
        
        ## singel subscription
        self.subscription = self.create_subscription(SensorImage, camera_topic, self.listener_callback, 10)
        # self.subscription_camerapose = self.create_subscription(Imu, "/imu/data", self.pose_callback, 10)
        self.subscription  # prevent unused variable warning
        # self.subscription_camerapose
        ### test multi topic subscriber
        # camera_img = Subscriber(self, SensorImage, camera_topic)
        # camera_pose = Subscriber(self, PoseWithCovarianceStamped, "/localization_pose")
        # ts = TimeSynchronizer([camera_img, camera_pose], 10)
        # ts.registerCallback(self.listener_callback)

        self.bridge = CvBridge()

        # Create a Lane detection topic to publish results on
        self.leftdetection_publisher = self.create_publisher(LaneParams, 'leftlanedetection', 10)
        self.rightdetection_publisher = self.create_publisher(LaneParams, 'rightlanedetection', 10)
        self.pub_projected_markings = self.create_publisher(LaneMarkingProjectedArrayBoth, 'lane_markings_left_right_projected', 10)

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
        self.left_window_data = np.zeros((window_size,3))
        self.right_window_data = np.zeros((window_size,3))

    def pose_callback(self, camera_pose):
        # self.get_logger().info("Subscription from /localization_pose")
        try: 
            pitch1, roll1, yaw1  = euler_from_quaternion([camera_pose.orientation.x, camera_pose.orientation.y,
                                                    camera_pose.orientation.z, camera_pose.orientation.w])
            # self.get_logger().info('pitch, yaw, roll: "%f, %f, %f"' % (pitch1, yaw1, roll1))
            self.CameraPose.Pitch = -np.rad2deg(pitch1+np.pi/2)
            self.CameraPose.Yaw = 0
            self.CameraPose.Roll = 0
        except (RuntimeError, TypeError, NameError):
            self.get_logger().info("no camera pose!!! ")

    def listener_callback(self, camera_img):
        # self.get_logger().info("Received an image! ")
        try:
          cv_image = self.bridge.imgmsg_to_cv2(camera_img, "bgr8")
        except CvBridgeError as e:
          cv_image = np.zeros((480,848)).astype(np.uint8)
          print(e)
        # self.get_logger().info('pitch, yaw, roll: "%f, %f, %f"' % (self.CameraPose.Pitch, self.CameraPose.Yaw, self.CameraPose.Roll))
        outdata = self.detect.run(cv_image)
        lanes = [lane.to_array(self.detect.cfg) for lane in outdata['lanes']]
        img = outdata['ori_img'].copy()
        img = cv2.resize(img, (int(self.detect.x_scale * img.shape[1]), int(self.detect.y_scale * img.shape[0])))
        binaryimg_original = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
        color_idx = 0
        lane_num = len(lanes)
        warpimage, unwarp_matrix, birdseyeview = bev_perspective(img, mtx, self.CameraPose, self.OutImageView, self.OutImageSize)
        lanes_list = [[] for _ in range(lane_num)]
        lanes_wc = [[] for _ in range(lane_num)]
        lane_idx = 0

        for lane in lanes:
            lane_color = COLOR_MAP[color_idx]
            for x, y in lane:
                if x <= 0 or y <= 0:
                    continue
                x = np.round(x * self.detect.x_scale)
                y = np.round(y * self.detect.y_scale)
                x, y = int(x), int(y)
                cv2.circle(binaryimg_original, (x, y), 3, 255, -1)
                cv2.circle(img, (x, y), 3, (0,255,0), -1)
                lanes_list[lane_idx].append(np.array([x,y]))
            color_idx = color_idx + 1
            lanes_wc[lane_idx], _ = birdseyeview.imagetovehicle(np.asarray(lanes_list[lane_idx]))
            lane_idx = lane_idx + 1
        # get left and right lanes
        left_lanes = []
        right_lanes = []
        egoleft = np.array([])
        egoright = np.array([])
        if lanes_wc:    #   !!if lanes_wc = [[],[]], there will give an error!
            for lane in lanes_wc:
                lateraloffset = lane[0][1]
                if lateraloffset>=0:
                    left_lanes.append(lane)
                else:
                    right_lanes.append(lane)
        if left_lanes:
            left_lateraloffset = []
            for lane in left_lanes: 
                left_lateraloffset.append(lane[0][1])
            idxmin = np.argmin(left_lateraloffset)
            egoleft = left_lanes[idxmin]
            if egoleft[0][1] > 3:
                egoleft = np.array([])
        if right_lanes:
            right_lateraloffset = []
            for lane in right_lanes: 
                right_lateraloffset.append(lane[0][1])
            idxmax = np.argmax(right_lateraloffset)
            egoright = right_lanes[idxmax]
            if egoright[0][1] < -3:
                egoright = np.array([])

        if egoleft.size==0 and egoright.size!=0:
            egoleft = egoright.copy()
            egoleft[:,1] = egoleft[:,1]+3.5
        if egoleft.size!=0 and egoright.size==0:
            egoright = egoleft.copy()
            egoright[:,1] = egoright[:,1]-3.5

        left_init_param = np.array([])
        right_init_param = np.array([])
        leftparam = get_fit_param(egoleft, left_init_param, self.left_fit_model)
        rightparam = get_fit_param(egoright, right_init_param, self.right_fit_model)

        if leftparam.size > 0:
            leftparam = moving_average(leftparam, self.left_window_data)
            self.left_window_data = np.vstack((self.left_window_data[1:], leftparam))
        if rightparam.size > 0:
            rightparam = moving_average(rightparam, self.right_window_data)
            self.right_window_data = np.vstack((self.right_window_data[1:], rightparam))

        left_lane_img = insertLaneBoundary(img, warpimage, leftparam, self.OutImageView, birdseyeview, (0,0,255))
        lane_img = insertLaneBoundary(left_lane_img, warpimage, rightparam, self.OutImageView, birdseyeview, (255,0,0))
            
        # imageX, imageY = np.where(warpimage)
        # xyBoundaryPoints = birdseyeview.bevimagetovehicle(np.column_stack((imageY,imageX)))

        # leftlane = xyBoundaryPoints[xyBoundaryPoints[:,1]>0]
        # rightlane = xyBoundaryPoints[xyBoundaryPoints[:,1]<=0]

        # left_init_param = np.array([])
        # right_init_param = np.array([])
        # leftparam = get_fit_param(leftlane, left_init_param, self.left_fit_model)
        # rightparam = get_fit_param(rightlane, right_init_param, self.right_fit_model)

        # left_lane_img = insertLaneBoundary(img, warpimage, leftparam, self.OutImageView, birdseyeview)
        # lane_img = insertLaneBoundary(left_lane_img, warpimage, rightparam, self.OutImageView, birdseyeview)
        
        cv2.imshow('view', lane_img)
        cv2.imshow('bev', warpimage)
        cv2.waitKey(1)
        # leftparam = np.array([])
        # rightparam = np.array([])
        left_detection = LaneParams()
        right_detection = LaneParams()
        combined_lanes = LaneMarkingProjectedArrayBoth()
        # lmp = LaneMarkingProjected()
        
        msg_stamp = self.get_clock().now().to_msg()
        
        left_detection.header.stamp = msg_stamp
        left_detection.header.frame_id =='left_lane'
        if leftparam.size == 0:
            left_detection.a = 0.0
            left_detection.b = 0.0
            left_detection.c = 0.0
            # lmp.x = 0.0
            # lmp.y = 0.0
            # lmp.z = 0.0
            # combined_lanes.markings_left.append(lmp)
        else:
            left_detection.a = leftparam[0]
            left_detection.b = leftparam[1]
            left_detection.c = leftparam[2]
            for lane in egoleft:
                lmp = LaneMarkingProjected()
                lmp.x = lane[0]
                lmp.y = lane[1]
                lmp.z = 0.0
                combined_lanes.markings_left.append(lmp)
        
        right_detection.header.stamp = msg_stamp
        right_detection.header.frame_id =='right_lane'
        if rightparam.size == 0:
            right_detection.a = 0.0
            right_detection.b = 0.0
            right_detection.c = 0.0
            # lmp.x = 0.0
            # lmp.y = 0.0
            # lmp.z = 0.0
            # combined_lanes.markings_right.append(lmp)
        else:
            right_detection.a = rightparam[0]
            right_detection.b = rightparam[1]
            right_detection.c = rightparam[2]
            for lane in egoright:
                lmp = LaneMarkingProjected()
                lmp.x = lane[0]
                lmp.y = lane[1]
                lmp.z = 0.0
                combined_lanes.markings_right.append(lmp)
        
        # Publishing the results onto the the lane detection vision_msgs format
        self.leftdetection_publisher.publish(left_detection)
        self.rightdetection_publisher.publish(right_detection)
        self.pub_projected_markings.publish(combined_lanes)

        ros_image = self.bridge.cv2_to_imgmsg(img)
        ros_image.header.frame_id = 'lane_detectimg'
        self.result_publisher.publish(ros_image)
        
        
        

