# ROS2 imports 
import rclpy
from rclpy.node import Node

import numpy as np
import os, sys, cv2, time
import pickle as pkl
import socket
from pykalman import KalmanFilter

# CV Bridge and message imports
from sensor_msgs.msg import Image as SensorImage
from std_msgs.msg import String
from lane_parameter_msg.msg import LaneParams
import cv2
from cv_bridge import CvBridge, CvBridgeError

# Torch import 
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Lane Detection imports
from .unet import UNet
from .utils.data_vis import plot_img_and_mask
from .utils.dataset import BasicDataset

from .utils.lane_parameter import DictObjHolder, bev_perspective, lanefit, drawlane
from .utils.lanecluster import lane_mask_coords

mtx = np.array([[1.121200e+03, 0.000000e+00, 9.634416e+02],
       [0.000000e+00, 1.131900e+03, 5.268709e+02],
       [0.000000e+00, 0.000000e+00, 1.000000e+00]])

dist = np.array([-0.3051, 0.0771, 0, 0, 0])

pitch = -1
yaw = -3
roll = 0
height = 1

distAheadOfSensor = 20
spaceToLeftSide = 4    
spaceToRightSide = 4
bottomOffset = 3

imgw = 1920
imgh = 1080


def predict_img(net,full_img,device,scale_factor=1,out_threshold=0.5):

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)            
        # probs = torch.sigmoid(output)
        probs = probs.squeeze(0)        
        tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ])
        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold
    
class LaneDetector(Node):

    def __init__(self):
        super().__init__('lanedetection_node')

        self.declare_parameter("camera_topic", "image_raw")
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

        self.model = UNet(n_channels=3, n_classes=1)
        
        # Weights and labels locations
        self.model_path = os.getenv("HOME")+ '/ros2_ws/src/lane_detection_package/lane_detection_package/model/laneunet.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
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
        
        ###### Lane Kalmanfilter
        
        L_init_lane_parameter = np.array([8.59772640e-07,  2.72346164e-02,  2.04014669e+02])
        R_init_lane_parameter = np.array([6.64430504e-06, -1.32049055e-02,  6.38566362e+02])


        L_KF_init_parameter = np.array([L_init_lane_parameter[0],0,
                                        L_init_lane_parameter[1],0,
                                        L_init_lane_parameter[2],0])
        R_KF_init_parameter = np.array([R_init_lane_parameter[0],0,
                                        R_init_lane_parameter[1],0,
                                        R_init_lane_parameter[2],0])
        transition_matrix = np.eye(6)
        transition_matrix[0,1] = 0.2
        transition_matrix[2,3] = 0.2
        transition_matrix[4,5] = 0.2
        
        observation_matrix = np.zeros((3,6))
        observation_matrix[0,0] = 1
        observation_matrix[1,2] = 1
        observation_matrix[2,4] = 1
        
        state_covariance = 1*np.eye(6)
        observation_covariance = 25*np.eye(3)
        self.C_stateleft = L_KF_init_parameter
        self.C_stateright = R_KF_init_parameter
        self.P_stateleft = state_covariance
        self.P_stateright = state_covariance
        
        self.L_kflane = KalmanFilter(transition_matrices = transition_matrix,
                              observation_matrices = observation_matrix,
                              transition_covariance  = state_covariance,
                              observation_covariance = observation_covariance,
                              initial_state_mean = L_KF_init_parameter)
        self.R_kflane = KalmanFilter(transition_matrices = transition_matrix,
                              observation_matrices = observation_matrix,
                              transition_covariance  = state_covariance,
                              observation_covariance = observation_covariance,
                              initial_state_mean = R_KF_init_parameter)
        
        
    def listener_callback(self, data):
        # self.get_logger().info("Received an image! ")
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          cv_image = np.ones((256,512)).astype(np.uint8)
          print(e)

        img_pil = Image.fromarray(cv_image)
        mask = predict_img(net=self.model,full_img=img_pil,scale_factor=1,
                            out_threshold=0.2,device=self.device)
        binaryimg = (mask*255).astype(np.uint8)
        # binaryimg_original = cv2.resize(binaryimg, (cv_image.shape[1],cv_image[0]))
        binaryimg_original = cv2.resize(binaryimg, (imgw,imgh))
        lanemask, lane_coords = lane_mask_coords(binaryimg)
        fit_params = []
        for i in range(len(lane_coords)):
            nonzero_y = lane_coords[i][:,0]
            nonzero_x = lane_coords[i][:,1]        
            fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
            fit_params.append(fit_param)
            
        # Bird's Eye View Image
        birdseyeviewimage, unwarp_matrix, birdseyeview = bev_perspective(cv_image, self.IntrinsicMatrix, self.CameraPose, self.OutImageView, self.OutImageSize)
        # lane parameter in vehicle coordination
        lane_fit_params = lanefit(binaryimg_original,self.IntrinsicMatrix,self.CameraPose, self.OutImageView, self.OutImageSize)
        ego_right_lane = lane_fit_params['ego_right_lane']
        ego_left_lane = lane_fit_params['ego_left_lane']
        # Tracking Lane with Kalman Filter
        if ego_left_lane is not None:
            (self.C_stateleft, self.P_stateleft) = self.L_kflane.filter_update(filtered_state_mean = self.C_stateleft,
                                                              filtered_state_covariance = self.P_stateleft,
                                                              observation = ego_left_lane)
            
            ego_left_lane = np.array([self.C_stateleft[0], self.C_stateleft[2], self.C_stateleft[4]])
        else:
            measureleft = np.array([self.C_stateleft[0], self.C_stateleft[2], self.C_stateleft[4]])
            (self.C_stateleft, self.P_stateleft) = self.L_kflane.filter_update(filtered_state_mean = self.C_stateleft,
                                                              filtered_state_covariance = self.P_stateleft,
                                                              observation = measureleft)
            ego_left_lane = np.array([self.C_stateleft[0], self.C_stateleft[2], self.C_stateleft[4]])
        if ego_right_lane is not None:
            (self.C_stateright, self.P_stateright) = self.R_kflane.filter_update(filtered_state_mean = self.C_stateright,
                                                                  filtered_state_covariance = self.P_stateright,
                                                                  observation = ego_right_lane)
            
            ego_right_lane = np.array([self.C_stateright[0], self.C_stateright[2], self.C_stateright[4]])
        else:
            measureright = np.array([self.C_stateright[0], self.C_stateright[2], self.C_stateright[4]])
            (self.C_stateright, self.P_stateright) = self.R_kflane.filter_update(filtered_state_mean = self.C_stateright,
                                                                  filtered_state_covariance = self.P_stateright,
                                                                  observation = measureright)
            ego_right_lane = np.array([self.C_stateright[0], self.C_stateright[2], self.C_stateright[4]])
    
        mask_img = drawlane(binaryimg_original, birdseyeviewimage, unwarp_matrix, ego_right_lane, ego_left_lane)
        g_b = np.zeros_like(mask_img).astype(np.uint8)
        add_img = cv2.merge((mask_img,g_b, g_b))
        out_image = cv2.addWeighted(cv2.resize(cv_image,(imgw, imgh)),1,add_img,1,0.0)
        
        left_detection = LaneParams()
        right_detection = LaneParams()
        
        left_detection.a = ego_left_lane[0]
        left_detection.b = ego_left_lane[1]
        left_detection.c = ego_left_lane[2]
        
        right_detection.a = ego_right_lane[0]
        right_detection.b = ego_right_lane[1]
        right_detection.c = ego_right_lane[2]
        
        cv2.imshow('outimage', cv2.resize(out_image,(512,256)))
        
        cv2.waitKey(1)
        
        # Publishing the results onto the the Detection2DArray vision_msgs format
        self.leftdetection_publisher.publish(left_detection)
        self.rightdetection_publisher.publish(right_detection)
        ros_image = self.bridge.cv2_to_imgmsg(cv_image)
        ros_image.header.frame_id = 'binaryimg_frame'
        self.result_publisher.publish(ros_image)
        
        
        

